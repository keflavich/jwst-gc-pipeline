"""A later phase must ADOPT an earlier phase's satstar fit, not recompute it.

The satstar fit runs once per cataloging phase (m12, m3, m4, m5, m6, m7) and
every phase redoes identical work: ``remove_saturated_stars`` reads the ORIGINAL
``_crf`` SCI array, which no phase writes, and recomputes its starting
parameters from the frame's DQ, so nothing can carry a prior solution in.
Verified on disk for brick F182M ``jw02221001001_07101_00001_nrca2`` (235 rows):
the six per-phase catalogs agree to maxdiff 0.000e+00 on ``x_init``, ``y_init``,
``flux_init`` AND ``x_fit``, ``y_fit``, ``flux_fit``.  The cache misses only
because its FILENAME carries the phase token.

Campaign scale: 44,633 of 57,619 per-exposure satstar catalogs (77%) are m3-m7
repeats, ~3,000-3,900 core-hours.  A reuse costs 0.030 s against a 124.4 s fit.

These tests pin the MECHANISM -- that the key is computed from the fit's INPUTS
and that a matching donor skips the fitter -- and the CORRECTNESS -- that what
the adopting phase gets is bit-identical to the donor, which is the property the
measured warm-start alternative could not provide (it moved the photometry by a
median 0.55 mmag / max 7.99 mmag against a cold-vs-cold baseline of exactly
0.000).
"""
import os
import shutil

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table

import jwst_gc_pipeline.photometry.crowdsource_catalogs_long as CL
from jwst_gc_pipeline.photometry import satstar_cache as SC

FRAME = 'jw02221001001_07101_00001_nrca2_destreak_o001_crf.fits'


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Start from a known switch state: the feature OFF and no keyed env set."""
    for name in [n for n in os.environ if 'SATSTAR' in n]:
        monkeypatch.delenv(name, raising=False)


def _frame(tmp_path, name=FRAME, sci=None, dq=None):
    """A stand-in cal/crf: primary header with the PSF-naming keywords, plus the
    two arrays the fit reads."""
    sci = np.arange(64, dtype='float32').reshape(8, 8) if sci is None else sci
    dq = np.zeros((8, 8), dtype='int32') if dq is None else dq
    hdr = fits.Header({'INSTRUME': 'NIRCAM', 'DETECTOR': 'NRCA2',
                       'FILTER': 'F182M'})
    fits.HDUList([fits.PrimaryHDU(header=hdr),
                  fits.ImageHDU(data=sci, name='SCI'),
                  fits.ImageHDU(data=dq, name='DQ')]).writeto(
                      tmp_path / name, overwrite=True)
    return str(tmp_path / name)


def _key(fn, tmp_path, **kw):
    kw.setdefault('path_prefix', str(tmp_path / 'psfs'))
    return SC.satstar_content_key(fn, **kw)


def _donor(fn, suffix, key, nrows=3, with_model=True):
    """Write a phase's satstar products: the catalog (stamped with ``key``) and
    the model image the next phase would otherwise refit to get."""
    stem = fn[:-len('.fits')]
    cat = f'{stem}{suffix}_satstar_catalog.fits'
    t = Table({'x_fit': np.linspace(1.0, 2.0, nrows),
               'y_fit': np.linspace(3.0, 4.0, nrows),
               'flux_fit': np.linspace(1e4, 2e4, nrows)})
    t.write(cat, overwrite=True)
    if key is not None:
        SC.stamp_content_key(cat, key)
    if with_model:
        fits.PrimaryHDU(data=np.full((8, 8), 7.5)).writeto(
            f'{stem}{suffix}_satstar_model.fits', overwrite=True)
    return cat


def _no_fit(monkeypatch):
    """Make the fitter a tripwire: reuse must not reach it."""
    def _boom(*a, **kw):
        raise AssertionError('remove_saturated_stars ran; the donor fit was '
                             'recomputed instead of adopted')
    monkeypatch.setattr(CL, 'remove_saturated_stars', _boom)


def _counting_fit(monkeypatch, nrows=1):
    calls = {'n': 0}

    def _stub(filename, overwrite=True, file_suffix='', **kw):
        calls['n'] += 1
        stem = filename[:-len('.fits')]
        Table({'x_fit': np.zeros(nrows), 'y_fit': np.zeros(nrows),
               'flux_fit': np.zeros(nrows)}).write(
                   f'{stem}{file_suffix}_satstar_catalog.fits', overwrite=True)
    monkeypatch.setattr(CL, 'remove_saturated_stars', _stub)
    return calls


# --------------------------------------------------------------------------
# MECHANISM: the key is computed from the fit's inputs
# --------------------------------------------------------------------------

def test_the_key_is_stable_for_unchanged_inputs(tmp_path):
    fn = _frame(tmp_path)
    assert _key(fn, tmp_path) == _key(fn, tmp_path)


def test_one_changed_SCI_pixel_changes_the_key(tmp_path):
    fn = _frame(tmp_path)
    before = _key(fn, tmp_path)
    sci = np.arange(64, dtype='float32').reshape(8, 8)
    sci[3, 3] += 1.0
    after = _key(_frame(tmp_path, sci=sci), tmp_path)
    assert after != before, "the key ignored the data the fit is run on"


def test_a_changed_DQ_changes_the_key(tmp_path):
    """The saturated mask -- and so the seed list and every starting
    parameter -- comes from DQ, not from SCI."""
    before = _key(_frame(tmp_path), tmp_path)
    dq = np.zeros((8, 8), dtype='int32')
    dq[2, 2] = 2
    assert _key(_frame(tmp_path, dq=dq), tmp_path) != before


def test_the_key_does_not_depend_on_the_phase_token(tmp_path):
    """The whole point: m12 and m5 differ ONLY in the output filename, so the
    key -- which takes no suffix at all -- must be the same for both."""
    fn = _frame(tmp_path)
    assert 'file_suffix' not in SC.satstar_content_key.__code__.co_varnames
    key = _key(fn, tmp_path)
    _donor(fn, '_m12', key)
    assert SC.find_reusable_satstar_catalog(fn, '_m5', key) is not None


def test_a_keyed_env_switch_changes_the_key(tmp_path, monkeypatch):
    """``cataloging`` sets NIRCAM_SATSTAR_LOCK_POS per run, and it changes the
    fit: with it on, x_0/y_0 are fixed and only the flux is free."""
    fn = _frame(tmp_path)
    off = _key(fn, tmp_path)
    monkeypatch.setenv('NIRCAM_SATSTAR_LOCK_POS', '1')
    on = _key(fn, tmp_path)
    assert on != off
    monkeypatch.setenv('NIRCAM_SATSTAR_LOCK_POS', '0')
    assert _key(fn, tmp_path) not in (on, off)


def test_a_future_SATSTAR_switch_is_keyed_without_editing_the_list(tmp_path,
                                                                  monkeypatch):
    fn = _frame(tmp_path)
    before = _key(fn, tmp_path)
    monkeypatch.setenv('NIRCAM_SATSTAR_SOMETHING_NEW', '3')
    assert _key(fn, tmp_path) != before


def test_an_unrelated_env_var_does_not_change_the_key(tmp_path, monkeypatch):
    """Otherwise the key would churn on SLURM_JOB_ID and never hit."""
    fn = _frame(tmp_path)
    before = _key(fn, tmp_path)
    monkeypatch.setenv('SLURM_JOB_ID', '12345678')
    monkeypatch.setenv('HOSTNAME', 'c0907a-s11')
    assert _key(fn, tmp_path) == before


def test_partner_band_seeds_change_the_key(tmp_path):
    """The one input that genuinely differs between phases: the partner-band
    satstar catalogs do not exist at m12 and do at m3+, so m12's fit saw fewer
    seeds and its products must NOT be adopted by a phase that has them."""
    fn = _frame(tmp_path)
    none_key = _key(fn, tmp_path, partner_sky=None)
    sky = SkyCoord([266.5, 266.6], [-28.7, -28.8], unit='deg')
    with_key = _key(fn, tmp_path, partner_sky=sky)
    assert with_key != none_key
    moved = SkyCoord([266.5, 266.61], [-28.7, -28.8], unit='deg')
    assert _key(fn, tmp_path, partner_sky=moved) != with_key


def test_the_psf_grid_is_keyed(tmp_path):
    """A rebuilt PSF grid is a different fit even on identical pixels."""
    fn = _frame(tmp_path)
    psfs = tmp_path / 'psfs'
    psfs.mkdir()
    grid = psfs / 'nircam_nrca2_f182m_fovp512_samp2_npsf16.fits'
    fits.PrimaryHDU(data=np.zeros((4, 4))).writeto(grid)
    before = _key(fn, tmp_path)
    fits.PrimaryHDU(data=np.ones((4, 4))).writeto(grid, overwrite=True)
    os.utime(grid, (0, 0))
    assert _key(fn, tmp_path) != before


@pytest.mark.parametrize('kw', [
    {'recovery_signature': 'zf1_ramp0_dil3'},
    {'deblend_with_zeroframe': True},
    {'forced_grid_search_radius': 0},
    {'oversub_clamp_percentile': 25.0},
    {'use_merged_psf_for_merged': True},
    {'outside_star_fit_box': 256},
    {'flux_overrides': {'a': 1.0}},
    {'flux_drops': ['a']},
    {'outside_star_pixels': [(10.0, 20.0)]},
    {'seed_gate_image': np.ones((4, 4))},
])
def test_every_fit_shaping_option_is_keyed(tmp_path, kw):
    fn = _frame(tmp_path)
    assert _key(fn, tmp_path, **kw) != _key(fn, tmp_path)


def test_override_maps_key_the_same_whatever_their_insertion_order(tmp_path):
    fn = _frame(tmp_path)
    a = _key(fn, tmp_path, flux_overrides={'x': 1.0, 'y': 2.0})
    b = _key(fn, tmp_path, flux_overrides={'y': 2.0, 'x': 1.0})
    assert a == b


# --------------------------------------------------------------------------
# MECHANISM: a matching donor skips the fitter
# --------------------------------------------------------------------------

def test_m5_adopts_m12s_fit_instead_of_rerunning_it(tmp_path, monkeypatch):
    """The saving.  With the tripwire fitter installed, returning at all proves
    no fit ran."""
    monkeypatch.setenv(SC.CROSS_PHASE_CACHE_ENV, '1')
    fn = _frame(tmp_path)
    _donor(fn, '_m12', _key(fn, tmp_path))
    _no_fit(monkeypatch)
    out = CL.load_or_make_satstar_catalog(fn, path_prefix=str(tmp_path / 'psfs'))
    assert out is not None and len(out) == 3


def test_the_feature_is_OFF_by_default(tmp_path, monkeypatch):
    """Lands during an observing window: an m12 donor sitting on disk must
    change nothing until the switch is thrown."""
    assert SC.CROSS_PHASE_CACHE_ENV not in os.environ
    fn = _frame(tmp_path)
    _donor(fn, '_m12', _key(fn, tmp_path))
    calls = _counting_fit(monkeypatch)
    CL.load_or_make_satstar_catalog(fn, path_prefix=str(tmp_path / 'psfs'))
    assert calls['n'] == 1, "reuse happened with the switch unset"


def test_a_donor_whose_key_differs_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv(SC.CROSS_PHASE_CACHE_ENV, '1')
    fn = _frame(tmp_path)
    _donor(fn, '_m12', 'a-key-from-different-inputs')
    calls = _counting_fit(monkeypatch)
    CL.load_or_make_satstar_catalog(fn, path_prefix=str(tmp_path / 'psfs'))
    assert calls['n'] == 1


def test_an_unstamped_legacy_catalog_is_never_adopted(tmp_path, monkeypatch):
    """Nothing is known about the inputs a pre-key catalog was fitted from."""
    monkeypatch.setenv(SC.CROSS_PHASE_CACHE_ENV, '1')
    fn = _frame(tmp_path)
    _donor(fn, '_m12', None)
    calls = _counting_fit(monkeypatch)
    CL.load_or_make_satstar_catalog(fn, path_prefix=str(tmp_path / 'psfs'))
    assert calls['n'] == 1


def test_the_phases_own_catalog_still_wins(tmp_path, monkeypatch):
    """The existing same-suffix cache is untouched: when this phase already has
    its own catalog it is served, donor or no donor."""
    monkeypatch.setenv(SC.CROSS_PHASE_CACHE_ENV, '1')
    fn = _frame(tmp_path)
    _donor(fn, '_m12', _key(fn, tmp_path), nrows=3)
    _donor(fn, '_m5', _key(fn, tmp_path), nrows=7)
    _no_fit(monkeypatch)
    out = CL.load_or_make_satstar_catalog(fn, path_prefix=str(tmp_path / 'psfs'),
                                          file_suffix='_m5')
    assert len(out) == 7


def test_a_fresh_fit_is_stamped_so_the_next_phase_can_adopt_it(tmp_path,
                                                              monkeypatch):
    monkeypatch.setenv(SC.CROSS_PHASE_CACHE_ENV, '1')
    fn = _frame(tmp_path)
    _counting_fit(monkeypatch, nrows=4)
    CL.load_or_make_satstar_catalog(fn, path_prefix=str(tmp_path / 'psfs'),
                                    file_suffix='_m12')
    stem = fn[:-len('.fits')]
    stamped = SC.read_content_key(f'{stem}_m12_satstar_catalog.fits')
    assert stamped == _key(fn, tmp_path)
    _no_fit(monkeypatch)
    out = CL.load_or_make_satstar_catalog(fn, path_prefix=str(tmp_path / 'psfs'),
                                          file_suffix='_m5')
    assert len(out) == 4


# --------------------------------------------------------------------------
# CORRECTNESS: what the adopting phase gets is the donor, byte for byte
# --------------------------------------------------------------------------

def test_the_adopted_catalog_is_bit_identical_to_the_donor(tmp_path, monkeypatch):
    """The tolerance is EXACTLY zero, and it is met by construction because the
    file is copied.  Zero is the right bar: the cold fit is reproducible to
    0.000e+00 across phases and processes, so any nonzero difference would be
    caused by the reuse alone.  (The warm-start alternative missed a 1e-4
    relative-flux tolerance by an order of magnitude at the median.)"""
    monkeypatch.setenv(SC.CROSS_PHASE_CACHE_ENV, '1')
    fn = _frame(tmp_path)
    donor_path = _donor(fn, '_m12', _key(fn, tmp_path))
    donor = Table.read(donor_path)
    _no_fit(monkeypatch)
    out = CL.load_or_make_satstar_catalog(fn, path_prefix=str(tmp_path / 'psfs'),
                                          file_suffix='_m5')
    for col in ('x_fit', 'y_fit', 'flux_fit'):
        assert np.max(np.abs(np.asarray(out[col]) - np.asarray(donor[col]))) == 0.0


def test_the_adopted_model_image_the_phase_subtracts_is_materialized(tmp_path,
                                                                    monkeypatch):
    """``_prepare_frame_for_photometry`` reads ``<frame><suffix>_satstar_model``
    back and subtracts it.  A reuse that produced only the catalog would leave
    the phase with no model and silently stop subtracting the saturated stars."""
    monkeypatch.setenv(SC.CROSS_PHASE_CACHE_ENV, '1')
    fn = _frame(tmp_path)
    _donor(fn, '_m12', _key(fn, tmp_path))
    _no_fit(monkeypatch)
    CL.load_or_make_satstar_catalog(fn, path_prefix=str(tmp_path / 'psfs'),
                                    file_suffix='_m5')
    stem = fn[:-len('.fits')]
    src = f'{stem}_m12_satstar_model.fits'
    dst = f'{stem}_m5_satstar_model.fits'
    assert os.path.exists(dst)
    assert open(src, 'rb').read() == open(dst, 'rb').read()


def test_adopt_copies_every_product_the_donor_wrote(tmp_path):
    fn = _frame(tmp_path)
    stem = fn[:-len('.fits')]
    _donor(fn, '_m12', 'k')
    for extra in ('_satstar_residual.fits', '_satstar_rejected.fits',
                  '_satstar_flags.fits'):
        shutil.copyfile(f'{stem}_m12_satstar_model.fits', f'{stem}_m12{extra}')
    written = SC.adopt_satstar_products(f'{stem}_m12_satstar_catalog.fits',
                                        fn, '_m5')
    assert written[0] == f'{stem}_m5_satstar_catalog.fits'
    assert set(os.path.basename(p) for p in written) == {
        os.path.basename(f'{stem}_m5{s}') for s in
        ('_satstar_catalog.fits', '_satstar_model.fits', '_satstar_residual.fits',
         '_satstar_rejected.fits', '_satstar_flags.fits')}


def test_an_extended_donor_is_preferred_and_keeps_its_family(tmp_path):
    """``force_union_satstar`` writes ``_extended_satstar_catalog``; the loader
    already prefers it over the plain one, and the reuse must not cross the two
    families."""
    fn = _frame(tmp_path)
    stem = fn[:-len('.fits')]
    _donor(fn, '_m12', 'k')
    ext = f'{stem}_m12_extended_satstar_catalog.fits'
    Table({'x_fit': [9.0], 'y_fit': [9.0], 'flux_fit': [9.0]}).write(ext)
    SC.stamp_content_key(ext, 'k')
    found = SC.find_reusable_satstar_catalog(fn, '_m5', 'k')
    assert found == ext
    written = SC.adopt_satstar_products(found, fn, '_m5')
    assert written[0] == f'{stem}_m5_extended_satstar_catalog.fits'
