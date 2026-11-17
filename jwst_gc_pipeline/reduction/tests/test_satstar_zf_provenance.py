"""The satstar products record how the ZEROFRAME fit anchor treated the frame (#681).

A saturated star is fitted either on the group-0 deep core with the rim
rewritten (mode ``rim``), on the deep core with nothing to rewrite
(``keepfinite``), or with the whole any-group SATURATED blob masked
(``blob``).  The treatments differ by 0.5-3.6 mag on marginally saturated
stars, and which one applies depends on whether a sibling ``_ramp.fits``
yields a usable group-0 scale.  The anchor now reports the outcome, and
``get_saturated_stars`` / ``remove_saturated_stars`` write it into per-row
columns and ``SATZF*`` keywords.  The fit itself is unchanged.
"""
import builtins
import os

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

import jwst_gc_pipeline.reduction.saturated_star_finding as SSF
from jwst_gc_pipeline.reduction.saturated_star_finding import (
    stamp_zeroframe_provenance, zeroframe_default_provenance,
    zeroframe_fit_anchor, zeroframe_reconcile_provenance,
    zeroframe_row_provenance)

SATBIT = 2


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in [n for n in os.environ if 'SATSTAR' in n]:
        monkeypatch.delenv(name, raising=False)


def _scene(rail=48000.0, r_true=0.17, calibrators=True, rim_g0=20000.0):
    """Dark frame, one saturated star at (60, 60): deep core at the group-0
    rail, a rim at ``rim_g0``, and (optionally) the bright unsaturated pixels
    the R estimator needs."""
    ny = nx = 120
    g0 = np.random.default_rng(7).normal(10, 3, (ny, nx))
    data = g0 * r_true + np.random.default_rng(8).normal(0, 0.05, (ny, nx))
    dq = np.zeros((ny, nx), dtype=np.uint32)
    if calibrators:
        vals = np.random.default_rng(9).uniform(21000, 40000, (8, 8))
        g0[100:108, 8:16] = vals
        data[100:108, 8:16] = vals * r_true
    yy, xx = np.mgrid[0:ny, 0:nx]
    rr = np.hypot(xx - 60, yy - 60)
    core = rr < 1.5
    rim = (rr >= 1.5) & (rr < 4)
    g0[core] = rail
    g0[rim] = rim_g0
    dq[core | rim] = SATBIT
    data[core] = np.nan
    data[rim] = rim_g0 * r_true * 1.15
    return data, dq, g0, core, rim


# --------------------------------------------------------------------------
# the anchor's provenance
# --------------------------------------------------------------------------

def test_rim_recovered_mode():
    data, dq, g0, core, rim = _scene()
    prov = {}
    out, deep, rim_mask, _ = zeroframe_fit_anchor(data, dq, g0,
                                                  provenance=prov)
    assert prov['mode'] == 'rim'
    assert prov['n_rim'] == int(rim_mask.sum()) > 0
    assert prov['n_deep'] == int(deep.sum()) > 0
    assert np.isfinite(prov['R'])


def test_no_usable_r_is_blob_mode():
    data, dq, g0, core, rim = _scene(calibrators=False)
    prov = {}
    _, deep, rim_mask, _ = zeroframe_fit_anchor(data, dq, g0, provenance=prov)
    assert deep is None
    assert prov['mode'] == 'blob'
    assert not np.isfinite(prov['R'])
    assert prov['n_rim'] == 0 and prov['n_deep'] == 0


def test_keep_finite_with_empty_rim_is_keepfinite_mode(monkeypatch):
    """Every SATURATED pixel below the ceiling keeps its ramp-fit rate, so
    nothing is rewritten, but the deep-core mask is still the fit mask."""
    monkeypatch.setenv('SATSTAR_ZF_KEEP_FINITE', '1')
    data, dq, g0, core, rim = _scene()
    prov = {}
    out, deep, rim_mask, delta = zeroframe_fit_anchor(data, dq, g0,
                                                      provenance=prov)
    assert not rim_mask.any() and deep is not None
    assert prov['mode'] == 'keepfinite'
    assert prov['n_rim'] == 0
    assert prov['n_deep'] == int(deep.sum()) > 0
    assert np.isfinite(prov['R'])


def test_finite_r_empty_rim_keep_finite_off_is_blob_with_r():
    """Nothing recoverable and KEEP_FINITE off: the blob is masked, and R is
    still recorded."""
    data, dq, g0, core, rim = _scene(rim_g0=48000.0)
    prov = {}
    _, deep, rim_mask, _ = zeroframe_fit_anchor(data, dq, g0, provenance=prov)
    assert deep is None and not rim_mask.any()
    assert prov['mode'] == 'blob'
    assert np.isfinite(prov['R'])


def test_return_values_do_not_depend_on_provenance():
    """The wrapper contract: same 4-tuple with or without the out-parameter,
    so existing callers and the fit are unaffected."""
    data, dq, g0, core, rim = _scene()
    a = zeroframe_fit_anchor(data, dq, g0)
    b = zeroframe_fit_anchor(data, dq, g0, provenance={})
    assert len(a) == len(b) == 4
    assert np.array_equal(a[0], b[0], equal_nan=True)
    assert np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2])
    assert a[3] is None and b[3] is None


# --------------------------------------------------------------------------
# per-row columns
# --------------------------------------------------------------------------

def _two_components():
    sources = np.zeros((40, 40), dtype=int)
    sources[5:10, 5:10] = 1
    sources[25:30, 25:30] = 2
    rim = np.zeros((40, 40), bool)
    rim[6, 6] = rim[6, 7] = True          # inside component 1 only
    return sources, rim


def test_row_flags_follow_the_component():
    sources, rim = _two_components()
    prov = {'mode': 'rim', 'n_rim': 2, 'n_deep': 4, 'R': 0.17}
    assert zeroframe_row_provenance(prov, rim, sources, 1) == (1, 1, 2)
    assert zeroframe_row_provenance(prov, rim, sources, 2) == (1, 0, 0)


def test_row_flags_with_a_component_slice_match_the_full_frame():
    sources, rim = _two_components()
    prov = {'mode': 'rim', 'n_rim': 2, 'n_deep': 4, 'R': 0.17}
    sl1 = (slice(5, 10), slice(5, 10))
    sl2 = (slice(25, 30), slice(25, 30))
    assert zeroframe_row_provenance(prov, rim, sources, 1, sl1) == (1, 1, 2)
    assert zeroframe_row_provenance(prov, rim, sources, 2, sl2) == (1, 0, 0)


def test_one_rim_pixel_is_rim_in_core():
    sources, rim = _two_components()
    rim[:] = False
    rim[6, 6] = True
    prov = {'mode': 'rim', 'n_rim': 1, 'n_deep': 4, 'R': 0.17}
    assert zeroframe_row_provenance(prov, rim, sources, 1) == (1, 1, 1)


def test_rim_pixels_of_another_component_do_not_count_with_a_slice():
    """The slice is the bounding box of the component, which can hold pixels of
    a neighbouring component; only ``sources == src_label`` counts."""
    sources, rim = _two_components()
    sources[7:9, 7:9] = 2                  # neighbour inside component 1's box
    rim[7, 7] = True
    prov = {'mode': 'rim', 'n_rim': 3, 'n_deep': 4, 'R': 0.17}
    sl1 = (slice(5, 10), slice(5, 10))
    assert zeroframe_row_provenance(prov, rim, sources, 1, sl1) == (1, 1, 2)
    assert zeroframe_row_provenance(prov, rim, sources, 2, sl1)[2] == 1


def test_blob_rows_read_zero():
    sources, rim = _two_components()
    rim[:] = False
    prov = zeroframe_default_provenance()
    assert zeroframe_row_provenance(prov, rim, sources, 1) == (0, 0, 0)
    # no zeroframe at all: no rim array
    assert zeroframe_row_provenance(prov, None, sources, 1) == (0, 0, 0)


def test_keepfinite_rows_are_flagged_but_have_no_rim_in_core():
    sources, rim = _two_components()
    rim[:] = False
    prov = {'mode': 'keepfinite', 'n_rim': 0, 'n_deep': 9, 'R': 0.17}
    assert zeroframe_row_provenance(prov, rim, sources, 1) == (1, 0, 0)


def test_forced_seed_has_no_component():
    sources, rim = _two_components()
    prov = {'mode': 'rim', 'n_rim': 2, 'n_deep': 4, 'R': 0.17}
    assert zeroframe_row_provenance(prov, rim, sources, None) == (1, 0, -1)
    blob = zeroframe_default_provenance()
    assert zeroframe_row_provenance(blob, None, sources, None) == (0, 0, -1)


# --------------------------------------------------------------------------
# the fit mask, not the anchor's report, decides the treatment
# --------------------------------------------------------------------------

def test_reconcile_deep_core_with_default_prov_is_a_deep_core_fit():
    deep = np.zeros((6, 6), bool)
    deep[2:4, 2:4] = True
    rim = np.zeros((6, 6), bool)
    rim[1, 1] = rim[1, 2] = True
    prov = zeroframe_reconcile_provenance(zeroframe_default_provenance(),
                                          deep, rim)
    assert prov['mode'] == 'rim' and prov['n_rim'] == 2 and prov['n_deep'] == 4
    prov = zeroframe_reconcile_provenance(zeroframe_default_provenance(),
                                          deep, np.zeros_like(rim))
    assert prov['mode'] == 'keepfinite' and prov['n_rim'] == 0
    prov = zeroframe_reconcile_provenance(zeroframe_default_provenance(),
                                          deep, None)
    assert prov['mode'] == 'keepfinite'


def test_reconcile_no_deep_core_is_blob_and_keeps_r():
    rim = np.ones((4, 4), bool)
    prov = {'mode': 'rim', 'n_rim': 16, 'n_deep': 3, 'R': 0.2}
    zeroframe_reconcile_provenance(prov, None, rim)
    assert prov == {'mode': 'blob', 'n_rim': 0, 'n_deep': 0, 'R': 0.2}


def test_reconcile_leaves_an_agreeing_report_alone():
    deep = np.ones((3, 3), bool)
    prov = {'mode': 'keepfinite', 'n_rim': 0, 'n_deep': 9, 'R': 0.2}
    before = dict(prov)
    zeroframe_reconcile_provenance(prov, deep, np.zeros((3, 3), bool))
    assert prov == before


def test_get_saturated_stars_stamps_from_the_mask_when_the_anchor_is_silent(
        monkeypatch):
    """An anchor that returns a deep core but never fills ``provenance`` still
    gives ``zf_fit = 1``: the mask is what the fit used."""
    n = 200
    sci = np.ones((n, n))
    dq = np.zeros((n, n), dtype=np.uint32)
    dq[98:103, 98:103] = SATBIT
    sci[dq != 0] = np.nan
    wcs_hdr = fits.Header({'CTYPE1': 'RA---TAN', 'CTYPE2': 'DEC--TAN',
                           'CRPIX1': 100, 'CRPIX2': 100, 'CRVAL1': 150.0,
                           'CRVAL2': 2.0, 'CDELT1': -1.7e-5, 'CDELT2': 1.7e-5,
                           'BUNIT': 'MJy/sr'})
    fh = fits.HDUList([
        fits.PrimaryHDU(header=fits.Header({
            'INSTRUME': 'NIRCAM', 'FILTER': 'F150W', 'PUPIL': 'CLEAR',
            'DETECTOR': 'NRCB3', 'MODULE': 'B', 'CHANNEL': 'SHORT'})),
        fits.ImageHDU(sci, header=wcs_hdr, name='SCI'),
        fits.ImageHDU(np.full((n, n), 0.1), name='ERR'),
        fits.ImageHDU(dq, name='DQ'),
        fits.ImageHDU(np.where(dq != 0, np.nan, 0.01), name='VAR_POISSON')])
    deep = np.zeros((n, n), bool)
    deep[99:102, 99:102] = True
    rim = (dq != 0) & ~deep

    def _silent_anchor(data, dq, zeroframe, provenance=None, **kw):
        return data, deep, rim, None

    real_switches = SSF.satstar_fit_switches

    def _stop(*a, **kw):
        # first read after the anchor (and after the stamp) ends the frame
        if hasattr(builtins, 'satstar_zf_provenance'):
            raise _AnchorReached
        return real_switches(*a, **kw)
    monkeypatch.setattr(SSF, 'zeroframe_fit_anchor', _silent_anchor)
    monkeypatch.setattr(SSF, 'satstar_fit_switches', _stop)
    monkeypatch.delattr(builtins, 'satstar_zf_provenance', raising=False)
    with pytest.raises(_AnchorReached):
        SSF.get_saturated_stars(fh, zeroframe=np.ones((n, n)), plot=False)
    prov = builtins.satstar_zf_provenance
    assert prov['mode'] == 'rim'
    assert prov['n_rim'] == int(rim.sum()) and prov['n_deep'] == 9
    hdr = fits.Header()
    stamp_zeroframe_provenance(hdr, prov)
    assert hdr['SATZFFIT'] is True
    assert zeroframe_row_provenance(prov, rim, np.ones((n, n), int), 1)[0] == 1


# --------------------------------------------------------------------------
# header stamps
# --------------------------------------------------------------------------

def test_stamp_rim_mode_on_a_header():
    hdr = fits.Header()
    prov = {'mode': 'rim', 'n_rim': 37, 'n_deep': 9, 'R': 0.1712}
    stamp_zeroframe_provenance(hdr, prov)
    assert hdr['SATZFFIT'] is True
    assert hdr['SATZFMOD'] == 'rim'
    assert hdr['SATZFRIM'] == 37 and hdr['SATZFDEP'] == 9
    assert hdr['SATZFR'] == pytest.approx(0.1712)


def test_stamp_blob_mode_omits_r_when_unusable():
    hdr = fits.Header()
    stamp_zeroframe_provenance(hdr, zeroframe_default_provenance())
    assert hdr['SATZFFIT'] is False
    assert hdr['SATZFMOD'] == 'blob'
    assert hdr['SATZFRIM'] == 0
    assert 'SATZFR' not in hdr


def test_stamp_blob_mode_keeps_a_finite_r():
    hdr = fits.Header()
    stamp_zeroframe_provenance(
        hdr, {'mode': 'blob', 'n_rim': 0, 'n_deep': 0, 'R': 0.2})
    assert hdr['SATZFFIT'] is False and hdr['SATZFR'] == pytest.approx(0.2)


# --------------------------------------------------------------------------
# get_saturated_stars and remove_saturated_stars plumbing
# --------------------------------------------------------------------------

class _AnchorReached(Exception):
    pass


def test_get_saturated_stars_passes_a_provenance_dict_to_the_anchor(
        monkeypatch):
    """``get_saturated_stars`` hands the anchor a dict to fill, which is the
    only way the group-0 scale R leaves the anchor."""
    n = 200
    sci = np.ones((n, n))
    dq = np.zeros((n, n), dtype=np.uint32)
    dq[98:103, 98:103] = SATBIT
    sci[dq != 0] = np.nan
    wcs_hdr = fits.Header({'CTYPE1': 'RA---TAN', 'CTYPE2': 'DEC--TAN',
                           'CRPIX1': 100, 'CRPIX2': 100, 'CRVAL1': 150.0,
                           'CRVAL2': 2.0, 'CDELT1': -1.7e-5, 'CDELT2': 1.7e-5,
                           'BUNIT': 'MJy/sr'})
    fh = fits.HDUList([
        fits.PrimaryHDU(header=fits.Header({
            'INSTRUME': 'NIRCAM', 'FILTER': 'F150W', 'PUPIL': 'CLEAR',
            'DETECTOR': 'NRCB3', 'MODULE': 'B', 'CHANNEL': 'SHORT'})),
        fits.ImageHDU(sci, header=wcs_hdr, name='SCI'),
        fits.ImageHDU(np.full((n, n), 0.1), name='ERR'),
        fits.ImageHDU(dq, name='DQ'),
        fits.ImageHDU(np.where(dq != 0, np.nan, 0.01), name='VAR_POISSON')])
    filled = {'mode': 'rim', 'n_rim': 5, 'n_deep': 3, 'R': 0.25}
    seen = {}

    def _anchor(data, dq, zeroframe, provenance=None, **kw):
        provenance.update(filled)
        seen['prov'] = provenance
        raise _AnchorReached
    monkeypatch.setattr(SSF, 'zeroframe_fit_anchor', _anchor)
    with pytest.raises(_AnchorReached):
        SSF.get_saturated_stars(fh, zeroframe=np.ones((n, n)), plot=False)
    assert seen['prov'] == filled


@pytest.mark.parametrize('prov, fit, mode', [
    ({'mode': 'rim', 'n_rim': 12, 'n_deep': 4, 'R': 0.2}, True, 'rim'),
    ({'mode': 'blob', 'n_rim': 0, 'n_deep': 0, 'R': float('nan')}, False,
     'blob'),
    (None, False, 'blob')])
def test_remove_saturated_stars_stamps_catalog_and_images(
        tmp_path, monkeypatch, prov, fit, mode):
    fn = tmp_path / 'jw03523005001_10101_00001_nrcb3_align_o005_crf.fits'
    fits.HDUList([fits.PrimaryHDU(header=fits.Header(
                      {'INSTRUME': 'NIRCAM', 'DETECTOR': 'NRCB3',
                       'FILTER': 'F150W', 'CRPIX1': 1})),
                  fits.ImageHDU(np.ones((8, 8), 'float32'), name='SCI'),
                  fits.ImageHDU(np.zeros((8, 8), 'int32'), name='DQ')]
                 ).writeto(fn)
    for attr in ('satstar_wingcal_measurements', 'satstar_rejected',
                 'satstar_resid', 'satstar_flagimg', 'satstar_zf_provenance'):
        monkeypatch.delattr(builtins, attr, raising=False)

    def _fit(fh, **kw):
        if prov is not None:
            builtins.satstar_zf_provenance = dict(prov)
        builtins.satstar_model = np.zeros((8, 8))
        builtins.satstar_resid = np.zeros((8, 8))
        return Table({'flux_fit': [1.0]})
    monkeypatch.setattr(SSF, 'get_saturated_stars', _fit)
    SSF.remove_saturated_stars(str(fn), recovery_signature='off')
    stem = str(fn)[:-len('.fits')]
    cat = Table.read(f'{stem}_satstar_catalog.fits')
    assert bool(cat.meta['SATZFFIT']) is fit
    assert cat.meta['SATZFMOD'] == mode
    for product in ('_satstar_model.fits', '_satstar_residual.fits'):
        hdr = fits.getheader(f'{stem}{product}')
        assert bool(hdr['SATZFFIT']) is fit
        assert hdr['SATZFMOD'] == mode
        assert ('SATZFR' in hdr) == (prov is not None
                                     and bool(np.isfinite(prov['R'])))


def test_deblend_stats_are_stamped_only_when_the_deblend_ran():
    """#1196: DBLNCMP / DBLNSPL / DBLFAIL / DBLNVET record the deblend counts,
    as bare integers in a table's meta and as (value, comment) cards in a
    FITS header."""
    from jwst_gc_pipeline.reduction.saturated_star_finding import (
        stamp_deblend_stats)
    keys = ('DBLNCMP', 'DBLNSPL', 'DBLFAIL', 'DBLNVET')
    meta = {}
    stamp_deblend_stats(meta, None)
    assert meta == {}
    stamp_deblend_stats(meta, {'n_components': 12, 'n_split': 3,
                               'n_failed': 1, 'n_vetoed': 7,
                               'psf_veto': {'applied': 12}})
    assert [meta[k] for k in keys] == [12, 3, 1, 7]
    assert all(type(meta[k]) is int for k in keys)
    meta = {}
    stamp_deblend_stats(meta, {})
    assert [meta[k] for k in keys] == [0, 0, 0, 0]
    hdr = fits.Header()
    stamp_deblend_stats(hdr, {'n_components': 12, 'n_vetoed': 7})
    assert (hdr['DBLNCMP'], hdr['DBLNVET']) == (12, 7)
    assert hdr.comments['DBLNVET']


def test_remove_saturated_stars_resets_a_stale_provenance(tmp_path,
                                                          monkeypatch):
    """A provenance left by a previous frame must not leak into this one when
    the fit never sets it."""
    fn = tmp_path / 'jw03523005001_10101_00001_nrcb3_align_o005_crf.fits'
    fits.HDUList([fits.PrimaryHDU(header=fits.Header(
                      {'INSTRUME': 'NIRCAM', 'DETECTOR': 'NRCB3',
                       'FILTER': 'F150W', 'CRPIX1': 1})),
                  fits.ImageHDU(np.ones((8, 8), 'float32'), name='SCI'),
                  fits.ImageHDU(np.zeros((8, 8), 'int32'), name='DQ')]
                 ).writeto(fn)
    for attr in ('satstar_wingcal_measurements', 'satstar_rejected',
                 'satstar_resid', 'satstar_flagimg'):
        monkeypatch.delattr(builtins, attr, raising=False)
    monkeypatch.setattr(builtins, 'satstar_zf_provenance',
                        {'mode': 'rim', 'n_rim': 9, 'n_deep': 9, 'R': 0.3},
                        raising=False)

    def _fit(fh, **kw):
        builtins.satstar_model = np.zeros((8, 8))
        builtins.satstar_resid = np.zeros((8, 8))
        return Table({'flux_fit': [1.0]})
    monkeypatch.setattr(SSF, 'get_saturated_stars', _fit)
    SSF.remove_saturated_stars(str(fn), recovery_signature='off')
    cat = Table.read(str(fn)[:-len('.fits')] + '_satstar_catalog.fits')
    assert bool(cat.meta['SATZFFIT']) is False
    assert cat.meta['SATZFMOD'] == 'blob'
