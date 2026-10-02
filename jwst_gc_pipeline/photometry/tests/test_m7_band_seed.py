"""m7 seed of one filter = cross-band seed UNION that filter's m6 vetted catalog
(opt-in, ``manual_m7_seed_own_band``).

The >=2-filter cross-band seed alone leaves out every source only ONE band's
m6 vetting accepted and m7 does not find again (a third of the m6 vetted
catalog in Brick F182M).  Own-band sources near a brighter seed source are not
added (companion cut).
"""
import os
import types

import numpy as np
import pytest
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from jwst_gc_pipeline.photometry.cataloging import (
    _build_i2d_augmented_seed, _build_m7_band_seed, annotate_independent_detection,
    crossband_seed_file, m7_band_seed_path)
from jwst_gc_pipeline.photometry.naming import vetted_to_i2dseed
from jwst_gc_pipeline.photometry.manual_defaults import MANUAL_DEFAULTS

RA0, DEC0 = 266.5, -28.8


def _sc(dx_mas, dy_mas):
    dx = np.asarray(dx_mas, float) / 3.6e6 / np.cos(np.deg2rad(DEC0))
    dy = np.asarray(dy_mas, float) / 3.6e6
    return SkyCoord((RA0 + dx) * u.deg, (DEC0 + dy) * u.deg)


def _write(tmp_path):
    # cross-band seed: three positions, no flux column (as _build_crossband_seed writes it)
    xb = Table({'skycoord': _sc([0, 1000, 2000], [0, 0, 0]),
                'n_filt_confirmed': [2, 2, 3]})
    xpath = os.path.join(tmp_path, 'crossband_seed_manual.fits')
    xb.write(xpath)
    # own-band m6 vetted: a 10 mas neighbour of seed 0, a 25 mas neighbour of
    # seed 1, two single-band sources, and one non-positive flux row
    own = Table({'skycoord': _sc([10, 1025, 5000, 6000, 7000], [0, 0, 0, 0, 0]),
                 'flux': [50.0, 70.0, 3.0, 4.0, -1.0]})
    opath = os.path.join(tmp_path, 'own_m6.fits')
    own.write(opath)
    return xpath, opath


def test_union_and_flux_transfer(tmp_path):
    xpath, opath = _write(str(tmp_path))
    out = Table.read(_build_m7_band_seed(xpath, opath, 'F182M', 'merged',
                                         max_sep_mas=30.0))
    origin = np.asarray(out['seed_origin']).astype(str)
    assert list(origin) == ['crossband'] * 3 + ['own_m6'] * 2
    # cross-band positions take the matched own-band flux, unmatched -> 1.0
    assert list(out['flux'][:3]) == [50.0, 70.0, 1.0]
    # the single-band sources are added with their m6 flux; the negative-flux
    # row and the two matched rows are not duplicated
    assert sorted(out['flux'][3:]) == [3.0, 4.0]
    sc = out['skycoord']
    assert np.allclose(sc[:3].ra.deg, _sc([0, 1000, 2000], [0, 0, 0]).ra.deg)


def test_max_sep_controls_dedup(tmp_path):
    xpath, opath = _write(str(tmp_path))
    out = Table.read(_build_m7_band_seed(xpath, opath, 'F182M', 'merged',
                                         max_sep_mas=20.0))
    # the 25 mas neighbour is now a separate own-band source
    assert len(out) == 6
    assert list(out['flux'][:3]) == [50.0, 1.0, 1.0]


def test_output_path_per_module_and_filter(tmp_path):
    xpath, opath = _write(str(tmp_path))
    pa = _build_m7_band_seed(xpath, opath, 'F182M', 'nrca')
    pb = _build_m7_band_seed(xpath, opath, 'F182M', 'nrcb')
    pc = _build_m7_band_seed(xpath, opath, 'F212N', 'nrca')
    assert len({pa, pb, pc}) == 3
    assert pa.endswith('crossband_seed_manual_nrca_f182m_vetted.fits')
    assert os.path.exists(xpath)            # the shared seed is not rewritten


def test_empty_own_catalog_still_writes_band_file(tmp_path):
    """The band seed is never the shared cross-band seed itself: the m7 caller
    adds the residual detections to it, and _build_i2d_augmented_seed writes
    next to a ``_vetted`` input (over a suffix-less one)."""
    xpath, _ = _write(str(tmp_path))
    before = Table.read(xpath)
    epath = os.path.join(str(tmp_path), 'empty.fits')
    Table({'skycoord': _sc([], []), 'flux': np.zeros(0)}).write(epath)
    out = _build_m7_band_seed(xpath, epath, 'F182M', 'merged')
    assert out != xpath
    assert out == m7_band_seed_path(xpath, 'merged', 'F182M')
    assert vetted_to_i2dseed(out) != out
    t = Table.read(out)
    assert list(np.asarray(t['seed_origin']).astype(str)) == ['crossband'] * 3
    assert list(t['flux']) == [1.0, 1.0, 1.0]
    after = Table.read(xpath)
    assert after.colnames == before.colnames and len(after) == len(before)


def test_i2d_augmented_seed_refuses_to_overwrite_its_input(tmp_path):
    xpath, _ = _write(str(tmp_path))
    with pytest.raises(ValueError, match='overwrite'):
        _build_i2d_augmented_seed('unused_i2d.fits', xpath, 'F182M')


def _write_i2d(path, stars, shape=(80, 80), sigma=0.9, noise=1.0, seed=0):
    """SCI/ERR/WHT co-add at 0.063"/px with Gaussian stars ``(x, y, peak)``."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    sci = rng.normal(0.0, noise, shape)
    for x, y, peak in stars:
        sci += peak * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crval = [RA0, DEC0]
    w.wcs.crpix = [shape[1] / 2, shape[0] / 2]
    w.wcs.cdelt = [-0.063 / 3600, 0.063 / 3600]
    hdr = w.to_header()
    fits.HDUList([fits.PrimaryHDU(),
                  fits.ImageHDU(sci, hdr, name='SCI'),
                  fits.ImageHDU(np.full(shape, noise), hdr, name='ERR'),
                  fits.ImageHDU(np.ones(shape), hdr, name='WHT')]).writeto(path)
    return w


def test_i2d_augmented_seed_marks_residual_detections(tmp_path):
    """Seed rows keep their origin; this round's residual detections are 'i2d',
    each at least DEDUPMAS from every earlier seed."""
    i2d = os.path.join(str(tmp_path), 'resid_i2d.fits')
    w = _write_i2d(i2d, [(20, 20, 200.0), (55, 30, 150.0), (40, 60, 120.0)])
    sky = w.pixel_to_world([20.0], [20.0])
    prev = Table({'skycoord': sky, 'flux': [500.0], 'seed_origin': ['own_m6']})
    ppath = os.path.join(str(tmp_path), 'band_seed_vetted.fits')
    prev.write(ppath)
    out = Table.read(_build_i2d_augmented_seed(i2d, ppath, 'F182M'))
    origin = list(np.asarray(out['seed_origin']).astype(str))
    assert origin == ['own_m6', 'i2d', 'i2d']
    assert out['flux'][0] == 500.0
    dedup = out.meta['DEDUPMAS']
    assert 60 < dedup < 200
    sc = out['skycoord']
    assert sc[1:].separation(sc[0]).to_value(u.mas).min() > dedup

    # a vetted catalog without seed_origin (m3..m6) reads as 'prev'
    del prev['seed_origin']
    prev.write(ppath, overwrite=True)
    out = Table.read(_build_i2d_augmented_seed(i2d, ppath, 'F182M'))
    assert list(np.asarray(out['seed_origin']).astype(str)) == ['prev', 'i2d', 'i2d']


def _annot_opts(**kw):
    o = types.SimpleNamespace(desaturated=False, bgsub=False, blur=False,
                              proposal_id='4147', field='012', modules='merged')
    for k, v in kw.items():
        setattr(o, k, v)
    return o


def test_annotate_counts_m7_residual_detections(tmp_path):
    """A source only the m7 residual daofind of one band found is an
    independent detection in that band (review of #1015: it was flagged False
    in the one band that detected it)."""
    cut_bp = str(tmp_path)
    os.makedirs(f'{cut_bp}/catalogs')
    opts = _annot_opts(manual_m7_seed_own_band=True)
    # m6 vetted F182M: source A only
    Table({'skycoord': _sc([0], [0]), 'flux': [100.0]}).write(
        f'{cut_bp}/catalogs/f182m_merged_indivexp_merged_resbgsub_m6_dao_basic_vetted.fits')
    # m7 F182M seed: A (own_m6), C (cross-band only), B (this band's residual)
    xb = crossband_seed_file(cut_bp, opts)
    seed = Table({'skycoord': _sc([0, 2000, 1000], [0, 0, 0]),
                  'flux': [100.0, 1.0, 5.0],
                  'seed_origin': ['own_m6', 'crossband', 'i2d']})
    seed.meta['DEDUPMAS'] = 63.0
    seed.write(vetted_to_i2dseed(m7_band_seed_path(xb, 'merged', 'F182M')))
    # merged catalog: A; B whose fit moved 45 mas from its residual centroid
    # (beyond the 30 mas m6 radius, inside DEDUPMAS); C
    mp = f'{cut_bp}/catalogs/merged_test.fits'
    Table({'skycoord_ref': _sc([5, 1045, 2000], [0, 0, 0])}).write(mp)

    annotate_independent_detection(mp, cut_bp, ['F182M'], opts)
    assert list(Table.read(mp)['independently_detected_f182m']) == [True, True, False]

    # own-band seed off: a band seed left by an earlier run is not read
    annotate_independent_detection(mp, cut_bp, ['F182M'],
                                   _annot_opts(manual_m7_seed_own_band=False))
    assert list(Table.read(mp)['independently_detected_f182m']) == [True, False, False]


def test_companion_of_brighter_seed_not_added(tmp_path):
    """Own-band sources within companion_fwhm FWHM of a brighter seed source
    are m6 fits in its PSF-mismatch ring (confirmed by an independent visit at
    the chance rate in Brick F182M; docs/evidence/faint_m7_seed_union) and are
    not added.  A fainter neighbour does not exclude a brighter source."""
    xpath = os.path.join(str(tmp_path), 'crossband_seed_manual.fits')
    Table({'skycoord': _sc([0], [0]), 'n_filt_confirmed': [2]}).write(xpath)
    opath = os.path.join(str(tmp_path), 'own_m6.fits')
    # the cross-band star itself (5 mas), faint sources 100 and 300 mas from
    # it, an own-band-only star at 3000 mas and a faint source 120 mas from it
    Table({'skycoord': _sc([5, 100, 300, 3000, 3120], [0] * 5),
           'flux': [100.0, 5.0, 6.0, 80.0, 4.0]}).write(opath)
    # 2.5 x 0.062" = 155 mas: the 100 mas and 3120 mas sources are left out
    out = Table.read(_build_m7_band_seed(xpath, opath, 'F182M', 'merged',
                                         companion_fwhm=2.5, fwhm_arcsec=0.062))
    assert list(out['flux']) == [100.0, 6.0, 80.0]
    assert out.meta['NCOMPAN'] == 2
    # the FWHM defaults to the filter's FWHM-table entry (F182M: 0.062")
    out_t = Table.read(_build_m7_band_seed(xpath, opath, 'F182M', 'merged',
                                           companion_fwhm=2.5))
    assert list(out_t['flux']) == [100.0, 6.0, 80.0]
    # companion_fwhm=0 disables the cut
    out0 = Table.read(_build_m7_band_seed(xpath, opath, 'F182M', 'merged',
                                          companion_fwhm=0))
    assert list(out0['flux']) == [100.0, 5.0, 6.0, 80.0, 4.0]
    assert out0.meta['NCOMPAN'] == 0


def test_pipeline_defaults():
    # opt-in; see docs/evidence/faint_m7_seed_union for the full-field realness
    assert MANUAL_DEFAULTS['manual_m7_seed_own_band'] is False
    assert MANUAL_DEFAULTS['manual_m7_seed_own_band_companion_fwhm'] == 2.5
