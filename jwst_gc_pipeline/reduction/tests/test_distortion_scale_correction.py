"""Unit tests for the per-detector F200W plate-scale correction
(distortion_scale_correction.py).

Hermetic: the synthetic GWCS and ImageModel builders are shared with the
rotation-correction tests.  The test on a real NIRCam crf runs only where one
is available (DSCL_TEST_CRF env var) and is skipped otherwise.
"""
import os
import shutil

import numpy as np
import pytest
from astropy.io import fits
from astropy.modeling import models as M
from astropy.table import Table

from jwst_gc_pipeline.reduction import distortion_rotation_correction as drc
from jwst_gc_pipeline.reduction import distortion_scale_correction as dsc
from jwst_gc_pipeline.reduction.tests.test_distortion_rotation_correction import (
    CEN, V2REF, V3REF, _anchor_model, _grid, _gwcs, _sep_mas, _sky, _synthetic_frame)

DETS = ['NRCA1', 'NRCA2', 'NRCA3', 'NRCA4', 'NRCB1', 'NRCB2', 'NRCB3', 'NRCB4']


def _scale_v2v3(model, ppm):
    c2, c3 = model(*CEN)
    return model | dsc.scale_about_pivot_model(c2, c3, ppm)


def _table(rows):
    return Table(rows=rows, names=dsc.TABLE_COLUMNS)


def _row(det='NRCA3', filt='F200W', pupil='CLEAR', ref='jwst_nircam_distortion_0284.asdf',
         corr=-21.4):
    return (det, filt, pupil, ref, corr, -corr, -corr)


# --- scale math and sign -----------------------------------------------------

def test_positive_ppm_moves_points_away_from_pivot():
    m = dsc.scale_about_pivot_model(V2REF, V3REF, 100.0)
    v2, v3 = m(V2REF + 10.0, V3REF - 20.0)
    assert v2 == pytest.approx(V2REF + 10.0 * (1 + 1e-4), abs=1e-12)
    assert v3 == pytest.approx(V3REF - 20.0 * (1 + 1e-4), abs=1e-12)


def test_correction_restores_anchor_and_wrong_sign_doubles():
    anchor = _anchor_model()
    s = 34.0                                              # measured band - anchor, ppm
    band = _scale_v2v3(anchor, s)
    x, y = _grid()
    a2, a3 = anchor(x, y)

    # stored correction is minus the measured scale; to first order in ppm
    fixed, _ = dsc.scaled_wcs(_gwcs(band), -s)
    f2, f3 = fixed.get_transform('detector', 'v2v3')(x, y)
    assert np.hypot(f2 - a2, f3 - a3).max() * 1e3 < 1e-4     # mas; (s*1e-6)^2 * lever

    b2, b3 = band(x, y)
    wrong, _ = dsc.scaled_wcs(_gwcs(band), +s)
    w2, w3 = wrong.get_transform('detector', 'v2v3')(x, y)
    err_before = np.hypot(b2 - a2, b3 - a3)
    err_wrong = np.hypot(w2 - a2, w3 - a3)
    good = err_before > 1e-6
    assert np.allclose(err_wrong[good] / err_before[good], 2.0, rtol=1e-4)


def test_reference_rotation_reads_back_the_scale():
    # the similarity fit used for the rotation table sees the same ppm
    anchor = _anchor_model()
    _, s, rms = drc.reference_rotation(_scale_v2v3(anchor, -21.4), anchor)
    assert s == pytest.approx(-21.4, abs=1e-3) and rms < 1e-6


def test_commutes_with_rotation_correction():
    w = _gwcs(_anchor_model())
    rs, _ = dsc.scaled_wcs(drc.rotated_wcs(w, -30.855)[0], -21.4)
    sr, _ = drc.rotated_wcs(dsc.scaled_wcs(w, -21.4)[0], -30.855)
    x, y = _grid()
    a = rs.get_transform('detector', 'v2v3')(x, y)
    b = sr.get_transform('detector', 'v2v3')(x, y)
    assert np.hypot(a[0] - b[0], a[1] - b[1]).max() * 1e3 < 1e-6


# --- pivot, subarrays, bounding box, inverse ---------------------------------

def test_detector_centre_is_fixed_point_and_corner_moves_radially():
    w = _gwcs(_anchor_model())
    out, pivot = dsc.scaled_wcs(w, -34.0)
    c0 = w.get_transform('detector', 'v2v3')(*CEN)
    c1 = out.get_transform('detector', 'v2v3')(*CEN)
    assert np.allclose(c0, pivot, atol=1e-12) and np.allclose(c1, c0, atol=1e-10)
    x0 = np.array(w.get_transform('detector', 'v2v3')(0.0, 0.0))
    x1 = np.array(out.get_transform('detector', 'v2v3')(0.0, 0.0))
    d0, d1 = x0 - np.array(c0), x1 - np.array(c0)
    assert np.hypot(*d1) / np.hypot(*d0) == pytest.approx(1 - 34e-6, abs=1e-12)
    assert abs(d0[0] * d1[1] - d0[1] * d1[0]) < 1e-12        # radial, no rotation


def test_subarray_pivot_is_full_frame_centre():
    xs, ys = 993, 1121
    full = _anchor_model()
    sub = (M.Shift(xs - 1) & M.Shift(ys - 1)) | full
    out_full, pv_full = dsc.scaled_wcs(_gwcs(full), 13.5)
    out_sub, pv_sub = dsc.scaled_wcs(_gwcs(sub, bbox=((-0.5, 63.5), (-0.5, 63.5))), 13.5,
                                     xstart=xs, ystart=ys)
    assert np.allclose(pv_sub, pv_full, atol=1e-10)
    x, y = _grid(5, 0, 63)
    a = out_sub.get_transform('detector', 'v2v3')(x, y)
    b = out_full.get_transform('detector', 'v2v3')(x + xs - 1, y + ys - 1)
    assert np.allclose(a, b, atol=1e-10)


def test_bounding_box_and_inverse_survive():
    w = _gwcs(_anchor_model())
    out, _ = dsc.scaled_wcs(w, -34.0)
    assert tuple(map(tuple, out.bounding_box)) == tuple(map(tuple, w.bounding_box))
    t = out.get_transform('detector', 'v2v3')
    x, y = _grid()
    xr, yr = t.inverse(*t(x, y))
    assert np.allclose(xr, x, atol=1e-8) and np.allclose(yr, y, atol=1e-8)
    v = w.get_transform('detector', 'v2v3')(x, y)
    assert np.allclose(v, _anchor_model()(x, y), atol=1e-12)


def test_verify_scale_rejects_other_transform():
    w = _gwcs(_anchor_model())
    out, pivot = dsc.scaled_wcs(w, -34.0)
    assert dsc.verify_scale(w, out, pivot, -34.0, (2048, 2048)) < 1e-6
    with pytest.raises(dsc.DistortionScaleError, match="verification failed"):
        dsc.verify_scale(w, out, pivot, +34.0, (2048, 2048))
    rot, pivot = drc.rotated_wcs(w, -1.0)
    with pytest.raises(dsc.DistortionScaleError, match="verification failed"):
        dsc.verify_scale(w, rot, pivot, 0.0, (2048, 2048))


# --- table lookup ------------------------------------------------------------

def test_lookup_none_without_rows():
    tbl = _table([_row()])
    assert dsc.lookup_correction_ppm('NRCA3', 'F150W', 'CLEAR', 'x.asdf', table=tbl) is None
    assert dsc.lookup_correction_ppm('NRCA1', 'F200W', 'CLEAR', 'x.asdf', table=tbl) is None


def test_lookup_match_strips_crds_prefix_and_case():
    tbl = _table([_row()])
    assert dsc.lookup_correction_ppm(
        'nrca3', 'f200w', None, 'crds://jwst_nircam_distortion_0284.asdf',
        table=tbl) == pytest.approx(-21.4)


def test_lookup_refuses_other_reference():
    tbl = _table([_row()])
    with pytest.raises(dsc.DistortionScaleError, match="re-measure"):
        dsc.lookup_correction_ppm('NRCA3', 'F200W', 'CLEAR',
                                  'jwst_nircam_distortion_9999.asdf', table=tbl)


def test_shipped_table_sanity():
    tbl = dsc.load_scale_table()
    assert set(tbl['filter']) == {'F200W'} and set(tbl['pupil']) == {'CLEAR'}
    assert sorted(tbl['detector']) == DETS
    # the scale was measured on the F200W references of the rotation table
    rot = drc.load_rotation_table()
    refs = {r['detector']: r['distortion_ref'] for r in rot
            if r['filter'] == 'F200W' and r['pupil'] == 'CLEAR'}
    for r in tbl:
        assert r['distortion_ref'] == refs[r['detector']]
        assert r['correction_ppm'] == pytest.approx(-(r['wd2_ppm'] + r['wd1_ppm']) / 2,
                                                    abs=0.051)
    assert np.all(np.abs(tbl['correction_ppm']) < 50)
    assert np.sqrt(np.mean((tbl['wd2_ppm'] - tbl['wd1_ppm']) ** 2)) < 5
    a = tbl[np.char.startswith(np.asarray(tbl['detector'], str), 'NRCA')]
    b = tbl[np.char.startswith(np.asarray(tbl['detector'], str), 'NRCB')]
    assert np.all(a['correction_ppm'] < 0) and np.all(b['correction_ppm'] > 0)


# --- preconditions and the apply path ----------------------------------------

def test_preconditions():
    assert dsc.check_apply_preconditions(fits.Header()) == 'go'
    assert dsc.check_apply_preconditions(fits.Header({dsc.MARKER: True})) == 'skip'
    with pytest.raises(dsc.DistortionScaleError, match="pending"):
        dsc.check_apply_preconditions(fits.Header({dsc.PENDING: True}))
    with pytest.raises(dsc.DistortionScaleError, match="RAOFFSET"):
        dsc.check_apply_preconditions(fits.Header({'RAOFFSET': 1e-5}))
    # a rotation-corrected frame is fine
    assert dsc.check_apply_preconditions(fits.Header({drc.MARKER: True})) == 'go'


def test_enabled_switch(monkeypatch):
    monkeypatch.delenv('DISTORTION_SCALE_CORRECTION', raising=False)
    assert not dsc.correction_enabled()
    monkeypatch.setenv('DISTORTION_SCALE_CORRECTION', '1')
    assert dsc.correction_enabled()
    monkeypatch.setenv('DISTORTION_SCALE_CORRECTION', 'true')
    assert not dsc.correction_enabled()


def _frame(path, **kw):
    kw.setdefault('filt', 'F200W')
    kw.setdefault('ref', 'crds://jwst_nircam_distortion_0284.asdf')
    return _synthetic_frame(path, **kw)


def test_apply_end_to_end(tmp_path):
    from jwst.datamodels import ImageModel
    tbl = _table([_row(corr=-500.0)])                    # large, so the shift is measurable
    fn = _frame(tmp_path / "syn_crf.fits")
    cx, cy = CEN[0] - 992, CEN[1] - 992
    s0, c0 = _sky(fn), _sky(fn, [cx], [cy])
    grid = np.mgrid[0:64:8, 0:64:8][::-1]
    v0 = ImageModel(fn).meta.wcs.get_transform('detector', 'v2v3')(*grid)

    assert dsc.apply_distortion_scale_correction(fn, table=tbl, verbose=False) == -500.0
    s1, c1 = _sky(fn), _sky(fn, [cx], [cy])
    assert _sep_mas(c1, c0).max() < 1e-4                # pivot fixed on the sky
    assert _sep_mas(s1, s0).min() > 1e-3                # the rest moved
    v1 = ImageModel(fn).meta.wcs.get_transform('detector', 'v2v3')(*grid)
    h = fits.getheader(fn, ('SCI', 1))
    exp = dsc.scale_about_pivot_model(h['DSCLPV2'], h['DSCLPV3'], -500.0)(*v0)
    assert np.hypot(v1[0] - exp[0], v1[1] - exp[1]).max() * 1e3 < 1e-6
    assert h[dsc.MARKER] is True and h[dsc.PENDING] is False
    assert h['DSCLPPM'] == -500.0 and h['DSCLREF'] == 'jwst_nircam_distortion_0284.asdf'
    assert h['SIPGWMAX'] < 0.05
    from astropy.wcs import WCS
    s = WCS(h).pixel_to_world_values(np.array([5.0, 50.0]), np.array([7.0, 40.0]))
    g = _sky(fn, [5.0, 50.0], [7.0, 40.0])
    assert np.allclose(s[0], g[0], atol=1e-7) and np.allclose(s[1], g[1], atol=1e-7)

    # idempotent
    assert dsc.apply_distortion_scale_correction(fn, table=tbl, verbose=False) is None
    s2 = _sky(fn)
    assert np.array_equal(s2[0], s1[0]) and np.array_equal(s2[1], s1[1])


def test_apply_after_rotation_correction(tmp_path):
    from jwst.datamodels import ImageModel
    rtbl = Table(rows=[('NRCA3', 'F200W', 'CLEAR', 'jwst_nircam_distortion_0284.asdf', 'F212N',
                        'jwst_nircam_distortion_0182.asdf', -30.855, 41.3, 1.0)],
                 names=drc.TABLE_COLUMNS)
    stbl = _table([_row(corr=-500.0)])
    fn = _frame(tmp_path / "syn_crf.fits")
    grid = np.mgrid[0:64:8, 0:64:8][::-1]
    v0 = ImageModel(fn).meta.wcs.get_transform('detector', 'v2v3')(*grid)
    drc.apply_distortion_rotation_correction(fn, table=rtbl, verbose=False)
    dsc.apply_distortion_scale_correction(fn, table=stbl, verbose=False)
    v1 = ImageModel(fn).meta.wcs.get_transform('detector', 'v2v3')(*grid)
    h = fits.getheader(fn, ('SCI', 1))
    assert h[drc.MARKER] is True and h[dsc.MARKER] is True
    assert h['DSCLPV2'] == pytest.approx(h['DROTPV2'], abs=1e-9)
    exp = dsc.scale_about_pivot_model(h['DSCLPV2'], h['DSCLPV3'], -500.0)(
        *drc.rotation_about_pivot_model(h['DROTPV2'], h['DROTPV3'], -30.855)(*v0))
    assert np.hypot(v1[0] - exp[0], v1[1] - exp[1]).max() * 1e3 < 1e-6


def test_apply_skips_frame_without_row(tmp_path):
    tbl = _table([_row()])
    fn = _frame(tmp_path / "syn_crf.fits", filt='F150W',
                ref='crds://jwst_nircam_distortion_0251.asdf')
    before = _sky(fn)
    assert dsc.apply_distortion_scale_correction(fn, table=tbl, verbose=False) is None
    h = fits.getheader(fn, ('SCI', 1))
    assert dsc.MARKER not in h and dsc.PENDING not in h
    after = _sky(fn)
    assert np.array_equal(before[0], after[0])


def test_apply_refuses_mismatched_reference(tmp_path):
    tbl = _table([_row()])
    fn = _frame(tmp_path / "syn_crf.fits", ref='crds://jwst_nircam_distortion_0999.asdf')
    with pytest.raises(dsc.DistortionScaleError, match="re-measure"):
        dsc.apply_distortion_scale_correction(fn, table=tbl, verbose=False)
    assert dsc.PENDING not in fits.getheader(fn, ('SCI', 1))


def test_apply_refuses_shift_aligned_frame(tmp_path):
    tbl = _table([_row()])
    fn = _frame(tmp_path / "syn_crf.fits", raoffset=1e-5)
    with pytest.raises(dsc.DistortionScaleError, match="RAOFFSET"):
        dsc.apply_distortion_scale_correction(fn, table=tbl, verbose=False)
    h = fits.getheader(fn, ('SCI', 1))
    assert dsc.MARKER not in h and dsc.PENDING not in h


@pytest.mark.skipif(not os.environ.get("DSCL_TEST_CRF"),
                    reason="set DSCL_TEST_CRF to an un-aligned NIRCam SW F200W crf")
def test_real_crf(tmp_path):
    from jwst.datamodels import ImageModel
    fn = str(tmp_path / os.path.basename(os.environ["DSCL_TEST_CRF"]))
    shutil.copy(os.environ["DSCL_TEST_CRF"], fn)
    m0 = ImageModel(fn)
    xs, ys = m0.meta.subarray.xstart or 1, m0.meta.subarray.ystart or 1
    cx, cy = CEN[0] - (xs - 1), CEN[1] - (ys - 1)
    c0, k0 = m0.meta.wcs(cx, cy), m0.meta.wcs(0.0, 0.0)
    corr = dsc.apply_distortion_scale_correction(fn, verbose=False)
    assert corr is not None
    m1 = ImageModel(fn)
    c1, k1 = m1.meta.wcs(cx, cy), m1.meta.wcs(0.0, 0.0)
    assert _sep_mas(c1, c0) < 0.01
    lever = _sep_mas(k0, c0)
    assert _sep_mas(k1, k0) == pytest.approx(lever * abs(corr) * 1e-6, rel=0.02)
    assert fits.getheader(fn, ('SCI', 1))['SIPGWMAX'] < 0.05
