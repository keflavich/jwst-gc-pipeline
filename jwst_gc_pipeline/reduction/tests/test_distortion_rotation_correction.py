"""Unit tests for the SW distortion-reference rotation correction
(distortion_rotation_correction.py).

Hermetic for the rotation math, sign convention, pivot, table lookup and the
apply path (synthetic GWCS built the way assign_wcs builds it).  The test on
a real NIRCam crf runs only where one is available (DROT_TEST_CRF env var)
and is skipped otherwise.
"""
import os
import shutil

import numpy as np
import pytest
from astropy.io import fits
from astropy.modeling import models as M
from astropy.table import Table

from jwst_gc_pipeline.reduction import distortion_rotation_correction as drc

V2REF, V3REF = -82.0, -497.0
PIX = 0.031
CEN = drc.FULL_FRAME_CENTRE


def _anchor_model():
    # linear full-frame detector -> V2V3 with a little shear so the two axes differ
    m = ((M.Shift(-CEN[0]) & M.Shift(-CEN[1]))
         | M.AffineTransformation2D(matrix=[[PIX, 2e-6], [-1e-6, -PIX * 1.00002]])
         | (M.Shift(V2REF) & M.Shift(V3REF)))
    return m


def _rotate_v2v3(model, angle_arcsec):
    c2, c3 = model(*CEN)
    return model | drc.rotation_about_pivot_model(c2, c3, angle_arcsec)


def _grid(n=9, lo=0.0, hi=2047.0):
    g = np.linspace(lo, hi, n)
    gx, gy = np.meshgrid(g, g)
    return gx.ravel(), gy.ravel()


def _gwcs(det2v, bbox=((-0.5, 2047.5), (-0.5, 2047.5))):
    from astropy import units as u
    from gwcs import coordinate_frames as cf
    from gwcs import wcs as gw
    det = cf.Frame2D(name='detector', axes_order=(0, 1), unit=(u.pix, u.pix))
    v2v3 = cf.Frame2D(name='v2v3', axes_order=(0, 1), unit=(u.arcsec, u.arcsec))
    w = gw.WCS([(det, det2v), (v2v3, None)])
    if bbox is not None:
        w.bounding_box = bbox
    return w


def _table(rows):
    return Table(rows=rows, names=drc.TABLE_COLUMNS)


def _row(det='NRCA3', filt='F150W', pupil='CLEAR', ref='jwst_nircam_distortion_0251.asdf',
         corr=-24.226):
    return (det, filt, pupil, ref, 'F212N', 'jwst_nircam_distortion_0182.asdf', corr, 15.4, 0.7)


# --- reference_rotation and the sign of the correction -----------------------

@pytest.mark.parametrize("rot", [-24.0, 7.2, 30.9])
def test_reference_rotation_recovers_rotation(rot):
    anchor = _anchor_model()
    band = _rotate_v2v3(anchor, rot)
    r, s, rms = drc.reference_rotation(band, anchor)
    assert r == pytest.approx(rot, abs=1e-6)
    assert abs(s) < 1e-3 and rms < 1e-6


def test_reference_rotation_separates_scale():
    anchor = _anchor_model()
    c2, c3 = anchor(*CEN)
    band = _rotate_v2v3(anchor, 20.0) | ((M.Shift(-c2) & M.Shift(-c3))
                                         | (M.Scale(1 + 40e-6) & M.Scale(1 + 40e-6))
                                         | (M.Shift(c2) & M.Shift(c3)))
    r, s, _ = drc.reference_rotation(band, anchor)
    assert r == pytest.approx(20.0, abs=1e-5)
    assert s == pytest.approx(40.0, abs=1e-3)


def test_correction_restores_anchor_and_wrong_sign_doubles():
    anchor = _anchor_model()
    band = _rotate_v2v3(anchor, 24.0)
    rot, _, _ = drc.reference_rotation(band, anchor)
    x, y = _grid()
    a2, a3 = anchor(x, y)

    fixed, _ = drc.rotated_wcs(_gwcs(band), -rot)
    f2, f3 = fixed.get_transform('detector', 'v2v3')(x, y)
    assert np.hypot(f2 - a2, f3 - a3).max() * 1e3 < 1e-6       # mas

    b2, b3 = band(x, y)
    wrong, _ = drc.rotated_wcs(_gwcs(band), +rot)
    w2, w3 = wrong.get_transform('detector', 'v2v3')(x, y)
    err_before = np.hypot(b2 - a2, b3 - a3)
    err_wrong = np.hypot(w2 - a2, w3 - a3)
    good = err_before > 1e-6
    assert np.allclose(err_wrong[good] / err_before[good], 2.0, rtol=1e-4)


def test_positive_angle_rotates_v2_toward_v3():
    m = drc.rotation_about_pivot_model(V2REF, V3REF, 3600.0 * 90)
    v2, v3 = m(V2REF + 1.0, V3REF)
    assert v2 == pytest.approx(V2REF, abs=1e-9) and v3 == pytest.approx(V3REF + 1.0, abs=1e-9)


# --- pivot, subarrays, bounding box, inverse ---------------------------------

def test_detector_centre_is_fixed_point():
    w = _gwcs(_anchor_model())
    out, pivot = drc.rotated_wcs(w, -24.0)
    c0 = w.get_transform('detector', 'v2v3')(*CEN)
    c1 = out.get_transform('detector', 'v2v3')(*CEN)
    assert np.allclose(c0, pivot, atol=1e-12) and np.allclose(c1, c0, atol=1e-10)
    # a corner moves by lever * angle
    x0 = w.get_transform('detector', 'v2v3')(0.0, 0.0)
    x1 = out.get_transform('detector', 'v2v3')(0.0, 0.0)
    lever = np.hypot(x0[0] - c0[0], x0[1] - c0[1])
    assert np.hypot(x1[0] - x0[0], x1[1] - x0[1]) == pytest.approx(
        lever * np.radians(24.0 / 3600.0), rel=1e-6)


def test_subarray_pivot_is_full_frame_centre():
    # assign_wcs prepends the subarray -> full-frame shift
    xs, ys = 993, 1121
    full = _anchor_model()
    sub = (M.Shift(xs - 1) & M.Shift(ys - 1)) | full
    _, pv_full = drc.rotated_wcs(_gwcs(full), -24.0)
    _, pv_sub = drc.rotated_wcs(_gwcs(sub, bbox=((-0.5, 63.5), (-0.5, 63.5))), -24.0,
                                xstart=xs, ystart=ys)
    assert np.allclose(pv_sub, pv_full, atol=1e-10)
    # the subarray-frame and full-frame corrections agree pixel for pixel
    out_full, _ = drc.rotated_wcs(_gwcs(full), -24.0)
    out_sub, _ = drc.rotated_wcs(_gwcs(sub, bbox=((-0.5, 63.5), (-0.5, 63.5))), -24.0,
                                 xstart=xs, ystart=ys)
    x, y = _grid(5, 0, 63)
    a = out_sub.get_transform('detector', 'v2v3')(x, y)
    b = out_full.get_transform('detector', 'v2v3')(x + xs - 1, y + ys - 1)
    assert np.allclose(a, b, atol=1e-10)


def test_bounding_box_and_inverse_survive():
    w = _gwcs(_anchor_model())
    out, _ = drc.rotated_wcs(w, -24.0)
    assert tuple(map(tuple, out.bounding_box)) == tuple(map(tuple, w.bounding_box))
    t = out.get_transform('detector', 'v2v3')
    x, y = _grid()
    xr, yr = t.inverse(*t(x, y))
    assert np.allclose(xr, x, atol=1e-8) and np.allclose(yr, y, atol=1e-8)
    # the input WCS is untouched
    v = w.get_transform('detector', 'v2v3')(x, y)
    assert np.allclose(v, _anchor_model()(x, y), atol=1e-12)


def test_no_bounding_box_is_fine():
    out, _ = drc.rotated_wcs(_gwcs(_anchor_model(), bbox=None), -24.0)
    assert out.get_transform('detector', 'v2v3') is not None


def test_verify_rotation_rejects_non_rotation():
    w = _gwcs(_anchor_model())
    out, pivot = drc.rotated_wcs(w, -24.0)
    assert drc.verify_rotation(w, out, pivot, -24.0, (2048, 2048)) < 1e-6
    with pytest.raises(drc.DistortionRotationError, match="verification failed"):
        drc.verify_rotation(w, out, pivot, -20.0, (2048, 2048))


# --- table lookup ------------------------------------------------------------

def test_lookup_none_without_rows():
    tbl = _table([_row()])
    assert drc.lookup_correction_arcsec('NRCA3', 'F212N', 'CLEAR', 'x.asdf', table=tbl) is None
    assert drc.lookup_correction_arcsec('NRCA1', 'F150W', 'CLEAR', 'x.asdf', table=tbl) is None


def test_lookup_match_strips_crds_prefix_and_case():
    tbl = _table([_row()])
    assert drc.lookup_correction_arcsec(
        'nrca3', 'f150w', 'clear', 'crds://jwst_nircam_distortion_0251.asdf',
        table=tbl) == pytest.approx(-24.226)


def test_lookup_refuses_other_reference():
    tbl = _table([_row()])
    with pytest.raises(drc.DistortionRotationError, match="regenerate"):
        drc.lookup_correction_arcsec('NRCA3', 'F150W', 'CLEAR',
                                     'jwst_nircam_distortion_9999.asdf', table=tbl)


def test_lookup_keys_on_pupil():
    tbl = _table([_row(filt='F150W2', pupil='F162M', ref='a.asdf', corr=-24.76),
                  _row(filt='F150W2', pupil='F164N', ref='b.asdf', corr=-24.90)])
    assert drc.lookup_correction_arcsec('NRCA3', 'F150W2', 'F164N', 'b.asdf',
                                        table=tbl) == pytest.approx(-24.90)
    assert drc.lookup_correction_arcsec('NRCA3', 'F150W2', 'CLEAR', 'a.asdf', table=tbl) is None
    assert drc.lookup_correction_arcsec('NRCA3', 'F150W2', None, 'a.asdf', table=tbl) is None


def test_shipped_table_sanity():
    tbl = drc.load_rotation_table()
    bands = {(r['filter'], r['pupil']) for r in tbl}
    assert bands == {('F115W', 'CLEAR'), ('F150W', 'CLEAR'), ('F200W', 'CLEAR'),
                     ('F150W2', 'F162M'), ('F150W2', 'F164N')}
    for f, p in bands:
        sel = tbl[(tbl['filter'] == f) & (tbl['pupil'] == p)]
        assert sorted(sel['detector']) == ['NRCA1', 'NRCA2', 'NRCA3', 'NRCA4',
                                           'NRCB1', 'NRCB2', 'NRCB3', 'NRCB4']
        assert len(set(sel['distortion_ref'])) == 8
    assert set(tbl['anchor_filter']) == {'F212N'}
    assert np.all(np.abs(tbl['correction_arcsec']) < 60)
    assert np.all(tbl['fit_rms_mas'] < 2)
    a23 = tbl[np.isin(tbl['detector'], ['NRCA2', 'NRCA3'])]
    assert np.all(a23['correction_arcsec'] < -15)


# --- preconditions and the apply path ----------------------------------------

def test_preconditions():
    assert drc.check_apply_preconditions(fits.Header()) == 'go'
    assert drc.check_apply_preconditions(fits.Header({drc.MARKER: True})) == 'skip'
    with pytest.raises(drc.DistortionRotationError, match="pending"):
        drc.check_apply_preconditions(fits.Header({drc.PENDING: True}))
    with pytest.raises(drc.DistortionRotationError, match="RAOFFSET"):
        drc.check_apply_preconditions(fits.Header({'RAOFFSET': 1e-5}))


def test_enabled_switch(monkeypatch):
    monkeypatch.delenv('DISTORTION_ROTATION_CORRECTION', raising=False)
    assert not drc.correction_enabled()
    monkeypatch.setenv('DISTORTION_ROTATION_CORRECTION', '1')
    assert drc.correction_enabled()


def _synthetic_frame(path, xstart=993, ystart=993, filt='F150W', pupil='CLEAR',
                     ref='crds://jwst_nircam_distortion_0251.asdf', raoffset=None):
    from astropy import coordinates as coord
    from astropy import units as u
    from gwcs import coordinate_frames as cf
    from gwcs import wcs as gw
    from jwst.assign_wcs import pointing
    from jwst.datamodels import ImageModel

    m = ImageModel((64, 64))
    m.meta.instrument.name = 'NIRCAM'
    m.meta.instrument.detector = 'NRCA3'
    m.meta.instrument.filter = filt
    m.meta.instrument.pupil = pupil
    m.meta.ref_file.distortion.name = ref
    m.meta.subarray.xstart, m.meta.subarray.ystart = xstart, ystart
    m.meta.subarray.xsize, m.meta.subarray.ysize = 64, 64
    wi = m.meta.wcsinfo
    wi.v2_ref, wi.v3_ref, wi.roll_ref = V2REF, V3REF, 90.2
    wi.ra_ref, wi.dec_ref = 266.85, -28.34
    wi.v3yangle, wi.vparity = -0.55, -1
    det2v = (M.Shift(xstart - 1) & M.Shift(ystart - 1)) | _anchor_model()
    frames = [cf.Frame2D(name=n, axes_order=(0, 1), unit=un) for n, un in
              (('detector', (u.pix, u.pix)), ('v2v3', (u.arcsec, u.arcsec)),
               ('v2v3vacorr', (u.arcsec, u.arcsec)))]
    world = cf.CelestialFrame(reference_frame=coord.ICRS(), name='world')
    m.meta.wcs = gw.WCS([(frames[0], det2v), (frames[1], M.Identity(2)),
                         (frames[2], pointing.v23tosky(m)), (world, None)])
    m.meta.wcs.bounding_box = ((-0.5, 63.5), (-0.5, 63.5))
    m.save(str(path))
    if raoffset is not None:
        with fits.open(path, mode='update') as h:
            h['SCI'].header['RAOFFSET'] = raoffset
    return str(path)


def _sky(fn, x=None, y=None):
    from jwst.datamodels import ImageModel
    if x is None:
        yy, xx = np.mgrid[0:64:8, 0:64:8]
        x, y = xx.ravel(), yy.ravel()
    return ImageModel(fn).meta.wcs(np.asarray(x, float), np.asarray(y, float))


def _sep_mas(a, b):
    return np.hypot((a[0] - b[0]) * np.cos(np.radians(b[1])), a[1] - b[1]) * 3.6e6


def test_apply_end_to_end(tmp_path):
    from jwst.datamodels import ImageModel
    tbl = _table([_row()])
    fn = _synthetic_frame(tmp_path / "syn_crf.fits")
    cx, cy = CEN[0] - 992, CEN[1] - 992                 # full-frame centre in this subarray
    s0, c0 = _sky(fn), _sky(fn, [cx], [cy])
    v0 = ImageModel(fn).meta.wcs.get_transform('detector', 'v2v3')(*np.mgrid[0:64:8, 0:64:8][::-1])

    assert drc.apply_distortion_rotation_correction(fn, table=tbl, verbose=False) == -24.226
    s1, c1 = _sky(fn), _sky(fn, [cx], [cy])
    assert _sep_mas(c1, c0).max() < 1e-4                # pivot fixed on the sky
    assert _sep_mas(s1, s0).min() > 1e-3                # the rest moved
    wcs1 = ImageModel(fn).meta.wcs
    v1 = wcs1.get_transform('detector', 'v2v3')(*np.mgrid[0:64:8, 0:64:8][::-1])
    h = fits.getheader(fn, ('SCI', 1))
    exp = drc.rotation_about_pivot_model(h['DROTPV2'], h['DROTPV3'], -24.226)(*v0)
    assert np.hypot(v1[0] - exp[0], v1[1] - exp[1]).max() * 1e3 < 1e-6
    assert h[drc.MARKER] is True and h[drc.PENDING] is False
    assert h['DROTARC'] == -24.226 and h['DROTREF'] == 'jwst_nircam_distortion_0251.asdf'
    assert h['SIPGWMAX'] < 0.05
    from astropy.wcs import WCS
    s = WCS(h).pixel_to_world_values(np.array([5.0, 50.0]), np.array([7.0, 40.0]))
    g = _sky(fn, [5.0, 50.0], [7.0, 40.0])
    assert np.allclose(s[0], g[0], atol=1e-7) and np.allclose(s[1], g[1], atol=1e-7)

    # idempotent
    assert drc.apply_distortion_rotation_correction(fn, table=tbl, verbose=False) is None
    s2 = _sky(fn)
    assert np.array_equal(s2[0], s1[0]) and np.array_equal(s2[1], s1[1])


def test_apply_skips_frame_without_row(tmp_path):
    tbl = _table([_row()])
    fn = _synthetic_frame(tmp_path / "syn_crf.fits", filt='F212N',
                          ref='crds://jwst_nircam_distortion_0182.asdf')
    before = _sky(fn)
    assert drc.apply_distortion_rotation_correction(fn, table=tbl, verbose=False) is None
    h = fits.getheader(fn, ('SCI', 1))
    assert drc.MARKER not in h and drc.PENDING not in h
    after = _sky(fn)
    assert np.array_equal(before[0], after[0])


def test_apply_refuses_mismatched_reference(tmp_path):
    tbl = _table([_row()])
    fn = _synthetic_frame(tmp_path / "syn_crf.fits", ref='crds://jwst_nircam_distortion_0999.asdf')
    with pytest.raises(drc.DistortionRotationError, match="regenerate"):
        drc.apply_distortion_rotation_correction(fn, table=tbl, verbose=False)
    assert drc.PENDING not in fits.getheader(fn, ('SCI', 1))


def test_apply_refuses_shift_aligned_frame(tmp_path):
    tbl = _table([_row()])
    fn = _synthetic_frame(tmp_path / "syn_crf.fits", raoffset=1e-5)
    with pytest.raises(drc.DistortionRotationError, match="RAOFFSET"):
        drc.apply_distortion_rotation_correction(fn, table=tbl, verbose=False)
    h = fits.getheader(fn, ('SCI', 1))
    assert drc.MARKER not in h and drc.PENDING not in h


@pytest.mark.skipif(not os.environ.get("DROT_TEST_CRF"),
                    reason="set DROT_TEST_CRF to an un-aligned NIRCam SW crf in a corrected band")
def test_real_crf(tmp_path):
    from jwst.datamodels import ImageModel
    fn = str(tmp_path / os.path.basename(os.environ["DROT_TEST_CRF"]))
    shutil.copy(os.environ["DROT_TEST_CRF"], fn)
    m0 = ImageModel(fn)
    xs, ys = m0.meta.subarray.xstart or 1, m0.meta.subarray.ystart or 1
    ny, nx = m0.data.shape
    cx, cy = CEN[0] - (xs - 1), CEN[1] - (ys - 1)
    c0, k0 = m0.meta.wcs(cx, cy), m0.meta.wcs(0.0, 0.0)
    corr = drc.apply_distortion_rotation_correction(fn, verbose=False)
    assert corr is not None
    m1 = ImageModel(fn)
    c1, k1 = m1.meta.wcs(cx, cy), m1.meta.wcs(0.0, 0.0)
    assert _sep_mas(c1, c0) < 0.01
    lever = np.hypot(cx, cy) * 31.0                     # mas, approx
    assert _sep_mas(k1, k0) == pytest.approx(lever * np.radians(abs(corr) / 3600.0), rel=0.05)
    assert fits.getheader(fn, ('SCI', 1))['SIPGWMAX'] < 0.05
