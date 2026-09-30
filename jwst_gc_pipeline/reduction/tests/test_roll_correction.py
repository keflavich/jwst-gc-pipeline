"""Unit tests for the field-rotation (roll) correction (roll_correction.py).

Hermetic for the rotation math, sign convention and table resolution.  The
GWCS round-trip test runs only where a real NIRCam crf is available
(ROLL_TEST_CRF env var) and is skipped otherwise.
"""
import os

import numpy as np
import pytest
from astropy.io import fits

from jwst_gc_pipeline.reduction import roll_correction as rc

PIV = (266.8528, -28.3426)


def _similarity_theta_pa(x, y, X, Y):
    """Fit X = a x - b y + tx, Y = b x + a y + ty; return theta_PA (arcsec),
    positive N->E, the convention of data-qa#346."""
    A = np.zeros((2 * len(x), 4))
    A[0::2] = np.c_[x, -y, np.ones_like(x), np.zeros_like(x)]
    A[1::2] = np.c_[y, x, np.zeros_like(x), np.ones_like(x)]
    rhs = np.empty(2 * len(x))
    rhs[0::2], rhs[1::2] = X, Y
    a, b, _, _ = np.linalg.lstsq(A, rhs, rcond=None)[0]
    return -np.degrees(np.arctan2(b, a)) * 3600.0


def test_tangent_roundtrip():
    ra = PIV[0] + np.linspace(-0.05, 0.05, 7)
    dec = PIV[1] + np.linspace(0.04, -0.04, 7)
    xi, eta = rc._tangent(ra, dec, *PIV)
    r2, d2 = rc._untangent(xi, eta, *PIV)
    assert np.allclose(r2, ra, atol=1e-11) and np.allclose(d2, dec, atol=1e-11)
    assert rc._untangent(0.0, 0.0, *PIV) == pytest.approx(PIV)


def test_pivot_is_fixed_point():
    r, d = rc.rotation_about_pivot(PIV[0], PIV[1], PIV[0], PIV[1], 25.0)
    assert r == pytest.approx(PIV[0], abs=1e-12)
    assert d == pytest.approx(PIV[1], abs=1e-12)


@pytest.mark.parametrize("theta", [-30.0, 5.0, 20.0])
def test_sign_convention_matches_similarity_fit(theta):
    rng = np.random.default_rng(1)
    ra = PIV[0] + rng.uniform(-0.04, 0.04, 200)
    dec = PIV[1] + rng.uniform(-0.03, 0.03, 200)
    r2, d2 = rc.rotation_about_pivot(ra, dec, *PIV, theta)
    x, y = rc._tangent(ra, dec, *PIV)
    X, Y = rc._tangent(r2, d2, *PIV)
    assert _similarity_theta_pa(x, y, X, Y) == pytest.approx(theta, abs=1e-6)


def test_point_north_of_pivot_moves_east_for_positive_roll():
    r, d = rc.rotation_about_pivot(PIV[0], PIV[1] + 0.05, *PIV, 20.0)
    assert r > PIV[0]            # East = increasing RA
    # displacement = lever * theta = 180" * 20"/206265" ~ 17.5 mas
    move = (r - PIV[0]) * np.cos(np.radians(d)) * 3.6e6
    assert move == pytest.approx(0.05 * 3600e3 * np.radians(20 / 3600.), rel=1e-3)


def _write_table(tmp_path, rows):
    p = tmp_path / "roll.csv"
    lines = ["# comment line",
             "program,observation,visit,delta_roll_arcsec,delta_roll_err_arcsec,reference,source"]
    lines += [",".join(map(str, r)) for r in rows]
    p.write_text("\n".join(lines) + "\n")
    return str(p)


def test_resolve_most_specific_row_wins(tmp_path, monkeypatch):
    monkeypatch.delenv("ROLL_CORRECTION_ARCSEC", raising=False)
    t = _write_table(tmp_path, [
        ("*", "*", "*", 10.0, 1, "gaia", "global"),
        (10678, "*", "*", 20.0, 1, "gaia", "program"),
        (10678, 135, "*", 22.0, 1, "gaia", "obs"),
        (10678, 135, 2, 23.0, 1, "gaia", "visit"),
    ])
    assert rc.resolve_roll_arcsec(10678, 135, 1, table=t) == 22.0
    assert rc.resolve_roll_arcsec("10678", "135", "002", table=t) == 23.0
    assert rc.resolve_roll_arcsec(10678, 40, 1, table=t) == 20.0
    assert rc.resolve_roll_arcsec(2221, 1, 1, table=t) == 10.0


def test_resolve_none_without_match(tmp_path, monkeypatch):
    monkeypatch.delenv("ROLL_CORRECTION_ARCSEC", raising=False)
    t = _write_table(tmp_path, [(2221, 1, "*", 5.0, 1, "gaia", "obs")])
    assert rc.resolve_roll_arcsec(1182, 4, 1, table=t) is None
    assert rc.resolve_roll_arcsec(1182, 4, 1, table=str(tmp_path / "missing.csv")) is None


def test_env_override(tmp_path, monkeypatch):
    t = _write_table(tmp_path, [("*", "*", "*", 10.0, 1, "gaia", "global")])
    monkeypatch.setenv("ROLL_CORRECTION_ARCSEC", "-7.5")
    assert rc.resolve_roll_arcsec(10678, 135, 1, table=t) == -7.5


def test_marker_gating():
    h = fits.Header()
    assert rc.roll_correction_needed(h)
    h[rc.MARKER] = True
    assert not rc.roll_correction_needed(h)


def test_shipped_table_parses(monkeypatch):
    monkeypatch.delenv("ROLL_CORRECTION_ARCSEC", raising=False)
    if not os.path.exists(rc.TABLE):
        pytest.skip("no shipped roll_corrections.csv")
    val = rc.resolve_roll_arcsec(10678, 135, 1)
    assert val is None or abs(val) < 120.0


@pytest.mark.skipif(not os.environ.get("ROLL_TEST_CRF"), reason="set ROLL_TEST_CRF to a NIRCam crf")
def test_gwcs_rotation_about_common_pivot():
    from jwst.datamodels import ImageModel
    m = ImageModel(os.environ["ROLL_TEST_CRF"])
    w0 = m.meta.wcs
    w1, (dra, ddec), piv = rc.roll_adjusted_wcs(w0, 20.0)
    worst = rc.verify_rotation(w0, w1, m.data.shape, piv, 20.0, tol_mas=0.05)
    assert worst < 0.05
    with pytest.raises(RuntimeError):
        rc.verify_rotation(w0, w1, m.data.shape, piv, -20.0, tol_mas=0.05)


# --------------------------------------------------------------------------
# End-to-end: apply_roll_correction on a synthetic NIRCam-like frame on disk.
# The GWCS carries the same v2v3 -> v2v3vacorr -> world (v23tosky) structure
# assign_wcs builds, so adjust_wcs, the pivot lookup and the SIP sync all run
# for real; no CRDS or real data needed.

def _synthetic_frame(path, raoffset=None):
    from astropy import coordinates as coord
    from astropy import units as u
    from astropy.modeling import models as M
    from gwcs import coordinate_frames as cf
    from gwcs import wcs as gw
    from jwst.assign_wcs import pointing
    from jwst.datamodels import ImageModel

    m = ImageModel((64, 64))
    m.meta.observation.program_number = '10678'
    m.meta.observation.observation_number = '135'
    m.meta.observation.visit_number = '001'
    wi = m.meta.wcsinfo
    wi.v2_ref, wi.v3_ref, wi.roll_ref = -82.0, -497.0, 90.2
    wi.ra_ref, wi.dec_ref = 266.85, -28.34
    wi.v3yangle, wi.vparity = -0.55, -1
    det2v = ((M.Shift(-32) & M.Shift(-32)) | (M.Scale(0.031) & M.Scale(0.031))
             | (M.Shift(wi.v2_ref) & M.Shift(wi.v3_ref)))
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
            h['SCI'].header['DEOFFSET'] = 0.0
    return str(path)


def _sky(fn):
    from jwst.datamodels import ImageModel
    yy, xx = np.mgrid[0:64:8, 0:64:8]
    return ImageModel(fn).meta.wcs(xx.ravel(), yy.ravel())


def test_apply_end_to_end(tmp_path, monkeypatch):
    monkeypatch.delenv("ROLL_CORRECTION_ARCSEC", raising=False)
    from jwst.datamodels import ImageModel
    fn = _synthetic_frame(tmp_path / "syn_crf.fits")
    ra0, dec0 = _sky(fn)
    pivot = rc.nircam_pivot_sky(ImageModel(fn).meta.wcs)

    assert rc.apply_roll_correction(fn, 20.0, verbose=False) == 20.0
    ra1, dec1 = _sky(fn)
    rap, decp = rc.rotation_about_pivot(ra0, dec0, pivot[0], pivot[1], 20.0)
    dev = np.hypot((ra1 - rap) * np.cos(np.radians(dec1)), dec1 - decp) * 3.6e6
    assert dev.max() < 0.05                                   # mas
    moved = np.hypot((ra1 - ra0) * np.cos(np.radians(dec1)), dec1 - dec0) * 3.6e6
    assert moved.min() > 1.0                                  # the frame did move

    h = fits.getheader(fn, ('SCI', 1))
    assert h[rc.MARKER] is True and h['ROLLARC'] == 20.0 and h[rc.PENDING] is False
    assert h['SIPGWMAX'] < 0.05
    # FITS header WCS follows the GWCS
    from astropy.wcs import WCS
    s = WCS(h).pixel_to_world_values(np.array([5.0, 50.0]), np.array([7.0, 40.0]))
    g = ImageModel(fn).meta.wcs(np.array([5.0, 50.0]), np.array([7.0, 40.0]))
    assert np.allclose(s[0], g[0], atol=1e-7) and np.allclose(s[1], g[1], atol=1e-7)

    # idempotent: second call is a no-op
    assert rc.apply_roll_correction(fn, 20.0, verbose=False) is None
    ra2, dec2 = _sky(fn)
    assert np.array_equal(ra2, ra1) and np.array_equal(dec2, dec1)


def test_apply_resolves_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ROLL_CORRECTION_ARCSEC", "-5.0")
    fn = _synthetic_frame(tmp_path / "syn_crf.fits")
    assert rc.apply_roll_correction(fn, verbose=False) == -5.0


def test_apply_refuses_shift_aligned_frame(tmp_path):
    fn = _synthetic_frame(tmp_path / "syn_crf.fits", raoffset=1e-5)
    before = _sky(fn)
    with pytest.raises(RuntimeError, match="RAOFFSET"):
        rc.apply_roll_correction(fn, 20.0, verbose=False)
    h = fits.getheader(fn, ('SCI', 1))
    assert rc.MARKER not in h and rc.PENDING not in h         # nothing written
    after = _sky(fn)
    assert np.array_equal(before[0], after[0]) and np.array_equal(before[1], after[1])


def test_apply_refuses_pending_frame(tmp_path):
    fn = _synthetic_frame(tmp_path / "syn_crf.fits")
    with fits.open(fn, mode='update') as h:
        h['SCI'].header[rc.PENDING] = True
    with pytest.raises(RuntimeError, match="pending"):
        rc.apply_roll_correction(fn, 20.0, verbose=False)


def test_resolve_handles_missing_header_values(monkeypatch):
    monkeypatch.delenv("ROLL_CORRECTION_ARCSEC", raising=False)
    assert rc.resolve_roll_arcsec(None, None) is None
    assert rc.resolve_roll_arcsec(None, '135') is None
    assert rc.resolve_roll_arcsec('10678', None) is None
    assert rc.resolve_roll_arcsec('10678', '135', None) == rc.resolve_roll_arcsec('10678', '135', '*')
