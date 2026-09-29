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
