"""Window-sweep + two-reference hardening of the sanctioned offset helper.

Regression guard for the brick-1182 v001 trap: a LARGE rigid offset must be
recovered (not read as ~0/incoherent) because a fixed narrow window has no true
pairs inside it.
"""
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord

from jwst_gc_pipeline.photometry.astrometry_offsets import (
    measure_offset, agree_across_references)


def _field(n=800, seed=0):
    rng = np.random.RandomState(seed)
    ra = 266.4 + rng.rand(n) * 0.05
    dec = -28.9 + rng.rand(n) * 0.05
    return ra, dec


def _shift(ra, dec, dra_arcsec, ddec_arcsec):
    cosd = np.cos(np.radians(dec))
    return SkyCoord((ra + dra_arcsec / 3600.0 / cosd) * u.deg,
                    (dec + ddec_arcsec / 3600.0) * u.deg)


def test_small_offset_recovered_at_narrow_window():
    ra, dec = _field()
    a = SkyCoord(ra * u.deg, dec * u.deg)
    b = _shift(ra, dec, 0.5, 0.3)
    r = measure_offset(a, b)
    assert r["ok"]
    assert abs(r["dra"] - 500) < 40 and abs(r["ddec"] - 300) < 40
    assert not r["swept"]           # offset (0.5") < initial window, no widening needed


def test_large_offset_recovered_by_sweep():
    # The brick-1182 v001 case: a ~20" rigid offset. A 3" window sees noise; the
    # sweep must widen and recover the true peak.
    ra, dec = _field(n=1500, seed=3)
    a = SkyCoord(ra * u.deg, dec * u.deg)
    b = _shift(ra, dec, 15.4, -13.5)          # ~20.4" offset
    r = measure_offset(a, b)                    # sweep ON by default
    assert r["ok"], f"large offset not recovered: {r}"
    assert abs(r["dra"] - 15400) < 300 and abs(r["ddec"] + 13500) < 300
    assert r["swept"] and r["window_arcsec"] >= 30.0


def test_no_sweep_misses_large_offset():
    # Prove the sweep is what saves it: with sweep off, a 3" window does NOT find
    # the 20" tie (this is the exact failure the sweep guards).
    ra, dec = _field(n=1500, seed=3)
    a = SkyCoord(ra * u.deg, dec * u.deg)
    b = _shift(ra, dec, 15.4, -13.5)
    r = measure_offset(a, b, sweep=False)
    assert (r is None) or (not r["ok"]) or (abs(r["dra"] - 15400) > 1000)


def test_two_reference_agreement():
    ra, dec = _field(n=1000, seed=7)
    a = SkyCoord(ra * u.deg, dec * u.deg)
    # both references are the same clean tie -> agree
    ref = _shift(ra, dec, -1.66, -0.98)
    sparse = ref[::5]
    out = agree_across_references(a, ref, sparse, label_a="dense", label_b="sparse")
    assert out["agree"], out
    assert out["sep_mas"] < 100
