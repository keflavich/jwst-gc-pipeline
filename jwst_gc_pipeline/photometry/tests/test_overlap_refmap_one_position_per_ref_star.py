"""The field-wide same-star reference map must measure something (issue #410).

``_samestar_ref_grid`` is fed ``allsrc``, the POOLED per-exposure detection list,
in which every star appears once per exposure.  ``local_residual_map`` keeps the
nearest reference star for each source and then requires that reference star to
be unique, so with N>1 copies every reference star serves N sources, the
uniqueness filter is all-False, and the map returns zero measured cells -- at
every cell size, on every field, against every reference.  Measured on brick
F405N vs its own 115032-star VIRAC2 catalogue (361892 pooled detections):
``n_measured = 0`` at 2/4/8/16/30".  A blocking gate's absolute-frame arm was
therefore reporting on nothing, and its message ("no residual-map cell held >= 20
matched stars") named the reference's star density for what is a property of the
caller.

These tests pin the collapse of the pooled list to ONE POSITION PER REFERENCE
STAR, and pin that it stays REPORT-ONLY: the verdict the gate acts on is still
derived from the pooled ladder, so no field's answer moves until
``OVERLAP_SAMESTAR_DEDUP_GATE=1`` is set deliberately.
"""
import importlib.util
import pathlib
import sys

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "check_interframe_overlap",
    REPO_ROOT / "scripts" / "release" / "check_interframe_overlap.py")
gate = importlib.util.module_from_spec(_SPEC)
sys.modules["check_interframe_overlap"] = gate
_SPEC.loader.exec_module(gate)

RA0, DEC0 = 266.5, -28.7
COSD = float(np.cos(np.deg2rad(DEC0)))


def _field(n=4000, seed=0, half_width_deg=0.02):
    rng = np.random.default_rng(seed)
    return (RA0 + rng.uniform(-half_width_deg, half_width_deg, n),
            DEC0 + rng.uniform(-half_width_deg, half_width_deg, n))


def _pooled(ra, dec, copies, jitter_mas=2.0, seed=1, dra_mas=None, mask=None):
    """The list ``build_groups`` hands the arbiter: one copy of every star per
    exposure, each copy carrying its own measurement jitter."""
    rng = np.random.default_rng(seed)
    r, d = np.repeat(ra, copies), np.repeat(dec, copies)
    if dra_mas:
        m = np.repeat(mask, copies)
        r = r + m * dra_mas / 3.6e6 / COSD
    r = r + rng.normal(0, jitter_mas / 3.6e6 / COSD, len(r))
    d = d + rng.normal(0, jitter_mas / 3.6e6, len(d))
    return SkyCoord(r * u.deg, d * u.deg)


def test_pooled_list_measures_zero_cells_and_the_dedup_map_does_not():
    """The issue, and the fix, in one assertion pair."""
    ra, dec = _field()
    ref = SkyCoord(ra * u.deg, dec * u.deg)
    src = _pooled(ra, dec, copies=3)
    g = gate._samestar_ref_grid(src, ref, max_off_mas=80.0)

    # today's ladder, over the pooled list: nothing, at every scale
    assert g["n_total"] == 0
    assert all(s["n_measured"] == 0 for s in g["scales"])
    assert g["measurable"] is False

    # one position per reference star: the same data becomes measurable
    ded = g["dedup"]
    assert ded["n_src"] == len(ref)
    assert ded["copies_median"] == 3
    assert ded["n_measured"] > 0
    assert any(s["n_measured"] > 0 for s in ded["scales"])


def _grid(n_side=25, step_arcsec=4.0, ra0=RA0, dec0=DEC0):
    """A regular field whose stars are far apart compared with the 0.3" match
    radius, so every detection belongs to exactly one reference star."""
    k = np.arange(n_side) * step_arcsec / 3600.0
    gx, gy = np.meshgrid(k, k)
    return (ra0 + gx.ravel() / COSD), (dec0 + gy.ravel())


def test_one_position_per_ref_star_averages_the_copies():
    ra, dec = _grid()
    ref = SkyCoord(ra * u.deg, dec * u.deg)
    src = _pooled(ra, dec, copies=6, jitter_mas=40.0, seed=4)
    mean, rsub, n_det = gate._one_position_per_ref_star(src, ref)
    assert len(mean) == len(rsub) == len(ref)
    assert set(np.unique(n_det).tolist()) == {6}
    # 40 mas per copy, 6 copies -> ~16 mas expected per star.  The pooled list
    # is what a real map would have had to throw away entirely.
    sep_mas = mean.separation(rsub).to(u.mas).value
    assert np.median(sep_mas) < 30.0 and np.max(sep_mas) < 100.0


def test_one_position_per_ref_star_is_wrap_safe():
    """A field straddling RA 0 must not average 359.999 with 0.001 into 180."""
    ra, dec = _grid(n_side=20, ra0=359.99)
    ra = ra % 360.0
    ref = SkyCoord(ra * u.deg, dec * u.deg)
    assert ra.max() > 359.9 and ra.min() < 0.1, "the field must straddle RA 0"
    src = _pooled(ra, dec, copies=4, jitter_mas=5.0, seed=8)
    mean, rsub, _ = gate._one_position_per_ref_star(src, ref)
    assert np.max(mean.separation(rsub).to(u.mas).value) < 20.0


def test_no_match_returns_none_rather_than_raising():
    ra, dec = _field(n=200, seed=5)
    ref = SkyCoord(ra * u.deg, dec * u.deg)
    far = SkyCoord((ra + 1.0) * u.deg, dec * u.deg)
    assert gate._one_position_per_ref_star(far, ref) == (None, None, None)


def _seamed(seed=11):
    """A quarter of the field displaced 150 mas: a local seam the map must see,
    small enough in the field average to clear the gross global-tie ceiling."""
    ra, dec = _field(seed=seed)
    ref = SkyCoord(ra * u.deg, dec * u.deg)
    mask = (ra > RA0) & (dec > DEC0)
    src = _pooled(ra, dec, copies=3, dra_mas=150.0, mask=mask, seed=seed + 1)
    return src, ref


def test_a_real_seam_is_recorded_but_does_not_gate_by_default(monkeypatch):
    monkeypatch.delenv("OVERLAP_SAMESTAR_DEDUP_GATE", raising=False)
    src, ref = _seamed()
    g = gate._samestar_ref_grid(src, ref, max_off_mas=80.0)
    # the verdict the gate acts on is exactly today's: could-not-verify
    assert g["measurable"] is False and g["n_total"] == 0
    # ... and the seam is on the record
    assert g["dedup"]["gating"] is False
    assert g["dedup"]["n_flagged"] > 0
    assert g["dedup"]["worst_off_mas"] > 30.0


def test_the_env_switch_promotes_the_dedup_map_to_the_verdict(monkeypatch):
    monkeypatch.setenv("OVERLAP_SAMESTAR_DEDUP_GATE", "1")
    src, ref = _seamed()
    g = gate._samestar_ref_grid(src, ref, max_off_mas=80.0)
    assert g["dedup"]["gating"] is True
    assert g["measurable"] is True
    assert g["clean"] is False
    assert g["n_total"] > 0
    assert g["worst_off_mas"] > 30.0


def test_scales_carry_the_per_cell_distribution():
    ra, dec = _field(seed=13)
    ref = SkyCoord(ra * u.deg, dec * u.deg)
    src = _pooled(ra, dec, copies=3, seed=14)
    g = gate._samestar_ref_grid(src, ref, max_off_mas=80.0)
    measured = [s for s in g["dedup"]["scales"] if s["n_measured"]]
    assert measured, "the deduplicated ladder measured nothing"
    for s in measured:
        assert np.isfinite(s["off_mas_p50"]) and np.isfinite(s["off_mas_p90"])
        assert s["off_mas_p50"] <= s["off_mas_p90"] <= s["worst_off_mas"] + 1e-9
        assert s["n_pairs"] > 0
