"""Hermetic tests for the same-star external-reference arbiter added to
check_interframe_overlap (the 2026-07 fix for the sparse 2221 nrca-long|nrcb-long
false FAIL).

Context: the inter-frame overlap gate could not measure the thin 2221 LW
inter-module overlap reference-free (0 mutual-coverage tiles), and its old
per-cell offset-histogram vs VIRAC (`measure_offset_grid`) was fooled by the
dense-field wrong-pair bias -- it read a 58" worst cell where the SAME-STAR tie
of the identical data is 3 mas.  `_samestar_ref_grid` uses `local_residual_map`
(real matched pairs, per-cell significance), which is density-immune, and gates a
gross / window-swept global tie directly.  These tests pin its behaviour on
controlled synthetic star fields -- no data files.
"""
import importlib.util
from pathlib import Path

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord

_spec = importlib.util.spec_from_file_location(
    "check_interframe_overlap",
    Path(__file__).with_name("check_interframe_overlap.py"))
ck = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ck)

RA0, DEC0 = 266.5, -28.7
COSD = float(np.cos(np.deg2rad(DEC0)))


def _field(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    ra = RA0 + rng.uniform(-0.02, 0.02, n)
    dec = DEC0 + rng.uniform(-0.02, 0.02, n)
    return ra, dec


def _sc(ra, dec):
    return SkyCoord(ra * u.deg, dec * u.deg)


def _shift(ra, dra_mas, mask=None):
    r = ra.copy()
    m = np.ones(len(ra), bool) if mask is None else mask
    r[m] = ra[m] + dra_mas / 3.6e6 / COSD
    return r


def test_clean_field_is_clean():
    ra, dec = _field()
    ref = _sc(ra, dec)
    g = ck._samestar_ref_grid(ref, ref, max_off_mas=80.0)
    assert g["clean"] and g["worst_off_mas"] == 0


def test_gross_coherent_offset_is_dirty():
    """A gross rigid offset (brick-1182 v001 ~20" class) -- the global tie gates it."""
    ra, dec = _field(seed=1)
    ref = _sc(ra, dec)
    src = _sc(_shift(ra, 20000.0), dec)     # 20 arcsec
    g = ck._samestar_ref_grid(src, ref, max_off_mas=80.0)
    assert not g["clean"] and g["worst_off_mas"] > 80.0


def test_moderate_coherent_offset_is_dirty():
    """A coherent 150 mas offset > max_off_mas is caught by the global tie."""
    ra, dec = _field(seed=2)
    ref = _sc(ra, dec)
    src = _sc(_shift(ra, 150.0), dec)
    g = ck._samestar_ref_grid(src, ref, max_off_mas=80.0)
    assert not g["clean"]


def test_localized_full_population_seam_is_dirty():
    """A localized region shifted 150 mas (whole cell population) is flagged by the
    per-cell same-star residual map -- the sensitivity the arbiter must keep."""
    ra, dec = _field(seed=3)
    ref = _sc(ra, dec)
    corner = (ra > RA0) & (dec > DEC0)
    src = _sc(_shift(ra, 150.0, corner), dec)
    g = ck._samestar_ref_grid(src, ref, max_off_mas=80.0)
    assert not g["clean"]


def test_within_tolerance_offset_is_clean():
    """A coherent offset below max_off_mas is within tolerance -> clean."""
    ra, dec = _field(seed=4)
    ref = _sc(ra, dec)
    src = _sc(_shift(ra, 40.0), dec)         # 40 mas < 80
    g = ck._samestar_ref_grid(src, ref, max_off_mas=80.0)
    assert g["clean"]


# Documented limitation (see the module docstring + follow-up issue): a LOCALIZED
# MINORITY offset right at the match radius (~300 mas = the dense-field NN spacing)
# can NN-collapse and be missed by any same-star matcher. It is out of scope for the
# deferred pairs (0-tile near-non-overlap slivers hold ~no common stars) and is owned
# by the reference-free per-tile frame-vs-frame layer for dense overlaps. Not asserted
# here so the test suite states what IS guaranteed, not a known gap.
