"""Footprint-scaled satstar consolidation dedup (fix 1) + big-footprint
inconsistent reject (fix 2).

A bright saturated star's per-frame satstar position scatters by ~the saturated
core radius, so a LARGE footprint (extended-emission saturated core) leaves
several un-merged rows at a fixed 0.15" radius.  Fix 1 scales the merge radius to
sqrt(sat_area/pi); fix 2 (extended-emission targets only) drops a pile of >=3
comparable-brightness but flux-inconsistent satstars on one big footprint.
"""
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

from jwst_gc_pipeline.photometry.merge_catalogs import _dedup_satstar_catalog

RA0 = 290.9354


def _tbl(dec_offsets_arcsec, fluxes, sat_area):
    n = len(fluxes)
    dec = 14.4977 + np.asarray(dec_offsets_arcsec) / 3600.0
    t = Table()
    t['skycoord_fit'] = SkyCoord(np.full(n, RA0) * u.deg, dec * u.deg)
    t['flux_fit'] = np.asarray(fluxes, dtype=float)
    if sat_area is not None:
        t['sat_area'] = np.asarray(sat_area, dtype=int)
    return t


def test_fix1_large_footprint_consistent_merges_to_one():
    # 4 scattered detections of ONE big-footprint star, consistent flux -> 1 row
    t = _tbl([0.0, 0.2, 0.4, 0.6], [440e3, 420e3, 410e3, 430e3],
             sat_area=[370, 370, 370, 370])
    out = _dedup_satstar_catalog(t, target='brick')   # fix2 off; fix1 still merges
    assert len(out) == 1


def test_fix1_flux_gate_keeps_distinct_neighbours():
    # two DISTINCT stars 0.3" apart (inside the footprint radius) but very
    # different flux -> the flux gate keeps them separate
    t = _tbl([0.0, 0.3], [300e3, 90e3], sat_area=[200, 200])
    out = _dedup_satstar_catalog(t, target='brick')
    assert len(out) == 2


def test_fix2_rejects_big_footprint_inconsistent_pile_on_ext_target():
    # 3 comparable-brightness but inconsistent satstars on one big footprint,
    # extended-emission target -> whole pile dropped
    t = _tbl([0.0, 0.25, 0.5], [350e3, 200e3, 150e3], sat_area=[370, 370, 370])
    out = _dedup_satstar_catalog(t, target='w51')
    assert len(out) == 0


def test_fix2_gated_off_for_nonext_target():
    # same pile on a non-extended target: reject disabled, so the pile is NOT
    # zeroed.  Fix 1 still merges the flux-consistent pair (350k absorbs 200k,
    # ratio 1.75 < 2); the inconsistent 150k (ratio 2.33) stays -> 2 kept
    # (contrast with 0 on the extended-emission target above).
    t = _tbl([0.0, 0.25, 0.5], [350e3, 200e3, 150e3], sat_area=[370, 370, 370])
    out = _dedup_satstar_catalog(t, target='brick')
    assert len(out) == 2


def test_backward_compat_without_sat_area():
    # no sat_area column -> base-radius behaviour: 0.1" dup absorbed, 0.5" kept
    t = _tbl([0.0, 0.1, 0.5], [300e3, 250e3, 200e3], sat_area=None)
    out = _dedup_satstar_catalog(t, target='w51')
    assert len(out) == 2
