"""Footprint-scaled satstar consolidation dedup keyed on component identity.

A bright saturated star's per-frame satstar position/flux scatters (the core is
NaN-masked), so a LARGE footprint (extended-emission saturated core) leaves
several un-merged rows at a fixed 0.15" radius.  The dedup scales the merge
radius to sqrt(sat_area/pi) (fix 1) and merges neighbours that share the same
saturated COMPONENT (its stable bbox-centre sky anchor sat_com_ra/dec) -- so an
extended blob seen N times collapses to one row while a real crowded cluster of
SEPARATE saturated cores is preserved.  Flux-consistency is only a fallback when
the anchor is missing.  A big-footprint inconsistent reject (fix 2) is available
opt-in via SATSTAR_FP_REJECT but off by default (it cannot distinguish a real
cluster from a blob).
"""
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

from jwst_gc_pipeline.photometry.merge_catalogs import _dedup_satstar_catalog

RA0 = 290.9354
DEC0 = 14.4977


def _tbl(dec_offsets_arcsec, fluxes, sat_area, anchor_dec_arcsec=None):
    n = len(fluxes)
    dec = DEC0 + np.asarray(dec_offsets_arcsec) / 3600.0
    t = Table()
    t['skycoord_fit'] = SkyCoord(np.full(n, RA0) * u.deg, dec * u.deg)
    t['flux_fit'] = np.asarray(fluxes, dtype=float)
    if sat_area is not None:
        t['sat_area'] = np.asarray(sat_area, dtype=int)
    if anchor_dec_arcsec is not None:
        t['sat_com_ra'] = np.full(n, RA0)
        t['sat_com_dec'] = DEC0 + np.asarray(anchor_dec_arcsec) / 3600.0
    return t


def test_same_component_merges_despite_flux_scatter(monkeypatch):
    # opt-in anchor merge (SATSTAR_FP_USE_ANCHOR): one extended blob, 3 detections
    # of WILDLY different flux sharing one component anchor -> collapse to ONE
    monkeypatch.setenv('SATSTAR_FP_USE_ANCHOR', '1')
    t = _tbl([0.0, 0.3, 0.6], [356e3, 156e3, 149e3], sat_area=[370, 400, 413],
             anchor_dec_arcsec=[0.30, 0.30, 0.30])
    out = _dedup_satstar_catalog(t, target='w51')
    assert len(out) == 1


def test_distinct_components_preserved_even_when_close(monkeypatch):
    # opt-in anchor merge: a real crowded cluster, 4 stars with SEPARATE component
    # anchors (>0.2" apart) within ~1.4" -> all preserved
    monkeypatch.setenv('SATSTAR_FP_USE_ANCHOR', '1')
    t = _tbl([0.0, 0.35, 0.7, 1.05], [164e3, 72e3, 58e3, 42e3],
             sat_area=[97, 45, 280, 377],
             anchor_dec_arcsec=[0.0, 0.35, 0.7, 1.05])
    out = _dedup_satstar_catalog(t, target='w51')
    assert len(out) == 4


def test_flux_fallback_without_anchor():
    # no anchor column -> flux-consistency fallback: consistent pair merges, the
    # inconsistent one stays
    t = _tbl([0.0, 0.3], [400e3, 410e3], sat_area=[370, 370])   # consistent -> 1
    assert len(_dedup_satstar_catalog(t, target='w51')) == 1
    t2 = _tbl([0.0, 0.3], [400e3, 90e3], sat_area=[370, 370])   # inconsistent -> 2
    assert len(_dedup_satstar_catalog(t2, target='w51')) == 2


def test_fix2_reject_is_opt_in(monkeypatch):
    # big-footprint comparable-but-inconsistent pile, NO anchor.  Default: reject
    # off -> flux fallback keeps survivors.  With SATSTAR_FP_REJECT=1 on an
    # extended target -> whole pile dropped.
    def _pile():
        return _tbl([0.0, 0.25, 0.5], [350e3, 200e3, 150e3], sat_area=[370, 370, 370])
    monkeypatch.delenv('SATSTAR_FP_REJECT', raising=False)
    assert len(_dedup_satstar_catalog(_pile(), target='w51')) >= 1
    monkeypatch.setenv('SATSTAR_FP_REJECT', '1')
    assert len(_dedup_satstar_catalog(_pile(), target='w51')) == 0
    # gated to extended-emission targets even when enabled
    assert len(_dedup_satstar_catalog(_pile(), target='brick')) >= 1


def test_backward_compat_without_sat_area():
    # no sat_area column -> base-radius behaviour: 0.1" dup absorbed, 0.5" kept
    t = _tbl([0.0, 0.1, 0.5], [300e3, 250e3, 200e3], sat_area=None)
    assert len(_dedup_satstar_catalog(t, target='w51')) == 2
