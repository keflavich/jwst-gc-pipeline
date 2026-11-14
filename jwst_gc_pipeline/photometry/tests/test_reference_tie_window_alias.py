"""The BULK reference tie must run the window-edge alias probe (issue #257).

``measure_offset`` has carried the alias rejector since #158, and
``diagnostics/astrometry_figs.py`` already passes ``confirm_windows=True``, but
``measure_reference_tie`` called ``measure_offset`` with the default
(``confirm_windows=False``).  So w51's checkpoints recorded

    off_mas 7827.079, bulk_source "histogram", swept true,
    window_consistent null, alias_rejected false

with ``window_edge_fraction`` 0.783 at m2 against 0.005 at m3, and the m2 peak
scaling with the search window at every scale (3"->2790, 10"->7827, 30"->19052,
60"->34766) while m3 read 32 mas same-star.  A real tie reads the same offset at
every window that can contain it; that one was a property of the window.

These tests are the reference-tie-level counterpart of
``test_sweep_window_alias.py`` (which covers ``measure_offset`` itself).  What
changes is the DIAGNOSIS, not any field's verdict: every swept reference tie in
the archive is w51's and already carried ``apply_ok: false``.
"""
import numpy as np
import pytest
import astropy.units as u
from astropy.coordinates import SkyCoord

from jwst_gc_pipeline.photometry import astrometry_offsets as ao
from jwst_gc_pipeline.photometry.visit_consensus import measure_reference_tie


@pytest.fixture(autouse=True)
def _confirmation_path_without_the_significance_gate(monkeypatch):
    """These tests pin the window-confirmation machinery, which a peak reaches
    only after ``ok`` is set.  Since issue #1232 the significance gate rejects
    a footprint-ridge alias at the wide windows before confirmation runs (see
    ``test_window_edge_alias_fails_the_significance_gate``), so the gate is
    switched off here to keep exercising the probes."""
    monkeypatch.setattr(ao, "SIGNIFICANCE_GATE_MIN_WINDOW_ARCSEC", np.inf)


def _clumpy_field(ra0, dec0, width_arcsec, seed, n_clump=300, per_clump=6,
                  sigma_arcsec=0.5):
    """A clustered star field -- the regime that makes the window-edge ridge
    sharp enough to clear the contrast floor (see test_sweep_window_alias).

    The clumps are 0.5" wide (1.5" before issue #1232).  The window-scaled bin is
    now capped at 0.1", and a ridge as smooth as a 1.5" clump spreads over so
    many fine bins that its contrast falls below the floor; the narrower clumps
    keep the alias sharp enough to clear it, which is the regime these tests
    guard."""
    rng = np.random.RandomState(seed)
    cra = ra0 + (rng.rand(n_clump) - 0.5) * width_arcsec / 3600.0
    cdec = dec0 + (rng.rand(n_clump) - 0.5) * width_arcsec / 3600.0
    ra = np.repeat(cra, per_clump) + rng.randn(n_clump * per_clump) * sigma_arcsec / 3600.0
    dec = np.repeat(cdec, per_clump) + rng.randn(n_clump * per_clump) * sigma_arcsec / 3600.0
    return SkyCoord(ra * u.deg, dec * u.deg)


def _disjoint_footprints():
    """Two ~160" footprints 170" apart in dec -- NO shared stars, so any peak
    between them is the truncated footprint cross-correlation ridge."""
    a = _clumpy_field(290.915, 14.529, 160.0, seed=1)
    b = _clumpy_field(290.915, 14.529 - 170.0 / 3600.0, 160.0, seed=2)
    return a, b


def _shifted_pair(dra_arcsec, ddec_arcsec, n=1200, seed=7):
    rng = np.random.RandomState(seed)
    ra = 290.9 + rng.rand(n) * 0.05
    dec = 14.5 + rng.rand(n) * 0.05
    a = SkyCoord(ra * u.deg, dec * u.deg)
    cosd = np.cos(np.radians(dec))
    b = SkyCoord((ra + dra_arcsec / 3600.0 / cosd) * u.deg,
                 (dec + ddec_arcsec / 3600.0) * u.deg)
    return a, b


def test_footprint_alias_is_rejected_by_the_reference_tie():
    """The defect: this read as an ordinary swept tie with window_consistent
    null.  It is now an explicit alias rejection."""
    ref_all, consensus = _disjoint_footprints()
    res = measure_reference_tie(consensus, ref_all, ref_all[::10],
                                dense=True, context="alias-test")
    vf = res["vs_full"]
    assert vf is not None and vf["swept"], vf
    assert vf["window_edge_fraction"] > 0.7, vf
    assert vf["alias_rejected"] is True, vf
    assert vf["window_consistent"] is False, vf
    assert vf["ok"] is False, vf
    assert res["apply_ok"] is False, res


def test_alias_record_keeps_the_probe_evidence():
    """A reviewer must be able to see WHY it was rejected, from the record."""
    ref_all, consensus = _disjoint_footprints()
    res = measure_reference_tie(consensus, ref_all, ref_all[::10],
                                dense=True, context="alias-test")
    conf = res["vs_full"]["window_confirmation"]
    probes = [p for p in conf["probes"] if p["dra"] is not None]
    assert probes, conf
    assert not any(p["agrees"] for p in probes), probes


def test_small_tie_is_untouched():
    """The common case: an un-swept mas-scale tie never runs the probe, so the
    reported numbers and apply_ok are unchanged."""
    ref_all, consensus = _shifted_pair(0.02, -0.01)
    res = measure_reference_tie(consensus, ref_all, ref_all[::10],
                                dense=True, context="small-tie")
    vf = res["vs_full"]
    assert vf["ok"] and not vf["swept"], vf
    assert vf["window_consistent"] is None and vf["alias_rejected"] is False, vf
    assert abs(res["dra_mas"] + 20.0) < 30.0 and abs(res["ddec_mas"] - 10.0) < 30.0, res


def test_genuine_large_offset_still_reported():
    """A real rigid gross shift reproduces at the wider probes, so the
    confirmation does not take it away (the brick-1182 v001 requirement)."""
    ref_all, consensus = _shifted_pair(-15.4, 13.5, seed=3)
    res = measure_reference_tie(consensus, ref_all, ref_all[::10],
                                dense=True, context="real-large-tie")
    vf = res["vs_full"]
    assert vf["swept"], vf
    assert vf["alias_rejected"] is False, vf
    assert vf["window_consistent"] is True, vf
    assert vf["ok"] is True, vf
    assert abs(vf["dra"] - 15400) < 400 and abs(vf["ddec"] + 13500) < 400, vf
