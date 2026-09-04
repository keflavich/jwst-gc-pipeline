"""Window-edge sweep alias + module-antisymmetry guards (issue #158).

W51 F335M/F360M/F405N read a fake ~28-30" per-exposure offset from the m2 visit
consensus, ANTISYMMETRIC across the two NIRCam modules (nrcalong at +d,
nrcblong at -d), `swept=True`, and with a razor-sharp error bar.  Two things had
to be true for that:

1. an internal tie between an nrcblong exposure and the nrcalong component union
   was accepted at the 60" sweep window although the two footprints do not
   overlap at all.  The "peak" there is the footprint cross-correlation ridge
   truncated by the window: measured on the real catalogs it sits AT the window
   edge and MOVES with it --
       window  55"   60"   66"   70"   80"   90"  100"
       off     54.8  59.8  64.7  67.2  75.9  89.0  99.5
   -- while a stable measurement reads the same offset at every window that can
   contain it (W51 F480M: (+20.6, -1.7) mas identically at 1/3/10/30/60").  That
   stability says the number is not a window artifact; whether the two modules
   really are that far apart is a separate question, and the reference-free
   measurement in the strip where they see the same stars says they are not
   (1.64 mas, issue #473).
2. that one bad tie merged both modules into a single consensus component, whose
   MEDIAN re-centring then split the error evenly between the modules -- which
   is why every exposure of one module read +d and every exposure of the other
   read -d.

The guards: a swept peak must REPRODUCE at an independent window
(``confirm_windows``), the per-exposure tie sweeps only to
``PER_EXPOSURE_SWEEP_WINDOWS``, and an antisymmetric per-module correction set
is refused outright (``detect_module_antisymmetry``).
"""
import pytest
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord

from jwst_gc_pipeline.photometry.astrometry_offsets import measure_offset
from jwst_gc_pipeline.photometry.visit_consensus import (
    DETECTOR_ANTISYMMETRY_MIN_MAS, MODULE_ANTISYMMETRY_MIN_MAS,
    PER_EXPOSURE_SWEEP_WINDOWS, detect_detector_antisymmetry,
    detect_module_antisymmetry, module_family)


def _clumpy_field(ra0, dec0, width_arcsec, seed, n_clump=300, per_clump=6,
                  sigma_arcsec=1.5):
    """A clustered star field — the nebulous-medium-band regime that makes the
    window-edge ridge sharp enough to clear the contrast floor."""
    rng = np.random.RandomState(seed)
    cra = ra0 + (rng.rand(n_clump) - 0.5) * width_arcsec / 3600.0
    cdec = dec0 + (rng.rand(n_clump) - 0.5) * width_arcsec / 3600.0
    ra = np.repeat(cra, per_clump) + rng.randn(n_clump * per_clump) * sigma_arcsec / 3600.0
    dec = np.repeat(cdec, per_clump) + rng.randn(n_clump * per_clump) * sigma_arcsec / 3600.0
    return SkyCoord(ra * u.deg, dec * u.deg)


def _adjacent_module_fields():
    """Two ~160" footprints 170" apart in dec — NO shared stars, exactly the
    nrcalong/nrcblong adjacency that produced the W51 alias."""
    a = _clumpy_field(290.915, 14.529, 160.0, seed=1)
    b = _clumpy_field(290.915, 14.529 - 170.0 / 3600.0, 160.0, seed=2)
    return a, b


def test_window_edge_alias_is_accepted_without_confirmation():
    """The bug: two footprints with no common star still produce a swept,
    contrast-clearing 'tie'.  This is the behaviour the guard has to remove."""
    a, b = _adjacent_module_fields()
    r = measure_offset(b, a, sweep=True)
    assert r is not None and r["ok"], r
    assert r["swept"], r
    assert r["off"] / 1000.0 > 10.0, r
    # the tell: the peak sits at the edge of its own search window
    assert r["window_edge_fraction"] > 0.7, r


def test_window_edge_alias_rejected_by_confirmation():
    a, b = _adjacent_module_fields()
    r = measure_offset(b, a, sweep=True, confirm_windows=True)
    assert r is not None
    assert not r["ok"], f"window-edge alias survived the confirmation: {r}"
    assert r["alias_rejected"] and r["window_consistent"] is False, r
    # every probe measured a DIFFERENT peak (it slid to the new window edge)
    probes = [p for p in r["window_confirmation"]["probes"] if p["dra"] is not None]
    assert probes and not any(p["agrees"] for p in probes), probes


def test_true_large_offset_survives_confirmation():
    """The brick-1182 v001 case must still be recovered: a genuine ~20" rigid
    offset reads the SAME at every window that can contain it."""
    rng = np.random.RandomState(3)
    ra = 266.4 + rng.rand(1500) * 0.05
    dec = -28.9 + rng.rand(1500) * 0.05
    a = SkyCoord(ra * u.deg, dec * u.deg)
    cosd = np.cos(np.radians(dec))
    b = SkyCoord((ra + 15.4 / 3600.0 / cosd) * u.deg, (dec - 13.5 / 3600.0) * u.deg)
    r = measure_offset(a, b, sweep=True, confirm_windows=True)
    assert r["ok"] and not r["alias_rejected"], r
    assert r["window_consistent"] is True, r
    assert abs(r["dra"] - 15400) < 300 and abs(r["ddec"] + 13500) < 300, r


def test_small_tie_is_numerically_untouched_by_confirmation():
    """F480M-unchanged, in hermetic form: the confirmation only ever runs on a
    SWEPT peak, so an ordinary mas/sub-arcsec tie is bit-for-bit identical."""
    rng = np.random.RandomState(11)
    ra = 266.4 + rng.rand(900) * 0.05
    dec = -28.9 + rng.rand(900) * 0.05
    a = SkyCoord(ra * u.deg, dec * u.deg)
    cosd = np.cos(np.radians(dec))
    b = SkyCoord((ra + 0.02 / 3600.0 / cosd) * u.deg, (dec + 0.01 / 3600.0) * u.deg)
    plain = measure_offset(a, b, sweep=True)
    guarded = measure_offset(a, b, sweep=True, confirm_windows=True)
    assert not plain["swept"], plain
    for k in ("dra", "ddec", "off", "contrast", "npairs", "window_arcsec", "ok"):
        assert plain[k] == guarded[k], (k, plain[k], guarded[k])
    assert guarded["window_consistent"] is None and not guarded["alias_rejected"]


# ---------------------------------------------------------------------------
# module antisymmetry, against the RECORDED W51 m2 measurements (2026-07-28,
# rebuilt from the on-disk m1 per-frame catalogs at /orange/.../w51/<FILT>/)
# ---------------------------------------------------------------------------

# F335M — the alias.  Every nrcalong exposure ~(+11.75", -27.57"), every
# nrcblong exposure the negation: the two modules 60" apart from each other.
_W51_F335M = {
    "nrcalong": [(11754.1, -27571.0), (11753.6, -27570.1), (11762.6, -27569.7),
                 (11761.8, -27569.6), (11753.1, -27563.8), (11750.7, -27563.8),
                 (11747.7, -27567.2), (11747.8, -27565.0)],
    "nrcblong": [(-11757.9, 27563.8), (-11758.3, 27564.5), (-11750.5, 27565.3),
                 (-11751.0, 27565.6), (-11760.3, 27571.0), (-11762.6, 27571.1),
                 (-11765.8, 27567.4), (-11766.1, 27569.4)],
}

# F480M — the control.  Correct, mas-scale, and it carries a REAL ~35 mas
# module split that is itself close to antisymmetric.  It must never be flagged.
_W51_F480M = {
    "nrcalong": [(20.6, -1.7), (19.9, -0.6), (28.9, 0.6), (28.1, 1.0),
                 (18.2, 5.4), (15.6, 5.7), (12.7, 1.3), (13.0, 3.3)],
    "nrcblong": [(-14.1, -20.7), (-14.9, -19.7), (-5.3, -19.5), (-6.0, -19.1),
                 (-12.7, -13.5), (-15.4, -13.4), (-19.5, -16.5), (-19.5, -14.7)],
}

# F410M — same field.  Exposure 1, (+17.2,+4.8) vs (-18.9,-11.8), is
# anti-parallel to cos = -0.96 with magnitudes within 20% and a 39.7 mas
# differential, so it is exactly the "+/-20 mas correction to the offset between
# modules" class the maintainer asked to discard (issue #473).  Measured
# reference-free in the strip where w51's two modules see the same stars, the
# F410M module relation is 2.40 mas and F480M's is 1.64 -- so the "~35 mas real
# module split" these fixtures were once said to carry is not in the products.
_W51_F410M = {
    "nrcalong": [(17.2, 4.8), (16.6, 5.6), (25.8, 6.8), (25.0, 7.2),
                 (14.8, 12.0), (12.6, 11.9), (9.5, 7.7), (9.8, 9.6)],
    "nrcblong": [(-18.9, -11.8), (-19.3, -11.0), (-9.6, -10.9), (-10.4, -10.4),
                 (-17.8, -4.7), (-20.0, -4.9), (-24.3, -7.8), (-24.4, -6.0)],
}


def _exposures(recorded, filtername):
    out = []
    for module, vals in recorded.items():
        for i, (dra, ddec) in enumerate(vals, start=1):
            out.append(dict(
                key=("1", i, module, filtername, "03103"),
                vs_consensus=dict(dra=dra, ddec=ddec,
                                  off=float(np.hypot(dra, ddec)), ok=True)))
    return out


def test_module_antisymmetry_detected_on_recorded_w51_alias():
    res = detect_module_antisymmetry(_exposures(_W51_F335M, "F335M"))
    assert res["detected"], res
    assert res["n_pairs_tested"] == 8 and res["n_antisymmetric"] == 8, res
    assert len(res["keys"]) == 16, res
    ex = res["examples"][0]
    assert ex["cos"] < -0.99, ex
    assert 55_000 < ex["separation_mas"] < 62_000, ex


def test_twenty_mas_module_differential_is_flagged():
    """W51 F480M/F410M carry +/-20 mas antisymmetric sets whose implied module
    differential is ~37-40 mas, where the reference-free strip measurement of
    the same field reads 1.6-2.4 mas.  Those corrections must be discarded, not
    applied (issue #473).  Before 2026-09-04 the 500 mas floor let every one of
    them through."""
    for recorded, filt in ((_W51_F480M, "F480M"), (_W51_F410M, "F410M")):
        res = detect_module_antisymmetry(_exposures(recorded, filt))
        assert res["detected"], (filt, res)
        assert res["n_pairs_tested"] == 8, (filt, res)
        assert res["keys"], (filt, res)
        for ex in res["examples"]:
            assert ex["separation_mas"] >= MODULE_ANTISYMMETRY_MIN_MAS, (filt, ex)


def test_the_floor_is_a_differential_not_a_per_module_magnitude():
    """+/-10 mas a side is a 20 mas differential and must fire.

    Issue #473 asks for "corrections of the order +/-20 mas to the offset
    BETWEEN modules" to be discarded, and the offset between modules is the A-B
    differential.  The scale test used to be on each module's OWN magnitude, so
    at any given floor it demanded twice as much as the issue asks for and twice
    as much as the reference-free strip measurement returns.  Here each module
    reads under the 15 mas floor while the pair still moves the two modules
    20 mas apart -- past every strip measurement on disk.
    """
    pair = {"nrcalong": [(10.0, 0.0), (9.8, 0.3)],
            "nrcblong": [(-10.0, 0.0), (-9.8, -0.3)]}
    res = detect_module_antisymmetry(_exposures(pair, "F410M"))
    assert res["detected"], res
    assert res["n_pairs_tested"] == 2 and res["n_antisymmetric"] == 2, res
    for ex in res["examples"]:
        assert np.hypot(ex["dra_a_mas"],
                        ex["ddec_a_mas"]) < MODULE_ANTISYMMETRY_MIN_MAS, ex
        assert ex["separation_mas"] >= MODULE_ANTISYMMETRY_MIN_MAS, ex


def test_module_relation_at_the_measured_scale_is_not_flagged():
    """The as-built module relation measured reference-free in the shared strip
    is 0.2-13 mas across brick/cloudc/sgrc/w51 (issue #473).  A consensus that
    reports a split at that scale is describing the products and must be left
    alone -- otherwise every field would be flagged."""
    measured = {
        "nrcalong": [(0.7, -0.3), (2.6, -1.1), (0.1, -0.1), (3.6, 1.6)],
        "nrcblong": [(-0.7, 0.3), (-2.6, 1.1), (-0.1, 0.1), (-3.6, -1.6)],
    }
    res = detect_module_antisymmetry(_exposures(measured, "F410M"))
    assert not res["detected"], res
    assert res["n_pairs_tested"] == 4, res
    assert not res["keys"], res


def test_antisymmetry_floor_is_below_the_appliable_ceiling():
    """The MODULE floor used to sit AT the per-exposure correction ceiling, so
    the guard could only ever fire on a correction already refused as
    unappliable -- which is why a +/-20 mas module differential was applied
    instead of discarded.  It is now well below that ceiling.

    The DETECTOR floor did not move.  The strip measurement that brought the
    module floor down is module-A stars against module-B stars and constrains
    nothing about two detectors of one module, so that guard keeps its old
    500 mas-a-side value restated as a 1000 mas differential.
    """
    from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
        MAX_CORRECTION_ARCSEC)
    from jwst_gc_pipeline.photometry.visit_consensus import (
        DETECTOR_ANTISYMMETRY_MIN_MAS, MODULE_ANTISYMMETRY_MIN_MAS)
    assert MODULE_ANTISYMMETRY_MIN_MAS < MAX_CORRECTION_ARCSEC * 1000.0
    # twice the old per-module floor, because this one is a DIFFERENTIAL: an
    # opposed, equal-magnitude pair at 500 mas a side is 1000 mas apart.
    assert DETECTOR_ANTISYMMETRY_MIN_MAS == 2.0 * 500.0
    assert MODULE_ANTISYMMETRY_MIN_MAS < DETECTOR_ANTISYMMETRY_MIN_MAS


def test_detector_floor_did_not_follow_the_module_floor_down():
    """A 20 mas antisymmetric DETECTOR pair inside one module is not flagged.

    The module floor came down on a module-vs-module overlap measurement; the
    o049 nrca3 divergence (23 mas, #585) is the detector-vs-detector class and
    nothing in that measurement covers it.  Sharing one constant discarded 56
    of the 194 corrections in the archive replay on evidence that is not about
    them.
    """
    exps = []
    for det, dra in (("nrca1", 10.0), ("nrca2", -10.0),
                     ("nrca3", 0.2), ("nrca4", -0.1)):
        exps.append(dict(key=("1", 1, det, "F212N", "03103"),
                         vs_consensus=dict(dra=dra, ddec=0.0, ok=True)))
    res = detect_detector_antisymmetry(exps)
    assert not res["detected"], res
    assert not res["keys"], res
    # ...and the same pair at the detector guard's own floor still fires
    gross = [dict(e, vs_consensus=dict(e["vs_consensus"],
                                       dra=e["vs_consensus"]["dra"] * 60.0))
             for e in exps]
    res = detect_detector_antisymmetry(gross)
    assert res["detected"], res
    assert {k[2] for k in res["keys"]} == {"nrca1", "nrca2"}, res


def test_module_family_mapping():
    assert module_family("nrcalong") == "a"
    assert module_family("nrca3") == "a"
    assert module_family("NRCB1") == "b"
    assert module_family("merged") == "merged"     # own family, never paired


def test_per_exposure_sweep_is_bounded_below_module_geometry():
    """A per-exposure tie is mas-scale; its sweep must not reach the scales at
    which module/mosaic geometry lives (the W51 ridge was ~56", the SIAF
    nrcalong<->nrcblong separation ~174")."""
    assert max(PER_EXPOSURE_SWEEP_WINDOWS) <= 15.0
    assert min(PER_EXPOSURE_SWEEP_WINDOWS) >= 1.0


def test_bounded_sweep_still_finds_a_confirmed_gross_offset():
    """The bound must not cost gross-frame DETECTION.  An offset outside
    ``PER_EXPOSURE_SWEEP_WINDOWS`` whose peak reproduces at an independent window
    is still found by the fallback; only an UNCONFIRMED wide peak is dropped.

    (The end-to-end version of this is
    ``test_visit_consensus.test_huge_misalignment_found_by_sweep``, which drives
    the same split through ``build_visit_consensus``.)"""
    rng = np.random.RandomState(5)
    ra = 266.4 + rng.rand(1200) * 0.05
    dec = -28.9 + rng.rand(1200) * 0.05
    a = SkyCoord(ra * u.deg, dec * u.deg)
    cosd = np.cos(np.radians(dec))
    b = SkyCoord((ra + 20.0 / 3600.0 / cosd) * u.deg, (dec + 4.0 / 3600.0) * u.deg)
    bounded = measure_offset(a, b, sweep=True,
                             sweep_windows=PER_EXPOSURE_SWEEP_WINDOWS,
                             confirm_windows=True)
    assert bounded is None or not bounded["ok"], bounded   # out of the bound
    wide = measure_offset(a, b, sweep=True, confirm_windows=True)
    assert wide["ok"] and wide["window_consistent"] is True, wide
    assert abs(wide["dra"] - 20000) < 400 and abs(wide["ddec"] - 4000) < 400, wide


# ---------------------------------------------------------------------------
# DETECTOR antisymmetry (issue #624), against the RECORDED ngc6334 m2 run
# (checkpoint_m2_F090W_j6778_latest.json, 2026-09-02, correcting=True,
# 35 corrections emitted).  nrca1 and nrca2 of the SAME module read
# equal-and-opposite ~22.9", and the module guard recorded
# detected=False / n_pairs_tested=4 / n_antisymmetric=0 -- it tested those
# groups and passed them, because the nrca family MEAN of
# (+22903, -22918, -1.5) is -5.4 mas, far below the 500 mas floor.
# ---------------------------------------------------------------------------

_NGC6334_F090W = {
    ("jw06778001001", 1): {"nrca1": (22903.0, 2153.8), "nrca2": (-22917.8, -2154.9),
                           "nrca3": (-1.5, -7.4), "nrcb4": (-7.7, 1.2)},
    ("jw06778001001", 2): {"nrcb4": (-6.4, 1.5)},
    ("jw06778001001", 3): {"nrca1": (22911.2, 2153.9), "nrca2": (-22904.0, -2156.0)},
    ("jw06778001001", 4): {"nrca1": (22912.2, 2158.3), "nrca2": (-22898.7, -2153.6),
                           "nrcb4": (5.6, 2.0)},
    ("jw06778001001", 5): {"nrca1": (22911.7, 2156.1), "nrca2": (-22898.9, -2155.6),
                           "nrcb4": (5.9, -0.8)},
    ("jw06778001002", 2): {"nrca2": (-8.0, -5.6), "nrca3": (-2.9, -3.3),
                           "nrcb3": (-10.6, -6.3)},
    ("jw06778001002", 3): {"nrca3": (4.8, 4.0)},
    ("jw06778001002", 4): {"nrca2": (4.0, -1.8)},
}


def _detector_exposures(recorded, filtername="F090W", vgroup="02103"):
    out = []
    for (visit, exposure), dets in recorded.items():
        for det, (dra, ddec) in dets.items():
            out.append(dict(
                key=(visit, exposure, det, filtername, vgroup),
                vs_consensus=dict(dra=dra, ddec=ddec,
                                  off=float(np.hypot(dra, ddec)), ok=True)))
    return out


def test_module_guard_misses_the_recorded_ngc6334_within_module_alias():
    """The #624 bug, pinned to the recorded run: the module guard TESTS these
    exposures and passes them, because averaging +22.9" with -22.9" inside one
    module collapses the family mean to a few mas."""
    res = detect_module_antisymmetry(_detector_exposures(_NGC6334_F090W))
    assert not res["detected"], res
    assert res["n_antisymmetric"] == 0, res
    assert not res["keys"], res


def test_detector_antisymmetry_detected_on_recorded_ngc6334_alias():
    res = detect_detector_antisymmetry(_detector_exposures(_NGC6334_F090W))
    assert res["detected"], res
    # exposures 1, 3, 4 and 5 of visit ...001 each carry one nrca1/nrca2 pair
    assert res["n_antisymmetric"] == 4, res
    flagged = {(k[0], k[1], k[2]) for k in res["keys"]}
    assert flagged == {("jw06778001001", e, d)
                       for e in (1, 3, 4, 5) for d in ("nrca1", "nrca2")}, flagged
    ex = next(e for e in res["examples"] if e["same_module"])
    assert ex["cos"] < -0.99, ex
    assert 45_000 < ex["separation_mas"] < 47_000, ex


def test_detector_guard_leaves_the_correct_detectors_alone():
    """nrca3 and nrcb4 measure at mas scale in the very same exposures and must
    not be swept up, and the all-correct visit ...002 must stay clean."""
    res = detect_detector_antisymmetry(_detector_exposures(_NGC6334_F090W))
    assert not any(k[2] in ("nrca3", "nrcb3", "nrcb4") for k in res["keys"]), res
    assert not any(k[0] == "jw06778001002" for k in res["keys"]), res


def test_detector_guard_still_catches_the_cross_module_w51_alias():
    """The detector guard subsumes the module case: W51 F335M fires in both."""
    res = detect_detector_antisymmetry(_exposures(_W51_F335M, "F335M"))
    assert res["detected"] and res["n_antisymmetric"] == 8, res
    assert len(res["keys"]) == 16, res
    assert not any(e["same_module"] for e in res["examples"]), res["examples"]


def test_detector_guard_does_not_flag_real_module_splits():
    """Same controls as the module guard, at detector granularity -- including
    W51 F410M, whose real split is anti-parallel at cos = -0.96."""
    for recorded, filt in ((_W51_F480M, "F480M"), (_W51_F410M, "F410M")):
        res = detect_detector_antisymmetry(_exposures(recorded, filt))
        assert not res["detected"], (filt, res)
        assert not res["keys"], (filt, res)


# ---------------------------------------------------------------------------
# What DETECTION does and does not establish, and what the checkpoint owes the
# release gate because of it (issue #473).
# ---------------------------------------------------------------------------

RA0, DEC0 = 266.5, -28.7
COSD = np.cos(np.radians(DEC0))


def _star_field(n=400, extent_arcsec=90.0, seed=42):
    rng = np.random.default_rng(seed)
    ra = RA0 + rng.uniform(0, extent_arcsec, n) / 3600.0 / COSD
    dec = DEC0 + rng.uniform(0, extent_arcsec, n) / 3600.0
    return ra, dec


def _frame_table(ra, dec, exposure, module, dra_mas=0.0, filtername="F212N"):
    """Synthetic per-frame catalog carrying a rigid on-sky offset."""
    from astropy.table import Table
    rng = np.random.default_rng(1000 + exposure)
    n = len(ra)
    tbl = Table()
    tbl["skycoord"] = SkyCoord(
        ra=(ra + (dra_mas + rng.normal(0, 1.0, n)) / 3.6e6 / COSD) * u.deg,
        dec=(dec + rng.normal(0, 1.0, n) / 3.6e6) * u.deg, frame="icrs")
    tbl["flux_fit"] = rng.uniform(1e3, 1e5, n)
    tbl["flux_err"] = tbl["flux_fit"] / 100.0
    tbl["qfit"] = rng.uniform(0.01, 0.05, n)
    tbl.meta.update(VISIT="001", EXPOSURE=f"{exposure:05d}", MODULE=module,
                    FILTER=filtername, RAOFFSET=0.1, DEOFFSET=-0.05)
    return tbl


def _split_visit(module_b_offset_mas, filtername="F212N"):
    """Two exposures x two modules, module B rigidly offset on sky."""
    ra, dec = _star_field()
    tables = []
    for e in (1, 2):
        tables.append(_frame_table(ra, dec, e, "nrca1", 0.0, filtername))
        tables.append(_frame_table(ra, dec, e, "nrcb1", module_b_offset_mas,
                                   filtername))
    return tables


@pytest.mark.parametrize("injected", [20.0, 200.0])
def test_antisymmetric_shape_is_forced_by_the_median_recentring(injected):
    """`detected` does NOT mean "alias" -- the shape is an identity, not evidence.

    `build_visit_consensus` re-centres each component on the MEDIAN of its
    members' relative offsets, so a component whose exposures split evenly
    between two module families comes back at exactly +D/2 and -D/2 whatever D
    is and wherever it came from.  Here D is a REAL rigid shift injected into
    module B's catalogs, and it still satisfies both shape tests to four
    significant figures.

    So the guard's cos/magnitude conditions carry no information for a
    two-family component (every NIRCam module split, and NIRCam LW at detector
    granularity), and a real inter-module misregistration -- the brick-1182
    F200W ~90 mas seam class -- is indistinguishable from the issue-158 alias.
    That is why the checkpoint must both discard the corrections AND refuse the
    visit rather than choosing between them.
    """
    from jwst_gc_pipeline.photometry.visit_consensus import (
        build_visit_consensus)
    cons = build_visit_consensus(_split_visit(injected), context="degeneracy")
    seen = {tuple(e["key"])[1:3]: e["vs_consensus"] for e in cons["exposures"]}
    for (_exp, module), res in seen.items():
        expected = injected / 2.0 * (1.0 if module == "nrca1" else -1.0)
        assert res["dra"] == pytest.approx(expected, abs=0.5), (module, res)
    res = detect_module_antisymmetry(cons["exposures"])
    assert res["detected"], res
    for ex in res["examples"]:
        assert ex["cos"] < -0.999, ex
        na = np.hypot(ex["dra_a_mas"], ex["ddec_a_mas"])
        nb = np.hypot(ex["dra_b_mas"], ex["ddec_b_mas"])
        assert abs(na - nb) < 0.01 * max(na, nb), ex
        assert ex["separation_mas"] == pytest.approx(injected, abs=1.0), ex


def test_module_antisymmetric_set_is_discarded_AND_refused(tmp_path):
    """A 200 mas module split emits no corrections and does not pass.

    Both halves matter.  Discarding alone is what let this class through: the
    exposures are dropped from the correction path, the visit is reported as
    merely `unverified`, `passed` stays True, and the release gate
    (check_astrometry_checkpoints.py, which reads `unverified_blocking`) never
    hears about a misregistration the pipeline measured and declined to act on.
    """
    from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
        run_visit_checkpoint)
    rec = run_visit_checkpoint(_split_visit(200.0), "m2", filtername="F212N",
                               record_dir=str(tmp_path), context="test")
    assert rec["correcting"]
    assert rec["corrections"] == [], rec["corrections"]
    blocking = [b for b in rec["unverified_blocking"]
                if "MODULE-ANTISYMMETRIC" in b]
    assert blocking, rec["unverified_blocking"]
    assert rec["passed"] is False, rec
    anti = rec["visits"][0]["module_antisymmetry"]
    assert anti["detected"] and anti["n_antisymmetric"] == 2, anti
    # the detector guard's own verdict now reaches disk as well, so a
    # detector-level discard is auditable instead of existing only as prose
    det = rec["visits"][0]["detector_antisymmetry"]
    assert det["min_mas"] == DETECTOR_ANTISYMMETRY_MIN_MAS, det
    assert det["n_pairs_tested"] == 2 and not det["detected"], det


def test_frozen_stage_movement_survives_the_alias_guard(tmp_path, monkeypatch):
    """A flagged exposure that MOVED since the m2 freeze still fails.

    The alias explanation is about the ABSOLUTE vs-consensus reading; a frozen
    stage compares a DELTA against m2's own value for the same exposure, and a
    static footprint-geometry alias cancels in that delta.  Suppressing the
    frozen branch as well as the correction branch was the one way an
    antisymmetric visit could move after the freeze and still report no
    failure.
    """
    monkeypatch.setenv("ASTROM_CHECKPOINT_ENFORCE", "release")
    from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
        run_visit_checkpoint)
    m2 = run_visit_checkpoint(_split_visit(100.0), "m2", filtername="F212N",
                              record_dir=str(tmp_path), context="test")
    assert m2["visits"][0]["module_antisymmetry"]["detected"], m2
    # the split widens to 300 mas: every exposure moves 100 mas, and the set is
    # still antisymmetric so the guard still flags all four
    m4 = run_visit_checkpoint(_split_visit(300.0), "m4", filtername="F212N",
                              record_dir=str(tmp_path), context="test")
    assert m4["visits"][0]["module_antisymmetry"]["detected"], m4
    moved = [f for f in m4["failures"] if "MOVED" in f]
    assert len(moved) == 4, m4["failures"]
    assert m4["passed"] is False, m4
