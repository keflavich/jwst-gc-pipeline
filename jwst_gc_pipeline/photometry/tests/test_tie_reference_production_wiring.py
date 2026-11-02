"""Pin that the PRODUCTION call site passes `resolve_tie_reference`'s return
value to `run_visit_checkpoint`, not the raw VIRAC2 `refcat` (PR #988 review,
mutant M5).

`astrometry_checkpoint.resolve_tie_reference` substitutes a field's reference
filter's own JWST consensus catalog for VIRAC2 when
`alignment_config.tie_through_reference_filter` is set
(`test_reference_filter_consensus_tie.py` covers that function in isolation).
But every one of those tests calls `run_visit_checkpoint` directly with a
hand-built reference dict, so none of them can see whether the ONE production
caller, `cataloging._run_astrometry_stage_checkpoint`, actually threads the
resolved value through. Reverting `refcat=tie_refcat` back to `refcat=refcat`
at that call site leaves every existing test green -- this is the mutant the
reviewer found surviving.

This test drives the real entry point (not a hand-rolled stand-in) with two
on-disk per-frame catalogs, stubs out the pieces that are expensive/irrelevant
here (`_drop_foreign_obs_duplicates`, the satstar consensus loader), and
monkeypatches `resolve_tie_reference` and `run_visit_checkpoint` at their
SOURCE module (`astrometry_checkpoint`) so the function's own local
`from ... import ...` picks up the stubs. It then asserts `run_visit_checkpoint`
receives exactly `resolve_tie_reference`'s return value as `refcat`.
"""
import types

from astropy.table import Table

from jwst_gc_pipeline.photometry import astrometry_checkpoint as ac
from jwst_gc_pipeline.photometry import cataloging as _cat


class _Opt:
    def __init__(self, **kw):
        self.target = "gc-treasury_o063"
        self.proposal_id = "10678"
        self.field = "063"
        self.modules = "nrcb"
        self.each_exposure = True
        self.cutout_region = ""
        for k, v in kw.items():
            setattr(self, k, v)


def _write_perframe(cut_bp, filt="F480M", module="nrcblong", n=2):
    d = cut_bp / filt
    d.mkdir(parents=True, exist_ok=True)
    for i in range(1, n + 1):
        t = Table({"x": [1.0]})
        t.write(str(d / f"{filt.lower()}_{module}_visit001_vgroup02101"
                        f"_exp{i:05d}_m2_daophot_basic.fits"))


def test_run_visit_checkpoint_receives_the_resolved_reference(tmp_path, monkeypatch):
    monkeypatch.delenv("ASTROM_CHECKPOINT", raising=False)
    monkeypatch.delenv("ASTROM_CHECKPOINT_WARN_ONLY", raising=False)
    monkeypatch.setenv("ASTROM_SATSTAR_CONSENSUS", "0")

    cut_bp = tmp_path / "cutouts" / "merged"
    _write_perframe(cut_bp)

    # Bypass the obs-dedup filter -- it is not what this test is about, and
    # its real logic needs a registered field/token setup covered elsewhere
    # (test_perframe_catalog_selection.py, test_foreign_obs_provenance.py).
    monkeypatch.setattr(_cat, "_drop_foreign_obs_duplicates",
                        lambda fns, *a, **k: fns)

    refcat_cache = {"refcat": "REAL_VIRAC2_REFCAT"}
    sentinel = {"all": "SUBSTITUTED_F212N_CONSENSUS", "sentinel": True}

    resolve_calls = []

    def _fake_resolve_tie_reference(refcat, *a, **k):
        resolve_calls.append(refcat)
        return sentinel

    monkeypatch.setattr(ac, "resolve_tie_reference", _fake_resolve_tie_reference)

    seen = {}

    def _fake_run_visit_checkpoint(tables, merge_label, refcat=None, **kw):
        seen["refcat"] = refcat
        return dict(passed=True, failures=[], corrections=[],
                   unverified_blocking=[], record_path="x")

    monkeypatch.setattr(ac, "run_visit_checkpoint", _fake_run_visit_checkpoint)

    _cat._run_astrometry_stage_checkpoint(
        "m2", "merged", "F480M", str(cut_bp), str(tmp_path), "10678",
        _Opt(), refcat_cache, context="test")

    # Sanity: resolve_tie_reference was actually reached, with the real
    # refcat as input, before asserting what came out of it was used.
    assert resolve_calls == ["REAL_VIRAC2_REFCAT"], resolve_calls
    assert "refcat" in seen, "run_visit_checkpoint was never called"
    assert seen["refcat"] is sentinel, (
        f"run_visit_checkpoint was called with {seen['refcat']!r} -- the "
        f"production call site must pass resolve_tie_reference's return "
        f"value (`tie_refcat`), never the raw refcat_cache value")
    # And refcat_cache itself must be untouched -- the m7 cross-filter
    # checkpoint shares this dict and must keep seeing the real VIRAC2 refcat.
    assert refcat_cache["refcat"] == "REAL_VIRAC2_REFCAT"
