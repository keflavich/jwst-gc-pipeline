"""A frozen-stage checkpoint is measured once per (stage, filter, inputs).

`_run_astrometry_stage_checkpoint` globs every detector of a filter, so the
nrca, nrcb and merged blocks of one run hand it the same per-frame catalogs.
At a frozen stage nothing else the measurement reads changes between those
calls, and the sgrb2 m7 finalize (2026-10) spent ~29 min per block measuring
the same thing three times.  The record measured first is reused; the verdict
handling still runs for every module.  A correcting stage always re-measures.
"""
import copy
import os
import types

import pytest
from astropy.table import Table

from jwst_gc_pipeline.photometry import astrometry_checkpoint as _ac
from jwst_gc_pipeline.photometry import cataloging as _cat


def _name(det, exp, label, filt="f200w"):
    seg = "" if label in _ac.CORRECTION_STAGES else "_resbgsub"
    return (f"{filt}_{det}_visit001_vgroup02101_exp{exp:05d}{seg}_{label}"
            f"_daophot_basic.fits")


def _layout(tmp_path, label, filt="f200w"):
    d = tmp_path / filt.upper()
    d.mkdir(parents=True, exist_ok=True)
    (tmp_path / "catalogs").mkdir(parents=True, exist_ok=True)
    paths = []
    for det in ("nrca1", "nrcb1"):
        for exp in (1, 2):
            p = d / _name(det, exp, label, filt)
            Table({"x": [1.0, 2.0]}).write(p, overwrite=True)
            paths.append(p)
    return paths


@pytest.fixture
def measured(monkeypatch):
    """Replace the measurement with a counter; return the list of contexts."""
    monkeypatch.delenv("PSF_EPSF_CORE_DIR", raising=False)
    monkeypatch.delenv("ASTROM_CHECKPOINT_WARN_ONLY", raising=False)
    monkeypatch.delenv("ASTROM_CHECKPOINT_REUSE", raising=False)
    monkeypatch.delenv(_ac.CHECKPOINT_ENFORCE_ENV, raising=False)
    monkeypatch.setenv("ASTROM_SATSTAR_CONSENSUS", "0")
    monkeypatch.setattr(_cat, "_astrom_checkpoint_refcat", lambda *a, **k: None)
    monkeypatch.setattr(_ac, "resolve_tie_reference",
                        lambda refcat, *a, **k: refcat)
    calls = []
    outcome = {"failures": []}

    def _fake(tables, stage, *, context="", **kw):
        calls.append(context)
        failures = list(outcome["failures"])
        return dict(stage=stage, context=context, failures=failures,
                    unverified_blocking=[], corrections=[],
                    passed=not failures,
                    record_path=f"/records/checkpoint_{stage}_{len(calls)}.json")

    monkeypatch.setattr(_ac, "run_visit_checkpoint", _fake)
    return calls, outcome


def _run(tmp_path, label, module, cache, filt="f200w"):
    _cat._run_astrometry_stage_checkpoint(
        label, module, filt, str(tmp_path), str(tmp_path), "4147",
        types.SimpleNamespace(cutout_region=""), cache,
        context=f"test {filt}/{module}")


@pytest.mark.parametrize("label", ["m3", "m5", "m7"])
def test_a_frozen_stage_is_measured_once_for_the_three_modules(
        tmp_path, measured, capsys, label):
    calls, _ = measured
    _layout(tmp_path, label)
    cache = {}
    for module in ("nrca", "nrcb", "merged"):
        _run(tmp_path, label, module, cache)
    assert calls == ["test f200w/nrca"]
    out = capsys.readouterr().out
    assert out.count("REUSING the record measured earlier in this run") == 2
    assert "/records/checkpoint_" + label + "_1.json" in out
    # every module still reports its own verdict
    for module in ("nrca", "nrcb", "merged"):
        assert f"[{label}] f200w/{module}: PASS" in out


@pytest.mark.parametrize("label", sorted(_ac.CORRECTION_STAGES))
def test_a_correcting_stage_is_always_remeasured(tmp_path, measured, capsys,
                                                 label):
    """Its corrections are applied by the caller and must come from the run
    that applies them."""
    calls, _ = measured
    _layout(tmp_path, label)
    cache = {}
    for module in ("nrca", "nrcb", "merged"):
        _run(tmp_path, label, module, cache)
    assert len(calls) == 3
    assert "REUSING" not in capsys.readouterr().out


def test_each_filter_is_measured_on_its_own(tmp_path, measured):
    calls, _ = measured
    _layout(tmp_path, "m5", filt="f200w")
    _layout(tmp_path, "m5", filt="f212n")
    cache = {}
    for module in ("nrca", "nrcb"):
        for filt in ("f200w", "f212n"):
            _run(tmp_path, "m5", module, cache, filt=filt)
    assert calls == ["test f200w/nrca", "test f212n/nrca"]


def test_a_rewritten_catalog_is_remeasured(tmp_path, measured):
    calls, _ = measured
    paths = _layout(tmp_path, "m5")
    cache = {}
    _run(tmp_path, "m5", "nrca", cache)
    Table({"x": [1.0, 2.0, 3.0]}).write(paths[0], overwrite=True)
    st = os.stat(paths[0])
    os.utime(paths[0], ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    _run(tmp_path, "m5", "nrcb", cache)
    _run(tmp_path, "m5", "merged", cache)
    assert calls == ["test f200w/nrca", "test f200w/nrcb"]


def test_a_different_input_set_is_remeasured(tmp_path, measured):
    calls, _ = measured
    _layout(tmp_path, "m5")
    cache = {}
    _run(tmp_path, "m5", "nrca", cache)
    Table({"x": [1.0]}).write(
        tmp_path / "F200W" / _name("nrcb2", 1, "m5"), overwrite=True)
    _run(tmp_path, "m5", "nrcb", cache)
    assert len(calls) == 2


def test_reuse_can_be_turned_off(tmp_path, measured, monkeypatch):
    calls, _ = measured
    monkeypatch.setenv("ASTROM_CHECKPOINT_REUSE", "0")
    _layout(tmp_path, "m5")
    cache = {}
    for module in ("nrca", "nrcb", "merged"):
        _run(tmp_path, "m5", module, cache)
    assert len(calls) == 3


def test_a_reused_failure_is_reported_for_every_module(tmp_path, measured,
                                                       capsys):
    """Reuse skips the measurement, never the verdict: a deferred frozen-stage
    failure prints for nrcb and merged exactly as it did for nrca."""
    calls, outcome = measured
    outcome["failures"] = ["visit 1 exp 1 moved 9 mas"]
    _layout(tmp_path, "m5")
    cache = {}
    for module in ("nrca", "nrcb", "merged"):
        _run(tmp_path, "m5", module, cache)
    assert len(calls) == 1
    out = capsys.readouterr().out
    for module in ("nrca", "nrcb", "merged"):
        assert f"[m5] f200w/{module}: FAILED -- 1 failure(s)" in out
    assert out.count("the chain CONTINUES") == 3


def test_the_stored_and_the_reused_record_are_copies(tmp_path, measured,
                                                     monkeypatch):
    """The verdict handling may annotate the record it is given; that must not
    leak into the record the next module reuses."""
    calls, _ = measured
    returned = []
    fake = _ac.run_visit_checkpoint

    def _keep(*a, **k):
        returned.append(fake(*a, **k))
        return returned[-1]

    monkeypatch.setattr(_ac, "run_visit_checkpoint", _keep)
    _layout(tmp_path, "m5")
    cache = {}
    _run(tmp_path, "m5", "nrca", cache)
    (stored,) = cache[_cat._FROZEN_STAGE_RECORDS_KEY].values()
    assert stored == returned[0]
    assert stored is not returned[0]
    assert stored["failures"] is not returned[0]["failures"]

    snapshot = copy.deepcopy(stored)
    seen = []
    real_deepcopy = _cat.copy.deepcopy

    def _spy(obj, *a, **k):
        out = real_deepcopy(obj, *a, **k)
        seen.append(out)
        return out

    monkeypatch.setattr(_cat.copy, "deepcopy", _spy)
    _run(tmp_path, "m5", "nrcb", cache)
    assert len(calls) == 1
    reused = [o for o in seen if o == snapshot and o is not stored]
    assert reused
    reused[0]["failures"].append("annotated by the verdict handling")
    assert stored == snapshot


def test_a_demoted_regression_is_not_cached(tmp_path, measured, monkeypatch):
    """warn_only demotes the raise and leaves no record to reuse, so the next
    module measures again (and is demoted again)."""
    calls, _ = measured
    monkeypatch.setenv("ASTROM_CHECKPOINT_WARN_ONLY", "1")

    def _raise(tables, stage, *, context="", **kw):
        calls.append(context)
        raise _ac.AstrometryRegressionError("moved")

    monkeypatch.setattr(_ac, "run_visit_checkpoint", _raise)
    _layout(tmp_path, "m5")
    cache = {}
    for module in ("nrca", "nrcb"):
        _run(tmp_path, "m5", module, cache)
    assert len(calls) == 2
    assert not cache.get(_cat._FROZEN_STAGE_RECORDS_KEY)
