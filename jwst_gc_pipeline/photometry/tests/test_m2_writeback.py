"""Write-back of m2's module-common per-exposure offsets into a locked table.

The table and record shapes are the live sgrb2 ones: one row per (visit,
exposure, filter, module family, vgroup), ``dra (arcsec)``/``ddec (arcsec)`` as
the pair fix_alignment reads, and m2 entries keyed
``(visit, exposure, detector, filter, vgroup)`` with the baked RAOFFSET/DEOFFSET
of the catalog they were measured on.
"""
import json
import os

import numpy as np
import pytest
from astropy.table import Table

from jwst_gc_pipeline.photometry.m2_writeback import (
    DEFAULT_MIN_MAS, M2WritebackError, TABLE_DEC_COL, TABLE_RA_COL,
    WRITEBACK_STAGE, apply_writeback, full_visit_token, writeback_plan)

PROP, OBS = "5365", "001"
VISIT = "jw05365001001"
DEC = -28.4
COSD = np.cos(np.radians(DEC))
SW = ("nrca1", "nrca2", "nrca3", "nrca4")
VG, VG_TABLE = "03101", 3101


def _baked(exp, fam):
    """A distinct baked offset per row (a collapsed table is refused)."""
    k = exp * 10 + (1 if fam.startswith("nrcb") else 0)
    return 0.01 + 0.0013 * k, -0.12 + 0.0007 * k


def _table(exposures=(1, 2), sw=("nrca", "nrcb"), lw=("nrcalong",)):
    rows = []
    for exp in exposures:
        for fam, filt in [(f, "F212N") for f in sw] + [(f, "F480M") for f in lw]:
            ra, de = _baked(exp, fam)
            rows.append((VISIT, exp, filt, fam, VG_TABLE, ra, de, ra, de))
    return Table(rows=rows, names=("Visit", "Exposure", "Filter", "Module",
                                   "Vgroup", "dra", "ddec", TABLE_RA_COL,
                                   TABLE_DEC_COL))


def _entry(exp, det, filt, dra, ddec, misaligned=False, baked=None, **kw):
    fam = det.rstrip("1234")
    ra, de = baked if baked is not None else _baked(exp, fam)
    e = dict(key=["1", exp, det, filt, VG], dra=dra, ddec=ddec,
             off=float(np.hypot(dra, ddec)), ok=True, unverified=False,
             alias_suspect=False, internal_tie=True, misaligned=misaligned,
             raoffset_meta=ra, deoffset_meta=de, dra_err=0.02, ddec_err=0.02)
    e.update(kw)
    return e


def _record(entries, passed=True, floor=4.0, corrections=(), filt="F212N"):
    return dict(stage="m2", date="2026-10-08T00:00:00Z", passed=passed,
                filtername=filt, tolerances=dict(correction_floor_mas=floor),
                visits=[dict(visit="1", exposures=entries)],
                corrections=list(corrections))


def _sw_row(exp, values, fam="nrca", **kw):
    dets = [f"{fam}{i}" for i in range(1, 5)]
    return [_entry(exp, d, "F212N", *v, **kw) for d, v in zip(dets, values)]


# exposure 1: module-common (+1.5, -1.0) plus a per-detector part summing to 0
ROW1 = [(+2.5, -1.0), (+0.5, -1.0), (+1.5, -0.2), (+1.5, -1.8)]
ROW1_MEAN = (1.5, -1.0)


def _plan(record, tbl=None, **kw):
    kw.setdefault("filtername", record.get("filtername"))
    return writeback_plan(record, _table() if tbl is None else tbl, PROP, OBS,
                          DEC, **kw)


def _row(plan, exp, fam):
    (r,) = [r for r in plan.rows if r.exposure == exp and r.family == fam]
    return r


def _write_table(tmp_path, tbl):
    path = os.path.join(tmp_path, "Offsets_JWST_Brick5365_VIRAC2locked.csv")
    tbl.write(path, overwrite=True)
    return path


# ---------------------------------------------------------------------------
# what is written
# ---------------------------------------------------------------------------

def test_the_written_value_is_the_mean_of_all_the_rows_detectors(tmp_path):
    rec = _record(_sw_row(1, ROW1))
    plan = _plan(rec)
    r = _row(plan, 1, "nrca")
    assert r.status == "write"
    assert (r.dra_mas, r.ddec_mas) == pytest.approx(ROW1_MEAN)
    tbl = _table()
    path = _write_table(tmp_path, tbl)
    out = apply_writeback(path, _plan(rec, tbl=tbl))
    after = Table.read(path)
    i = r.table_row
    assert after[TABLE_RA_COL][i] == pytest.approx(
        tbl[TABLE_RA_COL][i] + ROW1_MEAN[0] / 1000.0 / COSD, abs=1e-12)
    assert after[TABLE_DEC_COL][i] == pytest.approx(
        tbl[TABLE_DEC_COL][i] + ROW1_MEAN[1] / 1000.0, abs=1e-12)
    # the other pair is moved by the same increment
    assert after["dra"][i] - tbl["dra"][i] == pytest.approx(
        ROW1_MEAN[0] / 1000.0 / COSD, abs=1e-12)
    assert after["prov_stage"][i] == WRITEBACK_STAGE
    assert "2026-10-08" in after["prov_source"][i]
    assert out is not None


def test_the_mean_includes_detectors_m2_did_not_flag():
    # The one misaligned member reads (+3, 0); all four average to (+1, 0) and
    # their median is (+0.5, 0).  m2's APPLY path would pool the first; the
    # write-back writes the MEAN of all four, as the pooled writer computes it.
    vals = [(+3.0, 0.0), (+1.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
    ents = [_entry(1, d, "F212N", *v, misaligned=(v[0] > 2))
            for d, v in zip(SW, vals)]
    r = _row(_plan(_record(ents)), 1, "nrca")
    assert r.status == "write"
    assert (r.dra_mas, r.ddec_mas) == pytest.approx((1.0, 0.0))


def test_untouched_rows_are_unchanged(tmp_path):
    tbl = _table()
    path = _write_table(tmp_path, tbl)
    plan = _plan(_record(_sw_row(1, ROW1)), tbl=tbl)
    apply_writeback(path, plan)
    after = Table.read(path)
    for i in range(len(tbl)):
        if i != _row(plan, 1, "nrca").table_row:
            assert after[TABLE_RA_COL][i] == tbl[TABLE_RA_COL][i]
            assert after[TABLE_DEC_COL][i] == tbl[TABLE_DEC_COL][i]


def test_an_lw_row_is_its_one_detector():
    ents = [_entry(1, "nrcalong", "F480M", +1.2, -0.7)]
    r = _row(_plan(_record(ents, filt="F480M")), 1, "nrca")
    assert r.status == "write"
    assert r.detectors == ("nrcalong",)
    assert (r.dra_mas, r.ddec_mas) == pytest.approx((1.2, -0.7))


def test_a_row_below_the_minimum_is_not_written():
    small = [(+0.3, 0.0)] * 4
    plan = _plan(_record(_sw_row(1, small)))
    assert _row(plan, 1, "nrca").status == "below_min"
    assert plan.corrections == []
    assert _row(_plan(_record(_sw_row(1, small)), min_mas=0.25),
                1, "nrca").status == "write"
    assert DEFAULT_MIN_MAS == 0.5


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------

def test_a_row_m2_will_apply_itself_is_left_to_m2():
    # misaligned members pooling to 4.5 mas >= the 4 mas floor
    vals = [(+4.5, 0.0), (+4.5, 0.0), (0.0, 0.0), (0.0, 0.0)]
    ents = [_entry(1, d, "F212N", *v, misaligned=(v[0] > 2))
            for d, v in zip(SW, vals)]
    plan = _plan(_record(ents))
    assert _row(plan, 1, "nrca").status == "m2_actionable"
    assert plan.corrections == []


def test_a_row_missing_a_detector_is_incomplete():
    ents = _sw_row(1, ROW1) + _sw_row(2, ROW1)
    ents[5]["ok"] = False                     # exposure 2 nrca2 refused
    plan = _plan(_record(ents))
    assert _row(plan, 1, "nrca").status == "write"
    r2 = _row(plan, 2, "nrca")
    assert r2.status == "incomplete"
    assert "nrca2" in r2.reason
    # absent altogether is the same verdict
    ents = _sw_row(1, ROW1) + _sw_row(2, ROW1)[:3]
    assert _row(_plan(_record(ents)), 2, "nrca").status == "incomplete"


def test_a_row_whose_frames_were_built_from_another_table_is_stale():
    ents = _sw_row(1, ROW1, baked=(0.5, 0.5))
    r = _row(_plan(_record(ents)), 1, "nrca")
    assert r.status == "stale"
    assert "table row reads" in r.reason


def test_detectors_of_one_row_with_different_baked_offsets_are_stale():
    ents = _sw_row(1, ROW1)
    ents[0]["raoffset_meta"] += 1e-4
    r = _row(_plan(_record(ents)), 1, "nrca")
    assert r.status == "stale"
    assert "different baked" in r.reason


def test_a_key_the_reader_resolves_to_two_rows_is_not_written():
    tbl = _table()
    tbl.add_row(tbl[0])                        # a duplicate exp-1 nrca row
    r = _row(_plan(_record(_sw_row(1, ROW1)), tbl=tbl), 1, "nrca")
    assert r.status == "row_mismatch"


def test_a_second_pass_from_the_same_record_writes_nothing(tmp_path):
    tbl = _table()
    path = _write_table(tmp_path, tbl)
    rec = _record(_sw_row(1, ROW1) + _sw_row(2, ROW1))
    apply_writeback(path, _plan(rec, tbl=tbl))
    again = _plan(rec, tbl=Table.read(path))
    assert again.corrections == []
    assert {r.status for r in again.rows} == {"stale"}


def test_a_plan_made_before_the_table_changed_is_refused(tmp_path):
    tbl = _table()
    path = _write_table(tmp_path, tbl)
    plan = _plan(_record(_sw_row(1, ROW1)), tbl=tbl)
    apply_writeback(path, plan)
    before = Table.read(path)
    with pytest.raises(M2WritebackError, match="changed since the plan"):
        apply_writeback(path, plan)
    after = Table.read(path)
    assert list(after[TABLE_RA_COL]) == list(before[TABLE_RA_COL])


def test_an_empty_plan_writes_nothing(tmp_path):
    tbl = _table()
    path = _write_table(tmp_path, tbl)
    plan = _plan(_record(_sw_row(1, [(0.1, 0.1)] * 4)), tbl=tbl)
    assert apply_writeback(path, plan) is None
    assert not [f for f in os.listdir(tmp_path) if "pre_" in f]


# ---------------------------------------------------------------------------
# whole-record refusals
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("passed", [False, None])
def test_a_record_that_did_not_pass_is_refused(passed):
    with pytest.raises(M2WritebackError, match="passed"):
        _plan(_record(_sw_row(1, ROW1), passed=passed))


def test_a_record_of_another_stage_is_refused():
    rec = _record(_sw_row(1, ROW1))
    rec["stage"] = "m3"
    with pytest.raises(ValueError, match="not m2"):
        _plan(rec)


def test_a_joint_observation_is_refused():
    with pytest.raises(M2WritebackError, match="joint"):
        writeback_plan(_record(_sw_row(1, ROW1)), _table(), PROP, "002-998",
                       DEC, filtername="F212N")


def test_a_record_from_another_observation_is_refused():
    corr = dict(visit="jw05365002001", exposure=1, module="nrca1",
                filtername="F212N", vgroup=VG, dra_onsky_mas=3.0,
                ddec_onsky_mas=0.0, dec_deg=DEC, source="m2 visit-consensus")
    with pytest.raises(M2WritebackError, match="not among the visits"):
        _plan(_record(_sw_row(1, ROW1), corrections=[corr]))


def test_a_bulk_correction_from_another_observation_is_refused_too():
    bulk = dict(visit="jw05365002001", exposure=None, module=None,
                filtername="F212N", vgroup="", dra_onsky_mas=30.0,
                ddec_onsky_mas=0.0, dec_deg=DEC,
                source="m2 consensus->reference")
    with pytest.raises(M2WritebackError, match="not among the visits"):
        _plan(_record(_sw_row(1, ROW1), corrections=[bulk]))


def test_a_bare_visit_number_in_a_legacy_record_is_not_a_contradiction():
    # untokened records (cloudc) write the visit as '1'/'2'
    corr = dict(visit="1", exposure=1, module="nrca1", filtername="F212N",
                vgroup=VG, dra_onsky_mas=2.5, ddec_onsky_mas=0.0, dec_deg=DEC,
                source="m2 visit-consensus")
    assert _row(_plan(_record(_sw_row(1, ROW1), corrections=[corr])),
                1, "nrca").status == "write"


def test_a_table_without_the_reader_columns_is_refused():
    tbl = _table()
    tbl.remove_column(TABLE_RA_COL)
    with pytest.raises(M2WritebackError, match="lacks"):
        _plan(_record(_sw_row(1, ROW1)), tbl=tbl)


def test_a_nonfinite_declination_is_refused():
    with pytest.raises(M2WritebackError, match="dec_deg"):
        writeback_plan(_record(_sw_row(1, ROW1)), _table(), PROP, OBS,
                       float("nan"), filtername="F212N")


def test_visit_token():
    assert full_visit_token(5365, "001", "1") == VISIT
    assert full_visit_token("4147", "012", 2) == "jw04147012002"


# ---------------------------------------------------------------------------
# reaching the frames
# ---------------------------------------------------------------------------

def test_the_realign_tolerance_sees_every_written_row_as_stale():
    from jwst_gc_pipeline.reduction import unified_alignment as ua
    vals2 = [(+0.6, +0.1)] * 4
    plan = _plan(_record(_sw_row(1, ROW1) + _sw_row(2, vals2)))
    tol = plan.realign_tol_arcsec()
    # smallest per-axis coordinate change is exposure 2's dRA, 0.6 mas/cos(dec)
    assert tol == pytest.approx(0.4 * 0.6 / COSD / 1000.0)
    for r in plan.written():
        d_ra, d_dec = r.dra_mas / 1000.0 / COSD, r.ddec_mas / 1000.0
        # total-only frame, and a component frame whose change splits evenly
        # between bulk and jitter (the least either component can move)
        frames = [
            ({ua.TOTAL_RA_KEY: r.baked_ra, ua.TOTAL_DEC_KEY: r.baked_dec},
             dict(bulk_ra=r.baked_ra + d_ra, bulk_dec=r.baked_dec + d_dec)),
            ({ua.TOTAL_RA_KEY: r.baked_ra, ua.TOTAL_DEC_KEY: r.baked_dec,
              ua.BULK_RA_KEY: 0.0, ua.BULK_DEC_KEY: 0.0,
              ua.JITTER_RA_KEY: r.baked_ra, ua.JITTER_DEC_KEY: r.baked_dec},
             dict(bulk_ra=d_ra / 2, bulk_dec=d_dec / 2,
                  jitter_ra=r.baked_ra + d_ra / 2,
                  jitter_dec=r.baked_dec + d_dec / 2)),
        ]
        for hdr, comps in frames:
            shift = ua.AlignmentShift(source="TABLE_LOCKED",
                                      reference_frame="VIRAC2", **comps)
            assert ua.check_alignment_stale(hdr, shift, "f", tol_arcsec=tol)
            # ...and the default tolerance would call it current
            assert ua.check_alignment_stale(hdr, shift, "f") is None
    assert _plan(_record(_sw_row(1, [(0.1, 0.1)] * 4))).realign_tol_arcsec() is None


# ---------------------------------------------------------------------------
# the script
# ---------------------------------------------------------------------------

def _field(tmp_path, rec):
    base = tmp_path / "sgrb2"
    (base / "offsets").mkdir(parents=True)
    (base / "astrometry_checkpoints").mkdir()
    path = str(base / "offsets" / "Offsets_JWST_Brick5365_VIRAC2locked.csv")
    _table().write(path)
    with open(base / "astrometry_checkpoints"
              / "checkpoint_m2_F212N_o001_latest.json", "w") as fh:
        json.dump(rec, fh)
    return str(base), path


def _script():
    import importlib.util
    here = os.path.dirname(__file__)
    path = os.path.join(here, "..", "..", "..", "scripts", "reduction",
                        "m2_offsets_writeback.py")
    spec = importlib.util.spec_from_file_location("m2_offsets_writeback", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_script_is_a_dry_run_unless_told_to_write(tmp_path, capsys):
    corr = dict(visit=VISIT, exposure=1, module="nrca1", filtername="F212N",
                vgroup=VG, dra_onsky_mas=2.5, ddec_onsky_mas=-1.0, dec_deg=DEC,
                source="m2 visit-consensus")
    base, path = _field(tmp_path, _record(_sw_row(1, ROW1), corrections=[corr]))
    script = _script()
    before = Table.read(path)
    args = ["--basepath", base, "--proposal", PROP, "--field", OBS,
            "--filter", "F212N"]
    assert script.main(args) == 0
    assert "Dry run" in capsys.readouterr().out
    assert list(Table.read(path)[TABLE_RA_COL]) == list(before[TABLE_RA_COL])
    assert script.main(args + ["--write"]) == 0
    assert "WROTE 1 row(s)" in capsys.readouterr().out
    after = Table.read(path)
    assert after[TABLE_RA_COL][0] == pytest.approx(
        before[TABLE_RA_COL][0] + ROW1_MEAN[0] / 1000.0 / COSD, abs=1e-12)


def test_the_script_refuses_a_consensus_channel_field(tmp_path):
    with pytest.raises(SystemExit, match="only a module-locked"):
        _script().main(["--basepath", str(tmp_path), "--proposal", "6151",
                        "--field", "001", "--filter", "F212N"])


def test_the_script_reads_dec_from_the_consensus_catalog(tmp_path):
    cat = str(tmp_path / "cons.fits")
    Table(dict(DEC=[-28.0, -29.0, -28.5])).write(cat)
    assert _script()._dec_deg(dict(corrections=[], consensus_catalog=cat)) \
        == pytest.approx(-28.5)
    decs = [dict(dec_deg=-27.0), dict(dec_deg=-29.0), dict(dec_deg=-28.0)]
    assert _script()._dec_deg(dict(corrections=decs)) == -28.0
    with pytest.raises(M2WritebackError, match="consensus catalog"):
        _script()._dec_deg(dict(corrections=[], consensus_catalog=None))
