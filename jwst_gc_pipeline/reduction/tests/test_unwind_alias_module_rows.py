"""Coverage for scripts/reduction/unwind_alias_module_rows.py (issue #298).

The script edits a LIVE offsets table.  Nothing referenced it before, so a
mutation that made it delete the WRONG rows -- `choose_survivor` returning
`sorted(mods)[0]`, i.e. keeping observation 005's `nrcb` rows and deleting the
correct `nrcblong` ones -- passed the whole repo suite.
"""
import importlib.util
import json
import pathlib

import pytest
from astropy.table import Table

_SPEC = importlib.util.spec_from_file_location(
    "unwind_alias_module_rows",
    pathlib.Path(__file__).resolve().parents[3]
    / "scripts" / "reduction" / "unwind_alias_module_rows.py")
uw = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(uw)


def _table(rows):
    cols = dict(Visit=[], Filter=[], Exposure=[], Vgroup=[], Module=[])
    cols.update({"dra (arcsec)": [], "ddec (arcsec)": []})
    for mod, exp, vg, dra in rows:
        cols["Visit"].append("jw02092002001")
        cols["Filter"].append("F360M")
        cols["Exposure"].append(exp)
        cols["Vgroup"].append(vg)
        cols["Module"].append(mod)
        cols["dra (arcsec)"].append(dra)
        cols["ddec (arcsec)"].append(0.0)
    return Table(cols)


def _write(tmp_path, tbl, name="t.csv"):
    p = tmp_path / name
    tbl.write(p)
    return str(p)


def test_the_long_row_survives_and_the_bare_one_is_removed(tmp_path, capsys):
    """Kills the `choose_survivor -> sorted(mods)[0]` mutant, which keeps the
    stale bare row and deletes the correct detector one."""
    p = _write(tmp_path, _table([("nrcb", 1, "02101", 0.011),
                                 ("nrcblong", 1, "02101", -0.007)]))
    import sys
    sys.argv = ["x", p, "--apply"]
    uw.main()
    out = Table.read(p)
    assert list(out["Module"]) == ["nrcblong"], out
    assert float(out["dra (arcsec)"][0]) == pytest.approx(-0.007)
    assert "keeping 'nrcblong'" in capsys.readouterr().out


def test_apply_writes_a_backup_and_a_receipt(tmp_path):
    """Kills the `drop shutil.copy2` mutant: the script edits a live table and
    the as-built values must remain recoverable."""
    import sys
    p = _write(tmp_path, _table([("nrcb", 1, "02101", 0.011),
                                 ("nrcblong", 1, "02101", -0.007)]))
    sys.argv = ["x", p, "--apply"]
    uw.main()
    backups = list(tmp_path.glob("*.pre_unwind298_*"))
    receipts = list(tmp_path.glob("*.unwind298_*.json"))
    assert len(backups) == 1 and len(receipts) == 1
    assert len(Table.read(str(backups[0]), format="ascii.csv")) == 2  # as-built
    rec = json.loads(receipts[0].read_text())
    assert rec["issue"] == 298
    assert len(rec["removed"]) == 1
    assert rec["removed"][0]["removed_module"] == "nrcb"
    assert rec["removed"][0]["row"]["Module"] == "nrcb"   # full row recorded


def test_dry_run_writes_nothing(tmp_path):
    import sys
    p = _write(tmp_path, _table([("nrcb", 1, "02101", 0.011),
                                 ("nrcblong", 1, "02101", -0.007)]))
    before = pathlib.Path(p).read_bytes()
    sys.argv = ["x", p, "--dry-run"]
    uw.main()
    assert pathlib.Path(p).read_bytes() == before
    assert not list(tmp_path.glob("*.pre_unwind298_*"))


def test_apply_and_dry_run_are_mutually_exclusive(tmp_path):
    """Kills the mutant that drops the exclusivity check."""
    import sys
    p = _write(tmp_path, _table([("nrcb", 1, "02101", 0.011)]))
    sys.argv = ["x", p, "--apply", "--dry-run"]
    with pytest.raises(SystemExit):
        uw.main()


def test_distinct_detectors_are_not_an_alias(tmp_path):
    """Kills the mutant dropping the bare-token filter: nrcb3 and nrcb4 share a
    family but match nothing of each other's at read time."""
    tbl = _table([("nrcb3", 1, "02101", 0.001), ("nrcb4", 1, "02101", 0.002)])
    assert uw.find_alias_groups(tbl) == {}


def test_empty_vgroup_is_a_wildcard(tmp_path):
    """A legacy row with no Vgroup matches ANY vgroup at read time, so it
    aliases across them; grouping on the vgroup exactly would miss it."""
    tbl = _table([("nrcb", 1, "", 0.011), ("nrcblong", 1, "02101", -0.007)])
    groups = uw.find_alias_groups(tbl)
    assert len(groups) == 1, groups
    assert uw.choose_survivor(list(groups.values())[0]) == "nrcblong"


def test_distinct_non_empty_vgroups_do_not_alias():
    tbl = _table([("nrcb", 1, "02101", 0.011), ("nrcblong", 1, "04101", -0.007)])
    assert uw.find_alias_groups(tbl) == {}


def test_unsupported_schema_is_reported_not_a_traceback(tmp_path, capsys):
    """28 of 54 live offsets tables lack one of the identity columns."""
    import sys
    p = _write(tmp_path, Table({"Filter": ["F360M"], "dra": [0.0]}))
    sys.argv = ["x", p, "--dry-run"]
    uw.main()
    assert "lacks" in capsys.readouterr().out


def test_choose_survivor_refuses_what_it_cannot_judge():
    assert uw.choose_survivor({"nrcb": [0], "nrcb1": [1], "nrcb2": [2]}) is None
    assert uw.choose_survivor({"nrcb": [0]}) is None
    assert uw.choose_survivor({"nrcb": [0], "nrcblong": [1]}) == "nrcblong"
