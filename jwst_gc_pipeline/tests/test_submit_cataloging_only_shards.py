"""``ONLY_SHARDS`` reruns the timed-out shards of a fan-out array (issue #1172).

One TIMEOUT in a fan-out array leaves the finalize and every later phase on
``DependencyNeverSatisfied``.  Recovering meant hand-building ``sbatch
--array=1,5,9``.  ``submit_cataloging_perframe.sh`` now takes ``ONLY_SHARDS``
for the first phase of the call, and refuses it without ``SKIP_IF_DONE=1``
(every listed shard would refit the frames the killed task had finished) or with
a shard id the array does not have (a stale ``NSHARDS``).

The driver runs against a stub ``sbatch`` that records its arguments.
"""
import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = (Path(__file__).resolve().parents[2] / "scripts" / "reduction"
          / "submit_cataloging_perframe.sh")


def _run(tmp_path, phases="m4 m5", **env):
    bindir = tmp_path / "bin"
    bindir.mkdir(parents=True, exist_ok=True)
    (bindir / "sbatch").write_text(
        '#!/bin/bash\nprintf "%s\\n" "$*" >> "$SBATCH_LOG"\necho 1000\n')
    (bindir / "squeue").write_text("#!/bin/bash\ntrue\n")
    for f in ("sbatch", "squeue"):
        (bindir / f).chmod(0o755)
    log = tmp_path / "sbatch.log"
    child = {k: v for k, v in os.environ.items()
             if k not in ("ONLY_SHARDS", "SKIP_IF_DONE", "NSHARDS")}
    child.update(PATH=f"{bindir}:{os.environ['PATH']}", SBATCH_LOG=str(log),
                 PHASES=phases, TARGET="testfield", PROPOSAL="9999",
                 FIELD="001", FILTERS="F212N F480M",
                 BASEPATH=str(tmp_path / "tree"))
    child.update(env)
    done = subprocess.run(["bash", str(SCRIPT)], env=child,
                          capture_output=True, text=True)
    lines = log.read_text().splitlines() if log.exists() else []
    return done, lines


def _counting_stubs(tmp_path):
    """sbatch that hands out 1001, 1002, ... and scontrol that logs its call."""
    bindir = tmp_path / "bin"
    bindir.mkdir(parents=True, exist_ok=True)
    (bindir / "sbatch").write_text(
        '#!/bin/bash\nn=$(cat "$SBATCH_LOG" 2>/dev/null | wc -l)\n'
        'printf "%s\\n" "$*" >> "$SBATCH_LOG"\necho $((1001 + n))\n')
    (bindir / "squeue").write_text("#!/bin/bash\ntrue\n")
    (bindir / "scontrol").write_text(
        '#!/bin/bash\nprintf "%s\\n" "$*" >> "$SCONTROL_LOG"\n')
    for f in ("sbatch", "squeue", "scontrol"):
        (bindir / f).chmod(0o755)
    return bindir


def _run_counting(tmp_path, **env):
    bindir = _counting_stubs(tmp_path)
    log = tmp_path / "sbatch.log"
    child = {k: v for k, v in os.environ.items()
             if k not in ("ONLY_SHARDS", "SKIP_IF_DONE", "NSHARDS", "DEP",
                          "PER_FILTER_FINALIZE")}
    child.update(PATH=f"{bindir}:{os.environ['PATH']}", SBATCH_LOG=str(log),
                 SCONTROL_LOG=str(tmp_path / "scontrol.log"),
                 TARGET="testfield", PROPOSAL="9999", FIELD="001",
                 FILTERS="F212N F480M", BASEPATH=str(tmp_path / "tree"))
    child.update(env)
    done = subprocess.run(["bash", str(SCRIPT)], env=child,
                          capture_output=True, text=True)
    return done, log.read_text().splitlines() if log.exists() else []


def _arg(line, flag):
    """Value of ``--flag=value`` on one recorded sbatch line."""
    return line.split(flag + "=")[1].split()[0]


def _arrays(lines):
    """{job-name: --array value} for the fan-out submissions."""
    out = {}
    for line in lines:
        if "--array=" in line:
            name = line.split("--job-name=")[1].split()[0]
            out[name] = line.split("--array=")[1].split()[0]
    return out


def test_unset_keeps_the_full_array(tmp_path):
    done, lines = _run(tmp_path)
    assert done.returncode == 0, done.stderr
    assert set(_arrays(lines).values()) == {"0-15"}
    assert "ONLY_SHARDS" not in done.stdout
    assert "Leave ONLY_SHARDS" not in done.stderr


def test_subset_applies_to_the_first_phase_only(tmp_path):
    done, lines = _run(tmp_path, ONLY_SHARDS="1,5,9", SKIP_IF_DONE="1")
    assert done.returncode == 0, done.stderr
    assert _arrays(lines) == {
        "testfield9999-o001-m4-fanout": "1,5,9",
        "testfield9999-o001-m5-fanout": "0-15",
    }
    assert "ONLY_SHARDS=1,5,9 of NSHARDS=16" in done.stdout
    assert "(1,5,9)" in done.stdout


def test_ranges_and_a_custom_nshards_are_accepted(tmp_path):
    done, lines = _run(tmp_path, phases="m6", ONLY_SHARDS="2-4,23",
                       SKIP_IF_DONE="1", NSHARDS="24")
    assert done.returncode == 0, done.stderr
    assert set(_arrays(lines).values()) == {"2-4,23"}


@pytest.mark.parametrize("spec", ["1,3-5", "0,2-4,9", "7-9,12"])
def test_a_range_after_a_comma_is_accepted(tmp_path, spec):
    done, lines = _run(tmp_path, phases="m6", ONLY_SHARDS=spec,
                       SKIP_IF_DONE="1")
    assert done.returncode == 0, done.stderr
    assert set(_arrays(lines).values()) == {spec}


def test_a_subset_without_skip_if_done_is_refused(tmp_path):
    done, lines = _run(tmp_path, ONLY_SHARDS="1,5,9")
    assert done.returncode != 0
    assert "SKIP_IF_DONE=1" in done.stderr
    assert lines == []


@pytest.mark.parametrize("bad", ["1;5", "a,b", "1,,5", "16", "3,99", "4-20",
                                 "99999999999999999999",
                                 "1,99999999999999999999",
                                 "3-99999999999999999999", "1234567"])
def test_a_malformed_or_out_of_range_subset_is_refused(tmp_path, bad):
    done, lines = _run(tmp_path, ONLY_SHARDS=bad, SKIP_IF_DONE="1")
    assert done.returncode != 0, done.stdout
    assert "REFUSING" in done.stderr
    assert lines == []


@pytest.mark.parametrize("bad", ["08", "abc", "0", "-4", "1e3", "2.5"])
def test_a_malformed_nshards_is_refused_before_anything_is_submitted(
        tmp_path, bad):
    """``NSHARDS=08`` is read as bad octal after the first fan-out went in."""
    done, lines = _run(tmp_path, phases="m6 m7", ONLY_SHARDS="3",
                       SKIP_IF_DONE="1", NSHARDS=bad)
    assert done.returncode != 0, done.stdout
    assert "NSHARDS" in done.stderr
    assert lines == []


def test_a_subset_on_a_per_filter_finalize_chain_keeps_the_barrier(tmp_path):
    """The sgrb2 recovery path: held per-filter finalizes, then the next phase.

    Each held finalize hangs off the subset array; the next fan-out waits on
    every finalize and gets its full array.
    """
    done, lines = _run_counting(tmp_path, PHASES="m6 m7", ONLY_SHARDS="1,3",
                                SKIP_IF_DONE="1", PER_FILTER_FINALIZE="1")
    assert done.returncode == 0, done.stderr
    fan6, fin_a, fin_b, fan7, fin7 = lines[:5]
    assert _arg(fan6, "--array") == "1,3"
    assert "--dependency" not in fan6
    for fin in (fin_a, fin_b):
        assert "--hold" in fin
        assert _arg(fin, "--dependency") == "afterok:1001"
    assert _arg(fan7, "--array") == "0-15"
    assert _arg(fan7, "--dependency") == "afterok:1002:1003"
    assert _arg(fin7, "--dependency") == "afterok:1004"
    assert (tmp_path / "scontrol.log").read_text().split() == [
        "release", "1002,1003"]


def test_a_subset_behind_an_existing_dependency(tmp_path):
    done, lines = _run_counting(tmp_path, PHASES="m6 m7", ONLY_SHARDS="2,4",
                                SKIP_IF_DONE="1", DEP="afterok:777")
    assert done.returncode == 0, done.stderr
    fan6, fin6, fan7, fin7 = lines[:4]
    assert _arg(fan6, "--array") == "2,4"
    assert _arg(fan6, "--dependency") == "afterok:777"
    assert _arg(fin6, "--dependency") == "afterok:1001"
    assert _arg(fan7, "--array") == "0-15"
    assert _arg(fan7, "--dependency") == "afterok:1002"
    assert _arg(fin7, "--dependency") == "afterok:1003"


def _aborted(tmp_path, **extra):
    """A chain whose third sbatch call fails, i.e. m5's fan-out is refused."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "sbatch").write_text(
        '#!/bin/bash\nn=$(cat "$SBATCH_LOG" 2>/dev/null | wc -l)\n'
        'printf "%s\\n" "$*" >> "$SBATCH_LOG"\n'
        '[ "$n" -ge 2 ] && exit 1\necho 1000\n')
    (bindir / "squeue").write_text("#!/bin/bash\ntrue\n")
    for f in ("sbatch", "squeue"):
        (bindir / f).chmod(0o755)
    env = {k: v for k, v in os.environ.items()
           if k not in ("ONLY_SHARDS", "SKIP_IF_DONE", "NSHARDS")}
    env.update(PATH=f"{bindir}:{os.environ['PATH']}",
               SBATCH_LOG=str(tmp_path / "sbatch.log"), PHASES="m4 m5",
               TARGET="testfield", PROPOSAL="9999", FIELD="001",
               FILTERS="F212N F480M", BASEPATH=str(tmp_path / "tree"))
    env.update(extra)
    return subprocess.run(["bash", str(SCRIPT)], env=env,
                          capture_output=True, text=True)


def test_an_abort_after_the_first_phase_says_to_drop_the_subset(tmp_path):
    """The resume hint says "the same environment"; ONLY_SHARDS must not be in it.

    Left set, it would cut the first resumed phase's array to the subset.
    """
    done = _aborted(tmp_path, ONLY_SHARDS="1,5", SKIP_IF_DONE="1")
    assert done.returncode != 0
    assert "SUBMIT ABORTED" in done.stderr
    assert "PHASES='m5'" in done.stderr
    assert "Leave ONLY_SHARDS out of the resume" in done.stderr


def test_an_ordinary_abort_does_not_mention_the_subset(tmp_path):
    done = _aborted(tmp_path)
    assert done.returncode != 0
    assert "SUBMIT ABORTED" in done.stderr
    assert "PHASES='m5'" in done.stderr
    assert "ONLY_SHARDS" not in done.stderr
