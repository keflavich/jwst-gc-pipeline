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


def test_a_subset_without_skip_if_done_is_refused(tmp_path):
    done, lines = _run(tmp_path, ONLY_SHARDS="1,5,9")
    assert done.returncode != 0
    assert "SKIP_IF_DONE=1" in done.stderr
    assert lines == []


@pytest.mark.parametrize("bad", ["1;5", "a,b", "1,,5", "16", "3,99", "4-20"])
def test_a_malformed_or_out_of_range_subset_is_refused(tmp_path, bad):
    done, lines = _run(tmp_path, ONLY_SHARDS=bad, SKIP_IF_DONE="1")
    assert done.returncode != 0, done.stdout
    assert "REFUSING" in done.stderr
    assert lines == []


def test_an_abort_after_the_first_phase_says_to_drop_the_subset(tmp_path):
    """The resume hint says "the same environment"; ONLY_SHARDS must not be in it.

    Left set, it would cut the first resumed phase's array to the subset.
    """
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
               FILTERS="F212N F480M", BASEPATH=str(tmp_path / "tree"),
               ONLY_SHARDS="1,5", SKIP_IF_DONE="1")
    done = subprocess.run(["bash", str(SCRIPT)], env=env,
                          capture_output=True, text=True)
    assert done.returncode != 0
    assert "SUBMIT ABORTED" in done.stderr
    assert "PHASES='m5'" in done.stderr
    assert "Leave ONLY_SHARDS out of the resume" in done.stderr
