"""The per-filter finalize split keeps the phase barrier, and never splits m7.

``submit_cataloging_perframe.sh`` submits ONE finalize per phase, and that job
walks every (filter x module) combo serially.  On the campaign's largest field
that is the whole barrier: sgrb2 5365/001's m3 finalize (job 41424864) ran
43.2 h over 33 combos, 13.1 h of which was F187N alone.  ``PER_FILTER_FINALIZE=1``
submits one finalize per FILTER for m3-m6 instead, all afterok on the same
fan-out.

The property that makes the split safe is the BARRIER: phase p+1's fan-out must
not start until EVERY per-filter finalize of phase p has finished.  Lose one id
from that dependency list and phase p+1 reads that filter's merged catalog,
residual i2d and smoothed background from before its barrier ran -- with no
crash to catch it, because the finalize's strict marker verify only covers the
filters its own job was given.  So that is what these tests pin, including a
mutation: the barrier check is run against a driver deliberately patched to drop
ids, and has to fail.

The second pinned property is that m7 is NOT split.  m7 IS the cross-band merge
and the cross-filter astrometry anchor gate, and a single-filter run does not
even build an m7 phase -- ``phases`` gets m7 only when ``len(filternames) > 1``
(crowdsource_catalogs_long.py) -- so a split m7 would die on
``--manual-start-phase='m7' not in phases ['m12','m3','m4','m5','m6']``.  m12 is
excluded for a different reason: it runs the CORRECTING astrometry checkpoint,
whose correction path writes the field's single shared offsets CSV with no
locking.

These run the real driver with a stub ``sbatch`` on PATH and read the job names,
dependencies and ``--export`` it actually emits, so what is pinned is the
submitted chain rather than a constant in the script.
"""
import os
import re
import subprocess
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "reduction"
SCRIPT = SCRIPTS / "submit_cataloging_perframe.sh"

FILTERS = "F150W F187N F212N F480M"
SPLIT_PHASES = ("m3", "m4", "m5", "m6")
WHOLE_PHASES = ("m12", "m7")


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------
def _stubs(tmp_path):
    """A stub sbatch that logs its arguments and hands back a UNIQUE job id.

    Unique matters here: the whole point is which ids end up in the next
    phase's --dependency, and a stub that always says 1000 cannot tell a
    complete barrier from a dropped one.
    """
    bindir = tmp_path / "bin"
    bindir.mkdir(parents=True, exist_ok=True)
    (bindir / "sbatch").write_text(
        '#!/bin/bash\n'
        'printf "%s\\n" "$*" >> "$SBATCH_LOG"\n'
        'n=$(cat "$SBATCH_SEQ" 2>/dev/null || echo 1000)\n'
        'echo $((n + 1)) > "$SBATCH_SEQ"\n'
        'echo $((n + 1))\n')
    # The driver's duplicate-chain guard shells out to squeue; a real one here
    # would read this user's live queue.
    (bindir / "squeue").write_text("#!/bin/bash\ntrue\n")
    for f in ("sbatch", "squeue"):
        (bindir / f).chmod(0o755)
    return bindir


class Submit:
    """One sbatch invocation, as the driver emitted it."""

    def __init__(self, line):
        self.line = line
        self.name = re.search(r'--job-name=(\S+)', line).group(1)
        dep = re.search(r'--dependency=(\S+)', line)
        self.dependency = dep.group(1) if dep else ""
        # FILTERS is space-separated inside a comma-separated --export list.
        self.filters = re.search(r'FILTERS=([^,]*)', line).group(1).split()
        self.array = '--array=' in line
        self.mode = re.search(r'MODE=(\w+)', line).group(1)
        self.phase = re.search(r'PHASE=(\w+)', line).group(1)

    @property
    def dep_ids(self):
        parts = self.dependency.split(":")
        return set(parts[1:]) if parts and parts[0].startswith("after") else set()


def _run(tmp_path, script=SCRIPT, expect_rc=0, filters=FILTERS, **env):
    """Run the driver against the stub sbatch; return (completed, [Submit])."""
    bindir = _stubs(tmp_path)
    log = tmp_path / "sbatch.log"
    child = {k: v for k, v in os.environ.items()
             if not k.startswith(("FANOUT_", "FINALIZE_", "PER_FILTER_",
                                  "PHASES", "SKIP_IF_DONE"))}
    child.update(PATH=f"{bindir}:{os.environ['PATH']}", SBATCH_LOG=str(log),
                 SBATCH_SEQ=str(tmp_path / "seq"),
                 GC_SCRIPTS_DIR=str(SCRIPTS),
                 TARGET="sgrb2", PROPOSAL="5365", FIELD="001",
                 FILTERS=filters, BASEPATH=str(tmp_path / "no-such-tree"))
    child.update({k: str(v) for k, v in env.items()})
    done = subprocess.run(["bash", str(script)], env=child,
                          capture_output=True, text=True)
    assert done.returncode == expect_rc, (done.returncode, done.stdout,
                                          done.stderr)
    submits = ([Submit(l) for l in log.read_text().splitlines()]
               if log.exists() else [])
    return done, submits


def _ids(submits, log_text):
    """{job-name: job id} -- the id the stub handed back for each submit."""
    # The stub's ids are consecutive from 1001 in submission order.
    return {s.name: str(1001 + i) for i, s in enumerate(submits)}


def _barrier_violations(submits):
    """Phases whose next fan-out does not wait on every finalize of the phase.

    THE property.  Returns a list of human-readable violations; empty is pass.
    """
    ids = {s.name: str(1001 + i) for i, s in enumerate(submits)}
    order = [s for s in submits if s.mode == "fanout"]
    bad = []
    for i, fanout in enumerate(order[1:], start=1):
        prev_phase = order[i - 1].phase
        expected = {ids[s.name] for s in submits
                    if s.mode == "finalize" and s.phase == prev_phase}
        assert expected, f"no finalize submitted for phase {prev_phase}"
        missing = expected - fanout.dep_ids
        if missing:
            bad.append(
                f"{fanout.phase} fan-out waits on {fanout.dependency!r} but "
                f"phase {prev_phase} has finalize ids {sorted(expected)}; "
                f"missing {sorted(missing)}")
    return bad


@pytest.fixture(scope="module")
def split(tmp_path_factory):
    """The full default chain with the split ON."""
    return _run(tmp_path_factory.mktemp("split"), PER_FILTER_FINALIZE=1)[1]


@pytest.fixture(scope="module")
def unsplit(tmp_path_factory):
    """The same chain with the flag absent -- today's behaviour."""
    return _run(tmp_path_factory.mktemp("unsplit"))[1]


# --------------------------------------------------------------------------
# THE barrier
# --------------------------------------------------------------------------
def test_the_next_phase_waits_on_every_per_filter_finalize(split):
    """Phase p+1's fan-out depends on afterok of ALL of phase p's finalizes.

    Not the last one, not the first: all of them.  A missing id lets p+1 read
    a filter whose barrier never ran, and nothing downstream would notice --
    the finalize's marker verify covers only the filters its own job was given.
    """
    assert _barrier_violations(split) == []


def test_the_barrier_check_fails_on_a_driver_that_drops_an_id(tmp_path):
    """Mutation: keep only the LAST per-filter finalize in the dependency.

    That mutant is exactly the plausible bug -- ``prev_dep="afterok:$B"``
    written inside the per-filter loop instead of the list being accumulated --
    and it submits a chain that looks correct in the queue.  The test above has
    to reject it, or it pins nothing.
    """
    mutant = tmp_path / "mutant.sh"
    src = SCRIPT.read_text().replace(
        'fin_ids="${fin_ids}:$B"', 'fin_ids=":$B"', 1)
    assert src != SCRIPT.read_text(), "mutation did not apply"
    mutant.write_text(src)
    # _refuse_duplicate_chain.sh is sourced from the script's own directory.
    (tmp_path / "_refuse_duplicate_chain.sh").write_text(
        (SCRIPTS / "_refuse_duplicate_chain.sh").read_text())
    _, submits = _run(tmp_path, script=mutant, PER_FILTER_FINALIZE=1)
    violations = _barrier_violations(submits)
    assert violations, "the mutant's dropped dependencies went undetected"
    assert len(violations) == len(SPLIT_PHASES)


def test_every_per_filter_finalize_waits_on_its_own_phases_fanout(split):
    """The finalizes run in PARALLEL, each afterok on the one fan-out array.

    The saving is gone if they chain to each other, and the fan-out is not
    split -- one array already partitions every frame of every filter.
    """
    fanout = {s.phase: str(1001 + i) for i, s in enumerate(split)
              if s.mode == "fanout"}
    for i, s in enumerate(split):
        if s.mode == "finalize":
            assert s.dep_ids == {fanout[s.phase]}, s.line


def test_the_fanout_is_not_split(split):
    """One array per phase, carrying every filter."""
    for s in split:
        if s.mode == "fanout":
            assert s.array and s.filters == FILTERS.split(), s.line
    assert len([s for s in split if s.mode == "fanout"]) == 6


# --------------------------------------------------------------------------
# m7 (and m12) are never split
# --------------------------------------------------------------------------
@pytest.mark.parametrize("phase", WHOLE_PHASES)
def test_the_unsplittable_phases_stay_one_whole_finalize(phase, split):
    """m7 IS the cross-band merge; m12 writes the shared offsets table.

    A one-filter m7 has no m7 phase to start
    (``phases`` appends m7 only when multifilter), so the job would die with
    ``--manual-start-phase='m7' not in phases [...]``.  Eleven m12 finalizes
    would write one un-locked offsets CSV concurrently.
    """
    fins = [s for s in split if s.mode == "finalize" and s.phase == phase]
    assert len(fins) == 1, [s.name for s in fins]
    assert fins[0].filters == FILTERS.split()
    assert fins[0].name.endswith(f"-{phase}-finalize")


@pytest.mark.parametrize("phase", WHOLE_PHASES)
def test_asking_to_split_an_unsplittable_phase_is_refused(tmp_path, phase):
    """Refused at submit time, not quietly ignored and not attempted.

    An operator who reaches for this is reaching for the failure it prevents,
    so the message has to name the reason rather than the rule.
    """
    done, submits = _run(tmp_path, expect_rc=4, PER_FILTER_FINALIZE=1,
                         PER_FILTER_FINALIZE_PHASES=f"m3 {phase}")
    assert submits == [], "refused, but jobs were already submitted"
    assert "REFUSING" in done.stderr and phase in done.stderr, done.stderr
    assert "cross-band" in done.stderr and "offsets" in done.stderr


def test_a_single_filter_run_has_no_m7_to_split(tmp_path):
    """The driver's own phase list already drops m7 for one filter.

    Which is the same rule the split honours, from the other side.
    """
    _, submits = _run(tmp_path, filters="F212N", PER_FILTER_FINALIZE=1)
    assert {s.phase for s in submits} == {"m12", "m3", "m4", "m5", "m6"}
    assert _barrier_violations(submits) == []


# --------------------------------------------------------------------------
# what each split job is given
# --------------------------------------------------------------------------
@pytest.mark.parametrize("phase", SPLIT_PHASES)
def test_each_split_finalize_gets_exactly_one_filter_and_they_cover_all(
        phase, split):
    """One filter each, and the union is the run's whole filter list.

    A filter silently missing from the split is a filter whose barrier never
    runs at all.
    """
    fins = [s for s in split if s.mode == "finalize" and s.phase == phase]
    assert [s.filters for s in fins] == [[f] for f in FILTERS.split()]
    assert sorted(f for s in fins for f in s.filters) == sorted(
        FILTERS.split())


@pytest.mark.parametrize("phase", SPLIT_PHASES)
def test_the_job_name_carries_the_filter_and_still_reads_as_the_phase(
        phase, split):
    """``<target><program>-o<obsid>-<stage>-<FILTER>`` is the standing rule.

    The '-<phase>-' prefix is also what ``_refuse_duplicate_chain.sh`` greps
    for, so a re-submission of a split phase is still refused as a duplicate.
    """
    fins = [s for s in split if s.mode == "finalize" and s.phase == phase]
    for s, filt in zip(fins, FILTERS.split()):
        assert s.name == f"sgrb25365-o001-{phase}-finalize-{filt}"
        assert s.name.startswith(f"sgrb25365-o001-{phase}-")


def test_the_split_changes_nothing_but_the_finalize(split, unsplit):
    """Resources, mode flags and the fan-out are untouched by the split.

    One concern: a per-filter finalize does strictly less work than the whole
    one, so it keeps the same --cpus-per-task/--mem/--time it has today, and
    retuning those is a separate measurement.
    """
    def slice_(line):
        return re.findall(r'--(?:cpus-per-task|mem|time)=\S+', line)

    for phase in SPLIT_PHASES:
        s = [x for x in split if x.mode == "finalize" and x.phase == phase][0]
        u = [x for x in unsplit if x.mode == "finalize"
             and x.phase == phase][0]
        assert slice_(s.line) == slice_(u.line), (s.line, u.line)
    assert ([slice_(s.line) for s in split if s.mode == "fanout"]
            == [slice_(s.line) for s in unsplit if s.mode == "fanout"])


# --------------------------------------------------------------------------
# opt-in
# --------------------------------------------------------------------------
def test_off_by_default_submits_exactly_one_finalize_per_phase(unsplit):
    """sgrb2 is mid-chain and the 10678 tiles are landing.

    A run that does not ask for the split has to submit what it submits today,
    down to the job names.
    """
    fins = [s for s in unsplit if s.mode == "finalize"]
    assert [s.name for s in fins] == [
        f"sgrb25365-o001-{ph}-finalize"
        for ph in ("m12", "m3", "m4", "m5", "m6", "m7")]
    for s in fins:
        assert s.filters == FILTERS.split()
    assert _barrier_violations(unsplit) == []


def test_off_by_default_matches_the_previous_driver_submit_for_submit(
        tmp_path):
    """Byte-for-byte against the driver as it stands on the branch point.

    The flag-off path is not merely 'unsplit', it is unchanged.
    """
    head = subprocess.run(
        ["git", "-C", str(SCRIPTS), "show",
         "origin/main:scripts/reduction/submit_cataloging_perframe.sh"],
        capture_output=True, text=True)
    if head.returncode != 0:
        pytest.skip("origin/main not available in this checkout")
    old = tmp_path / "old" / "submit_cataloging_perframe.sh"
    old.parent.mkdir(parents=True, exist_ok=True)
    old.write_text(head.stdout)
    (old.parent / "_refuse_duplicate_chain.sh").write_text(
        (SCRIPTS / "_refuse_duplicate_chain.sh").read_text())

    _, before = _run(tmp_path / "a", script=old)
    _, after = _run(tmp_path / "b")
    strip = lambda ss: [re.sub(r'\S+submit_cataloging_perframe\S*', '', s.line)
                        for s in ss]
    assert strip(before) == strip(after)


def test_a_phase_outside_the_requested_set_is_not_split(tmp_path):
    """PER_FILTER_FINALIZE_PHASES narrows; it never widens past the allowlist."""
    _, submits = _run(tmp_path, PER_FILTER_FINALIZE=1,
                      PER_FILTER_FINALIZE_PHASES="m4")
    for phase in ("m12", "m3", "m5", "m6", "m7"):
        fins = [s for s in submits
                if s.mode == "finalize" and s.phase == phase]
        assert len(fins) == 1, phase
    assert len([s for s in submits
                if s.mode == "finalize" and s.phase == "m4"]) == 4
    assert _barrier_violations(submits) == []
