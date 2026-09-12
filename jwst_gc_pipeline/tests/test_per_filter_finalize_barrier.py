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
    """Stub sbatch/scontrol/scancel that log their arguments.

    sbatch hands back a UNIQUE job id: the whole point is which ids end up in
    the next phase's --dependency, and a stub that always says 1000 cannot tell
    a complete barrier from a dropped one.  It fails on the submission whose id
    ``SBATCH_FAIL_AT`` names, which is how the mid-loop abort is exercised.

    scontrol and scancel are stubs for the same reason: the split submits its
    finalizes --hold and releases them together, and an abort cancels what it
    held, so both calls are part of what this driver emits.
    """
    bindir = tmp_path / "bin"
    bindir.mkdir(parents=True, exist_ok=True)
    (bindir / "sbatch").write_text(
        '#!/bin/bash\n'
        'n=$(cat "$SBATCH_SEQ" 2>/dev/null || echo 1000)\n'
        'if [ -n "${SBATCH_FAIL_AT:-}" ] && [ "$n" -eq "$SBATCH_FAIL_AT" ]; then\n'
        '  echo "sbatch: error: stub refusing to submit" >&2; exit 1\n'
        'fi\n'
        'printf "%s\\n" "$*" >> "$SBATCH_LOG"\n'
        'echo $((n + 1)) > "$SBATCH_SEQ"\n'
        'echo $((n + 1))\n')
    # The driver's duplicate-chain guard shells out to squeue; a real one here
    # would read this user's live queue.
    (bindir / "squeue").write_text("#!/bin/bash\ntrue\n")
    for name in ("scontrol", "scancel"):
        (bindir / name).write_text(
            '#!/bin/bash\n'
            f'printf "{name} %s\\n" "$*" >> "$SLURM_CTL_LOG"\n')
    for f in ("sbatch", "squeue", "scontrol", "scancel"):
        (bindir / f).chmod(0o755)
    return bindir


def _crf_tree(tmp_path, counts, target="sgrb2", field="001"):
    """A crf tree with ``counts`` frames per filter, as the driver globs it.

    The driver sizes a split finalize from the filter's own crf count, so the
    tree is the input to that decision.  Shapes here mirror sgrb2 5365/001,
    whose real counts are F187N 384, four SW filters 192 and six LW 48.
    """
    root = tmp_path / "tree" / f"{target}_o{field}"
    for filt, n in counts.items():
        d = root / filt / "pipeline"
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            (d / f"jw05365001001_02101_{i:05d}_nrca1_destreak_o001_crf.fits"
             ).write_text("")
    return tmp_path / "tree"


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
        self.mem = re.search(r'--mem=(\S+)', line).group(1)
        self.time = re.search(r'--time=(\S+)', line).group(1)
        self.held = '--hold' in line

    @property
    def mem_gb(self):
        return int(re.sub(r'[^0-9]', '', self.mem))

    @property
    def dep_ids(self):
        parts = self.dependency.split(":")
        return set(parts[1:]) if parts and parts[0].startswith("after") else set()


def _run(tmp_path, script=SCRIPT, expect_rc=0, filters=FILTERS,
         basepath=None, **env):
    """Run the driver against the stubs; return (completed, [Submit]).

    ``completed.ctl`` carries the scontrol/scancel calls the run made.
    """
    bindir = _stubs(tmp_path)
    log = tmp_path / "sbatch.log"
    ctl = tmp_path / "slurmctl.log"
    child = {k: v for k, v in os.environ.items()
             if not k.startswith(("FANOUT_", "FINALIZE_", "PER_FILTER_",
                                  "PHASES", "SKIP_IF_DONE", "SBATCH_FAIL_AT"))}
    child.update(PATH=f"{bindir}:{os.environ['PATH']}", SBATCH_LOG=str(log),
                 SBATCH_SEQ=str(tmp_path / "seq"), SLURM_CTL_LOG=str(ctl),
                 GC_SCRIPTS_DIR=str(SCRIPTS),
                 TARGET="sgrb2", PROPOSAL="5365", FIELD="001",
                 FILTERS=filters,
                 BASEPATH=str(basepath or tmp_path / "no-such-tree"))
    child.update({k: str(v) for k, v in env.items()})
    done = subprocess.run(["bash", str(script)], env=child,
                          capture_output=True, text=True)
    assert done.returncode == expect_rc, (done.returncode, done.stdout,
                                          done.stderr)
    submits = ([Submit(l) for l in log.read_text().splitlines()]
               if log.exists() else [])
    done.ctl = ctl.read_text() if ctl.exists() else ""
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


def test_the_split_leaves_the_fanout_and_the_unsplit_phases_alone(split,
                                                                  unsplit):
    """The fan-out and every unsplit finalize keep their whole-field slice.

    Only the split finalizes are re-sized, and only downward: a split job does
    ONE filter's work, so it may never ask for more than the whole-field
    finalize it replaces.
    """
    def slice_(line):
        return re.findall(r'--(?:cpus-per-task|mem|time)=\S+', line)

    assert ([slice_(x.line) for x in split if x.mode == "fanout"]
            == [slice_(x.line) for x in unsplit if x.mode == "fanout"])
    for phase in WHOLE_PHASES:
        a = [x for x in split if x.mode == "finalize" and x.phase == phase][0]
        b = [x for x in unsplit if x.mode == "finalize"
             and x.phase == phase][0]
        assert slice_(a.line) == slice_(b.line), (a.line, b.line)
    for phase in SPLIT_PHASES:
        whole = [x for x in unsplit if x.mode == "finalize"
                 and x.phase == phase][0]
        for x in [y for y in split if y.mode == "finalize"
                  and y.phase == phase]:
            assert x.mem_gb <= whole.mem_gb, x.line
            assert _minutes(x.time) <= _minutes(whole.time), x.line


# --------------------------------------------------------------------------
# each split job is sized for ITS OWN filter
# --------------------------------------------------------------------------
#
# A split finalize that keeps the whole field's --mem/--time contradicts the
# reason the split exists.  sgrb2's m6 finalize -- 4 cpus asking a 3-day wall
# on a stage whose measured maximum is 13.7 h -- waited exactly its own
# walltime, because an over-ask can only be placed in a gap as wide as itself;
# eleven copies of that ask, against a shared astronomy-dept-b GrpTRES with 52
# treasury tiles landing, pays it eleven times.
#
# The scale is the filter's own crf count over the largest filter's, because
# that is what both quantities were measured to follow:
#   memory does not pool.  The whole 11-filter m3 finalize (41424864) recorded
#   MaxRSS 148 GiB; the single-filter F187N m12 finalize (39933196) recorded
#   151 GiB.  The peak is the biggest COMBO, and F187N IS it.
#   time is near-linear in frames: F187N has 384 crf to F182M's 192 and took
#   13.09 h to its 6.79 h inside the m3 finalize.
SGRB2_CRF = dict(F187N=384, F150W=192, F182M=192, F210M=192, F212N=192,
                 F300M=48, F360M=48, F405N=48, F410M=48, F466N=48, F480M=48)


def _minutes(t):
    """A SLURM --time to minutes."""
    days, _, hms = t.rpartition("-")
    h, m, sec = (int(x) for x in hms.split(":"))
    return int(days or 0) * 1440 + h * 60 + m + (1 if sec else 0)


@pytest.fixture(scope="module")
def sgrb2_split(tmp_path_factory):
    """The split on a tree shaped like sgrb2 5365/001's own crf counts."""
    tmp = tmp_path_factory.mktemp("sized")
    tree = _crf_tree(tmp, SGRB2_CRF)
    return _run(tmp, PER_FILTER_FINALIZE=1, PER_FILTER_FINALIZE_PHASES="m4",
                filters=" ".join(SGRB2_CRF), basepath=tree)[1]


def _m4_finalizes(submits):
    return {s.filters[0]: s for s in submits
            if s.mode == "finalize" and s.phase == "m4"}


def test_the_largest_filter_keeps_the_whole_field_request(sgrb2_split):
    """F187N is the peak the pooled job was sized for, so it is not squeezed.

    The whole-field m4 finalize on a large field asks --mem=256gb
    --time=2-00:00:00; the filter that sets that peak keeps both.
    """
    f187n = _m4_finalizes(sgrb2_split)["F187N"]
    assert f187n.mem == "256gb", f187n.line
    assert _minutes(f187n.time) == _minutes("2-00:00:00"), f187n.line


def test_a_smaller_filter_asks_for_less(sgrb2_split):
    """Half the frames, half the slice; an eighth of the frames, the floor.

    The floor is the phase sbatch's own hand-launch slice (--mem=64gb
    --time=12:00:00), the smallest anything in this chain has been run at.
    """
    fins = _m4_finalizes(sgrb2_split)
    for filt in ("F150W", "F182M", "F210M", "F212N"):   # 192 of 384 crf
        assert fins[filt].mem == "128gb", fins[filt].line
        assert _minutes(fins[filt].time) == _minutes("1-00:00:00")
    for filt in ("F300M", "F405N", "F480M"):            # 48 of 384 -> floor
        assert fins[filt].mem == "64gb", fins[filt].line
        assert _minutes(fins[filt].time) == _minutes("12:00:00")


def test_the_phases_memory_ask_drops_by_more_than_half(sgrb2_split):
    """The number this is for: what the field asks the shared QOS to hold.

    Eleven copies of 256 GiB is 2816 GiB for one phase of one field.
    """
    fins = _m4_finalizes(sgrb2_split)
    assert len(fins) == 11
    assert sum(f.mem_gb for f in fins.values()) == 1152
    assert 1152 < 11 * 256 / 2


def test_an_explicit_ask_is_never_scaled(tmp_path):
    """What the operator NAMED reaches every split job verbatim.

    Same rule the wall-clock overrides already follow: FINALIZE_TIME_<PHASE>
    exists because a runner knows something the crf count does not, and a
    driver that quietly halves it is the failure mode B4 describes from the
    other end.
    """
    tree = _crf_tree(tmp_path, SGRB2_CRF)
    _, submits = _run(tmp_path, PER_FILTER_FINALIZE=1,
                      PER_FILTER_FINALIZE_PHASES="m4",
                      filters=" ".join(SGRB2_CRF), basepath=tree,
                      FINALIZE_MEM="200gb", FINALIZE_TIME_M4="4-00:00:00")
    for filt, s in _m4_finalizes(submits).items():
        assert s.mem == "200gb", s.line
        assert s.time == "4-00:00:00", s.line


def test_with_no_crf_tree_every_split_job_keeps_the_field_request(split,
                                                                  unsplit):
    """No counts means no measurement, so nothing is scaled off a guess.

    CI and a fresh checkout have no data tree; the split there must ask exactly
    what the whole-field finalize asks, as it does today.
    """
    for phase in SPLIT_PHASES:
        whole = [x for x in unsplit if x.mode == "finalize"
                 and x.phase == phase][0]
        for x in [y for y in split if y.mode == "finalize"
                  and y.phase == phase]:
            assert (x.mem, x.time) == (whole.mem, whole.time), x.line


# --------------------------------------------------------------------------
# a phase is submitted all-or-nothing
# --------------------------------------------------------------------------
#
# `set -euo pipefail` aborts the driver on a failed sbatch.  Whole, that leaves
# at worst a fan-out with no finalize.  Split, it would leave phase p with SOME
# of its filters finalizing, the rest never submitted and nothing queued
# behind them -- every submitted job reaching COMPLETED on a field that is
# missing seven barriers, which nothing downstream can see (a finalize's marker
# verify covers only the filters its own job was given).  So the per-filter
# finalizes are submitted --hold and released together.


def test_the_split_finalizes_are_held_until_all_of_them_exist(sgrb2_split,
                                                              tmp_path):
    """Held at submit, released in one call once the barrier is whole."""
    for s in sgrb2_split:
        assert s.held == (s.mode == "finalize" and s.phase == "m4"), s.line
    tree = _crf_tree(tmp_path, SGRB2_CRF)
    done, submits = _run(tmp_path, PER_FILTER_FINALIZE=1,
                         PER_FILTER_FINALIZE_PHASES="m4",
                         filters=" ".join(SGRB2_CRF), basepath=tree)
    releases = [l for l in done.ctl.splitlines() if l.startswith("scontrol")]
    assert len(releases) == 1, done.ctl
    ids = releases[0].split()[-1].split(",")
    assert set(ids) == {str(1001 + i) for i, s in enumerate(submits)
                        if s.mode == "finalize" and s.phase == "m4"}


def test_a_failed_sbatch_mid_loop_leaves_no_half_finalized_phase(tmp_path):
    """The abort undoes the whole phase instead of stranding part of it.

    The held finalizes and the phase's own fan-out are cancelled -- neither has
    run -- so the chain ends at the last COMPLETE phase rather than at a phase
    that four filters finalized.
    """
    tree = _crf_tree(tmp_path, SGRB2_CRF)
    # ids run 1001.. in submission order: m12 fan-out/finalize, m3
    # fan-out/finalize, m4 fan-out (1005), then the eleven m4 finalizes from
    # 1006.  The stub refuses the submit that would have been 1009, so three
    # of the eleven exist when the driver aborts.
    done, submits = _run(tmp_path, expect_rc=1, PER_FILTER_FINALIZE=1,
                         PER_FILTER_FINALIZE_PHASES="m4",
                         filters=" ".join(SGRB2_CRF), basepath=tree,
                         SBATCH_FAIL_AT=1008)
    assert [s.name for s in submits][-1].endswith("-m4-finalize-F182M")
    cancels = [l for l in done.ctl.splitlines() if l.startswith("scancel")]
    assert len(cancels) == 1, done.ctl
    cancelled = set(cancels[0].split()[1:])
    # the three held finalizes AND m4's fan-out
    assert cancelled == {"1006", "1007", "1008", "1005"}, done.ctl
    assert "scontrol" not in done.ctl, "a part-submitted phase was released"
    # ... and nothing was left queued for the phases after it
    assert {s.phase for s in submits} == {"m12", "m3", "m4"}


def test_the_abort_prints_the_command_that_resumes_the_chain(tmp_path):
    """Loud AND recoverable: the barrier the next phase needs is a list of ids
    that is otherwise printed and then lost."""
    tree = _crf_tree(tmp_path, SGRB2_CRF)
    done, _ = _run(tmp_path, expect_rc=1, PER_FILTER_FINALIZE=1,
                   PER_FILTER_FINALIZE_PHASES="m4",
                   filters=" ".join(SGRB2_CRF), basepath=tree,
                   SBATCH_FAIL_AT=1008)
    err = done.stderr
    assert "SUBMIT ABORTED" in err
    assert "DEP='afterok:1004'" in err, err       # m3's finalize
    assert "PHASES='m4 m5 m6 m7'" in err, err


def test_an_abort_with_nothing_submitted_says_nothing(tmp_path):
    """The refusal paths submit zero jobs; they must not grow a recovery block
    telling the operator to cancel an empty list."""
    done, submits = _run(tmp_path, expect_rc=4, PER_FILTER_FINALIZE=1,
                         PER_FILTER_FINALIZE_PHASES="m3 m7")
    assert submits == []
    assert "SUBMIT ABORTED" not in done.stderr
    assert done.ctl == ""


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

    done_before, before = _run(tmp_path / "a", script=old)
    done_after, after = _run(tmp_path / "b")
    strip = lambda ss: [re.sub(r'\S+submit_cataloging_perframe\S*', '', s.line)
                        for s in ss]
    assert strip(before) == strip(after)
    # ...and the same on stdout.  The split prints more (the barrier is a list
    # of ids that a recovery has to be able to read back); with the flag off
    # there is nothing extra to say, so nothing extra is said.
    scrub = lambda t: t.replace(str(tmp_path / "a"), "<run>").replace(
        str(tmp_path / "b"), "<run>")
    assert scrub(done_before.stdout) == scrub(done_after.stdout)


def test_off_by_default_still_exits_zero(tmp_path):
    """The last statement of the driver decides its exit status.

    `cond && echo` as the final line makes the whole script exit 1 whenever
    cond is false -- and with the EXIT trap in place that non-zero status is
    reported as an aborted submission on a run that submitted the entire chain.
    `_run` asserts the return code, so this is the check; it is spelled out
    because the failure looks like a submit failure rather than a typo.
    """
    done, submits = _run(tmp_path)
    assert done.returncode == 0
    assert "SUBMIT ABORTED" not in done.stderr
    assert len(submits) == 12        # six phases, fan-out + finalize each


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
