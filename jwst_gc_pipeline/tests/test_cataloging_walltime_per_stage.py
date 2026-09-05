"""Cataloging wall time is sized per FIELD and per STAGE, not one flat value.

``submit_cataloging_perframe.sh`` asked 12:00:00 for every fan-out and every
finalize of every field.  That is below the observed maximum of the stages the
big fields run, and a stage killed on its time limit takes the whole ``afterok``
chain with it -- crowded_l3's eight m12-fanout shards all hit 12:00:03 on
2026-09-04, after they had written all 280 per-frame catalogs, and stranded the
field's other 11 jobs on Dependency (issue #737).

Too long is its own failure, and it is why the limit is keyed on the field as
well as the stage: a 4-cpu job asking a 3-day wall can only be placed in a
3-day-wide gap, and sgrb2's m6-finalize, submitted at 3-00:00:00 for a stage
whose measured maximum is 13.7 h, waited exactly its own walltime.  arches, m92,
ngc6397 and m4 have never run a cataloging stage past 6.5 h; handing
them sgrb2's wall would pay that cost on every one of the 52 treasury tiles.

These run the driver with a stub ``sbatch`` on PATH and read the ``--time`` it
actually emits, so what is pinned is the submitted job rather than a constant.
"""
import os
import re
import subprocess
from pathlib import Path

import pytest

SCRIPT = (Path(__file__).resolve().parents[2] / "scripts" / "reduction"
          / "submit_cataloging_perframe.sh")

# Longest COMPLETED-or-TIMEOUT run of each (stage, mode), in hours, from 14 days
# of `sacct` to 2026-09-04, split by the crf-count tier the driver itself uses
# for FINALIZE_MEM (small <200, mid 200-999, large >=1000).
#
#   small  arches 110 / m92 80 / ngc6397 120 / m4 150 / gc2211-o050 120
#   mid    sgra 216 / sgrc 240 / gc2211-o046 240 / crowded_l3 280 / wd1 696
#   large  w51 1120 / cloudc 1208 / sgrb2 1540 / brick 2016
#
# Every value below is a run that COMPLETED, so it is a real runtime, EXCEPT
# CENSORED: cloudc's large m4-finalize TIMED OUT at a 24 h limit, so its true
# runtime is unmeasured and above 24 h.  (The three other stars in this PR's
# first draft were wrong -- m12-finalize 55.3 h, m12-fanout 23.5 h and
# m7-finalize 20.7 h were all COMPLETED runs at limits their runners had already
# raised to 24-72 h, not runs truncated by the 12 h default.)
MEASURED = {
    "small": {
        ("m12", "fanout"): 6.5, ("m3", "fanout"): 3.3, ("m4", "fanout"): 3.3,
        ("m5", "fanout"): 1.5, ("m6", "fanout"): 1.5, ("m7", "fanout"): 1.5,
        ("m12", "finalize"): 3.6, ("m3", "finalize"): 2.4,
        ("m4", "finalize"): 3.0, ("m5", "finalize"): 2.8,
        ("m6", "finalize"): 3.5, ("m7", "finalize"): 3.1,
    },
    "mid": {
        ("m12", "fanout"): 23.5, ("m3", "fanout"): 5.8, ("m4", "fanout"): 6.5,
        ("m5", "fanout"): 5.0, ("m6", "fanout"): 6.1, ("m7", "fanout"): 21.8,
        ("m12", "finalize"): 20.0, ("m3", "finalize"): 18.2,
        ("m4", "finalize"): 16.2, ("m5", "finalize"): 12.7,
        ("m6", "finalize"): 18.6, ("m7", "finalize"): 13.9,
    },
    "large": {
        ("m12", "fanout"): 21.8, ("m3", "fanout"): 13.1,
        ("m4", "fanout"): 13.5, ("m5", "fanout"): 10.0,
        ("m6", "fanout"): 9.8, ("m7", "fanout"): 7.0,
        ("m12", "finalize"): 55.3, ("m3", "finalize"): 19.0,
        ("m4", "finalize"): 24.0, ("m5", "finalize"): 21.3,
        ("m6", "finalize"): 18.8, ("m7", "finalize"): 20.7,
    },
}
CENSORED = {("large", ("m4", "finalize"))}

TIERS = ("small", "mid", "large")
# crf counts that land inside each tier of the driver's own thresholds.
TIER_CRF = {"small": 24, "mid": 208, "large": 1008}

OLD_FLAT_DEFAULT_H = 12.0
QOS_MAXWALL_H = 4 * 24          # astronomy-dept-b, MaxWall=4-00:00:00
STAGES = sorted(MEASURED["small"])


def _hours(t):
    """SLURM D-HH:MM:SS / HH:MM:SS -> hours."""
    parts = [int(x) for x in t.split("-")[-1].split(":")]
    hours = parts[0] + (parts[1] / 60 if len(parts) > 1 else 0)
    return hours + (int(t.split("-")[0]) * 24 if "-" in t else 0)


def _upto(tier, stage):
    """Longest run of this stage at this tier or any SMALLER one.

    A bigger field is never assumed to be faster, so a limit is allowed to
    carry the worst run seen at or below its own size -- mid's 21.8 h
    m7-fanout is why the large tier gets 36 h there on a 7.0 h measurement.
    """
    return max(MEASURED[t][stage] for t in TIERS[:TIERS.index(tier) + 1])


def _crf_tree(tmp_path, count):
    """A data tree the driver's crf glob will count `count` files in."""
    base = tmp_path / "tree"
    d = base / "testfield_o001" / "F212N" / "pipeline"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        (d / f"jw_{i:05d}_crf.fits").touch()
    return base


def _stubs(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir(parents=True, exist_ok=True)
    (bindir / "sbatch").write_text(
        '#!/bin/bash\nprintf "%s\\n" "$*" >> "$SBATCH_LOG"\necho 1000\n')
    # The driver's duplicate-chain guard shells out to squeue; a real one here
    # would read this user's live queue.
    (bindir / "squeue").write_text("#!/bin/bash\ntrue\n")
    for f in ("sbatch", "squeue"):
        (bindir / f).chmod(0o755)
    return bindir


def _run(tmp_path, crf=0, phases="m12 m3 m4 m5 m6 m7", **env):
    """Run the driver against a stub sbatch; return (stdout, {(ph, mode): time})."""
    bindir = _stubs(tmp_path)
    log = tmp_path / "sbatch.log"
    child = {k: v for k, v in os.environ.items()
             if not k.startswith(("FANOUT_TIME", "FINALIZE_TIME"))}
    child.update(PATH=f"{bindir}:{os.environ['PATH']}", SBATCH_LOG=str(log),
                 PHASES=phases, TARGET="testfield", PROPOSAL="9999",
                 FIELD="001", FILTERS="F212N F480M",
                 BASEPATH=str(_crf_tree(tmp_path, crf)))
    child.update(env)
    done = subprocess.run(["bash", str(SCRIPT)], env=child,
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr

    out = {}
    for line in log.read_text().splitlines():
        name = re.search(r'--job-name=\S+-(m\d+)-(fanout|finalize)', line)
        time = re.search(r'--time=(\S+)', line)
        assert name and time, line
        out[(name.group(1), name.group(2))] = time.group(1)
    return done.stdout, out


@pytest.fixture(scope="module")
def emitted(tmp_path_factory):
    """{tier: {(phase, mode): --time}} -- one driver run per field tier."""
    return {tier: _run(tmp_path_factory.mktemp(tier), crf=TIER_CRF[tier])[1]
            for tier in TIERS}


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("stage", STAGES)
def test_every_stage_clears_its_own_measured_maximum(tier, stage, emitted):
    """The defect: one 12 h limit sat below what the big fields actually run."""
    observed = _upto(tier, stage)
    assert _hours(emitted[tier][stage]) >= 1.5 * observed, (
        f"{tier} {stage[0]}-{stage[1]}: --time={emitted[tier][stage]} against a "
        f"measured maximum of {observed} h")


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("stage", STAGES)
def test_nothing_is_shorter_than_the_flat_default_it_replaces(tier, stage,
                                                              emitted):
    """Per-field sizing must not shorten anything on its way to fixing this."""
    assert _hours(emitted[tier][stage]) >= OLD_FLAT_DEFAULT_H, (
        emitted[tier][stage])


@pytest.mark.parametrize("tier,stage", sorted(CENSORED))
def test_a_censored_maximum_is_sized_above_not_to(tier, stage, emitted):
    """cloudc's m4-finalize TIMED OUT at 24 h, so 24 h is a lower bound.

    Sizing to it would reproduce the kill on the next field that is slower.
    """
    assert _hours(emitted[tier][stage]) > MEASURED[tier][stage] + 12, (
        emitted[tier][stage])


@pytest.mark.parametrize("stage", STAGES)
def test_a_small_field_does_not_get_a_big_fields_wall(stage, emitted):
    """The backfill half of the bracket, on the axis that carries it.

    Every small-tier stage measured under 6.5 h over ~700 runs, and not one of
    the 35 TIMEOUTs in the window is a small field -- they are all sgrb2,
    crowded_l3, brick, cloudc and w51.  So the small tier keeps exactly the
    12 h it already had; combined with the "nothing is shortened" test above
    this pins it to that value.  sgrb2's m6-finalize asked 3 d for a 13.7 h
    stage and waited 3 d on an otherwise finished field; arches asking that is
    the same cost for a 1 h stage, and from 2026-09-10 it would be paid on 52
    treasury tiles.
    """
    assert _hours(emitted["small"][stage]) <= OLD_FLAT_DEFAULT_H, (
        f"{stage[0]}-{stage[1]}: a small field asks "
        f"--time={emitted['small'][stage]} for a stage that has never run past "
        f"{MEASURED['small'][stage]} h")


def test_a_small_fields_chain_asks_far_less_than_a_big_fields(emitted):
    """The twelve jobs are serial, so the whole chain carries the cost."""
    small = sum(_hours(t) for t in emitted["small"].values())
    large = sum(_hours(t) for t in emitted["large"].values())
    assert small * 2 <= large, f"small chain {small} h vs large {large} h"


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("stage", STAGES)
def test_no_ask_is_more_than_three_times_its_own_measurement(tier, stage,
                                                             emitted):
    """A limit is headroom over a measurement, not a round number chosen up.

    The floor is a day: below that the wall is not what excludes a job from
    backfill, so buying margin there is free.
    """
    allowed = max(24.0, 3 * _upto(tier, stage))
    assert _hours(emitted[tier][stage]) <= allowed, (
        f"{tier} {stage[0]}-{stage[1]}: --time={emitted[tier][stage]} is more "
        f"than 3x the {_upto(tier, stage)} h it has ever needed")


@pytest.mark.parametrize("stage", STAGES)
def test_a_bigger_field_never_gets_a_shorter_wall(stage, emitted):
    """Nothing about a larger field makes a stage faster."""
    walls = [_hours(emitted[t][stage]) for t in TIERS]
    assert walls == sorted(walls), f"{stage}: {walls}"


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("stage", STAGES)
def test_no_stage_exceeds_the_qos_wall_limit(tier, stage, emitted):
    """astronomy-dept-b is MaxWall=4-00:00:00 with DenyOnLimit.

    A longer --time is not a slow job, it is a rejected submit: sbatch prints
    "Job violates accounting/QOS policy" and exits 1, so the chain never starts.
    """
    assert _hours(emitted[tier][stage]) <= QOS_MAXWALL_H, emitted[tier][stage]


def test_the_long_tail_stage_gets_the_longest_limit(emitted):
    """m12-finalize on a large field is the one that has actually run 55.3 h."""
    longest = emitted["large"][("m12", "finalize")]
    assert _hours(longest) >= 1.5 * 55.3
    assert max(_hours(t) for e in emitted.values() for t in e.values()) == (
        _hours(longest))


def test_a_flat_explicit_walltime_still_wins_for_every_phase(tmp_path):
    """`FANOUT_TIME=... FINALIZE_TIME=... submit_...sh` is unchanged.

    An operator's explicit ask is never silently shortened, including where the
    table would have given more.
    """
    _, emitted = _run(tmp_path, crf=TIER_CRF["large"],
                      FANOUT_TIME="02:00:00", FINALIZE_TIME="6-00:00:00")
    assert {t for (ph, mode), t in emitted.items()
            if mode == "fanout"} == {"02:00:00"}
    assert {t for (ph, mode), t in emitted.items()
            if mode == "finalize"} == {"6-00:00:00"}


def test_one_phase_can_be_overridden_without_moving_the_other_five(tmp_path):
    """B1: FINALIZE_TIME is a single knob for six phases, and that is how the
    campaign's longest fields ended up asking their m12 number everywhere.

    pipeline-runners' run_sgrb2_5365_o001.sh sets FINALIZE_TIME=72:00:00, sized
    in its own comment for the m12 finalize; sgrb2's m3..m7 finalizes run
    12.5-17.3 h and inherited the 3-day wall, which is the queue case the
    maintainer measured.  The per-phase spelling is what lets that runner keep
    its m12 extrapolation and let the rest fall to the table.
    """
    _, plain = _run(tmp_path / "plain", crf=TIER_CRF["large"])
    _, emitted = _run(tmp_path / "over", crf=TIER_CRF["large"],
                      FINALIZE_TIME_M12="72:00:00", FANOUT_TIME_M12="30:00:00")
    assert emitted[("m12", "finalize")] == "72:00:00"
    assert emitted[("m12", "fanout")] == "30:00:00"
    for ph in ("m3", "m4", "m5", "m6", "m7"):
        for mode in ("fanout", "finalize"):
            assert emitted[(ph, mode)] == plain[(ph, mode)], (ph, mode)
    # and the flat spelling is what it replaces: that one DOES move all six.
    _, flat = _run(tmp_path / "flat", crf=TIER_CRF["large"],
                   FINALIZE_TIME="72:00:00")
    assert {t for (ph, mode), t in flat.items()
            if mode == "finalize"} == {"72:00:00"}


def test_the_per_phase_override_beats_the_flat_one(tmp_path):
    _, emitted = _run(tmp_path, crf=TIER_CRF["mid"],
                      FINALIZE_TIME="6-00:00:00", FINALIZE_TIME_M4="03:00:00")
    assert emitted[("m4", "finalize")] == "03:00:00"
    assert emitted[("m5", "finalize")] == "6-00:00:00"


def test_an_unrecognised_phase_is_not_left_on_the_old_default(tmp_path):
    """A phase outside the table (a hand-set PHASES) takes a safe default.

    It must not fall through to something shorter than what it replaces.
    """
    _, emitted = _run(tmp_path, crf=TIER_CRF["large"], phases="m9")
    assert _hours(emitted[("m9", "fanout")]) >= OLD_FLAT_DEFAULT_H
    assert _hours(emitted[("m9", "finalize")]) >= OLD_FLAT_DEFAULT_H


def test_the_submitted_limit_and_where_it_came_from_are_printed(tmp_path):
    """A TIMEOUT is diagnosed from the log, so the ask has to be in it -- and a
    blanket environment override has to be visible as one, next to what the
    table would have given.  That is how a stale runner-side FINALIZE_TIME is
    findable at all."""
    out, _ = _run(tmp_path, crf=TIER_CRF["large"], phases="m12")
    assert "--time=1-12:00:00 (large field)" in out, out
    assert "--time=4-00:00:00 (large field)" in out, out

    out, _ = _run(tmp_path, crf=TIER_CRF["large"], phases="m12",
                  FINALIZE_TIME="72:00:00")
    assert "--time=72:00:00 (FINALIZE_TIME, ALL phases; large-field table: " \
           "4-00:00:00)" in out, out

    out, _ = _run(tmp_path, crf=TIER_CRF["large"], phases="m12",
                  FINALIZE_TIME_M12="72:00:00")
    assert "--time=72:00:00 (FINALIZE_TIME_M12; large-field table: " \
           "4-00:00:00)" in out, out


@pytest.mark.parametrize("crf,tier", [(0, "small"), (199, "small"),
                                      (200, "mid"), (999, "mid"),
                                      (1000, "large")])
def test_the_tier_boundaries_are_the_memory_tiers(tmp_path, crf, tier):
    """One field-size proxy, used for both --mem and --time (#611, #737).

    0 crf is CI and a tree that is not where the driver looked; it takes the
    smallest tier, the same way the memory sizing already does.
    """
    out, emitted = _run(tmp_path, crf=crf, phases="m12")
    assert f"({tier} field)" in out, out
