"""Finalize memory is sized from the field, not a flat constant.

A flat 64 gb was the same request for sgrb2 -- 1540 crf over 16 filters -- as
for m92's 80.  sgrb2's m3-finalize was OOM-killed at 64 gb, and because the kill
landed inside a multiprocessing queue write the survivors deadlocked: the job
held its allocation at ~0 CPU for 44 h, and the `afterok` chain behind it died
with it.  The same run completed in 12 h 26 at 256 gb.

These tests pin the policy in the submit driver rather than executing it: the
script is shell and runs sbatch, so the check is that the thresholds exist, are
ordered, and map the measured field sizes to the right tier.
"""
import os
import re
import subprocess
from pathlib import Path

import pytest

SCRIPT = (Path(__file__).resolve().parents[2] / "scripts" / "reduction"
          / "submit_cataloging_perframe.sh")


def _tiers():
    """[(threshold, mem)] parsed from the driver, highest threshold first."""
    text = SCRIPT.read_text()
    block = text[text.index("FINALIZE MEMORY SCALES WITH THE FIELD"):]
    block = block[:block.index("FINALIZE_TIME")]
    tiers = [(int(n), mem) for n, mem in
             re.findall(r'-ge\s+(\d+)\s*\];\s*then\s+FINALIZE_MEM=(\d+gb)', block)]
    floor = re.search(r'else\s+FINALIZE_MEM=(\d+gb)', block)
    assert floor, "no default tier"
    return tiers, floor.group(1)


def _resolve(crf):
    tiers, floor = _tiers()
    for threshold, mem in tiers:
        if crf >= threshold:
            return mem
    return floor


def test_the_tiers_are_ordered():
    """A lower threshold must not hand out more memory than a higher one."""
    tiers, floor = _tiers()
    assert tiers, "no tiers found in the driver"
    thresholds = [t for t, _ in tiers]
    assert thresholds == sorted(thresholds, reverse=True), thresholds
    mems = [int(m[:-2]) for _, m in tiers] + [int(floor[:-2])]
    assert mems == sorted(mems, reverse=True), mems


@pytest.mark.parametrize("field,crf,expected", [
    ("m92", 80, "64gb"),
    ("arches", 110, "64gb"),
    ("ngc6397", 120, "64gb"),
    ("m4", 150, "64gb"),
    ("sgrc", 240, "128gb"),
    ("sickle", 536, "128gb"),
    ("cloudef", 640, "128gb"),
    ("wd1", 696, "128gb"),
    ("w51", 1120, "256gb"),
    ("cloudc", 1208, "256gb"),
    ("sgrb2", 1540, "256gb"),
    ("brick", 2016, "256gb"),
])
def test_measured_fields_land_in_the_right_tier(field, crf, expected):
    """Counts measured from the live trees on 2026-09-02."""
    assert _resolve(crf) == expected, field


def test_sgrb2_would_no_longer_be_starved():
    """The case that produced the 44 h deadlock.

    It was OOM-killed at 64 gb with MaxRSS recorded at 155 G, so its tier has to
    clear that, not merely exceed the old flat value.
    """
    assert int(_resolve(1540)[:-2]) > 155


def test_an_explicit_request_still_wins():
    """`FINALIZE_MEM=... submit_...sh` must override, for one-off reruns."""
    text = SCRIPT.read_text()
    assert 'if [ -z "${FINALIZE_MEM:-}" ]; then' in text, (
        "the tiering must be skipped when the caller set FINALIZE_MEM")


def test_fanout_memory_is_untouched():
    """Fan-out is per-shard; only the finalize merges the whole field."""
    text = SCRIPT.read_text()
    assert re.search(r'FANOUT_MEM=\$\{FANOUT_MEM:-\d+gb\}', text)


def test_a_missing_data_tree_does_not_abort_the_submit():
    """The driver runs under `set -euo pipefail`.

    `ls` exits non-zero when its glob matches nothing, so a bare
    `ls ... | wc -l` killed the whole submit on any host without the data tree
    -- CI, a fresh checkout, a new field before its first reduce.  The count
    must swallow that status and fall to the smallest tier.
    """
    src = SCRIPT.read_text()
    assert "set -euo pipefail" in src, "premise: the driver still uses set -e"
    block = src[src.index("FINALIZE MEMORY SCALES WITH THE FIELD"):]
    block = block[:block.index("FINALIZE_TIME")]
    assert "|| true; } | wc -l" in block, (
        "the crf count must not abort the submit when the tree is absent")


def test_the_count_is_scoped_to_this_target():
    """A glob over the whole archive would size every field by the largest."""
    src = SCRIPT.read_text()
    block = src[src.index("FINALIZE MEMORY SCALES WITH THE FIELD"):]
    block = block[:block.index("FINALIZE_TIME")]
    assert '/$TARGET"' in block


# --- the count has to look where the data actually is -------------------------
#
# The tests above read the policy out of the driver.  These EXECUTE it: the
# sizing block is self-contained, so it can be lifted out and run against a
# synthetic tree without reaching sbatch.

def _sizing_block():
    """The `if [ -z "${FINALIZE_MEM:-}" ]` block, runnable on its own."""
    text = SCRIPT.read_text()
    start = text.index('if [ -z "${FINALIZE_MEM:-}" ]; then')
    end = text.index('FINALIZE_MEM=${FINALIZE_MEM:-64gb}', start)
    return text[start:end]


def _run_sizing(tmp_path, target, field, trees):
    """Run the block with `trees` = {directory name: number of crf} on disk."""
    for tree, n in trees.items():
        pipeline = tmp_path / tree / "F200W" / "pipeline"
        pipeline.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            (pipeline / f"jw_{i:05d}_destreak_crf.fits").touch()
    env = {k: v for k, v in os.environ.items() if k != "FINALIZE_MEM"}
    env.update(TARGET=target, FIELD=field, BASEPATH=str(tmp_path))
    # `set -e` is the driver's own state (line 28); the block must survive it.
    done = subprocess.run(
        ["bash", "-c", "set -euo pipefail\n" + _sizing_block()
         + '\necho "MEM=$FINALIZE_MEM"\n'],
        env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return done.stdout


def test_a_split_tree_field_is_counted_under_its_own_observation(tmp_path):
    """TARGET=gc2211 FIELD=046 -- the data is in gc2211_o046/, not gc2211/.

    886 job logs invoke the driver with the target and the observation split
    apart, and bare gc2211/ holds 0 crf.  Keyed on $TARGET alone o046's 240 crf
    read as 0 and took the smallest tier -- and o046 is one of the OOMs this
    sizing exists to prevent.
    """
    out = _run_sizing(tmp_path, "gc2211", "046",
                      {"gc2211_o046": 240, "gc2211": 0})
    assert "MEM=128gb" in out, out
    assert "gc2211_o046" in out, out


def test_the_per_observation_tree_is_tried_before_the_plain_one(tmp_path):
    """Both spellings present: the observation's own tree is the measurement.

    gc2211/ carries shared products (catalogs, offsets, cutouts); counting it
    instead of the observation would size o023 by the wrong field.
    """
    out = _run_sizing(tmp_path, "gc2211", "023",
                      {"gc2211_o023": 80, "gc2211": 1200})
    assert "MEM=64gb" in out, out
    assert "80 crf" in out, out


def test_a_single_tree_field_still_falls_back_to_the_plain_target(tmp_path):
    """sgrb2/001 has no sgrb2_o001/; the fallback is what keeps it at 256gb."""
    out = _run_sizing(tmp_path, "sgrb2", "001", {"sgrb2": 1540})
    assert "MEM=256gb" in out, out
    assert "1540 crf" in out, out


def test_a_target_that_already_carries_its_obsid_still_counts(tmp_path):
    """TARGET=gc2211_o046 FIELD=046 -- gc2211_o046_o046/ does not exist."""
    out = _run_sizing(tmp_path, "gc2211_o046", "046", {"gc2211_o046": 240})
    assert "MEM=128gb" in out, out


def test_an_empty_count_is_not_silent(tmp_path):
    """0 means two different things, and only one of them is a memory decision.

    An absent tree is a supported state (CI, a fresh checkout, a field before
    its first reduce), so "count 0" is also what "I looked in the wrong place"
    looks like -- the split-tree bug above.  The log has to distinguish them, so
    a zero count names every path it tried and says the count came back empty.
    """
    out = _run_sizing(tmp_path, "gc2211", "023", {})
    assert "MEM=64gb" in out, out
    assert "EMPTY" in out, out
    tokens = out.replace(";", " ").split()
    assert str(tmp_path / "gc2211_o023") in tokens, out
    assert str(tmp_path / "gc2211") in tokens, out


def test_a_nonzero_count_reports_the_path_it_measured(tmp_path):
    """The counted path is the one to check when a tier looks wrong."""
    out = _run_sizing(tmp_path, "gc2211", "049", {"gc2211_o049": 320})
    assert "EMPTY" not in out, out
    assert str(tmp_path / "gc2211_o049") in out, out
