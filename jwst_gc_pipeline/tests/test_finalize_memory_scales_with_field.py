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
import re
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
