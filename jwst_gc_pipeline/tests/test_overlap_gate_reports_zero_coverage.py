"""A band with nothing to compare is reported, not counted as a pass.

`check_interframe_overlap.py` is the blocking release gate, and CLAUDE.md is
explicit that a green `registration_failsafes` is not sufficient without it.  On
a single-visit module-split field it provides no coverage at all and exits 0:

    arches     F212N: 88 crf -> 2 groups, 0 overlapping pairs, 0 FAIL, 0 could-not-verify
    quintuplet F212N: 96 crf -> 2 groups, 0 overlapping pairs, 0 FAIL, 0 could-not-verify

nrca and nrcb point at different sky, so there is no seam to check and the gate
is behaving correctly for the geometry -- but the release chain could not tell
that apart from a verified pass, and a reader totalling green gates over-counted.

The exit code deliberately stays 0: the geometry will never produce pairs, so
blocking would be permanent rather than actionable.  What changes is that the
gate says so.  Those fields' registration evidence comes from their m7
cross-filter checkpoints (arches 1.01 mas, quintuplet 0.44 mas) instead.
"""
import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "scripts" / "release" / "check_interframe_overlap.py"


def test_check_filter_reports_how_many_pairs_it_compared():
    """Without a count the caller cannot distinguish 'agreed' from 'compared nothing'."""
    src = SRC.read_text()
    assert "n_overlapping=len(overlapped)" in src


def test_zero_coverage_is_accumulated_separately_from_the_two_refusals():
    """could-not-verify means pairs existed and nothing could arbitrate them.

    Zero coverage means there were no pairs.  Conflating them would either block
    a field whose geometry can never yield pairs, or hide that nothing was
    measured.
    """
    src = SRC.read_text()
    assert "no_coverage = []" in src
    assert 'if not r.get("n_overlapping"):' in src
    assert "no_coverage.append" in src


def test_the_gate_says_so_on_the_band_line():
    src = SRC.read_text()
    assert "NO COVERAGE" in src
    assert "contributes" in src and "no evidence" in src


def test_the_summary_names_the_bands_and_refuses_the_word_pass():
    """The line has to be readable by someone totalling gates, not just parseable."""
    src = SRC.read_text()
    block = src[src.index("OVERLAP GATE: NO COVERAGE"):]
    block = block[:1200]
    # The message is assembled from adjacent f-string literals, so assert on
    # fragments that do not span a line break rather than on the rendered text.
    assert "{', '.join(sorted(no_coverage))}" in block, "must name which bands"
    assert "NOT a verified pass" in block
    assert "cross-filter checkpoint" in block, "must point at the real evidence"


def test_exit_code_is_not_changed_by_zero_coverage():
    """Blocking on a geometry that can never produce pairs would be permanent.

    The zero-coverage branch must print and fall through -- no return, no
    mutation of the refusal flags.
    """
    src = SRC.read_text()
    start = src.index("if no_coverage:")
    block = src[start:src.index("# Print EVERY refusal that applies", start)]
    assert "return" not in block, "zero coverage must not change the exit code"
    assert "any_fail" not in block and "any_noverify" not in block
