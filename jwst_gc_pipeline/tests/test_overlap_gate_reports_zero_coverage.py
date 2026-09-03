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
import importlib.util
import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "scripts" / "release" / "check_interframe_overlap.py"

_spec = importlib.util.spec_from_file_location("check_interframe_overlap", SRC)
ck = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ck)


def _run(monkeypatch, capsys, verdicts):
    """Run ``main --field f --scan`` over canned per-filter verdicts.

    Hermetic: ``check_filter`` and ``field_filters`` are replaced, so nothing
    touches disk and each verdict dict is exactly one of ``check_filter``'s
    return shapes.
    """
    by_filt = {v["filt"]: v for v in verdicts}
    monkeypatch.setattr(ck, "field_filters", lambda field: list(by_filt))
    monkeypatch.setattr(ck, "check_filter",
                        lambda field, filt, **kw: dict(by_filt[filt], field=field))
    rc = ck.main(["--field", "testfield", "--scan"])
    return rc, capsys.readouterr().out


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
    assert 'if r.get("n_overlapping") == 0:' in src
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


# `check_filter` has SIX return shapes and only the full result carries
# `n_overlapping`.  The four early returns below omit it, so an absent key must
# not be read as a measured zero -- see the accumulator's comment.


def test_a_band_that_never_got_far_enough_to_compare_is_not_zero_coverage(
        monkeypatch, capsys):
    """The four early returns omit the key; absence is not a measured zero.

    A falsy `not r.get("n_overlapping")` listed all four under a summary
    blaming single-visit module-split geometry -- naming a cause that did not
    occur, on exactly the bands whose next move is to check the file naming or
    the glob.
    """
    early = [
        # crf on disk, none parse as a per-exposure crf name (wd1/F200W)
        dict(filt="F200W", PASS=False, could_not_verify=True,
             unreadable_names=["x.fits"],
             note="3 crf on disk, none parseable"),
        # glob matched nothing at all
        dict(filt="F212N", PASS=False, could_not_verify=True,
             note="no crf frames matched"),
        # frames found, detection produced nothing usable
        dict(filt="F323N", PASS=False, could_not_verify=True,
             note="no detections from any crf"),
        # a genuine single exposure-group: nothing to pairwise-check, PASS=True
        dict(filt="F410M", PASS=True, note="single exposure-group (4 crf)"),
    ]
    rc, out = _run(monkeypatch, capsys, early)
    assert "NO COVERAGE" not in out
    for v in early:
        assert v["filt"] not in out.split("OVERLAP GATE")[-1]
    # each keeps the verdict it already had: three could-not-verify still block
    assert rc == 2


def test_the_single_exposure_group_pass_is_left_alone(monkeypatch, capsys):
    """On its own it is still a pass, and still not a zero-coverage band."""
    rc, out = _run(monkeypatch, capsys,
                   [dict(filt="F410M", PASS=True,
                         note="single exposure-group (4 crf)")])
    assert rc == 0
    assert "NO COVERAGE" not in out


def test_an_explicit_zero_is_zero_coverage(monkeypatch, capsys):
    """The intended case: the full result, measured, zero pairs (arches)."""
    rc, out = _run(monkeypatch, capsys,
                   [dict(filt="F212N", PASS=True, could_not_verify=False,
                         ext_fail=False, n_fail=0, n_overlapping=0)])
    assert "OVERLAP GATE: NO COVERAGE" in out
    assert "F212N" in out.split("OVERLAP GATE: NO COVERAGE")[-1]
    assert rc == 0


def test_measured_pairs_are_not_zero_coverage(monkeypatch, capsys):
    rc, out = _run(monkeypatch, capsys,
                   [dict(filt="F212N", PASS=True, could_not_verify=False,
                         ext_fail=False, n_fail=0, n_overlapping=7)])
    assert "NO COVERAGE" not in out
    assert rc == 0


def test_only_the_measured_zero_is_named(monkeypatch, capsys):
    """Mixed field: the summary must name the measured-zero band and only it."""
    rc, out = _run(monkeypatch, capsys, [
        dict(filt="F212N", PASS=True, could_not_verify=False,
             ext_fail=False, n_fail=0, n_overlapping=0),
        dict(filt="F410M", PASS=True, note="single exposure-group (4 crf)"),
    ])
    summary = out.split("OVERLAP GATE: NO COVERAGE")[-1]
    assert "F212N" in summary
    assert "F410M" not in summary
    assert "1/2 band(s)" in summary
    assert rc == 0
