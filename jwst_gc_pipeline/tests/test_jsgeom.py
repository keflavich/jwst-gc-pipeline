"""Execute the shared hit-testing JavaScript, rather than grep for its name.

`inPoly` is a ray-casting loop: the kind of code that looks right and is wrong
on one boundary case.  Asserting that the rendered page contains the string
``inPoly`` says nothing about whether a point is inside a polygon, which is why
these run it under node.
"""
import json
import shutil
import subprocess

import pytest

from jwst_gc_pipeline import jsgeom

pytestmark = pytest.mark.skipif(shutil.which('node') is None,
                                reason='node not available')

#: A NIRCam footprint is a quadrilateral; the concave case is here because the
#: even-odd rule is what makes it work and a winding-number version would not.
SQUARE = [[0, 0], [10, 0], [10, 10], [0, 10]]
CONCAVE = [[0, 0], [10, 0], [10, 10], [5, 5], [0, 10]]
ROTATED = [[5, 0], [10, 5], [5, 10], [0, 5]]


def _run(js, expr):
    done = subprocess.run([shutil.which('node'), '-e', js + '\n' + expr],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def _in_poly(poly, pts):
    expr = ('console.log(JSON.stringify(%s.map(function (p) '
            '{ return inPoly(p[0], p[1], %s); })));'
            % (json.dumps(pts), json.dumps(poly)))
    return _run(jsgeom.POINT_IN_POLY_JS, expr)


def test_inside_and_outside_a_square():
    got = _in_poly(SQUARE, [[5, 5], [1, 1], [9, 9],      # inside
                            [-1, 5], [11, 5], [5, -1], [5, 11]])   # outside
    assert got == [True, True, True, False, False, False, False]


def test_a_concave_footprint_uses_the_even_odd_rule():
    """The notch has to read as outside.  A winding-number test would call it
    inside, and two abutting tiles would then both claim the same pixel."""
    got = _in_poly(CONCAVE, [[5, 2], [1, 8], [9, 8], [5, 8]])
    assert got == [True, True, True, False]


def test_a_rotated_quadrilateral():
    """Footprints arrive at whatever position angle the visit was scheduled
    at, so an axis-aligned test would not exercise the interpolation."""
    got = _in_poly(ROTATED, [[5, 5], [5, 1], [1, 1], [9, 9]])
    assert got == [True, True, False, False]


def test_a_point_on_a_shared_horizontal_edge_is_claimed_once():
    """Two tiles that abut along an edge must not both return true for a pixel
    on it, or the hit test picks whichever it happened to visit first."""
    lower = [[0, 0], [10, 0], [10, 5], [0, 5]]
    upper = [[0, 5], [10, 5], [10, 10], [0, 10]]
    on_edge = [[5, 5]]
    assert _in_poly(lower, on_edge)[0] != _in_poly(upper, on_edge)[0]


def test_winding_order_does_not_change_the_answer():
    assert _in_poly(SQUARE, [[5, 5]]) == _in_poly(SQUARE[::-1], [[5, 5]])


def _area(poly):
    return _run(jsgeom.POLY_AREA_JS,
                'console.log(JSON.stringify(polyArea(%s)));' % json.dumps(poly))


def test_the_area_is_the_shoelace_area():
    assert _area(SQUARE) == pytest.approx(100.0)
    assert _area([[0, 0], [10, 0], [0, 10]]) == pytest.approx(50.0)
    assert _area(ROTATED) == pytest.approx(50.0)


def test_the_area_is_unsigned():
    """A projected footprint can come out either way round depending on the
    projection and the hemisphere; the caller compares sizes and a negative
    area would make the largest tile win."""
    assert _area(SQUARE) == pytest.approx(_area(SQUARE[::-1]))
    assert _area(SQUARE) > 0


def test_the_smallest_of_two_overlapping_polygons_is_the_smaller_area():
    """The rule the sky view picks by, in the geometry it picks with."""
    big = [[0, 0], [20, 0], [20, 20], [0, 20]]
    assert _area(SQUARE) < _area(big)


def test_the_snippets_are_emitted_with_single_braces():
    """`skyview` interpolates these into an f-string.  Doubling them here would
    put literal `{{` in the page -- a syntax error in a script that renders
    fine and whose every control is dead."""
    for src in (jsgeom.POINT_IN_POLY_JS, jsgeom.POLY_AREA_JS):
        assert '{{' not in src and '}}' not in src


def test_both_snippets_parse_on_their_own():
    node = shutil.which('node')
    done = subprocess.run([node, '--check', '-'], input=jsgeom.HIT_TEST_JS,
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
