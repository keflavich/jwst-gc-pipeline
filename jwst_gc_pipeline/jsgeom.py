"""Screen-space polygon geometry, as JavaScript source, shared by both viewers.

Two pages hit-test footprints by projecting them to the screen and testing the
cursor against them: the monitor's sky view
(``jwst_gc_pipeline.monitoring.skyview``, which uses this) and the
colour-magnitude explorer (``jwst_gc_pipeline.quicklook.cmdview``, on the
branch for #884, which still carries a character-identical copy and should
switch to this once both have landed -- kept separate for now so neither PR
has to merge before the other).  Two copies of the same ray-casting loop, in
two f-string JavaScript blobs, neither reachable by a test.

Keeping the source here means one copy, and -- because it is a plain string --
one place a test can hand to ``node`` and actually EXECUTE.  A ray-casting loop
is exactly the kind of code that looks right and is off by one boundary case;
asserting that the page contains the word ``inPoly`` proves nothing about
whether a point is inside a polygon.

The snippets are emitted with SINGLE braces.  ``skyview`` interpolates them
into an f-string, where the interpolated value is inserted literally and not
re-scanned for braces, so they must NOT be doubled here.  The tests run the
rendered page through ``node --check`` and the snippets themselves through
``node``, which is what catches it if that ever stops being true.
"""

#: Even-odd ray casting.  ``pts`` is ``[[x, y], ...]`` in screen pixels.
#:
#: The half-open rule ``(yi > py) !== (yj > py)`` is what keeps a point that
#: lies exactly on a shared horizontal edge from being counted twice, so two
#: footprints that abut do not both claim it.
POINT_IN_POLY_JS = """
  function inPoly(px, py, pts) {
    var inside = false;
    for (var i = 0, j = pts.length - 1; i < pts.length; j = i++) {
      var xi = pts[i][0], yi = pts[i][1], xj = pts[j][0], yj = pts[j][1];
      if (((yi > py) !== (yj > py)) &&
          (px < (xj - xi) * (py - yi) / (yj - yi) + xi)) { inside = !inside; }
    }
    return inside;
  }
"""

#: The shoelace area, unsigned -- the caller compares sizes and does not care
#: which way round the vertices run, and a projected footprint can come out
#: either way depending on the projection and the hemisphere.
POLY_AREA_JS = """
  function polyArea(pts) {
    var a = 0;
    for (var i = 0, j = pts.length - 1; i < pts.length; j = i++) {
      a += (pts[j][0] + pts[i][0]) * (pts[j][1] - pts[i][1]);
    }
    return Math.abs(a / 2);
  }
"""

#: Both, for a caller that hit-tests overlapping polygons and has to choose.
HIT_TEST_JS = POINT_IN_POLY_JS + POLY_AREA_JS
