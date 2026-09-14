"""The colour-magnitude explorer page: sky on the left, CMD on the right.

The page is a thin renderer.  Every number it draws was computed by
``build_cmd_viewer.py`` and shipped in a sidecar JSON: the whole-sample hexmap,
one hexmap per pointing, and the pointing footprints.  Nothing is recomputed in
the browser, so hovering a footprint is a canvas redraw of a few thousand
hexagons rather than a re-bin of ten million stars.

Two maps share one grid.  The grey background is every matched star in the
programme; the viridis overlay is the pointing under the cursor, binned on the
SAME cells so the overlay lands on the background rather than beside it.  Grey
is deliberately the whole sample: the question the page answers is "how does
this tile differ from the survey", and that only reads if the comparison is
underneath.

Hit-testing is done here rather than through Aladin's own object events.  The
footprints are projected to screen with ``world2pix`` and tested against the
mouse pixel directly -- forty vertices per frame, which costs nothing, and it
does not depend on which of the v3 hover APIs this build of Aladin Lite
exposes.
"""
import html
import json

#: Aladin Lite v3, the same build the other viewers on this site load.
ALADIN_JS = 'https://aladin.cds.unistra.fr/AladinLite/api/v3/latest/aladin.js'

#: HiPS layers under the footprints, in paint order (last on top).  Same URLs
#: and the same order as the main viewer, so the two pages show the same sky.
_AVM = 'https://starformation.astro.ufl.edu/avm_images/'
HIPS_LAYERS = (
    ('tmiri', _AVM + 'jwst_gc_treasury_miri_hips/', 'Treasury MIRI (F770W)'),
    ('tnir', _AVM + 'jwst_gc_treasury_hips/', 'Treasury NIRCam (F212N/F480M)'),
)

#: Fallback background, used while the treasury layers are sparse.
BASE_SURVEY = 'CDS/P/VISTA/VVV/DR4/H/Bulge'

CSS = """
:root { --bg:#0d1117; --panel:#161b22; --fg:#e6edf3; --muted:#8b949e;
        --accent:#58a6ff; --border:#30363d; }
* { box-sizing:border-box; }
html, body { margin:0; height:100%; background:var(--bg); color:var(--fg);
  font:14px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; }
a { color:var(--accent); }
#wrap { display:flex; height:100%; }
#sky { flex:1 1 auto; min-width:0; position:relative; }
#side { flex:0 0 400px; background:var(--panel); border-left:1px solid var(--border);
        overflow-y:auto; padding:14px 16px; }
h1 { font-size:1rem; margin:0 0 2px; }
h2 { font-size:.72rem; text-transform:uppercase; letter-spacing:.08em;
     color:var(--muted); margin:18px 0 8px; font-weight:600; }
p.sub, .muted { color:var(--muted); font-size:.78rem; }
p.sub { margin:0 0 14px; }
#cmd { width:100%; display:block; background:#05070a;
       border:1px solid var(--border); border-radius:6px; }
#cmdlabel { font-size:.82rem; margin:8px 0 0; min-height:2.6em; }
#cmdlabel b { color:var(--accent); }
table.tiles { width:100%; border-collapse:collapse; font-size:.76rem;
              margin-top:6px; }
table.tiles td { padding:2px 4px; border-bottom:1px solid var(--border);
                 cursor:pointer; }
table.tiles tr:hover td { background:#1f2937; }
table.tiles tr.on td { background:#1f2937; color:var(--accent); }
table.tiles tr.partial td:first-child { color:#d29922; }
td.num { text-align:right; color:var(--muted); font-variant-numeric:tabular-nums; }
.warn { border-left:2px solid #d29922; padding-left:9px; color:var(--muted);
        font-size:.75rem; margin-top:14px; }
button { background:#21262d; color:var(--fg); border:1px solid var(--border);
         border-radius:5px; padding:4px 9px; font-size:.78rem; cursor:pointer; }
button:hover { border-color:var(--accent); }
.bar { display:flex; gap:6px; flex-wrap:wrap; margin-top:8px; }
code { color:var(--accent); font-size:.75rem; }
@media (max-width:820px) {
  #wrap { flex-direction:column; }
  #sky { height:55vh; }
  #side { flex:1 1 auto; border-left:0; border-top:1px solid var(--border); }
}
"""


DATA_FILE = 'cmd_explorer_data.json'
PAGE_FILE = 'cmd_explorer.html'


def render(data, data_href=DATA_FILE,
           title='JWST GC colour-magnitude explorer'):
    """The page.  ``data`` is only read for the summary line rendered server-side;
    the drawing all happens from ``data_href`` in the browser."""
    fields = data['fields']
    nstars = sum(f.get('n', 0) for f in fields)
    sources = sorted({f.get('source', '?') for f in fields})
    waiting = data.get('incomplete', {})
    blue = data['bands']['blue'].upper()
    red = data['bands']['red'].upper()

    partial = [f for f in fields if f.get('partial')]
    partial_html = ''
    if partial:
        names = ', '.join(f"{html.escape(f['label'])} ({'+'.join(f['modules'])})"
                          for f in partial)
        partial_html = (
            f"<div class='warn'><b>Part of the tile:</b> {names}. These "
            f"pointings have no combined-module catalog yet, so only the "
            f"listed NIRCam module is plotted &mdash; roughly 60% of the "
            f"tile's sky, under the whole tile's name. Marked \u26a0 in the "
            f"table.</div>")

    waiting_html = ''
    if waiting:
        items = ', '.join(f"{html.escape(o)} (needs {html.escape('/'.join(b).upper())})"
                          for o, b in sorted(waiting.items()))
        waiting_html = (f"<div class='warn'><b>Waiting on a band:</b> {items}. "
                        f"These pointings have one filter reduced and not the "
                        f"other, so they carry no colour yet.</div>")

    return f"""<!doctype html>
<html lang=en>
<head>
<meta charset=utf-8>
<meta name=viewport content='width=device-width,initial-scale=1'>
<title>{html.escape(title)}</title>
<style>{CSS}</style>
</head>
<body>
<div id=wrap>
  <div id=sky></div>
  <div id=side>
    <h1>{html.escape(blue)} &minus; {html.escape(red)} by pointing</h1>
    <p class=sub>Hover a footprint on the sky (or a row below) to see that
      pointing's colour-magnitude diagram in <b>viridis</b> over the whole
      sample in <b>grey</b>. Click to pin it.</p>

    <canvas id=cmd width=760 height=860></canvas>
    <p id=cmdlabel class=muted>Whole sample: {nstars:,} stars matched in both
      bands across {len(fields)} pointings.</p>
    <div class=bar>
      <button id=unpin>clear selection</button>
      <button id=fitsky>zoom to data</button>
    </div>

    <h2>Pointings</h2>
    <table class=tiles id=tiles></table>

    <h2>Provenance</h2>
    <p class=muted>Catalogs: {html.escape(', '.join(sources) or 'none')}.
      Magnitudes are <b>Vega</b>. Built {html.escape(str(data.get('built', '')))}.
      Grid {data['grid']['nx']} hexagons wide.</p>
    {waiting_html}
    {partial_html}
    <div class=warn><b>Quicklook, not a release.</b> These are the catalogs that
      exist right now, cross-matched between the two filters at
      {data.get('match_arcsec', 0.1)}&Prime;. They have not been through the
      release gates, and the Treasury astrometry has no measured tie to
      VIRAC2/Gaia yet.</div>
    <p class=muted id=status>loading&hellip;</p>
  </div>
</div>
<script src="{ALADIN_JS}" charset="utf-8"></script>
<script>
const DATA_URL = {json.dumps(data_href)};
const HIPS = {json.dumps([{'id': i, 'url': u, 'name': n} for i, u, n in HIPS_LAYERS])};
const BASE_SURVEY = {json.dumps(BASE_SURVEY)};
{_SCRIPT}
</script>
</body>
</html>
"""


# The browser half.  Kept out of the f-string above so that the JavaScript can
# use braces without doubling every one of them.
_SCRIPT = r"""
var statusEl = document.getElementById('status');
var cmdEl = document.getElementById('cmd');
var labelEl = document.getElementById('cmdlabel');
var tilesEl = document.getElementById('tiles');
var DATA = null, aladin = null, byId = {}, hovered = null, pinned = null;

// Viridis at 9 stops, interpolated.  Enough for a density map: the eye reads
// the ramp, not the individual stops, and a 256-entry table is 8x the bytes
// for no visible difference.
var VIRIDIS = [[68,1,84],[72,40,120],[62,74,137],[49,104,142],[38,130,142],
               [31,158,137],[53,183,121],[109,205,89],[180,222,44],[253,231,37]];

function ramp(t) {
  t = Math.max(0, Math.min(1, t));
  var x = t * (VIRIDIS.length - 1), i = Math.floor(x), f = x - i;
  var a = VIRIDIS[i], b = VIRIDIS[Math.min(i + 1, VIRIDIS.length - 1)];
  return 'rgb(' + Math.round(a[0] + f * (b[0] - a[0])) + ',' +
                  Math.round(a[1] + f * (b[1] - a[1])) + ',' +
                  Math.round(a[2] + f * (b[2] - a[2])) + ')';
}

function grey(t) {
  // Floor at 45 so the sparsest occupied cell is still visible against the
  // panel; a linear ramp from 0 puts the CMD's outskirts below the background.
  var v = Math.round(45 + 175 * Math.max(0, Math.min(1, t)));
  return 'rgb(' + v + ',' + v + ',' + v + ')';
}

// Log counts: a crowded-field CMD spans four decades between the giant clump
// and the sparse blue end, and on a linear ramp everything but the clump is black.
function level(count, max) {
  if (max <= 1) return 1;
  return Math.log(count) / Math.log(max);
}

var PAD = {l: 62, r: 12, t: 12, b: 46};

function plotBox() {
  return {x: PAD.l, y: PAD.t,
          w: cmdEl.width - PAD.l - PAD.r, h: cmdEl.height - PAD.t - PAD.b};
}

// Normalised (u, v) -> canvas pixels.  The magnitude axis is INVERTED: v = 0
// is ymin, the brightest magnitude, and it goes at the TOP, which is how a
// colour-magnitude diagram is read.  Every y here follows that convention --
// the tick labels below use the same expression, so the two cannot disagree.
function toPix(u, v) {
  var b = plotBox();
  return [b.x + u * b.w, b.y + v * b.h];
}

function hexPath(ctx, cx, cy, rx, ry) {
  // Pointy-top hexagon, drawn with separate x/y radii: the grid is regular in
  // NORMALISED space, and the box it is drawn into is not square, so a single
  // radius would leave gaps along one axis.
  ctx.beginPath();
  for (var k = 0; k < 6; k++) {
    var a = Math.PI / 180 * (60 * k - 90);
    var x = cx + rx * Math.cos(a), y = cy + ry * Math.sin(a);
    if (k === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.closePath();
}

function drawCells(ctx, cells, max, colour) {
  var g = DATA.grid, R = g.radius, b = plotBox();
  // The hexagon's own half-width/half-height in normalised units, scaled into
  // the box independently on each axis.
  var rx = R * b.w * 1.08, ry = R * b.h * 1.08;
  for (var i = 0; i < cells.length; i++) {
    var q = cells[i][0], r = cells[i][1], c = cells[i][2];
    var u = R * Math.sqrt(3) * (q + r / 2), v = R * 1.5 * r;
    var p = toPix(u, v);
    ctx.fillStyle = colour(level(c, max));
    hexPath(ctx, p[0], p[1], rx, ry);
    ctx.fill();
  }
}

function drawAxes(ctx) {
  var g = DATA.grid, e = g.extent, b = plotBox();
  ctx.strokeStyle = '#30363d'; ctx.fillStyle = '#8b949e';
  ctx.lineWidth = 1; ctx.font = '12px system-ui, sans-serif';
  ctx.strokeRect(b.x, b.y, b.w, b.h);
  ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  var k;
  for (k = 0; k <= 4; k++) {
    var u = k / 4, x = b.x + u * b.w;
    ctx.beginPath(); ctx.moveTo(x, b.y + b.h); ctx.lineTo(x, b.y + b.h + 5); ctx.stroke();
    ctx.fillText((e[0] + u * (e[1] - e[0])).toFixed(1), x, b.y + b.h + 8);
  }
  ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  for (k = 0; k <= 5; k++) {
    var v = k / 5, p = toPix(0, v);
    ctx.beginPath(); ctx.moveTo(b.x - 5, p[1]); ctx.lineTo(b.x, p[1]); ctx.stroke();
    ctx.fillText((e[2] + v * (e[3] - e[2])).toFixed(1), b.x - 8, p[1]);
  }
  ctx.fillStyle = '#e6edf3'; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
  ctx.fillText(DATA.labels.x, b.x + b.w / 2, cmdEl.height - 6);
  ctx.save();
  ctx.translate(14, b.y + b.h / 2); ctx.rotate(-Math.PI / 2);
  ctx.textBaseline = 'top';
  ctx.fillText(DATA.labels.y, 0, 0);
  ctx.restore();
}

function drawCMD(field) {
  var ctx = cmdEl.getContext('2d');
  ctx.clearRect(0, 0, cmdEl.width, cmdEl.height);
  drawCells(ctx, DATA.all.cells, DATA.all.max, grey);
  if (field) drawCells(ctx, field.cells, field.max, ramp);
  drawAxes(ctx);
  if (field) {
    labelEl.textContent = field.label + ' (' + field.id + ') \u2014 ' +
      field.n.toLocaleString() + ' stars, ' + field.source + ' catalogs' +
      (field.partial
        ? ' \u2014 \u26a0 ' + field.modules.join('+') + ' only, part of the tile'
        : '') +
      (pinned && pinned.id === field.id ? ' \u2014 pinned' : '');
  } else {
    labelEl.textContent = 'Whole sample: ' + DATA.all.n.toLocaleString() +
      ' stars matched in both bands across ' + DATA.fields.length + ' pointings.';
  }
  Array.prototype.forEach.call(tilesEl.rows, function (row) {
    row.classList.toggle('on', !!field && row.dataset.id === field.id);
  });
}

function show(field) {
  hovered = field;
  drawCMD(field || pinned);
}

function cell(tr, text, cls) {
  var td = tr.insertCell();
  if (cls) { td.className = cls; }
  td.textContent = text;              // not innerHTML: these are data, not markup
  return td;
}

function buildTable() {
  DATA.fields.forEach(function (f) {
    var tr = tilesEl.insertRow();
    tr.dataset.id = f.id;
    cell(tr, f.label + (f.partial ? ' \u26a0' : ''));
    if (f.partial) {
      // One NIRCam module is about 60% of a tile.  Saying `GC_128` over 60% of
      // its sky with nothing to mark it is the page asserting coverage it does
      // not have.
      tr.title = f.label + ': ' + f.modules.join('+') + ' only \u2014 part of '
               + 'the tile, no merged catalog for it yet';
      tr.classList.add('partial');
    }
    cell(tr, f.n.toLocaleString(), 'num');
    cell(tr, f.partial ? f.modules.join('+') : f.source, 'num');
    tr.addEventListener('mouseenter', function () { show(f); });
    tr.addEventListener('mouseleave', function () { show(null); });
    tr.addEventListener('click', function () {
      pinned = (pinned && pinned.id === f.id) ? null : f;
      drawCMD(hovered || pinned);
      if (aladin && f.centre) { aladin.gotoRaDec(f.centre[0], f.centre[1]); }
    });
  });
}

// Screen-space point-in-polygon.  Projecting the footprint and testing the
// mouse pixel keeps this correct under any projection and any rotation, and
// costs one world2pix per vertex per mousemove.
function inPoly(px, py, pts) {
  var inside = false;
  for (var i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    var xi = pts[i][0], yi = pts[i][1], xj = pts[j][0], yj = pts[j][1];
    if (((yi > py) !== (yj > py)) &&
        (px < (xj - xi) * (py - yi) / (yj - yi) + xi)) inside = !inside;
  }
  return inside;
}

function fieldAt(px, py) {
  for (var i = 0; i < DATA.fields.length; i++) {
    var f = DATA.fields[i];
    for (var k = 0; k < f.polys.length; k++) {
      var screen = [];
      for (var v = 0; v < f.polys[k].length; v++) {
        var p = aladin.world2pix(f.polys[k][v][0], f.polys[k][v][1]);
        if (!p) { screen = null; break; }
        screen.push(p);
      }
      if (screen && inPoly(px, py, screen)) return f;
    }
  }
  return null;
}

function drawFootprints() {
  var ov = A.graphicOverlay({color: '#58a6ff', lineWidth: 1.4,
                             name: 'Treasury pointings with catalogs'});
  aladin.addOverlay(ov);
  DATA.fields.forEach(function (f) {
    f.polys.forEach(function (poly) { ov.add(A.polygon(poly)); });
  });
}

function boot() {
  buildTable();
  drawCMD(null);
  A.init.then(function () {
    aladin = A.aladin('#sky', {survey: BASE_SURVEY, target: '0 0', fov: 2.0,
                               cooFrame: 'galactic', showCooGridControl: true,
                               showLayersControl: true, showFullscreenControl: true,
                               showProjectionControl: false});
    HIPS.forEach(function (L) {
      var h = (typeof A.HiPS === 'function') ? A.HiPS(L.url, {name: L.name, imgFormat: 'png'})
                                             : A.imageHiPS(L.url, {name: L.name});
      aladin.setOverlayImageLayer(h, L.id);
    });
    drawFootprints();
    if (DATA.centre) { aladin.gotoRaDec(DATA.centre[0], DATA.centre[1]); }
    if (DATA.fov) { aladin.setFoV(DATA.fov); }

    var sky = document.getElementById('sky');
    sky.addEventListener('mousemove', function (ev) {
      var box = sky.getBoundingClientRect();
      var f = fieldAt(ev.clientX - box.left, ev.clientY - box.top);
      if (f !== hovered) show(f);
    });
    sky.addEventListener('mouseleave', function () { show(null); });
    sky.addEventListener('click', function (ev) {
      var box = sky.getBoundingClientRect();
      var f = fieldAt(ev.clientX - box.left, ev.clientY - box.top);
      if (f) { pinned = (pinned && pinned.id === f.id) ? null : f; drawCMD(f); }
    });
    statusEl.textContent = DATA.fields.length + ' pointings drawn.';
  }).catch(function (err) {
    statusEl.textContent = 'Aladin Lite failed to initialise: ' + err +
      ' -- the diagram on the right still works.';
  });
}

document.getElementById('unpin').addEventListener('click', function () {
  pinned = null; drawCMD(hovered);
});
document.getElementById('fitsky').addEventListener('click', function () {
  if (aladin && DATA.centre) { aladin.gotoRaDec(DATA.centre[0], DATA.centre[1]);
                               aladin.setFoV(DATA.fov || 2.0); }
});

fetch(DATA_URL).then(function (r) {
  if (!r.ok) throw new Error(r.status + ' ' + r.statusText);
  return r.json();
}).then(function (d) { DATA = d; boot(); }).catch(function (err) {
  // The commonest cause is opening the file from disk: fetch() refuses a
  // file:// sibling.  Say so, rather than leaving a blank canvas.
  statusEl.textContent = 'could not load ' + DATA_URL + ': ' + err +
    (location.protocol === 'file:'
      ? ' -- this page has to be served over http, not opened from disk.' : '');
});
"""
