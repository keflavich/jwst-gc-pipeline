"""A page that pans slowly across the Treasury mosaic at full resolution.

The sky viewer answers "what is at this position"; this one answers "what does
the survey look like", which is a different question and is not served by a
viewer you have to drive. It walks the tiles that have imagery, at the HiPS's
own pixel scale, so the detail on screen is the detail in the data rather than
a zoomed-out summary of it.

The tour visits only pointings the coadd actually contains -- a tile that has
been observed but not yet rendered would pan the viewer across blank sky and
look like a fault. The builder takes that list from the per-field layers the
coadd was built from, not from the observing schedule.
"""
import json

ALADIN_CSS = 'https://aladin.cds.unistra.fr/AladinLite/api/v3/latest/aladin.css'
ALADIN_JS = 'https://aladin.cds.unistra.fr/AladinLite/api/v3/latest/aladin.js'

#: The layer this page exists to show, and the one Adam made the default
#: everywhere else: fixed cuts, so a given surface brightness is the same
#: colour in every field.  A per-field stretch would make the pan read as
#: brightness steps at every tile edge.
SURVEY_URL = ('https://starformation.astro.ufl.edu/avm_images/'
              'jwst_gc_treasury_vminmax_hips/')

#: Degrees across the viewport.  The HiPS is order 14, so its pixels are about
#: 0.019", and 0.008 deg (~29") is a little under 1:1 on a 1500 px window --
#: full zoom without magnifying past the data.
DEFAULT_FOV = 0.008

#: Arcseconds per second of wall clock.  At 2"/s a NIRCam tile (~2.2') takes
#: about a minute to cross, and the whole survey a little over half an hour.
DEFAULT_RATE = 2.0

PAGE_FILE = 'slow_panner.html'
DATA_FILE = 'slow_panner_tour.json'

CSS = """
:root { --bg:#05070c; --fg:#e6edf3; --muted:#8b949e; --accent:#58a6ff; }
* { box-sizing:border-box; }
html, body { margin:0; height:100%; background:var(--bg); color:var(--fg);
  font:14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
#sky { position:absolute; inset:0; }
.bar { position:absolute; left:0; right:0; bottom:0; z-index:5;
  display:flex; gap:.75rem; align-items:center; flex-wrap:wrap;
  padding:.6rem .9rem; background:rgba(5,7,12,.82);
  border-top:1px solid #1f2937; backdrop-filter:blur(3px); }
.bar button, .bar select { background:#0d1117; color:var(--fg);
  border:1px solid #30363d; border-radius:6px; padding:.3rem .7rem;
  font:inherit; cursor:pointer; }
.bar button:hover, .bar select:hover { border-color:var(--accent); }
.where { color:var(--muted); margin-left:auto; text-align:right;
  font-variant-numeric:tabular-nums; }
.where b { color:var(--fg); }
.head { position:absolute; top:0; left:0; right:0; z-index:5;
  padding:.55rem .9rem; background:linear-gradient(rgba(5,7,12,.85),transparent);
  color:var(--muted); }
.head a { color:var(--accent); }
.head b { color:var(--fg); }
@media (max-width:640px) { .where { margin-left:0; text-align:left; } }
"""

_SCRIPT = r"""
var TOUR = null, aladin = null, playing = true, rate = 1.0;
var leg = 0, along = 0, last = null;

function fmt(deg, isRa) {
  var v = isRa ? deg / 15 : Math.abs(deg);
  var a = Math.floor(v), b = Math.floor((v - a) * 60), c = ((v - a) * 60 - b) * 60;
  var sign = (!isRa && deg < 0) ? '-' : '';
  return sign + a + (isRa ? 'h' : '°') + (b < 10 ? '0' : '') + b +
         (isRa ? 'm' : "'") + (c < 10 ? '0' : '') + c.toFixed(1) + (isRa ? 's' : '"');
}

// Great-circle interpolation. Linear interpolation in RA/Dec would run visibly
// faster in RA near the pole and, more to the point here, would not hold the
// constant on-sky rate the page promises: at dec -29 a degree of RA is 0.87
// degrees of sky.
function unit(ra, dec) {
  var r = ra * Math.PI / 180, d = dec * Math.PI / 180;
  return [Math.cos(d) * Math.cos(r), Math.cos(d) * Math.sin(r), Math.sin(d)];
}

function fromUnit(v) {
  var ra = Math.atan2(v[1], v[0]) * 180 / Math.PI;
  if (ra < 0) { ra += 360; }
  return [ra, Math.asin(Math.max(-1, Math.min(1, v[2]))) * 180 / Math.PI];
}

function slerp(a, b, t) {
  var u = unit(a[0], a[1]), w = unit(b[0], b[1]);
  var dot = Math.max(-1, Math.min(1, u[0] * w[0] + u[1] * w[1] + u[2] * w[2]));
  var ang = Math.acos(dot);
  if (ang < 1e-9) { return a.slice(); }
  var s = Math.sin(ang);
  var k1 = Math.sin((1 - t) * ang) / s, k2 = Math.sin(t * ang) / s;
  return fromUnit([k1 * u[0] + k2 * w[0], k1 * u[1] + k2 * w[1],
                   k1 * u[2] + k2 * w[2]]);
}

function sep(a, b) {
  var u = unit(a[0], a[1]), w = unit(b[0], b[1]);
  var dot = Math.max(-1, Math.min(1, u[0] * w[0] + u[1] * w[1] + u[2] * w[2]));
  return Math.acos(dot) * 180 / Math.PI * 3600;        // arcsec
}

function legPoints(i) {
  var pts = TOUR.stops;
  return [pts[i % pts.length], pts[(i + 1) % pts.length]];
}

function step(now) {
  requestAnimationFrame(step);
  if (!aladin || !TOUR) { return; }
  if (last === null) { last = now; }
  var dt = Math.min(0.25, (now - last) / 1000);       // clamp: a backgrounded
  last = now;                                          // tab must not jump
  if (!playing) { return; }

  var ends = legPoints(leg);
  // A leg the builder marked as a cut is crossed instantly: the survey is not
  // one contiguous block, and panning the gap in real time is minutes of empty
  // sky rather than a view of the data.
  // Bounded: an all-cut tour would spin here forever inside
  // requestAnimationFrame and freeze the tab with nothing in the console. The
  // builder refuses to write one, and this is the second line of defence for a
  // hand-edited tour file.
  for (var guard = 0; ends[0].jump && guard < TOUR.stops.length; guard++) {
    leg = (leg + 1) % TOUR.stops.length;
    along = 0;
    ends = legPoints(leg);
  }
  if (ends[0].jump) { return; }
  var span = sep([ends[0].ra, ends[0].dec], [ends[1].ra, ends[1].dec]);
  if (span < 1e-6) { leg = (leg + 1) % TOUR.stops.length; along = 0; return; }
  along += (TOUR.rate * rate * dt) / span;
  while (along >= 1) {
    along -= 1;
    leg = (leg + 1) % TOUR.stops.length;
    ends = legPoints(leg);
    for (var g2 = 0; ends[0].jump && g2 < TOUR.stops.length; g2++) {
      leg = (leg + 1) % TOUR.stops.length;
      along = 0;
      ends = legPoints(leg);
    }
    if (ends[0].jump) { return; }
    span = sep([ends[0].ra, ends[0].dec], [ends[1].ra, ends[1].dec]);
    if (span < 1e-6) { along = 0; break; }
  }
  var p = slerp([ends[0].ra, ends[0].dec], [ends[1].ra, ends[1].dec], along);
  aladin.gotoRaDec(p[0], p[1]);
  var near = along < 0.5 ? ends[0] : ends[1];
  document.getElementById('where').innerHTML =
    '<b>' + near.label + '</b> &mdash; ' + fmt(p[0], true) + ' ' + fmt(p[1], false);
}

function boot() {
  A.init.then(function () {
    aladin = A.aladin('#sky', {survey: TOUR.survey, fov: TOUR.fov,
                               target: TOUR.stops[0].ra + ' ' + TOUR.stops[0].dec,
                               cooFrame: 'icrs', showLayersControl: false,
                               showFullscreenControl: true,
                               showCooGridControl: false,
                               showProjectionControl: false,
                               showZoomControl: false, showGotoControl: false});
    requestAnimationFrame(step);
  }).catch(function (err) {
    document.getElementById('where').textContent =
      'Aladin Lite failed to load: ' + err;
  });
}

document.getElementById('play').addEventListener('click', function () {
  playing = !playing;
  this.textContent = playing ? '⏸ Pause' : '▶ Play';
});
document.getElementById('rate').addEventListener('change', function () {
  rate = parseFloat(this.value);
});
document.getElementById('skip').addEventListener('click', function () {
  leg = (leg + 1) % TOUR.stops.length; along = 0;
});

fetch(DATA_URL, {cache: 'no-cache'})
  .then(function (r) { if (!r.ok) { throw new Error('HTTP ' + r.status); }
                       return r.json(); })
  .then(function (d) { TOUR = d; boot(); })
  .catch(function (err) {
    document.getElementById('where').textContent =
      'could not load the tour: ' + err;
  });
"""


def render_page(data_url=DATA_FILE):
    """The page itself.  The tour is fetched rather than inlined so a rebuilt
    tour does not require rewriting the HTML."""
    return f"""<!doctype html>
<html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>JWST Galactic Center — slow panner</title>
<link rel=stylesheet href="{ALADIN_CSS}">
<style>{CSS}</style>
</head><body>
<div id=sky></div>
<div class=head>Programme 10678, <b>F212N + F480M</b> at full resolution —
  drifting across the tiles that have imagery.
  <a href="index.html">back to the release</a></div>
<div class=bar>
  <button id=play>&#9208; Pause</button>
  <button id=skip>&#9197; Next tile</button>
  <select id=rate title="pan rate">
    <option value="0.5">0.5&times;</option>
    <option value="1" selected>1&times;</option>
    <option value="2">2&times;</option>
    <option value="4">4&times;</option>
  </select>
  <span class=where id=where>loading the tour&hellip;</span>
</div>
<script src="{ALADIN_JS}" charset=utf-8></script>
<script>
const DATA_URL = {json.dumps(data_url)};
{_SCRIPT}
</script>
</body></html>
"""
