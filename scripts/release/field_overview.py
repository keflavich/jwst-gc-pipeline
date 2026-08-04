"""Whole-CMZ overview panel for the release index: where each field is on sky.

The index is a grid of cards, which says what exists but not *where*.  This adds
one framed panel showing every Galactic Centre field's real footprint over the
CMZ, with each footprint a link to that field's release page.

Two rendering paths, same geometry -- the arrangement ``jwst_gc_pipeline.monitoring.skyview``
arrived at, for the same reason:

* a **static SVG map**, inline, no script and no network.  Each field is an
  ``<a>`` wrapping its polygons, so the click-through is a plain link that works
  with JavaScript off, under a strict CSP, and from ``file://``;
* an **optional Aladin Lite upgrade**, loaded only when the reader asks for it,
  which swaps in pannable/zoomable sky with the survey's own JWST CMZ HiPS
  underneath.  Clicking a field there navigates to the same page.

Geometry comes from the staged mosaics themselves (``images/<FILT>/*_i2d.fits``):
an ``i2d`` is a rectified plain ``RA---TAN`` grid with no SIP, so
``astropy.wcs.WCS(header)`` on it is exact -- this is the case the
frame-WCS rule in CLAUDE.md explicitly exempts.  Only headers are read.

Drawn in **Galactic** coordinates: the CMZ fields span l ~ -0.6..+0.7, b ~ -0.15..+0.1,
so in l/b they lie along a grid line instead of diagonally across one, and the
panel can be a wide strip rather than mostly blank sky.  Without astropy the
module degrades to ICRS and labels the axes it actually has, rather than
claiming Galactic and drawing something else.
"""
import glob
import html
import json
import math
import os

#: The survey's own imagery, first; the rest are context.  Same list the monitor
#: uses, so a reader recognises the two pages as one survey.
SURVEYS = (
    ('JWST CMZ', 'https://starformation.astro.ufl.edu/avm_images/jwst_cmz_hips/'),
    ('CMZ RGB', 'https://starformation.astro.ufl.edu/avm_images/rgb_final_uncropped_hips/'),
    ('GLIMPSE', 'P/Spitzer/GLIMPSE360'),
    ('2MASS', 'P/2MASS/color'),
    ('DSS', 'P/DSS2/color'),
)

ALADIN_JS = 'https://aladin.cds.unistra.fr/AladinLite/api/v3/latest/aladin.js'

#: Distinguishable at small size against a dark background, in a fixed order so
#: a field keeps its colour between builds (the index is regenerated often).
PALETTE = ('#46bcd6', '#f6c453', '#a78bfa', '#4ade80', '#ff8ba0',
           '#7dd3fc', '#fb923c', '#c4b5fd', '#86efac', '#fca5a5',
           '#67e8f9', '#fcd34d')

#: Padding around the drawn footprints, as a fraction of the larger span.
PAD_FRAC = 0.06
#: Width of the plotted sky box; the viewBox is this plus the margins.
SVG_W = 1000.0
#: Clamp on the drawn height so one stray field cannot make a page-tall panel.
SVG_H_MAX = 460.0
#: Room outside the sky box for tick labels and axis titles.  Without it the
#: leftmost longitude tick is half off the canvas and the axis title sits on the
#: ticks.
MARGIN_LEFT, MARGIN_RIGHT, MARGIN_TOP, MARGIN_BOTTOM = 42.0, 26.0, 18.0, 40.0


def _science_mosaics(field_dir):
    """Staged science mosaics (not the residual/model images) under a field."""
    pattern = os.path.join(str(field_dir), 'images', '**', '*_i2d.fits')
    return sorted(p for p in glob.glob(pattern, recursive=True)
                  if '_residual' not in os.path.basename(p)
                  and '_model' not in os.path.basename(p))


def footprint_polys(field_dir):
    """Sky polygons (lists of ``(ra, dec)`` in degrees) for one staged field.

    One polygon per staged science mosaic.  Returns ``[]`` when the field has no
    readable mosaic -- a missing or corrupt image must cost the field its outline,
    never the whole index.
    """
    try:
        from astropy.io import fits
        from astropy.wcs import WCS
    except ImportError:
        return []
    polys = []
    for path in _science_mosaics(field_dir):
        try:
            header = fits.getheader(path, 'SCI')
            corners = WCS(header).calc_footprint(header)
        except (OSError, KeyError, ValueError):
            continue
        if corners is None or len(corners) < 3:
            continue
        polys.append([(float(a), float(d)) for a, d in corners])
    return polys


def to_galactic(polys):
    """``(polys_in_frame, frame_name)``.  Galactic when astropy is importable.

    Longitudes are wrapped at 180 deg so the CMZ (which straddles l = 0) is one
    continuous strip rather than two clumps at either end of a 0..360 axis.
    """
    if not polys:
        return [], 'icrs'
    try:
        import numpy as np
        from astropy.coordinates import SkyCoord
    except ImportError:
        return polys, 'icrs'
    counts = [len(p) for p in polys]
    flat = np.array([pt for poly in polys for pt in poly], dtype=float)
    gal = SkyCoord(flat[:, 0], flat[:, 1], unit='deg', frame='icrs').galactic
    lon = gal.l.wrap_at('180d').deg
    lat = gal.b.deg
    out, i = [], 0
    for n in counts:
        out.append([(float(lon[j]), float(lat[j])) for j in range(i, i + n)])
        i += n
    return out, 'galactic'


def collect(entries):
    """Geometry for each field that has one.

    ``entries`` is an iterable of ``(field, field_dir, href)``.  Fields whose
    mosaics cannot be read are dropped, so the panel shows what it can rather
    than nothing.
    """
    out = []
    for field, field_dir, href in entries:
        polys = footprint_polys(field_dir)
        if polys:
            out.append({'field': field, 'href': href, 'polys': polys})
    return out


def _mean_point(polys):
    pts = [pt for poly in polys for pt in poly]
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def _nice_step(span):
    """A round grid step giving roughly 4-8 lines across ``span``."""
    if span <= 0:
        return 1.0
    raw = span / 6.0
    mag = 10.0 ** math.floor(math.log10(raw))
    for mult in (1.0, 2.0, 2.5, 5.0, 10.0):
        if raw <= mult * mag:
            return mult * mag
    return 10.0 * mag


def _projection(geoms, frame):
    """Linear sky->pixel mapping plus the drawn extent.

    Plate carrée with the longitude axis compressed by ``cos(lat)``, which is an
    equal-scale projection to well under a pixel over the ~1 deg the CMZ spans.
    Longitude increases to the LEFT, the convention for a Galactic map.
    """
    pts = [pt for g in geoms for poly in g['polys'] for pt in poly]
    lon = [p[0] for p in pts]
    lat = [p[1] for p in pts]
    lon0, lon1 = min(lon), max(lon)
    lat0, lat1 = min(lat), max(lat)
    coslat = math.cos(math.radians((lat0 + lat1) / 2.0)) or 1.0
    pad = PAD_FRAC * max((lon1 - lon0) * coslat, lat1 - lat0, 1e-6)
    lon0 -= pad / coslat
    lon1 += pad / coslat
    lat0 -= pad
    lat1 += pad
    w_sky = (lon1 - lon0) * coslat
    h_sky = lat1 - lat0
    scale = SVG_W / w_sky if w_sky > 0 else 1.0
    height = h_sky * scale
    if height > SVG_H_MAX:                 # keep the panel a strip, not a wall
        scale *= SVG_H_MAX / height
        height = SVG_H_MAX
    width = w_sky * scale

    def project(lon_deg, lat_deg):
        # longitude increases leftward, latitude upward; offset into the margins
        x = MARGIN_LEFT + (lon1 - lon_deg) * coslat * scale
        y = MARGIN_TOP + (lat1 - lat_deg) * scale
        return x, y

    return project, (lon0, lon1, lat0, lat1), (width, height)


#: Vertical separation between two field labels, in viewBox units (font is 14px
#: in the same units, so this is one line plus a little air).
LABEL_DY = 17.0
#: Approximate advance width per character at that font size.  Only used to
#: decide whether two labels would overlap, so an estimate is enough.
LABEL_CHAR_W = 7.6


def label_half_width(name):
    return 0.5 * LABEL_CHAR_W * len(name)


def _spread_labels(labels):
    """Nudge overlapping labels apart vertically.

    The CMZ fields are not evenly spread -- arches, quintuplet, sickle and
    gc2211 sit within ~0.1 deg of each other -- so centroid labels collide and
    become unreadable exactly where the map is busiest.  A label that would
    overlap an already-placed one is pushed below it; the marker (the polygon)
    never moves, only its name.

    ``labels`` is a list of ``[x, y, half_width]``; modified in place and
    returned.
    """
    for index in range(len(labels)):
        x, y, half = labels[index]
        for _ in range(len(labels)):        # bounded: one pass per label, at most
            moved = False
            for other_x, other_y, other_half in labels[:index]:
                if (abs(x - other_x) < half + other_half
                        and abs(y - other_y) < LABEL_DY):
                    y = other_y + LABEL_DY
                    moved = True
            if not moved:
                break
        labels[index] = [x, y, half]
    return labels


def _fmt(value):
    text = f'{value:.3f}'.rstrip('0').rstrip('.')
    return text or '0'


def _graticule(project, bounds, size, frame):
    lon0, lon1, lat0, lat1 = bounds
    width, height = size
    top, bottom = MARGIN_TOP, MARGIN_TOP + height
    left, right = MARGIN_LEFT, MARGIN_LEFT + width
    lon_label = 'l' if frame == 'galactic' else 'RA'
    lat_label = 'b' if frame == 'galactic' else 'Dec'
    parts = []
    step = _nice_step(lon1 - lon0)
    value = math.ceil(lon0 / step) * step
    while value <= lon1 + 1e-9:
        x, _ = project(value, (lat0 + lat1) / 2.0)
        parts.append(f'<line class="ov-grid" x1="{x:.1f}" y1="{top:.1f}" '
                     f'x2="{x:.1f}" y2="{bottom:.1f}"/>')
        parts.append(f'<text class="ov-tick" x="{x:.1f}" y="{bottom + 15:.1f}" '
                     f'text-anchor="middle">{_fmt(value)}</text>')
        value += step
    step = _nice_step(lat1 - lat0)
    value = math.ceil(lat0 / step) * step
    while value <= lat1 + 1e-9:
        _, y = project((lon0 + lon1) / 2.0, value)
        parts.append(f'<line class="ov-grid" x1="{left:.1f}" y1="{y:.1f}" '
                     f'x2="{right:.1f}" y2="{y:.1f}"/>')
        parts.append(f'<text class="ov-tick" x="{left - 6:.1f}" y="{y + 4:.1f}" '
                     f'text-anchor="end">{_fmt(value)}</text>')
        value += step
    parts.append(f'<text class="ov-axis" x="{left + width / 2:.1f}" '
                 f'y="{bottom + 33:.1f}" text-anchor="middle">{lon_label} (deg)</text>')
    parts.append(f'<text class="ov-axis" x="{left - 6:.1f}" y="{top - 5:.1f}" '
                 f'text-anchor="end">{lat_label}</text>')
    return parts


CSS = """
.overview { border:1px solid var(--border); border-radius:8px; background:var(--panel);
            margin:1.25rem 0 2rem; overflow:hidden; }
.overview > h3 { margin:0; padding:.7rem 1rem; font-size:1rem;
                 border-bottom:1px solid var(--border); }
.overview .ov-note { padding:.5rem 1rem 0; color:var(--muted); font-size:.85rem; }
.ov-stage { position:relative; background:#05080a; }
.ov-stage svg { display:block; width:100%; height:auto; }
.ov-grid { stroke:#26343a; stroke-width:1; stroke-dasharray:3 4; }
.ov-tick { fill:#7d919a; font-size:12px; font-family:ui-monospace,monospace; }
.ov-axis { fill:#9fb4bc; font-size:12px; font-family:ui-monospace,monospace; }
/* The dark halo that keeps a label readable over a filled footprint is a
   SEPARATE stroked copy drawn underneath, not `paint-order:stroke` on one
   element: without paint-order support the stroke paints OVER the fill and a
   14px label becomes an unreadable black smudge (rsvg does exactly that). Two
   elements need no such support anywhere. */
.ov-name { font-size:14px; font-weight:600; }
.ov-halo { stroke:#05080a; stroke-width:3px; stroke-linejoin:round; fill:none; }
.ov-field polygon { fill-opacity:.18; stroke-width:1.6; }
.ov-field:hover polygon, .ov-field:focus polygon { fill-opacity:.42; stroke-width:2.6; }
.ov-field:focus { outline:none; }
.ov-legend { display:flex; flex-wrap:wrap; gap:.4rem .9rem; padding:.7rem 1rem; }
.ov-legend a { display:inline-flex; align-items:center; gap:.35rem; font-size:.85rem; }
.ov-sw { width:14px; height:4px; border-radius:2px; display:inline-block; }
.ov-live { padding:.6rem 1rem; border-top:1px solid var(--border); }
.ov-live button { cursor:pointer; font:inherit; font-size:.82rem; padding:.3rem .8rem;
                  border-radius:4px; border:1px solid var(--border);
                  background:transparent; color:inherit; }
.ov-live button:hover { border-color:var(--muted); }
.ov-aladin { position:absolute; inset:0; }
.ov-msg { padding:.4rem 1rem .8rem; color:var(--muted); font-size:.8rem; }
"""


def static_svg(geoms, frame):
    """Inline SVG of the footprints; each field is one ``<a>``."""
    project, bounds, (width, height) = _projection(geoms, frame)
    svg_w = width + MARGIN_LEFT + MARGIN_RIGHT
    svg_h = height + MARGIN_TOP + MARGIN_BOTTOM
    parts = [f'<svg viewBox="0 0 {svg_w:.1f} {svg_h:.1f}" '
             f'role="img" aria-label="Footprints of the released fields on sky" '
             f'xmlns="http://www.w3.org/2000/svg">']
    parts += _graticule(project, bounds, (width, height), frame)
    # place every label first, so the de-collision sees them all
    labels = _spread_labels(
        [[*project(*_mean_point(g['polys'])), label_half_width(g['field'])]
         for g in geoms])
    for index, geom in enumerate(geoms):
        color = PALETTE[index % len(PALETTE)]
        name = html.escape(geom['field'])
        parts.append(f'<a class="ov-field" href="{html.escape(geom["href"], quote=True)}" '
                     f'aria-label="{name} release page">')
        parts.append(f'<title>{name}</title>')
        for poly in geom['polys']:
            pts = ' '.join(f'{x:.1f},{y:.1f}' for x, y in
                           (project(lon, lat) for lon, lat in poly))
            parts.append(f'<polygon points="{pts}" fill="{color}" stroke="{color}"/>')
        cx, cy = labels[index][0], labels[index][1]
        parts.append(f'<text class="ov-name ov-halo" x="{cx:.1f}" y="{cy:.1f}" '
                     f'text-anchor="middle" aria-hidden="true">{name}</text>')
        parts.append(f'<text class="ov-name" x="{cx:.1f}" y="{cy:.1f}" '
                     f'text-anchor="middle" fill="{color}">{name}</text>')
        parts.append('</a>')
    parts.append('</svg>')
    return '\n'.join(parts)


def section(geoms, title='The fields on sky', aladin_src=ALADIN_JS,
            surveys=SURVEYS):
    """The whole panel, or ``''`` when there is no geometry to draw.

    Returning ``''`` rather than an empty frame matters: a release built where
    astropy is unavailable, or before any field is staged, should look like the
    index always did, not like a broken widget.
    """
    if not geoms:
        return ''
    framed, frame = to_galactic([p for g in geoms for p in g['polys']])
    # put the reframed polygons back on their fields, in order
    reframed, index = [], 0
    for geom in geoms:
        count = len(geom['polys'])
        reframed.append({**geom, 'polys': framed[index:index + count]})
        index += count

    legend = ''.join(
        f'<a href="{html.escape(g["href"], quote=True)}">'
        f'<span class="ov-sw" style="background:{PALETTE[i % len(PALETTE)]}"></span>'
        f'{html.escape(g["field"])}</a>'
        for i, g in enumerate(reframed))

    payload = json.dumps({
        'fields': [{'field': g['field'], 'href': g['href'],
                    # Aladin works in ICRS, so it gets the ORIGINAL sky polygons
                    'polys': geoms[i]['polys'],
                    'color': PALETTE[i % len(PALETTE)]}
                   for i, g in enumerate(reframed)],
        'surveys': [{'name': n, 'id': s} for n, s in surveys],
        'aladin': aladin_src,
    })

    frame_note = ('Galactic coordinates' if frame == 'galactic'
                  else 'equatorial coordinates (astropy unavailable, so the '
                       'Galactic transform was skipped)')
    return f"""<section class=overview>
<h3>{html.escape(title)}</h3>
<div class=ov-note>Released footprints over the Central Molecular Zone, in
{frame_note}. Click a field to open its release page.</div>
<div class=ov-stage id=ov-stage>{static_svg(reframed, frame)}</div>
<div class=ov-legend>{legend}</div>
<div class=ov-live><button type=button id=ov-load>Load interactive sky view
(Aladin Lite, ~1.8&nbsp;MB)</button> <span id=ov-status class=muted></span></div>
<script id=ov-data type="application/json">{payload}</script>
<script>
(function () {{
  var btn = document.getElementById('ov-load');
  var status = document.getElementById('ov-status');
  var stage = document.getElementById('ov-stage');
  if (!btn || !stage) {{ return; }}
  var data;
  try {{ data = JSON.parse(document.getElementById('ov-data').textContent); }}
  catch (err) {{ btn.disabled = true; return; }}
  function fail(msg) {{
    status.textContent = msg;
    btn.disabled = false;
    btn.textContent = 'Retry interactive sky view';
  }}
  btn.addEventListener('click', function () {{
    btn.disabled = true;
    status.textContent = 'loading\\u2026';
    var script = document.createElement('script');
    script.src = data.aladin;
    script.onerror = function () {{
      // the static map is still on screen; say what is missing and stop
      fail('Aladin Lite could not be loaded (offline, or blocked by a content '
           + 'security policy). The map above is unaffected.');
    }};
    script.onload = function () {{
      if (typeof A === 'undefined' || !A.init) {{ fail('Aladin Lite did not initialise.'); return; }}
      A.init.then(function () {{
        var host = document.createElement('div');
        host.className = 'ov-aladin';
        // the static SVG stays in the DOM underneath; if Aladin throws we
        // remove the host and the reader is back where they started
        stage.appendChild(host);
        var aladin;
        try {{
          aladin = A.aladin(host, {{
            survey: data.surveys[0].id, projection: 'AIT', cooFrame: 'galactic',
            target: 'galactic 0 0', fov: 1.8, showReticle: false,
            showCooGrid: true, showFullscreenControl: false
          }});
        }} catch (err) {{
          stage.removeChild(host);
          fail('This browser could not start Aladin Lite (' + err + '). The map above is unaffected.');
          return;
        }}
        var cat = A.catalog({{name: 'released fields', sourceSize: 14, onClick: 'showPopup'}});
        aladin.addCatalog(cat);
        data.fields.forEach(function (f) {{
          var ov = A.graphicOverlay({{color: f.color, lineWidth: 2, name: f.field}});
          aladin.addOverlay(ov);
          var lon = 0, lat = 0, n = 0;
          f.polys.forEach(function (poly) {{
            ov.add(A.polygon(poly));
            poly.forEach(function (pt) {{ lon += pt[0]; lat += pt[1]; n += 1; }});
          }});
          if (n) {{ cat.addSources([A.source(lon / n, lat / n, {{field: f.field, href: f.href}})]); }}
        }});
        aladin.on('objectClicked', function (src) {{
          if (src && src.data && src.data.href) {{ window.location.href = src.data.href; }}
        }});
        status.textContent = 'click a field to open its release page';
        btn.textContent = 'Interactive sky view loaded';
      }}).catch(function (err) {{
        fail('Aladin Lite failed to start (' + err + '). The map above is unaffected.');
      }});
    }};
    document.head.appendChild(script);
  }});
}})();
</script>
</section>"""
