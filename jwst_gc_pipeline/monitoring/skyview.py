"""The Aladin Lite sky view: where the survey plans to observe, and what it has.

Modelled on ``roman_footprint_gbtds.html`` from the galactic-plane-surveys repo,
with the layer set inverted for this page's purpose: **only the JWST footprints
are on by default**, and the Roman GBTDS tiles are available but off, because a
pipeline monitor is about the JWST survey and the Roman geometry is context.

Three JWST layers, deliberately separate:

* **planned NIRCam** -- the prime instrument's 8 SW detectors per pointing;
* **planned MIRI** -- the coordinated parallel, which lands ~7.5' away from the
  prime and so covers *different sky*.  Drawing it in the NIRCam colour would
  suggest the survey covers a contiguous area it does not;
* **observed** -- its own colour and its own toggle, filled in from the APT
  visit status.  Today it is empty; the layer and its toggle still render, so
  the page shows "nothing observed yet" rather than hiding the question.

Loading
-------
The panel draws the footprints **twice**, by two routes with different
dependencies:

* a **static map** -- an inline SVG on a tangent plane, built here at render
  time.  No script, no fetch, no tiles: it is part of the HTML.  This is what
  renders in a published artifact, from a ``file://`` path, or anywhere else the
  network is unavailable, and it is what the reader sees before deciding whether
  the interactive view is worth 1.8 MB.
* the **interactive view** -- Aladin Lite over remote HiPS imagery, loaded
  lazily on request.  The script is served from this page's own directory
  (``report.publish`` links a copy in), so no third-party CDN is involved, but
  the HiPS tiles are unavoidably remote -- that is what a HiPS is.

Where the interactive route is blocked the static map simply stays up and the
caption says which part was unavailable.  Earlier this panel had only the Aladin
route and covered the map with a full-bleed overlay on failure; that overlay
also swallowed every pointer event.  There is no overlay now, which removes that
class of bug along with the blank rectangle.
"""
import json
import math
import os

#: Where a copy of Aladin Lite is hardlinked from when publishing.  Same origin
#: as the page, so no third-party CDN is involved.
ALADIN_SOURCE = '/orange/adamginsburg/web/public/ACES_Aladin_tour/aladin.js'
ALADIN_LOCAL = 'aladin.js'

#: Written next to the page by ``report.write_report``.
FOOTPRINTS_JSON = 'footprints.json'
#: Roman GBTDS context geometry.  The static map draws these layers directly
#: (~14 kB), so their toggles work without the interactive view; this file is
#: still fetched for the interactive view, which wants the full-precision
#: polygons rather than the map's rounded ones.
ROMAN_JSON = 'roman_gbtds.json'

#: Layer colours.  JWST uses the page's own accent family; Roman keeps the
#: blue/orange of the source page so the two are recognisably the same layers.
COLOR_NIRCAM_PLANNED = '#46bcd6'
COLOR_MIRI_PLANNED = '#a78bfa'
COLOR_OBSERVED = '#4ade80'
COLOR_SPRING = '#1E90FF'
COLOR_AUTUMN = '#FF8C00'
COLOR_TARGET_AREA = '#ff6b6b'

#: Backgrounds worth having here.  The first is the one the view opens on, so it has to be all-sky.  The survey's
#: own imagery is a CMZ-only HiPS (``hips_initial_fov`` 0.077 deg, no tiles below
#: order ~8); opening on it at a 1.6 deg field showed a black rectangle, because
#: there is nothing there to draw at that scale.  It is worth having, but as a
#: choice you zoom into, not as the default.
#:
#: IDs checked against the CDS MOCServer rather than copied: ``P/DSS2/color``,
#: ``P/2MASS/color`` and ``P/allWISE/color`` resolve (to ``CDS/P/...``), but
#: ``P/Spitzer/GLIMPSE360`` matches nothing at all -- that button named a HiPS
#: that has never existed under that ID.  GLIMPSE360 is served by IPAC.
#:
#: The third element is an optional note for the reader: the CMZ mosaic has no
#: tiles below order 8, so it is blank until you are zoomed in past ~10'.
SURVEYS = (
    ('DSS', 'P/DSS2/color', ''),
    ('2MASS', 'P/2MASS/color', ''),
    ('GLIMPSE', 'IPAC/P/GLIMPSE360', ''),
    ('WISE', 'P/allWISE/color', ''),
    ('JWST CMZ', 'https://starformation.astro.ufl.edu/avm_images/jwst_cmz_hips/',
     'the survey mosaic — zoom in past ~10′ for it to appear'),
)


def load_footprints(path):
    """The footprint JSON, or ``None``.  Built by ``scripts/monitoring/build_footprints.py``."""
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _esc(text):
    import html
    return html.escape('' if text is None else str(text), quote=True)


CSS = """
.gcm-sky { margin-top: .75rem; }
.gcm-sky-wrap { position: relative; height: 560px; border-radius: 3px;
                overflow: hidden; border: 1px solid var(--rule);
                background: #05080a; }
#gcm-aladin { position: absolute; inset: 0; }
.gcm-sky-ui { position: absolute; top: 8px; right: 8px; z-index: 10; width: 208px;
              background: rgba(10,16,20,.86); color: #dfe9ec;
              border: 1px solid rgba(255,255,255,.13); border-radius: 4px;
              font-size: 11.5px; overflow: hidden;
              font-family: var(--sans); backdrop-filter: blur(6px); }
.gcm-sky-ui h4 { margin: 0; padding: 6px 10px; font-size: 11px; font-weight: 600;
                 letter-spacing: .06em; text-transform: uppercase;
                 background: rgba(255,255,255,.06); color: #9fb4bc;
                 border-bottom: 1px solid rgba(255,255,255,.1);
                 font-family: var(--mono); }
.gcm-sky-sec { padding: 6px 9px; border-bottom: 1px solid rgba(255,255,255,.07); }
.gcm-sky-sec:last-child { border-bottom: 0; }
.gcm-sky-lab { font-size: 9.5px; text-transform: uppercase; letter-spacing: .07em;
               color: #7d919a; margin-bottom: 4px; font-family: var(--mono); }
.gcm-sky-row { display: flex; gap: 4px; flex-wrap: wrap; }
.gcm-sky-btn { cursor: pointer; padding: 3px 7px; border-radius: 3px;
               border: 1px solid rgba(255,255,255,.18);
               background: rgba(255,255,255,.06); color: #cfdde2;
               font-size: 10.5px; font-family: var(--mono); }
.gcm-sky-btn:hover { background: rgba(255,255,255,.16); color: #fff; }
.gcm-sky-btn:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
.gcm-sky-btn.on { border-color: currentColor; font-weight: 600; }
.gcm-sky-count { color: #6b8089; font-size: 9.5px; margin-left: .25rem; }
.gcm-sky-empty { color: #8b6f3a; }
/* Hiding is done with this class, not the `hidden` attribute: an author
   `display` beats the UA's [hidden]{display:none} on origin, and the version of
   this panel that relied on the attribute left an overlay covering the map,
   swallowing every pointer and wheel event.  !important puts this above any
   normal author `display`; only another !important could still override it. */
.gcm-sky-off { display: none !important; }

.gcm-sky-static { position: absolute; inset: 0; width: 100%; height: 100%;
                  touch-action: none; cursor: grab;
                  -webkit-user-select: none; user-select: none; }
.gcm-sky-static.gcm-dragging { cursor: grabbing; }
.gcm-sky-nostatic { position: absolute; inset: 0; display: flex;
                    align-items: center; justify-content: center;
                    color: #9fb4bc; font-size: .85rem; }
.gcm-grid-eq { fill: none; stroke: rgba(159,180,188,.17); }
.gcm-grid-gal { fill: none; stroke: rgba(255,190,120,.32); stroke-dasharray: 6 6; }
.gcm-grid-lab { fill: #61777f; font-family: var(--mono); font-size: 10px; }
.gcm-grid-lab-gal { fill: #8d7550; }
.gcm-sky-bar { stroke: #9fb4bc; stroke-width: 1.4; fill: none; }
.gcm-sky-bar text { fill: #b7c9cf; stroke: none; font-family: var(--mono);
                    font-size: 11px; }
.gcm-lyr path { stroke-linejoin: round; }
.gcm-sky-foot { font-size: .82rem; color: var(--muted); line-height: 1.65;
                margin: .55rem 0 0; }
.gcm-sky-note { display: block; margin-top: .3rem; color: var(--warn, #b08a3c); }
.gcm-sky-btn[disabled] { opacity: .38; cursor: not-allowed; }
.gcm-sky-load { display: inline-block; cursor: pointer; margin-top: .6rem;
                padding: .35rem .8rem; border-radius: 3px;
                border: 1px solid var(--accent); color: var(--accent);
                background: none; font-family: var(--mono); font-size: .78rem; }
.gcm-sky-legend { display: grid; grid-template-columns: 15px 1fr; gap: 3px 6px;
                  align-items: center; }
.gcm-sky-sw { width: 15px; height: 3px; border-radius: 2px; }
"""


def _safe_num(value, default=None):
    """A float, or ``default``.  The sky view must never take the page down.

    ``render_page`` calls this section unguarded, so a ``footprints.json`` whose
    schema has drifted -- a null ``pa_v3``, a one-element range, a string where a
    number belongs -- would raise during formatting and lose ``monitor.html``
    entirely: every pipeline check discarded for a decorative panel.
    """
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if out == out else default          # NaN -> default


# --------------------------------------------------------------------------
# Static map: the footprints as inline SVG, with no network dependency at all.
# --------------------------------------------------------------------------

#: Width of the static map's user-coordinate system.  Coordinates are rounded
#: to one unit-decimal, which over this survey's ~1.3 deg extent is ~0.5" --
#: two orders of magnitude below a detector edge -- and roughly halves the
#: inline SVG against writing full float repr.
STATIC_W = 1000.0
#: Fraction of the field added as margin on each side.
STATIC_PAD = 0.045
#: Sample points per graticule line.  The lines are near-straight over a degree;
#: this is enough that the curvature reads as a curve and not as a polygon.
GRID_SAMPLES = 33


DEG = '°'

#: The frames the static map can be drawn in. The footprint file is always ICRS;
#: anything else is a transform away, and needs astropy.
_FRAMES = {
    'galactic': {'astropy': 'galactic', 'lon': 'l', 'lat': 'b',
                 'lon_label': 'l', 'lat_label': 'b', 'wrap': 180.0},
    'icrs': {'astropy': 'icrs', 'lon': 'ra', 'lat': 'dec',
             'lon_label': 'RA', 'lat_label': 'Dec', 'wrap': 360.0},
}

#: Draw in Galactic coordinates by default. This is a Galactic-centre survey:
#: in l/b the tiled strip runs along a grid line instead of diagonally across
#: one, and every conversation about where the survey goes is already in l/b.
DEFAULT_FRAME = 'galactic'


def _transform(pairs, src, dst):
    """``[(lon, lat)]`` from frame ``src`` to frame ``dst``, in degrees.

    Returns ``None`` -- not an empty list -- when astropy is unavailable, so a
    caller can tell "nothing to convert" from "cannot convert" and fall back to
    the untransformed frame instead of silently drawing an empty map.
    """
    if src == dst:
        return list(pairs)
    if not pairs:
        return []
    try:
        from astropy.coordinates import SkyCoord
        import astropy.units as u
    except ImportError:
        return None
    a, b = _FRAMES[src], _FRAMES[dst]
    coord = SkyCoord(**{a['lon']: [p[0] for p in pairs] * u.deg,
                        a['lat']: [p[1] for p in pairs] * u.deg},
                     frame=a['astropy']).transform_to(b['astropy'])
    lon = getattr(coord, b['lon'])
    if b['wrap'] == 180.0:
        lon = lon.wrap_at(180 * u.deg)
    return list(zip([float(v) for v in lon.deg],
                    [float(v) for v in getattr(coord, b['lat']).deg]))


def _reframe(footprints, frame):
    """``(footprints in`` ``frame``, the frame actually used)``.

    Every vertex goes through **one** vectorised transform rather than one call
    per polygon: 5000 vertices as 5000 SkyCoord round trips would dominate the
    whole page build.
    """
    if frame == 'icrs' or not isinstance(footprints, dict):
        return footprints, 'icrs'
    flat, index = [], []
    for group in ('planned', 'observed'):
        for i, pointing in enumerate(footprints.get(group) or []):
            if not isinstance(pointing, dict):
                continue
            for key in ('nircam', 'miri'):
                for j, poly in enumerate(pointing.get(key) or []):
                    for k, vertex in enumerate(poly or []):
                        if len(vertex) < 2:
                            continue
                        lon, lat = _safe_num(vertex[0]), _safe_num(vertex[1])
                        if lon is None or lat is None:
                            continue
                        index.append((group, i, key, j, k))
                        flat.append((lon, lat))
    moved = _transform(flat, 'icrs', frame)
    if moved is None:                     # no astropy: draw it in ICRS instead
        return footprints, 'icrs'
    out = dict(footprints)
    for group in ('planned', 'observed'):
        out[group] = [dict(p) if isinstance(p, dict) else p
                      for p in (footprints.get(group) or [])]
        for pointing in out[group]:
            if isinstance(pointing, dict):
                for key in ('nircam', 'miri'):
                    if pointing.get(key):
                        pointing[key] = [[list(v) for v in (poly or [])]
                                         for poly in pointing[key]]
    for (group, i, key, j, k), value in zip(index, moved):
        out[group][i][key][j][k] = [value[0], value[1]]
    return out, frame


def _circ_mean_deg(values):
    """Mean of angles in degrees, wrap-safe.

    This survey does not cross RA = 0, but a footprint file for one that did
    would otherwise centre the map 180 deg from the data and project every
    vertex onto the far side of the sky.
    """
    if not values:
        return 0.0
    rad = math.pi / 180.0
    sin_sum = sum(math.sin(v * rad) for v in values)
    cos_sum = sum(math.cos(v * rad) for v in values)
    return math.degrees(math.atan2(sin_sum, cos_sum)) % 360.0


class _TanFrame(object):
    """Gnomonic (TAN) tangent plane plus the SVG mapping onto it.

    TAN because it is the projection the mosaics themselves carry, so a
    detector edge stays straight and the drawn outline is the outline the WCS
    would give.  Over a degree the choice of projection is far below the
    linewidth; what matters is that north is up and **east is left**, the
    orientation every other image of this field is shown in.
    """

    def __init__(self, ra0, dec0, xmin, xmax, ymin, ymax):
        self.ra0, self.dec0 = ra0, dec0
        span_x = max(xmax - xmin, 1e-9)
        span_y = max(ymax - ymin, 1e-9)
        pad_x, pad_y = span_x * STATIC_PAD, span_y * STATIC_PAD
        self.xmin, self.xmax = xmin - pad_x, xmax + pad_x
        self.ymin, self.ymax = ymin - pad_y, ymax + pad_y
        self.scale = STATIC_W / (self.xmax - self.xmin)      # units per degree
        self.width = STATIC_W
        self.height = round((self.ymax - self.ymin) * self.scale, 1)

    def plane(self, ra, dec):
        """Tangent-plane degrees ``(x, y)``, or ``None`` if over the horizon."""
        rad = math.pi / 180.0
        sin_d0, cos_d0 = math.sin(self.dec0 * rad), math.cos(self.dec0 * rad)
        sin_d, cos_d = math.sin(dec * rad), math.cos(dec * rad)
        dra = (ra - self.ra0) * rad
        cos_dra, sin_dra = math.cos(dra), math.sin(dra)
        denom = sin_d0 * sin_d + cos_d0 * cos_d * cos_dra
        if denom <= 0:                       # >90 deg from the tangent point
            return None
        xi = cos_d * sin_dra / denom / rad
        eta = (cos_d0 * sin_d - sin_d0 * cos_d * cos_dra) / denom / rad
        return (-xi, eta)                    # -xi: RA increases leftwards

    def xy(self, ra, dec):
        """SVG user coordinates, or ``None``."""
        pt = self.plane(ra, dec)
        if pt is None:
            return None
        return ((pt[0] - self.xmin) * self.scale,
                (self.ymax - pt[1]) * self.scale)          # y down in SVG

    def inside(self, xy, slack=0.0):
        return (xy is not None
                and -slack <= xy[0] <= self.width + slack
                and -slack <= xy[1] <= self.height + slack)


def _vertices(footprints):
    """Every (ra, dec) drawn on the static map, planned and observed."""
    out = []
    for group in ('planned', 'observed'):
        for pointing in footprints.get(group) or []:
            if not isinstance(pointing, dict):
                continue
            for key in ('nircam', 'miri'):
                for poly in pointing.get(key) or []:
                    for vertex in poly or []:
                        ra = _safe_num(vertex[0]) if len(vertex) > 1 else None
                        dec = _safe_num(vertex[1]) if len(vertex) > 1 else None
                        if ra is not None and dec is not None:
                            out.append((ra, dec))
    return out


def _frame_for(footprints):
    """A ``_TanFrame`` sized to the footprints, or ``None`` if there are none."""
    verts = _vertices(footprints)
    if len(verts) < 3:
        return None
    frame = _TanFrame(_circ_mean_deg([v[0] for v in verts]),
                      sum(v[1] for v in verts) / len(verts),
                      -1.0, 1.0, -1.0, 1.0)                 # provisional
    planar = [frame.plane(ra, dec) for ra, dec in verts]
    planar = [p for p in planar if p is not None]
    if len(planar) < 3:
        return None
    return _TanFrame(frame.ra0, frame.dec0,
                     min(p[0] for p in planar), max(p[0] for p in planar),
                     min(p[1] for p in planar), max(p[1] for p in planar))


def _num(value):
    """One decimal, without the trailing ``.0`` on whole numbers."""
    out = round(float(value), 1)
    return ('%d' % out) if out == int(out) else ('%.1f' % out)


def _polys_path(frame, polys):
    """One SVG path covering every polygon in ``polys`` (each a closed subpath)."""
    parts = []
    for poly in polys or []:
        pts = []
        for vertex in poly or []:
            if len(vertex) < 2:
                continue
            ra, dec = _safe_num(vertex[0]), _safe_num(vertex[1])
            xy = frame.xy(ra, dec) if ra is not None and dec is not None else None
            if xy is None:
                pts = []
                break
            pts.append(xy)
        if len(pts) >= 3:
            parts.append('M' + 'L'.join('%s %s' % (_num(x), _num(y))
                                        for x, y in pts) + 'Z')
    return ''.join(parts)


def _nice_step(span, target_lines=6):
    """A 1/2/5-times-power-of-ten step giving roughly ``target_lines`` lines."""
    if span <= 0:
        return 1.0
    raw = span / max(target_lines, 1)
    power = 10.0 ** math.floor(math.log10(raw))
    for mult in (1.0, 2.0, 5.0, 10.0):
        if raw <= mult * power:
            return mult * power
    return 10.0 * power


def _grid_line(frame, points):
    """A polyline path through ``points``, clipped to points on the plane."""
    drawn = [frame.xy(ra, dec) for ra, dec in points]
    drawn = [p for p in drawn if p is not None]
    if len(drawn) < 2:
        return '', None
    path = 'M' + 'L'.join('%s %s' % (_num(x), _num(y)) for x, y in drawn)
    labelled = [p for p in drawn if frame.inside(p)]
    return path, (labelled[-1] if labelled else None)


def _spread(lines, min_sep=26.0):
    """Drop labels that would land on top of one another.

    Lines of constant longitude and latitude can leave the drawn box within a
    few units of each other, and two graticules doubles the chance; the labels
    then overprint into something unreadable. Dropping one is better than
    stacking both -- the line is still there and still on a round value.
    """
    out, placed = [], []
    for path, label, text in lines:
        if label is not None:
            if any(abs(label[0] - x) < min_sep and abs(label[1] - y) < min_sep
                   for x, y in placed):
                label = None
            else:
                placed.append(label)
        out.append((path, label, text))
    return out


def _bbox_sky(frame):
    """The map's plane bounding box back on the sky, as ``(lon, lat)`` corners.

    Inverse gnomonic. The graticule needs to know what range of sky the drawn
    area covers; over a degree the corners bound it closely enough.
    """
    rad = math.pi / 180.0
    out = []
    for x, y in ((frame.xmin, frame.ymin), (frame.xmax, frame.ymin),
                 (frame.xmin, frame.ymax), (frame.xmax, frame.ymax),
                 ((frame.xmin + frame.xmax) / 2, (frame.ymin + frame.ymax) / 2)):
        xi, eta = -x * rad, y * rad
        rho = math.hypot(xi, eta)
        if rho == 0:
            out.append((frame.ra0, frame.dec0))
            continue
        c = math.atan(rho)
        lat = math.asin(math.cos(c) * math.sin(frame.dec0 * rad)
                        + eta * math.sin(c) * math.cos(frame.dec0 * rad) / rho)
        lon = frame.ra0 * rad + math.atan2(
            xi * math.sin(c),
            rho * math.cos(frame.dec0 * rad) * math.cos(c)
            - eta * math.sin(frame.dec0 * rad) * math.sin(c))
        out.append((math.degrees(lon), math.degrees(lat)))
    return out


def _graticule(frame, grid_frame, map_frame):
    """Lines of constant longitude and latitude, as ``(path, label_xy, text)``.

    ``grid_frame`` is the frame whose grid this is; ``map_frame`` is the frame
    the map itself is drawn in. When they differ the lines are generated in the
    grid's own coordinates and transformed onto the map, which is what makes the
    second graticule curve across the first instead of paralleling it.
    """
    corners = _bbox_sky(frame)
    if grid_frame != map_frame:
        corners = _transform(corners, map_frame, grid_frame)
        if corners is None:                  # no astropy: no overlay graticule
            return []
    spec = _FRAMES[grid_frame]
    lons = [c[0] for c in corners]
    if spec['wrap'] == 360.0 and max(lons) - min(lons) > 180.0:
        lons = [(v - 360.0 if v > 180.0 else v) for v in lons]
    lon_lo, lon_hi = min(lons), max(lons)
    lat_lo, lat_hi = min(c[1] for c in corners), max(c[1] for c in corners)
    lon_step = _nice_step(lon_hi - lon_lo)
    lat_step = _nice_step(lat_hi - lat_lo)

    def line(points):
        if grid_frame != map_frame:
            points = _transform(points, grid_frame, map_frame)
            if points is None:
                return '', None
        return _grid_line(frame, points)

    def span(lo, hi):
        return [lo + (hi - lo) * i / (GRID_SAMPLES - 1.0)
                for i in range(GRID_SAMPLES)]

    lines = []
    value = math.ceil(lon_lo / lon_step) * lon_step
    while value <= lon_hi:
        path, label = line([(value, v) for v in span(lat_lo, lat_hi)])
        if path:
            lines.append((path, label,
                          '%s %g' % (spec['lon_label'], round(value, 4)) + DEG))
        value += lon_step
    value = math.ceil(lat_lo / lat_step) * lat_step
    while value <= lat_hi:
        path, label = line([(v, value) for v in span(lon_lo, lon_hi)])
        if path:
            # `%+g` renders the equator as "+0", which reads as a rounded small
            # positive number rather than as the plane itself.
            shown = round(value, 4)
            text = ('%s 0' % spec['lat_label'] if shown == 0
                    else '%s %+g' % (spec['lat_label'], shown))
            lines.append((path, label, text + DEG))
        value += lat_step
    return lines


def _reframe_polys(polys, frame_name):
    """``[[(lon, lat), ...], ...]`` from ICRS into ``frame_name``.

    One vectorised transform for the whole set. Returns the input unchanged when
    the transform is unavailable, which is the same fallback ``_reframe`` makes:
    an untransformed overlay is wrong, but a *missing* one is invisible, and the
    frame fallback applies to the whole map at once.
    """
    if frame_name == 'icrs':
        return [[tuple(v[:2]) for v in (poly or []) if len(v) >= 2]
                for poly in polys]
    flat, shape = [], []
    for poly in polys:
        kept = [v for v in (poly or []) if len(v) >= 2]
        shape.append(len(kept))
        flat.extend((v[0], v[1]) for v in kept)
    moved = _transform(flat, 'icrs', frame_name)
    if moved is None:
        return [[tuple(v[:2]) for v in (poly or []) if len(v) >= 2]
                for poly in polys]
    out, at = [], 0
    for count in shape:
        out.append(moved[at:at + count])
        at += count
    return out


def _roman_polys(roman):
    """``(spring, autumn, target)`` polygon lists from the Roman GBTDS file."""
    spring, autumn, target = [], [], []
    if not isinstance(roman, dict):
        return spring, autumn, target
    tiles = roman.get('tiles')
    if isinstance(tiles, dict):
        for tile in tiles.values():
            if not isinstance(tile, dict):
                continue
            spring.extend(tile.get('spring') or [])
            autumn.extend(tile.get('autumn') or [])
    if roman.get('target_area'):
        target.append(roman['target_area'])
    return spring, autumn, target


def _plain_layer(frame, polys, color, fill_opacity, layer_id, label, width=1.2):
    """A single group of polygons with no per-item titles."""
    path = _polys_path(frame, polys)
    return ('<g id="%s" class="gcm-lyr gcm-sky-off" data-label="%s" stroke="%s" '
            'fill="%s" fill-opacity="%s" stroke-width="%s" '
            'vector-effect="non-scaling-stroke">%s</g>'
            % (layer_id, _esc(label), color, color, fill_opacity, width,
               ('<path d="%s"/>' % path) if path else ''))


def _pointing_title(pointing):
    filters = pointing.get('filters') or {}
    bands = '/'.join(str(v) for v in (filters.get('ShortFilter'),
                                      filters.get('LongFilter')) if v)
    ra, dec = _safe_num(pointing.get('ra')), _safe_num(pointing.get('dec'))
    where = ('  %.4f %+.4f' % (ra, dec)) if ra is not None and dec is not None else ''
    return '%s%s%s' % (pointing.get('target') or ('#%s' % pointing.get('number')),
                       ('  ' + bands) if bands else '', where)


def _layer(frame, pointings, key, color, fill_opacity, layer_id, label):
    """One instrument's polygons for one group, as a titled ``<g>`` per pointing."""
    body = []
    count = 0
    for pointing in pointings:
        if not isinstance(pointing, dict):
            continue
        path = _polys_path(frame, pointing.get(key))
        if not path:
            continue
        count += 1
        body.append('<g><title>%s</title><path d="%s"/></g>'
                    % (_esc(_pointing_title(pointing)), path))
    return ('<g id="%s" class="gcm-lyr" data-label="%s" stroke="%s" fill="%s" '
            'fill-opacity="%s" stroke-width="1.1" vector-effect="non-scaling-stroke">'
            '%s</g>' % (layer_id, _esc(label), color, color,
                        fill_opacity, ''.join(body))), count


def static_map(footprints, roman=None, frame_name=DEFAULT_FRAME):
    """The footprints as a self-contained inline SVG.

    Returns ``(svg, info)``; ``svg`` is ``''`` when there is nothing drawable,
    in which case the caller should say so rather than showing an empty box.
    ``info['frame']`` is the frame actually drawn, which is ICRS rather than the
    requested one when astropy is missing.
    """
    data, frame_name = _reframe(footprints, frame_name)
    frame = _frame_for(data)
    if frame is None:
        return '', {}
    planned = [p for p in (data.get('planned') or []) if isinstance(p, dict)]
    observed = [p for p in (data.get('observed') or []) if isinstance(p, dict)]
    other = 'icrs' if frame_name == 'galactic' else 'galactic'

    grid = []
    native = _graticule(frame, frame_name, frame_name)
    overlay = _graticule(frame, other, frame_name)
    # Native labels get priority; the overlay's then have to avoid both those
    # and each other, so the two graticules cannot print over one another.
    native = _spread(native)
    overlay = _spread(native + overlay)[len(native):]
    for path, label, text in native:
        grid.append('<path class="gcm-grid-eq" d="%s"/>' % path)
        if label is not None:
            grid.append('<text class="gcm-grid-lab" x="%s" y="%s">%s</text>'
                        % (_num(label[0] + 3), _num(label[1] - 3), _esc(text)))
    gal = []
    for path, label, text in overlay:
        gal.append('<path class="gcm-grid-gal" d="%s"/>' % path)
        if label is not None:
            gal.append('<text class="gcm-grid-lab gcm-grid-lab-gal" x="%s" '
                       'y="%s">%s</text>'
                       % (_num(label[0] + 3), _num(label[1] - 3), _esc(text)))

    nircam, n_nircam = _layer(frame, planned, 'nircam', COLOR_NIRCAM_PLANNED,
                              '.09', 'stat-nircam', 'NIRCam planned')
    miri, n_miri = _layer(frame, planned, 'miri', COLOR_MIRI_PLANNED,
                          '.13', 'stat-miri', 'MIRI parallel planned')
    obs_n, n_obs_n = _layer(frame, observed, 'nircam', COLOR_OBSERVED,
                            '.18', 'stat-obs-nircam', 'observed NIRCam')
    obs_m, n_obs_m = _layer(frame, observed, 'miri', COLOR_OBSERVED,
                            '.18', 'stat-obs-miri', 'observed MIRI')

    # Roman is drawn here too, not only in the interactive view. It used to be
    # Aladin-only, which made its three toggles do nothing at all until someone
    # loaded 1.8 MB of script -- they looked like live buttons and were not.
    spring_p, autumn_p, target_p = _roman_polys(roman)
    n_roman = len(spring_p) + len(autumn_p) + len(target_p)
    reframed = _reframe_polys(spring_p + autumn_p + target_p, frame_name)
    spring = _plain_layer(frame, reframed[:len(spring_p)], COLOR_SPRING, '.05',
                          'stat-spring', 'Roman GBTDS spring')
    autumn = _plain_layer(frame, reframed[len(spring_p):len(spring_p) + len(autumn_p)],
                          COLOR_AUTUMN, '.05', 'stat-autumn', 'Roman GBTDS autumn')
    target = _plain_layer(frame, reframed[len(spring_p) + len(autumn_p):],
                          COLOR_TARGET_AREA, '0', 'stat-target',
                          'JWST target area', width=2.0)

    # The HUD is rendered at zoom 1 here so it exists without JavaScript, and
    # rewritten by the viewer's zoom handler: the bar has to stay a real angular
    # length (it grows on screen as you zoom in) while its label and the compass
    # have to stay a constant *screen* size, which no single SVG unit does.
    bar = frame.scale / 60.0                      # one arcminute, user units
    bar_y = frame.height - 26
    hud = (
        '<g id="gcm-sky-hud" class="gcm-sky-bar">'
        '<path id="gcm-sky-barline" d="M28 %s h%s" '
        'vector-effect="non-scaling-stroke"/>'
        '<text id="gcm-sky-barlab" x="%s" y="%s">1′</text>'
        '<g id="gcm-sky-compass">'
        '<path d="M52 62 V26 M52 62 H16" vector-effect="non-scaling-stroke"/>'
        '<text x="52" y="20" text-anchor="middle">%s</text>'
        '<text x="10" y="66" text-anchor="end">%s</text></g></g>'
        % (_num(bar_y), _num(bar), _num(28), _num(bar_y - 7),
           # The axes are whatever the map's frame is, so the compass must say
           # so: labelling a Galactic map N/E would name the wrong directions.
           _esc(_FRAMES[frame_name]['lat_label'] + '+'),
           _esc(_FRAMES[frame_name]['lon_label'] + '+')))

    axes = ('Galactic longitude increasing left, latitude up'
            if frame_name == 'galactic' else 'north up, east left')
    svg = ('<svg id="gcm-sky-static" class="gcm-sky-static" '
           'viewBox="0 0 %s %s" preserveAspectRatio="xMidYMid meet" '
           'role="group" aria-label="Survey footprints on the sky, %s">'
           '<g id="gcm-sky-pan">%s%s%s%s%s%s%s%s%s</g>%s</svg>'
           % (_num(frame.width), _num(frame.height), _esc(axes),
              ''.join(grid), ''.join(gal), spring, autumn, target,
              nircam, miri, obs_n, obs_m, hud))
    info = {'n_nircam': n_nircam, 'n_miri': n_miri,
            'n_observed': n_obs_n + n_obs_m, 'n_roman': n_roman,
            'width': frame.width, 'height': frame.height,
            'scale': frame.scale,          # SVG user units per degree
            'frame': frame_name, 'axes': axes,
            'overlay_grid': bool(gal)}
    return svg, info


def section(footprints, roman=None, aladin_src=ALADIN_LOCAL,
            data_url=FOOTPRINTS_JSON, roman_url=ROMAN_JSON,
            frame_name=DEFAULT_FRAME):
    """The whole sky-view section, or a note when there is no footprint data."""
    if not isinstance(footprints, dict) or not footprints:
        return ('<section class="gcm-sec" id="skyview"><h2>Sky view</h2>'
                '<p class="gcm-empty">No footprint data — run '
                '<code>scripts/monitoring/build_footprints.py</code> to generate '
                '<code>footprints.json</code>.</p></section>')

    n_planned = _safe_num(footprints.get('n_planned'), 0) or 0
    n_observed = _safe_num(footprints.get('n_observed'), 0) or 0
    n_planned, n_observed = int(n_planned), int(n_observed)
    rng = footprints.get('pa_v3_range') or []
    lo = _safe_num(rng[0]) if len(rng) > 0 else None
    hi = _safe_num(rng[1]) if len(rng) > 1 else None
    pa = _safe_num(footprints.get('pa_v3'))
    pa_note = ((f'PA_V3 {pa:.0f}°' if pa is not None else 'PA_V3 unrecorded')
               + (f' (program allows {lo:.0f}–{hi:.0f}°)'
                  if lo is not None and hi is not None else ''))
    dither = footprints.get('dither') or {}
    dither_note = ''
    if dither.get('PrimaryDitherType'):
        dither_note = (
            f" Each pointing dithers "
            f"({'/'.join(dither.get('PrimaryDitherType', []))}"
            f"{', ' + '/'.join(dither.get('PrimaryDithers', [])) if dither.get('PrimaryDithers') else ''})"
            f" and only the nominal position is drawn — the real covered area is"
            f" a little larger, and a FULLBOX pattern exists precisely to fill"
            f" the inter-module gap the outline shows as a hole.")

    observed_cls = 'gcm-sky-empty' if not n_observed else ''

    def _survey_button(index, name, url, note):
        title = f' title="{_esc(note)}"' if note else ''
        return (f'<button class="gcm-sky-btn survey'
                f'{" on" if index == 0 else ""}"{title} '
                f'data-survey="{_esc(url)}">{_esc(name)}</button>')

    surveys = ''.join(_survey_button(i, *entry)
                      for i, entry in enumerate(SURVEYS))

    static_svg, static_info = static_map(footprints, roman, frame_name)
    if not static_svg:
        static_svg = ('<div class="gcm-sky-nostatic">Footprint data contains no '
                      'drawable polygons.</div>')
    view_w = _safe_num(static_info.get('width'), 1000.0)
    view_h = _safe_num(static_info.get('height'), 1000.0)
    frame_scale = _safe_num(static_info.get('scale'), 0.0)
    drawn_frame = static_info.get('frame', 'icrs')
    spec = _FRAMES[drawn_frame]
    other = _FRAMES['icrs' if drawn_frame == 'galactic' else 'galactic']
    axes_note = ('Galactic coordinates, <em>l</em> increasing left'
                 if drawn_frame == 'galactic' else 'North up, east left')
    grid_note = ('grid %s/%s solid%s'
                 % (spec['lon_label'], spec['lat_label'],
                    (', %s/%s dashed' % (other['lon_label'], other['lat_label']))
                    if static_info.get('overlay_grid') else ''))
    spring_p, autumn_p, _target_p = _roman_polys(roman)
    n_spring, n_autumn = len(spring_p), len(autumn_p)
    # Enough zoom-out to reach the Roman tiles, which sit outside the JWST
    # bounding box that sets the home view; a little more when they are absent
    # would only show empty sky.
    zoom_out_max = 8 if static_info.get('n_roman') else 3

    return f"""
<section class="gcm-sec gcm-sky" id="skyview"><h2>Sky view — survey footprints</h2>
<p class="gcm-note">Program {_esc(footprints.get('program'))},
<em>{_esc(footprints.get('title'))}</em>: {n_planned} planned pointings, NIRCam
prime with MIRI as a coordinated parallel. The MIRI parallel sits ~7.5′ from the
prime and so covers <em>different sky</em>, which is why it is drawn separately.
{_esc(pa_note)} — not fixed until each visit is scheduled, and the program may yet
request the 180° flip, so these positions are <em>indicative</em>: across the
allowed range alone the MIRI parallel moves ~125″, about the width of a NIRCam
module, and a flip moves it ~15′. The NIRCam layer draws the eight short-wave
detectors; the long-wave arrays cover the same two modules without the
intra-module gaps.{dither_note} <strong>{n_observed}</strong> pointings observed
so far, from the APT visit status.</p>

<div class="gcm-sky-wrap">
  <div id="gcm-aladin" class="gcm-sky-off"></div>
  {static_svg}

  <div class="gcm-sky-ui" id="gcm-sky-ui">
    <h4>Footprints</h4>

    <div class="gcm-sky-sec">
      <div class="gcm-sky-lab">JWST — planned</div>
      <div class="gcm-sky-row">
        <button class="gcm-sky-btn on" id="lyr-nircam"
                style="color:{COLOR_NIRCAM_PLANNED}">NIRCam
          <span class="gcm-sky-count">{n_planned}</span></button>
        <button class="gcm-sky-btn on" id="lyr-miri"
                style="color:{COLOR_MIRI_PLANNED}">MIRI ∥
          <span class="gcm-sky-count">{n_planned}</span></button>
      </div>
    </div>

    <div class="gcm-sky-sec">
      <div class="gcm-sky-lab">JWST — observed</div>
      <div class="gcm-sky-row">
        <button class="gcm-sky-btn on" id="lyr-observed"
                style="color:{COLOR_OBSERVED}">observed
          <span class="gcm-sky-count {observed_cls}">{n_observed or 'none yet'}</span></button>
      </div>
    </div>

    <div class="gcm-sky-sec">
      <div class="gcm-sky-lab">Roman GBTDS (context)</div>
      <div class="gcm-sky-row">
        <button class="gcm-sky-btn" id="lyr-spring"
                style="color:{COLOR_SPRING}">spring
          <span class="gcm-sky-count">{n_spring}</span></button>
        <button class="gcm-sky-btn" id="lyr-autumn"
                style="color:{COLOR_AUTUMN}">autumn
          <span class="gcm-sky-count">{n_autumn}</span></button>
        <button class="gcm-sky-btn" id="lyr-target"
                style="color:{COLOR_TARGET_AREA}">target area</button>
      </div>
    </div>

    <div class="gcm-sky-sec">
      <div class="gcm-sky-lab">Background</div>
      <div class="gcm-sky-row">{surveys}</div>
    </div>

    <div class="gcm-sky-sec">
      <div class="gcm-sky-lab">Legend</div>
      <div class="gcm-sky-legend">
        <div class="gcm-sky-sw" style="background:{COLOR_NIRCAM_PLANNED}"></div>
        <div>NIRCam planned</div>
        <div class="gcm-sky-sw" style="background:{COLOR_MIRI_PLANNED}"></div>
        <div>MIRI parallel planned</div>
        <div class="gcm-sky-sw" style="background:{COLOR_OBSERVED}"></div>
        <div>observed</div>
      </div>
    </div>
  </div>
</div>

<p class="gcm-sky-foot">{axes_note}; {grid_note}. Hover a pointing for its
target and filters. Drag to pan; click the map, then scroll to zoom. The
interactive view adds sky imagery you can pan across.
<button class="gcm-sky-load" id="gcm-sky-reset" type="button">reset view</button>
<button class="gcm-sky-load" id="gcm-sky-load" type="button">load interactive view</button>
<span class="gcm-sky-note" id="gcm-sky-note"></span></p>
</section>

<script>
(function () {{
  var DATA_URL = {json.dumps(data_url)};
  var ALADIN_SRC = {json.dumps(aladin_src)};
  var ROMAN_URL = {json.dumps(roman_url)};
  var ROMAN = {{}};
  var VIEW = {json.dumps([view_w, view_h])};
  var UNITS_PER_DEG = {frame_scale:.6f};
  var C = {json.dumps({'nircam': COLOR_NIRCAM_PLANNED, 'miri': COLOR_MIRI_PLANNED,
                       'observed': COLOR_OBSERVED, 'spring': COLOR_SPRING,
                       'autumn': COLOR_AUTUMN, 'target': COLOR_TARGET_AREA})};

  // Which static <g> each toggle owns.  `observed` owns two, because the
  // observed layer draws both instruments in one colour.
  var STATIC_GROUPS = {{
    nircam: ['stat-nircam'], miri: ['stat-miri'],
    observed: ['stat-obs-nircam', 'stat-obs-miri'],
    spring: ['stat-spring'], autumn: ['stat-autumn'],
    target: ['stat-target']
  }};

  var svg = document.getElementById('gcm-sky-static');
  var note = document.getElementById('gcm-sky-note');
  var loadBtn = document.getElementById('gcm-sky-load');
  var resetBtn = document.getElementById('gcm-sky-reset');
  var aladinDiv = document.getElementById('gcm-aladin');

  // Layer state is shared: a toggle drives the static groups now and the Aladin
  // overlays later, so the view you built survives the upgrade.
  var on = {{ nircam: true, miri: true, observed: true,
             spring: false, autumn: false, target: false }};
  var L = null;                      // Aladin overlays, once they exist

  function applyLayers() {{
    Object.keys(STATIC_GROUPS).forEach(function (k) {{
      STATIC_GROUPS[k].forEach(function (id) {{
        var g = document.getElementById(id);
        if (g) {{ g.classList.toggle('gcm-sky-off', !on[k]); }}
      }});
    }});
    if (L) {{
      Object.keys(L).forEach(function (k) {{ on[k] ? L[k].show() : L[k].hide(); }});
    }}
  }}

  [['lyr-nircam', 'nircam'], ['lyr-miri', 'miri'], ['lyr-observed', 'observed'],
   ['lyr-spring', 'spring'], ['lyr-autumn', 'autumn'], ['lyr-target', 'target']
  ].forEach(function (pair) {{
    var el = document.getElementById(pair[0]);
    if (!el) {{ return; }}
    el.classList.toggle('on', on[pair[1]]);
    el.addEventListener('click', function () {{
      on[pair[1]] = !on[pair[1]];
      el.classList.toggle('on', on[pair[1]]);
      applyLayers();
    }});
  }});

  // ---------------------------------------------------------------- static
  // Pan and zoom by rewriting the viewBox.  Everything scales together, which
  // keeps the angular scale bar truthful; the HUD text and the compass are
  // counter-scaled below so they stay a constant size on screen.
  var HOME = [0, 0, VIEW[0], VIEW[1]];
  // Roman's tiles lie outside the JWST bounding box that sets the home
  // view, so enabling that layer has to be followed by zooming out far
  // enough to reach it.
  var ZOOM_OUT_MAX = {zoom_out_max};
  var vb = HOME.slice();
  var armed = false;                 // wheel zooms only after a click on the map
  var BARS = [[600, '10′'], [300, '5′'], [120, '2′'], [60, '1′'],
              [30, '30″'], [10, '10″'], [5, '5″'], [2, '2″'], [1, '1″']];

  function drawHUD() {{
    var hud = document.getElementById('gcm-sky-hud');
    var line = document.getElementById('gcm-sky-barline');
    var lab = document.getElementById('gcm-sky-barlab');
    var compass = document.getElementById('gcm-sky-compass');
    if (!hud || !line || !lab) {{ return; }}
    var k = vb[2] / HOME[2];                       // user units per screen unit
    var pick = BARS[BARS.length - 1];
    for (var i = 0; i < BARS.length; i++) {{
      if (UNITS_PER_DEG * BARS[i][0] / 3600.0 < vb[2] * 0.32) {{ pick = BARS[i]; break; }}
    }}
    var len = UNITS_PER_DEG * pick[0] / 3600.0;
    var x0 = vb[0] + vb[2] * 0.035;
    var y0 = vb[1] + vb[3] * 0.955;
    line.setAttribute('d', 'M' + x0 + ' ' + y0 + ' h' + len);
    lab.setAttribute('x', x0);
    lab.setAttribute('y', y0 - 7 * k);
    lab.setAttribute('font-size', 11 * k);
    lab.textContent = pick[1];
    if (compass) {{
      compass.setAttribute('transform',
        'translate(' + (vb[0] + vb[2] * 0.03) + ',' + (vb[1] + vb[3] * 0.02) +
        ') scale(' + k + ')');
    }}
    var labels = svg ? svg.querySelectorAll('.gcm-grid-lab') : [];
    for (var j = 0; j < labels.length; j++) {{
      labels[j].setAttribute('font-size', 10 * k);
    }}
  }}

  function setVB() {{
    if (!svg) {{ return; }}
    svg.setAttribute('viewBox', vb.join(' '));
    drawHUD();
  }}

  function viewScale() {{
    // preserveAspectRatio="meet" letterboxes, so the on-screen scale is the
    // smaller of the two ratios -- using the wrong one makes cursor-anchored
    // zoom drift away from the pointer.
    var r = svg.getBoundingClientRect();
    return Math.min(r.width / vb[2], r.height / vb[3]);
  }}

  function toUser(ev) {{
    var r = svg.getBoundingClientRect();
    var s = viewScale();
    return [vb[0] + (ev.clientX - r.left - (r.width - vb[2] * s) / 2) / s,
            vb[1] + (ev.clientY - r.top - (r.height - vb[3] * s) / 2) / s];
  }}

  if (svg) {{
    svg.addEventListener('pointerdown', function (ev) {{
      armed = true;
      var start = toUser(ev);
      var moved = false;
      svg.setPointerCapture(ev.pointerId);
      svg.classList.add('gcm-dragging');
      function move(e2) {{
        var now = toUser(e2);
        vb[0] -= now[0] - start[0];
        vb[1] -= now[1] - start[1];
        moved = true;
        setVB();
      }}
      function up() {{
        svg.removeEventListener('pointermove', move);
        svg.removeEventListener('pointerup', up);
        svg.classList.remove('gcm-dragging');
        if (!moved) {{ return; }}
      }}
      svg.addEventListener('pointermove', move);
      svg.addEventListener('pointerup', up);
    }});
    svg.addEventListener('mouseleave', function () {{ armed = false; }});
    svg.addEventListener('wheel', function (ev) {{
      // Only after the map has been clicked: a page-long report that eats the
      // scroll wheel whenever the cursor crosses a figure is worse than a map
      // that needs one click first.
      if (!armed) {{ return; }}
      ev.preventDefault();
      var at = toUser(ev);
      var f = ev.deltaY > 0 ? 1.15 : 1 / 1.15;
      var w = Math.min(Math.max(vb[2] * f, HOME[2] / 60), HOME[2] * ZOOM_OUT_MAX);
      var h = w * HOME[3] / HOME[2];
      vb[0] = at[0] - (at[0] - vb[0]) * (w / vb[2]);
      vb[1] = at[1] - (at[1] - vb[1]) * (h / vb[3]);
      vb[2] = w;
      vb[3] = h;
      setVB();
    }}, {{ passive: false }});
    drawHUD();
  }}

  if (resetBtn) {{
    resetBtn.addEventListener('click', function () {{
      vb = HOME.slice();
      setVB();
    }});
  }}

  // ----------------------------------------------------------- interactive
  function describe(e) {{
    // aladin.js throws a bare string ('WebGL2 not supported by your browser'),
    // so e.message is undefined for the most likely real failure.
    if (!e) {{ return 'unknown error'; }}
    return (typeof e === 'string') ? e : (e.message || String(e));
  }}

  function restore() {{
    // Put the map back and take Aladin's container away.  Called from BOTH
    // failure paths: a synchronous throw inside start() -- `A` undefined, or
    // `A.init` not a thenable because the script was truncated -- happens while
    // evaluating `A.init.then` and so escapes past the inner handler to the
    // outer one, which used to report a fetch failure over a black box.
    if (aladinDiv) {{ aladinDiv.classList.add('gcm-sky-off'); }}
    if (svg) {{ svg.classList.remove('gcm-sky-off'); }}
  }}

  function fail(what) {{
    // The map stays up, so this is a note beside it, not an overlay across it:
    // the reader keeps a working footprint map either way.
    if (note) {{
      note.textContent = 'Sky imagery could not be loaded (' + what +
        '). The footprint map above is unaffected.';
    }}
    loading = false;
    if (loadBtn) {{ loadBtn.disabled = false; loadBtn.textContent = 'retry'; }}
  }}

  var aladin = null;          // the live view, once there is one
  var wanted = null;          // a background asked for before it existed
  var loading = false;

  function loadInteractive() {{
    if (aladin || loading) {{ return; }}
    loading = true;
    if (loadBtn) {{
      loadBtn.disabled = true;
      loadBtn.textContent = 'loading…';
    }}
    if (note) {{ note.textContent = ''; }}
    var s = document.createElement('script');
    s.src = ALADIN_SRC;
    s.onerror = function () {{ fail('the viewer could not be loaded'); }};
    s.onload = function () {{
      fetch(DATA_URL).then(function (r) {{
        if (!r.ok) {{ throw new Error('http ' + r.status); }}
        return r.json();
      }}).then(function (fp) {{
        // Roman is context and every one of its layers is off by default, so a
        // failure to load it must not stop the JWST view from rendering.
        return fetch(ROMAN_URL)
          .then(function (r) {{ return r.ok ? r.json() : {{}}; }})
          .catch(function () {{ return {{}}; }})
          .then(function (rm) {{ ROMAN = rm || {{}}; return fp; }});
      }}).then(start).catch(function (e) {{
        restore();
        fail(describe(e));
      }});
    }};
    document.body.appendChild(s);
  }}

  if (loadBtn) {{ loadBtn.addEventListener('click', loadInteractive); }}

  // Background buttons are live from the start.  They used to be disabled until
  // the interactive view existed, which meant the row a reader sees first was a
  // row of dead controls -- and the thing they select is exactly the reason to
  // load that view, so asking for one is a perfectly good way to ask for it.
  function applySurvey(target) {{
    if (!aladin) {{
      wanted = target;
      loadInteractive();
      return;
    }}
    aladin.setImageSurvey(target.indexOf('http') === 0 ? A.HiPS(target) : target);
  }}

  document.querySelectorAll('#gcm-sky-ui button.survey').forEach(function (b) {{
    b.addEventListener('click', function () {{
      document.querySelectorAll('#gcm-sky-ui button.survey')
        .forEach(function (o) {{ o.classList.remove('on'); }});
      b.classList.add('on');
      applySurvey(b.dataset.survey);
    }});
  }});

  function start(fp) {{
    // Aladin measures its container at init, so it has to be visible first --
    // initialising into a display:none div yields a 0x0 canvas.  The map stays
    // up underneath and is hidden only once there is a working view to replace
    // it with, so no failure between here and there can empty the panel.
    if (aladinDiv) {{ aladinDiv.classList.remove('gcm-sky-off'); }}
    return A.init.then(function () {{
      // Open on whichever background was asked for, so a click on '2MASS' that
      // triggered this load arrives as 2MASS rather than as the default.
      aladin = A.aladin('#gcm-aladin', {{
        survey: wanted || {json.dumps(SURVEYS[0][1])},
        target: '0 0', fov: 1.6, cooFrame: 'galactic'
      }});
      if (wanted && wanted.indexOf('http') === 0) {{
        // A HiPS given by URL has to be built, not named.
        aladin.setImageSurvey(A.HiPS(wanted));
      }}
      wanted = null;
      if (svg) {{ svg.classList.add('gcm-sky-off'); }}
      if (loadBtn) {{
        loadBtn.textContent = 'interactive view loaded';
        loadBtn.disabled = true;
      }}
      // Pan/zoom belongs to Aladin now; the static map's reset does nothing.
      if (resetBtn) {{ resetBtn.disabled = true; }}

      function layer(name, color, width) {{
        var ov = A.graphicOverlay({{ color: color, lineWidth: width || 1.2,
                                     name: name }});
        aladin.addOverlay(ov);
        return ov;
      }}

      L = {{
        nircam: layer('JWST NIRCam (planned)', C.nircam, 1.1),
        miri: layer('JWST MIRI parallel (planned)', C.miri, 1.1),
        observed: layer('JWST observed', C.observed, 2.0),
        spring: layer('Roman GBTDS spring', C.spring, 1.2),
        autumn: layer('Roman GBTDS autumn', C.autumn, 1.2),
        target: layer('JWST target area', C.target, 2.0)
      }};

      (fp.planned || []).forEach(function (p) {{
        (p.nircam || []).forEach(function (poly) {{ L.nircam.add(A.polygon(poly)); }});
        (p.miri || []).forEach(function (poly) {{ L.miri.add(A.polygon(poly)); }});
      }});
      (fp.observed || []).forEach(function (p) {{
        (p.nircam || []).forEach(function (poly) {{ L.observed.add(A.polygon(poly)); }});
        (p.miri || []).forEach(function (poly) {{ L.observed.add(A.polygon(poly)); }});
      }});
      Object.keys(ROMAN.tiles || {{}}).forEach(function (name) {{
        var t = ROMAN.tiles[name];
        (t.spring || []).forEach(function (poly) {{ L.spring.add(A.polygon(poly)); }});
        (t.autumn || []).forEach(function (poly) {{ L.autumn.add(A.polygon(poly)); }});
      }});
      if (ROMAN.target_area) {{ L.target.add(A.polygon(ROMAN.target_area)); }}

      applyLayers();

    }}).catch(function (e) {{
      restore();
      fail(describe(e));
    }});
  }}
}})();
</script>"""
