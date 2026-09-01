#!/usr/bin/env python
"""Turn the Roman Galactic Plane Survey definition into sky polygons.

Companion to ``build_footprints.py`` (JWST) and to the Roman GBTDS geometry the
monitor already carries.  The GBTDS file is per-tile detector geometry projected
through ``pysiaf``; RGPS is nothing like that shape.  Its survey definition is a
set of **Galactic-coordinate regions** -- longitude/latitude boxes for the
wide-area and time-domain components, and pointing+radius circles for the deep
spectroscopic fields -- so this script converts regions, not apertures.

    python scripts/monitoring/build_rgps_footprints.py --out rgps.json

Source
------
``config/rgps_survey_definitions.json`` in the RGPS repository
(https://github.com/rachel3834/rgps).  The definition lists each region once per
filter that observes it; the geometry is identical across filters, so regions
are de-duplicated on ``(name, l, b)`` and the observing filters recorded
alongside.  A region with neither a box nor a pointing is skipped rather than
guessed at.

Coordinates
-----------
The definition is Galactic and everything downstream of ``skyview`` works in
ICRS (``_reframe_polys`` converts to whatever frame the map draws).  So the
conversion happens here, once, and the output is ICRS degrees -- the same
convention as ``roman_gbtds.json``.

Boxes are emitted with intermediate vertices along each edge, not just four
corners: a Galactic box spanning 117 deg of longitude is a curve in ICRS, and
four corners would draw a straight line through a region the survey does not
cover.  ``--edge-step`` controls the sampling.
"""
import argparse
import json
import math
import os
import sys

#: Where the RGPS definition lives when the repo is checked out beside this one.
DEFAULT_SPEC = os.path.expanduser(
    '/blue/adamginsburg/adamginsburg/repos/rgps/config/rgps_survey_definitions.json')

#: Survey components, in the order they are drawn.  Each becomes one toggleable
#: layer.  The keys are the component names in the definition file.
COMPONENTS = ('wide_area', 'time_domain', 'deep_spec')

#: Degrees between sampled vertices along a box edge.  2 deg keeps the 117-deg
#: Disk box to ~60 vertices per long edge, which is smooth at any zoom the page
#: offers without making the JSON large.
DEFAULT_EDGE_STEP = 2.0

#: Vertices used to draw a deep-field circle.
CIRCLE_VERTICES = 48


def _regions(spec):
    """De-duplicated regions per component: ``{component: [region, ...]}``.

    A region appears once per filter in the definition with identical geometry,
    so the filters are collected and the geometry taken once.
    """
    out = {}
    for component in COMPONENTS:
        info = spec.get(component)
        if not isinstance(info, dict):
            continue
        seen = {}
        for filtername, entries in info.items():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                lon = entry.get('l') or []
                lat = entry.get('b') or []
                pointing = entry.get('pointing') or []
                key = (entry.get('name'), tuple(lon), tuple(lat),
                       tuple(pointing))
                region = seen.setdefault(key, {
                    'name': entry.get('name'),
                    'l': list(lon), 'b': list(lat),
                    'pointing': list(pointing),
                    'filters': [],
                })
                if filtername not in region['filters']:
                    region['filters'].append(filtername)
        # Skip anything with no geometry at all rather than inventing one.
        out[component] = [r for r in seen.values()
                          if (len(r['l']) == 2 and len(r['b']) == 2)
                          or len(r['pointing']) >= 3]
    return out


def _lon_edge(l0, l1, b, step):
    """Vertices from ``(l0, b)`` to ``(l1, b)``, sampled every ``step`` deg."""
    n = max(1, int(math.ceil(abs(l1 - l0) / step)))
    return [(l0 + (l1 - l0) * (i / float(n)), b) for i in range(n + 1)]


def _lat_edge(b0, b1, l, step):
    """Vertices from ``(l, b0)`` to ``(l, b1)``, sampled every ``step`` deg."""
    n = max(1, int(math.ceil(abs(b1 - b0) / step)))
    return [(l, b0 + (b1 - b0) * (i / float(n))) for i in range(n + 1)]


def _box_galactic(lon, lat, step):
    """A Galactic box as a closed vertex ring, sampled along its edges.

    Both edge helpers return ``(l, b)`` in that order.  An earlier version had
    one function for both directions that built the latitude edge as ``(b, a)``
    and then swapped the result back, which silently emitted ``(b, l)`` for the
    two vertical edges -- the Carina boxes came out with vertices at RA 350 and
    RA 160 in the same four-sided region.  Two explicit functions cost three
    lines and cannot express that mistake.
    """
    l0, l1 = float(lon[0]), float(lon[1])
    b0, b1 = float(lat[0]), float(lat[1])
    ring = []
    ring += _lon_edge(l0, l1, b0, step)          # bottom, +l
    ring += _lat_edge(b0, b1, l1, step)[1:]      # right, +b
    ring += _lon_edge(l1, l0, b1, step)[1:]      # top, -l
    ring += _lat_edge(b1, b0, l0, step)[1:]      # left, -b
    return ring


def _circle_galactic(pointing, n=CIRCLE_VERTICES):
    """A deep field's ``[l, b, radius]`` as a ring in Galactic degrees."""
    l0, b0, radius = (float(pointing[0]), float(pointing[1]),
                      float(pointing[2]))
    ring = []
    for i in range(n):
        theta = 2.0 * math.pi * i / n
        db = radius * math.sin(theta)
        # Longitude circles converge towards the poles; at |b| < 7 deg this is a
        # sub-percent correction, but it costs nothing to be right.
        scale = math.cos(math.radians(b0 + db))
        dl = radius * math.cos(theta) / (scale if abs(scale) > 1e-6 else 1e-6)
        ring.append((l0 + dl, b0 + db))
    return ring


def _to_icrs(rings):
    """Galactic ``(l, b)`` rings to ICRS ``[ra, dec]`` rings."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    out = []
    for ring in rings:
        if not ring:
            out.append([])
            continue
        lons = [p[0] for p in ring]
        lats = [p[1] for p in ring]
        coord = SkyCoord(l=lons * u.deg, b=lats * u.deg,
                         frame='galactic').icrs
        out.append([[round(float(r), 5), round(float(d), 5)]
                    for r, d in zip(coord.ra.deg, coord.dec.deg)])
    return out


def build(spec, edge_step=DEFAULT_EDGE_STEP):
    """``{component: {name: {polys, filters, ...}}}`` in ICRS degrees."""
    regions = _regions(spec)
    out = {}
    for component, items in regions.items():
        rings, meta = [], []
        for region in sorted(items, key=lambda r: str(r.get('name'))):
            if len(region['l']) == 2 and len(region['b']) == 2:
                rings.append(_box_galactic(region['l'], region['b'], edge_step))
                shape = 'box'
            else:
                rings.append(_circle_galactic(region['pointing']))
                shape = 'circle'
            meta.append({'name': region['name'], 'shape': shape,
                         'filters': sorted(region['filters']),
                         'l': region['l'], 'b': region['b'],
                         'pointing': region['pointing']})
        icrs = _to_icrs(rings)
        out[component] = [dict(m, poly=p) for m, p in zip(meta, icrs)]
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--spec', default=DEFAULT_SPEC,
                        help='rgps_survey_definitions.json from the RGPS repo')
    parser.add_argument('--out', default='rgps.json')
    parser.add_argument('--edge-step', type=float, default=DEFAULT_EDGE_STEP,
                        help='degrees between sampled vertices along a box edge')
    args = parser.parse_args(argv)

    if not os.path.exists(args.spec):
        parser.error('RGPS survey definition not found: %s\n'
                     'Clone https://github.com/rachel3834/rgps and pass --spec.'
                     % args.spec)
    with open(args.spec) as fh:
        spec = json.load(fh)

    components = build(spec, edge_step=args.edge_step)
    payload = {
        'components': components,
        'source': ('rgps_survey_definitions.json (RGPS repo, '
                   'https://github.com/rachel3834/rgps); Galactic regions '
                   'converted to ICRS'),
        'edge_step_deg': args.edge_step,
    }
    with open(args.out, 'w') as fh:
        json.dump(payload, fh)
    total = sum(len(v) for v in components.values())
    print('%s: %d regions across %d components (%s)'
          % (args.out, total, len(components),
             ', '.join('%s=%d' % (k, len(v)) for k, v in components.items())))
    return 0


if __name__ == '__main__':
    sys.exit(main())
