#!/usr/bin/env python
"""The full JWST focal plane at each Treasury observation's own attitude.

The sky viewer draws where the survey's NIRCam tiles landed. This answers a
different question: where the WHOLE observatory was pointing when it took them
-- MIRI's parallel, the FGS guiders, and NIRCam together, at that
observation's position angle.

Every aperture is projected through ONE attitude, anchored on the aperture APT
points at (`NRCALL_FULL` for this programme). That anchoring is what puts the
parallel instruments where they actually were rather than on top of the prime;
see `build_footprints.aperture_polygons`, which this reuses rather than
reimplements.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..')))

import build_footprints as bf            # noqa: E402

DEFAULT_FOOTPRINTS = '/orange/adamginsburg/jwst/monitor/footprints.json'
DEFAULT_OUT = '/orange/adamginsburg/jwst/htdocs_staging/jwst_gc_focal_plane.json'

#: One entry per instrument, drawn as its own colour. NIRCam is the eight
#: science detectors rather than `NRCALL_FULL`: the gap between modules is real
#: sky the survey does not cover, and a single outline hides it.
INSTRUMENTS = (
    ('NIRCam', '#58a6ff', ('NRCA1_FULL', 'NRCA2_FULL', 'NRCA3_FULL',
                           'NRCA4_FULL', 'NRCB1_FULL', 'NRCB2_FULL',
                           'NRCB3_FULL', 'NRCB4_FULL')),
    ('NIRCam LW', '#7fd0ff', ('NRCA5_FULL', 'NRCB5_FULL')),
    ('MIRI', '#ffb454', ('MIRIM_ILLUM',)),
    ('FGS', '#8cff8c', ('FGS1_FULL', 'FGS2_FULL')),
    ('NIRISS', '#c792ea', ('NIS_CEN',)),
)


def build(args):
    with open(args.footprints) as fh:
        data = json.load(fh)
    default_pa = data.get('pa_v3')

    entries = []
    for entry in data.get('observed', []) + data.get('planned', []):
        number = str(entry.get('number', '')).strip()
        ra, dec = entry.get('ra'), entry.get('dec')
        if not number.isdigit() or ra is None or dec is None:
            continue
        pa = entry.get('pa_v3', default_pa)
        if pa is None:
            # Without an attitude there is no footprint, and a default angle
            # would draw the observatory somewhere it never pointed.
            print(f'note: o{int(number):03d} has no PA_V3; skipped')
            continue
        shapes = []
        for label, colour, apertures in INSTRUMENTS:
            try:
                polys = bf.aperture_polygons(float(ra), float(dec), float(pa),
                                             apertures, args.anchor)
            except (KeyError, ValueError) as err:
                print(f'note: {label} skipped for o{int(number):03d}: {err}')
                continue
            shapes.append({'label': label, 'colour': colour,
                           'polys': [polys[name] for name in apertures
                                     if name in polys]})
        entries.append({
            'id': f'o{int(number):03d}',
            'label': entry.get('target') or f'o{int(number):03d}',
            'ra': float(ra), 'dec': float(dec), 'pa_v3': float(pa),
            'status': entry.get('status') or '',
            'observed': entry in data.get('observed', []),
            'shapes': shapes,
        })
    entries.sort(key=lambda e: e['id'])
    if not entries:
        raise SystemExit(f'no pointing in {args.footprints} has a position '
                         f'and an attitude')

    out = {'programme': args.programme, 'anchor': args.anchor,
           'default_pa_v3': default_pa,
           'instruments': [{'label': l, 'colour': c} for l, c, _ in INSTRUMENTS],
           'pointings': entries}
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(out, fh, separators=(',', ':'))
    observed = sum(1 for e in entries if e['observed'])
    print(f'{len(entries)} pointings ({observed} observed), '
          f'{len(INSTRUMENTS)} instruments, anchor {args.anchor}')
    print(f'wrote {args.out}')
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--footprints', default=DEFAULT_FOOTPRINTS)
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--anchor', default=bf.NIRCAM_ANCHOR,
                    help='the aperture APT points at; the attitude is built here')
    ap.add_argument('--programme', default='10678')
    return build(ap.parse_args(argv))


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    sys.exit(main())
