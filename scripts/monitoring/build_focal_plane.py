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
#: Delivered frames, which carry the attitude the telescope actually held.
DEFAULT_EXPOSURES = ('/orange/adamginsburg/jwst/releases/v1.8-2026.09/'
                     'gc-treasury/exposures')

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


def observed_attitude(exposures_dir):
    """``{observation: {ra, dec, pa_v3, anchor, ...}}`` as flown.

    The plan's PA is not the attitude.  10678 was planned at PA_V3 = 87.0 and
    every delivered frame reads 90.150 to 90.152 -- 3.15 degrees out, uniform
    across the survey and stable to 0.002 within an observation.  Drawn at the
    planned angle the NIRCam boxes are visibly rotated and the parallels, which
    sit 8 to 15 arcmin off the anchor, are displaced by arcminutes.

    Read per observation from extension 1 of one frame: ``RA_REF``/``DEC_REF``
    are that aperture's reference point on the sky and ``PA_V3`` the telescope
    roll, which is exactly the triple `aperture_polygons` needs, with the
    frame's own ``APERNAME`` as the anchor.  Reconstructing NRCA1_FULL and
    NRCB4_FULL this way lands within 0.2-0.3" of the same frame's GWCS corners.
    """
    from astropy.io import fits

    out = {}
    if not os.path.isdir(exposures_dir):
        return out
    seen = {}
    for root, _dirs, names in os.walk(exposures_dir):
        for name in sorted(names):
            if not name.endswith('.fits'):
                continue
            path = os.path.join(root, name)
            try:
                hdr0 = fits.getheader(path, 0)
                hdr1 = fits.getheader(path, 1)
            except (OSError, IndexError, ValueError):
                continue
            number = hdr0.get('OBSERVTN')
            aper = hdr0.get('APERNAME')
            if number is None or not aper:
                continue
            obs = f'o{int(number):03d}'
            pa, ra, dec = (hdr1.get('PA_V3'), hdr1.get('RA_REF'),
                           hdr1.get('DEC_REF'))
            if None in (pa, ra, dec):
                continue
            entry = seen.setdefault(obs, {'pa': [], 'first': None})
            entry['pa'].append(float(pa))
            if entry['first'] is None:
                entry['first'] = {'ra': float(ra), 'dec': float(dec),
                                  'pa_v3': float(pa), 'anchor': str(aper),
                                  'instrument': str(hdr0.get('INSTRUME', ''))}
    for obs, entry in seen.items():
        rec = dict(entry['first'])
        rec['n_frames'] = len(entry['pa'])
        # Reported rather than averaged away: a spread beyond a few mdeg would
        # mean the observation was not held at one attitude, and then a single
        # footprint is the wrong picture regardless of which angle it uses.
        rec['pa_v3_spread_deg'] = round(max(entry['pa']) - min(entry['pa']), 5)
        out[obs] = rec
    return out


def build(args):
    with open(args.footprints) as fh:
        data = json.load(fh)
    default_pa = data.get('pa_v3')

    flown = observed_attitude(args.exposures) if args.exposures else {}
    if flown:
        spread = max(r['pa_v3_spread_deg'] for r in flown.values())
        print(f'{len(flown)} observation(s) carry an as-flown attitude '
              f'(worst within-observation PA_V3 spread {spread:.4f} deg)')

    entries = []
    for entry in data.get('observed', []) + data.get('planned', []):
        number = str(entry.get('number', '')).strip()
        ra, dec = entry.get('ra'), entry.get('dec')
        if not number.isdigit() or ra is None or dec is None:
            continue
        obs_id = f'o{int(number):03d}'
        as_flown = flown.get(obs_id)
        anchor = args.anchor
        pa = entry.get('pa_v3', default_pa)
        if as_flown:
            # The measured attitude replaces BOTH the angle and the anchor: the
            # plan's RA/Dec is the target, the frame's RA_REF/DEC_REF is where
            # that aperture's reference point actually landed.
            ra, dec, pa = (as_flown['ra'], as_flown['dec'], as_flown['pa_v3'])
            anchor = as_flown['anchor']
        if pa is None:
            # Without an attitude there is no footprint, and a default angle
            # would draw the observatory somewhere it never pointed.
            print(f'note: o{int(number):03d} has no PA_V3; skipped')
            continue
        shapes = []
        for label, colour, apertures in INSTRUMENTS:
            try:
                polys = bf.aperture_polygons(float(ra), float(dec), float(pa),
                                             apertures, anchor)
            except (KeyError, ValueError) as err:
                print(f'note: {label} skipped for o{int(number):03d}: {err}')
                continue
            shapes.append({'label': label, 'colour': colour,
                           'polys': [polys[name] for name in apertures
                                     if name in polys]})
        entries.append({
            'id': obs_id,
            'label': entry.get('target') or obs_id,
            'ra': float(ra), 'dec': float(dec), 'pa_v3': float(pa),
            'status': entry.get('status') or '',
            'observed': entry in data.get('observed', []),
            # Which attitude drew this: the one the telescope held, or the one
            # the plan asked for. They differ by 3.15 deg for this programme,
            # so a viewer that does not say which is showing one of them as if
            # it were the other.
            'attitude': 'as-flown' if as_flown else 'planned',
            'anchor': anchor,
            'shapes': shapes,
        })
    entries.sort(key=lambda e: e['id'])
    if not entries:
        raise SystemExit(f'no pointing in {args.footprints} has a position '
                         f'and an attitude')

    out = {'programme': args.programme, 'anchor': args.anchor,
           'default_pa_v3': default_pa,
           'n_as_flown': sum(1 for e in entries if e['attitude'] == 'as-flown'),
           'instruments': [{'label': l, 'colour': c} for l, c, _ in INSTRUMENTS],
           'pointings': entries}
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(out, fh, separators=(',', ':'))
    observed = sum(1 for e in entries if e['observed'])
    print(f'{len(entries)} pointings ({observed} observed, '
          f"{out['n_as_flown']} drawn at the as-flown attitude), "
          f'{len(INSTRUMENTS)} instruments, default anchor {args.anchor}')
    print(f'wrote {args.out}')
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--footprints', default=DEFAULT_FOOTPRINTS)
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--exposures', default=DEFAULT_EXPOSURES,
                    help='delivered frames to read the as-flown attitude from; '
                         'empty string to draw every pointing at the plan PA')
    ap.add_argument('--anchor', default=bf.NIRCAM_ANCHOR,
                    help='the aperture APT points at; the attitude is built here')
    ap.add_argument('--programme', default='10678')
    return build(ap.parse_args(argv))


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    sys.exit(main())
