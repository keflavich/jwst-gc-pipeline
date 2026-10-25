"""Solve the in-detector rotation of NIRCam SW distortion references against
the F212N reference of the same detector, and (with --write) regenerate
``jwst_gc_pipeline/reduction/data/distortion_rotations.ecsv``.

The rotation is a property of the reference pair alone: a similarity fit of
the band's pixel -> V2/V3 map against the F212N map on a 9x9 pixel grid
(``distortion_rotation_correction.reference_rotation``).  The stored
``correction_arcsec`` is minus that rotation.

References are resolved through CRDS (``crds.getrecommendations``) for the
given context and observation date and read from the local CRDS cache.  Every
SW filter is surveyed and printed; only ``--filters`` go into the table.  See
issue #1135 for why the F212N group is taken as correct and which filters have
been validated against data.

Example::

    CRDS_MODE=local CRDS_PATH=/orange/adamginsburg/jwst/crds \\
        python scripts/analysis/solve_distortion_rotations.py --context jwst_1596.pmap --write
"""
import argparse
import os

import asdf
import crds
from astropy.table import Table

from jwst_gc_pipeline.reduction.distortion_rotation_correction import (TABLE, TABLE_COLUMNS,
                                                                       reference_rotation)

DETECTORS = ['NRCA1', 'NRCA2', 'NRCA3', 'NRCA4', 'NRCB1', 'NRCB2', 'NRCB3', 'NRCB4']
ANCHOR = ('F212N', 'CLEAR')
SURVEY = ([(f, 'CLEAR') for f in ['F070W', 'F090W', 'F115W', 'F140M', 'F150W', 'F150W2', 'F182M',
                                  'F187N', 'F200W', 'F210M', 'F212N']]
          + [('F150W2', 'F162M'), ('F150W2', 'F164N')])
# validated against wd2 per-frame catalogs (#1135); F070W/F090W/F140M are not
VALIDATED = ['F115W', 'F150W', 'F162M', 'F164N', 'F200W']


def band_label(filtername, pupil):
    return filtername if pupil == 'CLEAR' else pupil


def split_band(label):
    """'F162M' -> ('F150W2', 'F162M'); 'F150W' -> ('F150W', 'CLEAR')."""
    if label in ('F162M', 'F164N'):
        return 'F150W2', label
    return label, 'CLEAR'


def distortion_reference(detector, filtername, pupil, context, date):
    pars = {'META.EXPOSURE.TYPE': 'NRC_IMAGE', 'META.INSTRUMENT.DETECTOR': detector,
            'META.INSTRUMENT.CHANNEL': 'SHORT', 'META.INSTRUMENT.PUPIL': pupil,
            'META.INSTRUMENT.FILTER': filtername, 'META.INSTRUMENT.NAME': 'NIRCAM',
            'META.OBSERVATION.DATE': date, 'META.OBSERVATION.TIME': '00:00:00'}
    return crds.getrecommendations(pars, reftypes=['distortion'], context=context,
                                   observatory='jwst')['distortion']


def load_model(refname):
    path = crds.locate_file(refname, observatory='jwst')
    if not os.path.exists(path):
        return None
    with asdf.open(path, lazy_load=False, memmap=False) as af:
        return af.tree['model']


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--context', default='jwst_1596.pmap')
    ap.add_argument('--date', default='2023-06-01')
    ap.add_argument('--filters', nargs='+', default=VALIDATED,
                    help='band labels to tabulate (pupil name for F162M/F164N)')
    ap.add_argument('--write', action='store_true', help=f'write {TABLE}')
    ap.add_argument('--output', default=TABLE)
    args = ap.parse_args(argv)

    rows, survey = [], {}
    for det in DETECTORS:
        anchor_ref = distortion_reference(det, *ANCHOR, args.context, args.date)
        anchor = load_model(anchor_ref)
        if anchor is None:
            raise FileNotFoundError(f"{anchor_ref} ({det} {ANCHOR[0]}) is not in the CRDS cache")
        for filt, pup in SURVEY:
            ref = distortion_reference(det, filt, pup, args.context, args.date)
            model = load_model(ref)
            if model is None:
                survey[(det, filt, pup)] = (ref, None)
                continue
            rot, scale, rms = reference_rotation(model, anchor)
            survey[(det, filt, pup)] = (ref, (rot, scale, rms))
            if band_label(filt, pup) in args.filters:
                rows.append({'detector': det, 'filter': filt, 'pupil': pup,
                             'distortion_ref': ref, 'anchor_filter': ANCHOR[0],
                             'anchor_ref': anchor_ref, 'correction_arcsec': round(-rot, 3),
                             'ref_scale_ppm': round(scale, 1), 'fit_rms_mas': round(rms, 3)})

    print(f"rotation (arcsec) of each reference relative to {ANCHOR[0]}, v2+i*v3 sense "
          f"({args.context}, {args.date})")
    print('band            ' + ' '.join(f'{d:>7s}' for d in DETECTORS))
    for filt, pup in SURVEY:
        vals = [survey[(d, filt, pup)][1] for d in DETECTORS]
        print(f'{band_label(filt, pup):14s}  '
              + ' '.join('    n/a' if v is None else f'{v[0]:+7.2f}' for v in vals))
    print(f'\nscale (ppm) relative to {ANCHOR[0]}')
    for filt, pup in SURVEY:
        vals = [survey[(d, filt, pup)][1] for d in DETECTORS]
        print(f'{band_label(filt, pup):14s}  '
              + ' '.join('    n/a' if v is None else f'{v[1]:+7.1f}' for v in vals))

    missing = [b for b in args.filters
               if sum(r['filter'] == split_band(b)[0] and r['pupil'] == split_band(b)[1]
                      for r in rows) != len(DETECTORS)]
    if missing:
        raise RuntimeError(f"no complete set of {len(DETECTORS)} references for {missing}")

    tbl = Table(rows=rows, names=TABLE_COLUMNS)
    tbl.meta['comments'] = [
        'In-detector rotation correction for NIRCam SW distortion references (#1135).',
        'correction_arcsec = -(rotation of distortion_ref relative to anchor_ref), applied in',
        'V2V3 about the full-frame detector centre; positive rotates +V2 toward +V3.',
        'Rotation-only: the reference scale difference (ref_scale_ppm) is physical and is not applied.',
        f'Solved by scripts/analysis/solve_distortion_rotations.py, context {args.context}, '
        f'date {args.date}.',
        f'Bands: {" ".join(args.filters)}.',
    ]
    if args.write:
        tbl.write(args.output, format='ascii.ecsv', overwrite=True)
        print(f'\nwrote {len(tbl)} rows to {args.output}')
    else:
        print('\n(dry run; pass --write to update the table)')
        tbl.pprint(max_lines=-1, max_width=-1)


if __name__ == '__main__':
    main()
