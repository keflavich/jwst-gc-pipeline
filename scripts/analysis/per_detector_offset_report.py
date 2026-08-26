#!/usr/bin/env python
"""Per-detector offsets from the module mean, across every field and epoch.

WHY THIS EXISTS
---------------
The m2 retie loop measures each exposure's offset from its visit consensus
PER DETECTOR, but the offsets table stores one row PER MODULE, so the
per-detector component has to be pooled away before anything is written.
Issues #340/#342 proposed extending the table to per-detector rows to stop
discarding it.

That is only justified if the per-detector term is a real, static, uncorrected
distortion -- the detectors are mechanically locked and are not moving at mas
level years into the mission, so a term that varies observation to observation
is noise, and applying it would inject that noise into the astrometry.

This script is the standing test of that, and it is what makes the decision
reviewable rather than argued: run it, read the last column.

METHOD
------
For each exposure entry in every ``checkpoint_m2_*.json``:

  1. take the measured offset from the visit consensus (on-sky mas);
  2. within one (field, filter, visit, exposure, vgroup, module) group -- the
     detectors of one module in one exposure -- subtract the group MEAN.  That
     removes the common-mode per-exposure term (guide-star jitter etc., which
     the module-locked table CAN express) and leaves only detector-to-detector
     structure;
  3. aggregate per detector, per field, sigma-clipped (the bulk-repair epochs
     put arcsecond outliers in the same array as a mas-scale term);
  4. compare the between-FIELD scatter of a detector's mean against the mean
     itself.  A static distortion term is the same in every field; a term whose
     between-field scatter exceeds its mean is not static.

The MEAN is the estimator, not the median: with n=4 detectors ``np.median``
averages the middle two and discards the extremes, which is ~2x the variance of
the mean and cannot respond to one detector being genuinely offset.

THE ROLL DEGENERACY, AND WHY STEP 4 IS NOT SUFFICIENT ON ITS OWN
----------------------------------------------------------------
Step 4 is run on ON-SKY dRA/dDec.  A detector's placement error is fixed in the
INSTRUMENT frame, so on sky it is rotated by the observation's roll -- and this
archive's fields do not share a roll.  Measured from the ``PA_V3`` of one
``_crf`` per band:

    arches 88.2   sgrb2 88.7-90.0   gc2211_o049 88.1   brick 89.1 AND 275.5
    sgrc 91.4     w51 105.0-247.1   wd2 140.8   m92 170.0   ngc6397 267.3
    cloudef 272.2 gc2211_o046 272.7 gc2211_o050 279.6   wd1 284.8   m4 290.1

Two clusters ~180 deg apart.  A term that is perfectly static in the instrument
frame therefore MUST reverse sign on sky between a PA~89 field and a PA~275
one, which is exactly the observation ("each changes sign between fields") that
the report uses to conclude the term is not static.  Read on sky alone, the
test cannot separate "not static" from "static, and you rotated the telescope".

So the report runs the SAME test a second time on de-rotated deviations
(``derotate``), and a third time with the rolls SHUFFLED between bands
(``shuffle_rolls``) as the control that says whether any improvement is the
rotation or is just axis mixing -- de-rotating by ~90 deg swaps the two axes,
and averaging a noisy axis into a quiet one shrinks the scatter of both
whatever the angle was.  A real instrument-frame term shows up in the de-rotated
column and NOT in the shuffled one.  Measured on the archive as it stands
(2026-08-25), it shows up in neither: the shuffled control reproduces the whole
of the de-rotated scatter reduction, so the "not static" verdict survives the
one objection the on-sky test could not answer.
"""
import argparse
import collections
import glob
import json
import math
import os

import numpy as np

MODULE_LEVEL = ('nrca', 'nrcb', 'nrcalong', 'nrcblong')


def _robust(a, nsigma=4.0, niter=5, min_keep=8):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) < 2:
        return float('nan'), float('nan'), len(a)
    for _ in range(niter):
        med = np.median(a)
        mad = 1.4826 * np.median(np.abs(a - med))
        if mad == 0:
            # More than half the sample is identical.  Bailing out here leaves
            # the outliers in and lets one arcsecond-scale bulk-repair value
            # carry the mean (a 61-point test array went to 148 mas); fall back
            # to the std, which is inflated by exactly the points to be cut.
            sd = a.std()
            if sd == 0:
                break
            keep = np.abs(a - med) < nsigma * sd
        else:
            keep = np.abs(a - med) < nsigma * mad
        if keep.sum() < min_keep or keep.all():
            break
        a = a[keep]
    return float(a.mean()), float(a.std(ddof=1) / math.sqrt(len(a))), len(a)


def collect(root='/orange/adamginsburg/jwst'):
    """(field, filter, visit, exposure, vgroup, module, detector) -> offsets."""
    rows = []
    for path in sorted(glob.glob(f'{root}/*/astrometry_checkpoints/checkpoint_m2_*.json')):
        if path.endswith('_latest.json'):
            continue                       # a duplicate of a timestamped record
        field = path.split(f'{root}/')[1].split('/')[0]
        try:
            with open(path) as fh:
                rec = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        if rec.get('stage') != 'm2':
            continue
        date = str(rec.get('date', ''))[:19]
        for visit in rec.get('visits') or []:
            for exp in visit.get('exposures') or []:
                key = exp.get('key')
                if not key or len(key) < 4 or not exp.get('ok'):
                    continue
                if exp.get('dra') is None or exp.get('ddec') is None:
                    continue
                det = str(key[2])
                if not det.startswith('nrc') or det in MODULE_LEVEL:
                    continue
                rows.append(dict(
                    field=field, date=date, filt=str(key[3]), visit=str(key[0]),
                    exp=key[1], vgroup=(str(key[4]) if len(key) > 4 else ''),
                    module=det[:4] + ('long' if det.endswith('long') else ''),
                    det=det, dra=float(exp['dra']), ddec=float(exp['ddec'])))
    return rows


def roll_by_band(root='/orange/adamginsburg/jwst', per_dir=1):
    """(field, filter) -> PA_V3 in degrees, read from that band's own frames.

    ``PA_V3`` lives in the SCI header of a ``_crf``, not the primary one.  One
    frame per band is enough: within a band the spread is 0.00-0.07 deg
    everywhere it was checked.  Fields that hold two epochs at opposite rolls
    (brick, cloudc, sickle, ngc6334, w51) get a DIFFERENT value per band, which
    is the point -- pooling those two rolls into one field-level angle would
    reintroduce the degeneracy this function exists to break.
    """
    from astropy.io import fits
    out = {}
    for d in sorted(glob.glob(f'{root}/*/*/pipeline')):
        parts = d.split('/')
        field, filt = parts[-3], parts[-2]
        fns = sorted(glob.glob(os.path.join(d, 'jw*_crf.fits')))[:per_dir]
        for fn in fns:
            try:
                pa = fits.getheader(fn, 1).get('PA_V3')
            except (OSError, IndexError, KeyError):
                continue
            if pa is not None:
                out[(field, filt)] = float(pa)
                break
    return out


def derotate(rows, roll):
    """Rotate each row's on-sky (dRA, dDec) into the instrument frame.

    ``roll`` maps ``(field, filter) -> PA_V3`` (degrees); a band with no entry
    is DROPPED rather than assigned an angle, because a wrong angle is a
    rotation of a real vector into a wrong one and would be indistinguishable
    from noise in the output.

    Applied to ROWS, before ``deviations``: the rotation is linear and every
    member of a module-group shares one PA, so rotating then subtracting the
    group mean is the same operation as subtracting the group mean then
    rotating.  Doing it here keeps ``deviations`` and ``analyse`` untouched, so
    the on-sky and instrument-frame verdicts come out of the same estimator.
    """
    out = []
    for r in rows:
        pa = roll.get((r['field'], r['filt']))
        if pa is None:
            continue
        th = math.radians(pa)
        c, s = math.cos(th), math.sin(th)
        q = dict(r)
        q['dra'] = c * r['dra'] - s * r['ddec']
        q['ddec'] = s * r['dra'] + c * r['ddec']
        out.append(q)
    return out


def shuffle_rolls(roll, seed=0):
    """The control: the same angles, dealt to the wrong bands.

    De-rotating by ~90 deg exchanges the two axes, so it mixes a noisy axis
    with a quiet one and shrinks the between-field scatter of both regardless
    of whether any instrument-frame term exists.  Re-running the de-rotated
    test with the angles permuted measures exactly that artifact: a reduction
    the shuffled control reproduces is axis mixing and is not evidence.
    """
    import random
    rng = random.Random(seed)
    keys = sorted(roll)
    vals = [roll[k] for k in keys]
    rng.shuffle(vals)
    return dict(zip(keys, vals))


def deviations(rows, min_detectors=3):
    """Per-detector deviation from its module's mean, within one exposure."""
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r['field'], r['filt'], r['visit'], r['exp'], r['vgroup'],
                r['module'], r['date'])].append(r)
    dev = collections.defaultdict(list)
    ngroups = 0
    for g in groups.values():
        if len(g) < min_detectors:
            continue
        ngroups += 1
        mra = float(np.mean([x['dra'] for x in g]))
        mde = float(np.mean([x['ddec'] for x in g]))
        for x in g:
            dev[x['det']].append((x['dra'] - mra, x['ddec'] - mde, x['field']))
    return dev, ngroups


def analyse(dev, min_per_field=20, exclude_fields=()):
    """Per detector: per-field means, and the static-term test."""
    out = {}
    for det in sorted(dev):
        byfield = collections.defaultdict(list)
        for dra, ddec, field in dev[det]:
            if field in exclude_fields:
                continue
            byfield[field].append((dra, ddec))
        per = {}
        for field, vals in byfield.items():
            if len(vals) < min_per_field:
                continue
            mra, sra, _ = _robust([v[0] for v in vals])
            mde, sde, n = _robust([v[1] for v in vals])
            per[field] = dict(dra=mra, dra_sem=sra, ddec=mde, ddec_sem=sde, n=n)
        if len(per) < 3:
            continue
        v = np.array([p['ddec'] for p in per.values()])
        out[det] = dict(per_field=per, mean=float(v.mean()),
                        between_field_sd=float(v.std(ddof=1)),
                        static=bool(abs(v.mean()) > v.std(ddof=1)))
    return out


def make_figure(res, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dets = sorted(res)
    fields = sorted({f for d in dets for f in res[d]['per_field']})
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5),
                                  gridspec_kw=dict(width_ratios=[2.1, 1]))
    cmap = plt.get_cmap('tab10')
    for i, field in enumerate(fields):
        xs, ys, es = [], [], []
        for j, det in enumerate(dets):
            p = res[det]['per_field'].get(field)
            if p:
                xs.append(j + (i - len(fields) / 2) * 0.055)
                ys.append(p['ddec'])
                es.append(p['ddec_sem'])
        ax.errorbar(xs, ys, yerr=es, fmt='o', ms=4, lw=1, capsize=0,
                    color=cmap(i % 10), label=field, alpha=0.85)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(range(len(dets)))
    ax.set_xticklabels(dets)
    ax.set_ylabel(r'$\Delta\delta$ from module mean (mas)')
    ax.set_title('Per-detector Dec offset, by field\n'
                 'a static distortion term would line up horizontally')
    ax.legend(fontsize=7, ncol=2, loc='best')
    ax.grid(alpha=0.25)

    m = [res[d]['mean'] for d in dets]
    s = [res[d]['between_field_sd'] for d in dets]
    ax2.plot(s, m, 'o', color='k')
    for d, x, y in zip(dets, s, m):
        ax2.annotate(d, (x, y), fontsize=7, xytext=(4, 2),
                     textcoords='offset points')
    lim = max(max(s), max(abs(np.array(m)))) * 1.25
    ax2.plot([0, lim], [0, lim], 'r--', lw=1)
    ax2.plot([0, lim], [0, -lim], 'r--', lw=1,
             label='|mean| = between-field sd')
    ax2.set_xlim(0, lim)
    ax2.set_xlabel('between-field scatter (mas)')
    ax2.set_ylabel('mean over fields (mas)')
    ax2.set_title('Static term test\ninside the wedge = NOT static')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default='/orange/adamginsburg/jwst')
    ap.add_argument('--out', default='reports/figures/per_detector_offsets.png')
    ap.add_argument('--exclude-field', action='append', default=[],
                    help='field to drop (e.g. a known-broken epoch)')
    ap.add_argument('--skip-roll-frame', action='store_true',
                    help='skip the instrument-frame repeat of the test '
                         '(it opens one FITS header per band)')
    ap.add_argument('--control-seed', type=int, action='append', default=None,
                    help='seed(s) for the shuffled-roll control '
                         '(default 1 2 3)')
    args = ap.parse_args()
    if args.control_seed is None:
        args.control_seed = [1, 2, 3]

    rows = collect(args.root)
    dev, ngroups = deviations(rows)
    res = analyse(dev, exclude_fields=tuple(args.exclude_field))
    print(f'{len(rows)} per-detector measurements, {ngroups} module-groups, '
          f'{len({r["field"] for r in rows})} fields')
    print(f'\n{"det":9s} {"fields":>6s} {"mean dDec":>10s} {"between-field sd":>17s}  verdict')
    for det, r in res.items():
        print(f'{det:9s} {len(r["per_field"]):6d} {r["mean"]:+10.3f} '
              f'{r["between_field_sd"]:17.3f}  '
              f'{"STATIC" if r["static"] else "not static"}')
    if not any(r['static'] for r in res.values()):
        print('\nNo detector shows a static offset: for every one, the '
              'between-field scatter exceeds the mean.\n'
              'Per-detector corrections would inject field-specific noise.')

    if not args.skip_roll_frame:
        roll = roll_by_band(args.root)
        excl = tuple(args.exclude_field)
        inst = analyse(deviations(derotate(rows, roll))[0], exclude_fields=excl)
        ctrl = [analyse(deviations(derotate(rows, shuffle_rolls(roll, s)))[0],
                        exclude_fields=excl) for s in args.control_seed]
        print(f'\nROLL FRAME -- {len(roll)} (field, band) angles, '
              f'{min(roll.values()):.1f} to {max(roll.values()):.1f} deg')
        print('An instrument-frame term reverses sign on sky between the ~89 '
              'and ~275 deg\nclusters, so the on-sky column above cannot '
              'reject it.  De-rotating removes that.\nThe shuffled column is '
              'the control: it de-rotates by the WRONG angle, so any\n'
              'reduction it also produces is axis mixing rather than a '
              'recovered term.')
        print(f'\n{"det":9s} {"sd on sky":>10s} {"sd derot":>9s} '
              f'{"sd shuffled":>12s}  verdict (de-rotated)')
        for det in sorted(res):
            if det not in inst:
                continue
            sh = [c[det]['between_field_sd'] for c in ctrl if det in c]
            print(f'{det:9s} {res[det]["between_field_sd"]:10.3f} '
                  f'{inst[det]["between_field_sd"]:9.3f} '
                  f'{(float(np.mean(sh)) if sh else float("nan")):12.3f}  '
                  f'{"STATIC" if inst[det]["static"] else "not static"}')
        _sum = lambda t: sum(t[d]['between_field_sd'] for d in t)
        print(f'\nsummed between-field sd:  on sky {_sum(res):.3f}   '
              f'de-rotated {_sum(inst):.3f}   '
              f'shuffled {np.mean([_sum(c) for c in ctrl]):.3f}')
        if not any(r['static'] for r in inst.values()):
            print('No detector is static in the INSTRUMENT frame either, so '
                  'the on-sky verdict\nis not an artifact of the archive\'s '
                  'two roll clusters.')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print('\nwrote', make_figure(res, args.out))


if __name__ == '__main__':
    main()
