"""Survey measured ZEROFRAME rim R vs header R on non-wd2 NIRCam fields (read-only use of run_frames5/satrefit_core)."""
import os, sys, glob, json, pickle, time
import numpy as np
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit')
import run_frames5 as R5
import satrefit_core as C

OUT = os.path.dirname(os.path.abspath(__file__))
JW = '/orange/adamginsburg/jwst'
GBINS = [(200, 500), (500, 1000), (1000, 2000), (2000, 4000)]
FIELDS = {'brick': ['F115W', 'F200W', 'F356W', 'F444W'],
          'sgrb2': ['F150W', 'F210M', 'F360M', 'F480M'],
          'sgrc': ['F115W', 'F162M', 'F360M', 'F480M']}
SW = ['nrca1', 'nrca2', 'nrca3', 'nrca4', 'nrcb1', 'nrcb2', 'nrcb3', 'nrcb4']
LW = ['nrcalong', 'nrcblong']


def pick(field, band):
    fl = []
    d = f'{JW}/{field}/{band}/pipeline'
    for det in (LW if band[1:4] >= '250' and False else SW + LW):
        c = sorted(glob.glob(f'{d}/jw*_{det}_destreak_*_crf.fits'))
        c = [x for x in c if os.path.exists(R5.S._find_ramp_for(x) or '')]
        if c:
            fl.append((det, c[0]))
    return fl


def survey(field, band, det, fn):
    F = R5.load_frame(fn)
    out = dict(field=field, band=band, det=det, file=fn, Rhdr=float(F['Rhdr']))
    for N in (0, 25):
        r = R5.pipeline_curve_N(fn, F, N)
        out[N] = dict(R2440=float(r['R2440']), ngood=r['ngood'], rebuilt=bool(r['rebuilt']), fallback=bool(r['fallback']), satfac=float(r['satfac']))
    g0, cal, sat, edt = F['g0'], F['cal'], F['sat'], F['edt']
    dnu = (F['dq'] & 1) != 0
    far = np.isfinite(cal) & np.isfinite(g0) & ~sat & ~dnu & (cal > 0) & (g0 > 200) & (edt >= 25)
    out['far'] = {}
    for lo, hi in GBINS:
        s = far & (g0 >= lo) & (g0 < hi)
        n = int(s.sum())
        out['far'][f'{lo}-{hi}'] = (n, float(np.median(cal[s] / g0[s]) / F['Rhdr']) if n >= 50 else float('nan'))
    return out


if __name__ == '__main__':
    res = []
    for field, bands in FIELDS.items():
        for band in bands:
            lw = int(band[1:4]) >= 250
            for det, fn in pick(field, band):
                if lw != det.endswith('long'):
                    continue
                t = time.time()
                try:
                    res.append(survey(field, band, det, fn))
                except (OSError, KeyError, ValueError, TypeError, IndexError) as e:
                    print('SKIP', field, band, det, fn, type(e).__name__, e, flush=True)
                    continue
                r = res[-1]
                print('done', field, band, det, os.path.basename(fn), 'Rh=%.4g' % r['Rhdr'], 'N0/H=%.4f' % (r[0]['R2440'] / r['Rhdr']),
                      'N25/H=%.4f' % (r[25]['R2440'] / r['Rhdr']), 'ngood', r[0]['ngood'], 'reb', r[0]['rebuilt'], 'fb', r[0]['fallback'],
                      '%.0fs' % (time.time() - t), flush=True)
                pickle.dump(res, open(OUT + '/survey_fields.pkl', 'wb'))
    json.dump(res, open(OUT + '/survey_fields.json', 'w'), default=str)
    print('ALLDONE', len(res), flush=True)
