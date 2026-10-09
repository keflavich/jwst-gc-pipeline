"""Per-detector ZP of unsaturated main2 stars against dolphot, compared with the change of PHOTMJSR between the
2023-10 photom references (jwst_nircam_photom_0150-0159) and the ones in our crf headers (2026-03, 0166-0175).

For each band and detector: stars inside the footprint of that detector only (all 4 exposures, 10 px margin),
unsaturated and not replaced, inside the band's ZP window; median dm = ours - dolphot - ZP.
pred = -2.5 log10(PHOTMJSR_ours / PHOTMJSR_2023) is the shift of our magnitudes if dolphot carries the 2023 values.
Both columns are shown with their N-weighted mean over detectors removed, since the band ZP absorbs a common offset.
usage: python photdet.py [arm] > photdet.txt"""
import sys
import glob
import json
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an

CRDS = '/orange/adamginsburg/jwst/crds/references/jwst/nircam'
OLD = {fits.getheader(f)['DETECTOR'].lower(): f for f in (f'{CRDS}/jwst_nircam_photom_{k:04d}.fits' for k in range(150, 160))}
an.ZPWIN.update(an.zp_windows())
arm = sys.argv[1] if len(sys.argv) > 1 else 'main2'
A = an.Arm(arm)
tree = f'{an.Q}/tree_main2'
sky = A.sky[A.idx]


def photmjsr(fn, filt, pupil):
    d = fits.getdata(fn, 1)
    s = (np.char.strip(d['filter'].astype(str)) == filt) & (np.char.strip(d['pupil'].astype(str)) == pupil)
    return float(d['photmjsr'][s][0]) if s.any() else np.nan


res = {}
for band in an.BANDS:
    frames = sorted(glob.glob(f'{tree}/F{band}/pipeline/jw03523005001_*_align_o005_crf.fits'))
    if not frames:
        continue
    dets = sorted({fits.getheader(f)['DETECTOR'].lower() for f in frames})
    on, info = {}, {}
    for det in dets:
        inside = np.zeros(A.n, bool)
        for fn in frames:
            h0 = fits.getheader(fn)
            if h0['DETECTOR'].lower() != det:
                continue
            h = fits.getheader(fn, 'SCI')
            ny, nx = h['NAXIS2'], h['NAXIS1']
            x, y = WCS(h).world_to_pixel(sky)
            inside |= (x > 10) & (x < nx - 10) & (y > 10) & (y < ny - 10)
            new = float(h['PHOTMJSR'])
            info[det] = dict(new=new, old=photmjsr(OLD[det], h0['FILTER'], h0['PUPIL']), ref=h0['R_PHOTOM'].split('//')[-1])
        on[det] = inside
    nd = np.sum([on[d] for d in dets], axis=0)
    dm = A.dm(band)
    lo, hi = an.ZPWIN.get(band, (0, 19))
    ok = A.matched & np.isfinite(dm) & ~A.rep[band] & ~A.sat[band] & (A.ref[band] >= lo) & (A.ref[band] < hi)
    rows = []
    for det in dets:
        s = ok & on[det] & (nd == 1)
        if s.sum() < 10:
            continue
        med = float(np.median(dm[s]))
        err = float(1.2533 * 1.4826 * np.median(np.abs(dm[s] - med)) / np.sqrt(s.sum()))
        pred = -2.5 * np.log10(info[det]['new'] / info[det]['old'])
        rows.append(dict(det=det, n=int(s.sum()), med=med, err=err, pred=float(pred), **info[det]))
    if not rows:
        continue
    w = np.array([r['n'] for r in rows], float)
    mm = np.sum(w * [r['med'] for r in rows]) / w.sum()
    mp = np.sum(w * [r['pred'] for r in rows]) / w.sum()
    print(f'\n### F{band} (ZP {A.zp[band]:+.3f}, window {lo:.1f}-{hi:.1f}; N-weighted mean pred {mp:+.4f})')
    print('| detector | N | measured dm | measured - mean | pred | pred - mean | residual | PHOTMJSR ours / 2023 |')
    print('|---|---|---|---|---|---|---|---|')
    for r in rows:
        r['dmm'], r['dpm'] = r['med'] - mm, r['pred'] - mp
        print(f"| {r['det']} | {r['n']} | {r['med']:+.4f} ± {r['err']:.4f} | {r['dmm']:+.4f} | {r['pred']:+.4f} | {r['dpm']:+.4f} | "
              f"{r['dmm'] - r['dpm']:+.4f} | {r['new']:.4g} / {r['old']:.4g} ({r['ref']}) |")
    res[band] = dict(zp=float(A.zp[band]), mean_pred=float(mp), rows=rows)

X = np.array([r['dpm'] for b in res.values() for r in b['rows']])
Y = np.array([r['dmm'] for b in res.values() for r in b['rows']])
E = np.array([r['err'] for b in res.values() for r in b['rows']])
k = np.sum(X * Y / E ** 2) / np.sum(X ** 2 / E ** 2)
print(f'\nall bands, detectors: {len(X)} points; slope measured-mean vs pred-mean (through 0, 1/err^2) {k:.3f}; '
      f'rms measured-mean {np.std(Y):.4f}, rms after subtracting pred {np.std(Y - X):.4f}, after subtracting slope*pred {np.std(Y - k * X):.4f}')
ZP = np.array([b['zp'] for b in res.values()])
MP = np.array([b['mean_pred'] for b in res.values()])
print('band ZP vs mean pred: ' + ', '.join(f'F{b} {v["zp"]:+.3f}/{v["mean_pred"]:+.3f}' for b, v in res.items()))
json.dump(res, open(f'{an.Q}/photomver/photdet_{arm}.json', 'w'), indent=1)
