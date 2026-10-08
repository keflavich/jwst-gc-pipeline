"""Task C: CRDS reference files for the SW frames and per-detector saturation/linearity/gain comparison.
Usage: nice -19 python -u refdata.py   (reads headers of the crf files, CRDS references and the rows_<band>.npz star positions)"""
import glob
import json
import os
import re
import numpy as np
from astropy.io import fits

TREE = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/tree_main2'
OUT = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/nrcb3'
CRDS = '/orange/adamginsburg/jwst/crds/references/jwst/nircam'
BANDS = ['F150W', 'F162M', 'F182M', 'F200W']
SW = ['nrca1', 'nrca2', 'nrca3', 'nrca4', 'nrcb1', 'nrcb2', 'nrcb3', 'nrcb4']
KEYS = ['R_SATURA', 'R_LINEAR', 'R_GAIN', 'R_READNO', 'R_DARK', 'R_FLAT', 'R_SUPERB', 'R_PHOTOM', 'R_DFLAT', 'R_SFLAT']


def resolve(v):
    return f"{CRDS}/{v.replace('crds://', '')}"


def main():
    hdr = {}
    for band in BANDS:
        for fn in sorted(glob.glob(f'{TREE}/{band}/pipeline/*_align_o005_crf.fits')):
            mm = re.search(r'_(nrc[ab][1-4])_align', fn)
            if not mm:
                continue
            h = fits.getheader(fn, 0)
            hdr[(band, mm.group(1), os.path.basename(fn))] = {k: h.get(k) for k in KEYS + ['CRDS_CTX']}
    # distinct references per detector
    L = ['### R_* references used by the SW crf frames (distinct per band/detector)', '', '| band | det | ' + ' | '.join(KEYS[:6]) + ' | CRDS_CTX |', '|---|---|' + '---|' * 7]
    refs = {}
    for (band, det, fn), d in sorted(hdr.items()):
        refs.setdefault((band, det), set()).add(tuple(d[k] for k in KEYS[:6]) + (d['CRDS_CTX'],))
    for (band, det), s in sorted(refs.items()):
        for t in sorted(s, key=str):
            L.append(f'| {band} | {det} | ' + ' | '.join(str(x).replace('crds://', '') for x in t) + ' |')
    # per-reference-file statistics
    files = {}
    for (band, det, fn), d in hdr.items():
        for k in ('R_SATURA', 'R_LINEAR', 'R_GAIN'):
            files.setdefault((k, det, d[k]), 0)
            files[(k, det, d[k])] += 1
    res = {}
    # star positions on each detector (satstar rows) for the core statistics
    pos = {}
    for band in BANDS:
        z = np.load(f'{OUT}/rows_{band}.npz', allow_pickle=True)
        for det in SW:
            s = z['S_det'] == det
            pos[(band, det)] = (z['S_x'][s], z['S_y'][s])
    fracs = [0.5, 0.7, 0.9]
    L += ['', '### Per detector: saturation threshold, linearity correction, gain', '',
          '| band | det | sat ref | sat med full-det (DN) | sat med at star rows (DN) | lin ref | lin corr at 50/70/90% sat, full-det median | lin corr at star rows | gain ref | gain med full-det | gain at star rows |', '|---|---|---|---|---|---|---|---|---|---|---|']
    for band in BANDS:
        for det in SW:
            d0 = [d for (b, dt, fn), d in hdr.items() if b == band and dt == det]
            if not d0:
                continue
            d0 = d0[0]
            with fits.open(resolve(d0['R_SATURA'])) as h:
                sat = np.asarray(h['SCI'].data, float)
            with fits.open(resolve(d0['R_LINEAR'])) as h:
                co = np.asarray(h['COEFFS'].data, float)
            with fits.open(resolve(d0['R_GAIN'])) as h:
                gain = np.asarray(h['SCI'].data, float)
            x, y = pos[(band, det)]
            xi = np.clip(np.rint(x).astype(int), 0, 2047)
            yi = np.clip(np.rint(y).astype(int), 0, 2047)
            satpos = sat[yi, xi] if len(xi) else np.array([np.nan])
            sat_ok = np.isfinite(sat) & (sat > 1000) & (sat < 70000)
            lin_full, lin_pos = [], []
            for f in fracs:
                # linearity: corrected = sum_k c_k x^k with x in DN; correction ratio = corrected / x at x = f * sat
                def corr(s_, c_):
                    xx = f * s_
                    cc = np.zeros_like(xx)
                    for k in range(c_.shape[0]):
                        cc = cc + c_[k] * xx ** k
                    return cc / xx
                sub = (slice(None, None, 8), slice(None, None, 8))
                ok_ = np.isfinite(sat[sub]) & (sat[sub] > 1000) & (sat[sub] < 70000)
                lin_full.append(float(np.nanmedian(corr(sat[sub][ok_], co[(slice(None),) + sub][:, ok_]))))
                if len(xi):
                    lin_pos.append(float(np.nanmedian(corr(satpos, co[:, yi, xi]))))
                else:
                    lin_pos.append(np.nan)
            gfull = float(np.nanmedian(gain))
            gpos = float(np.nanmedian(gain[yi, xi])) if len(xi) else np.nan
            res[f'{band}_{det}'] = dict(sat_full=float(np.nanmedian(sat[sat_ok])), sat_pos=float(np.nanmedian(satpos)), lin_full=lin_full, lin_pos=lin_pos,
                                        gain_full=gfull, gain_pos=gpos, refs=[d0['R_SATURA'], d0['R_LINEAR'], d0['R_GAIN']], nstar=int(len(x)))
            L.append(f"| {band} | {det} | {d0['R_SATURA'].replace('crds://', '')} | {res[f'{band}_{det}']['sat_full']:.0f} | {res[f'{band}_{det}']['sat_pos']:.0f} | "
                     f"{d0['R_LINEAR'].replace('crds://', '')} | " + '/'.join(f'{v:.4f}' for v in lin_full) + ' | ' + '/'.join(f'{v:.4f}' for v in lin_pos) +
                     f" | {d0['R_GAIN'].replace('crds://', '')} | {gfull:.3f} | {gpos:.3f} |")
            print(band, det, res[f'{band}_{det}'], flush=True)
    json.dump(res, open(f'{OUT}/refdata.json', 'w'), indent=1)
    open(f'{OUT}/refdata_tables.md', 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
