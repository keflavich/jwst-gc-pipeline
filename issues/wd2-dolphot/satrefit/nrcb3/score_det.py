"""Task D: satrefit round-3 variants split by detector (nrcb1 vs nrcb3), read-only on satrefit/.
Variant magnitudes follow score3.py, applied row by row: dm_var = dm_row_final - 2.5 log(a_cat/a_raw) - 2.5 log(a_eff/a_base)
(a_eff = a_variant, or min(a_variant, cap) for '+cap', cap = a_raw where a_raw < 0.999 a_cat).  dm_row_final comes from rows_<band>.npz.
Usage: nice -19 python -u score_det.py -> det_variants.json, det_variants_tables.md, rcurve_tables.md"""
import glob
import json
import os
import re
import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
OUT = f'{Q}/nrcb3'
BANDS = {'150W': 'F150W', '200W': 'F200W'}
BINS = {'150W': [(14, 15), (15, 16), (16, 17), (17, 18), (18, 19)], '200W': [(13, 14.5), (14.5, 15), (15, 15.5), (15.5, 16), (16, 18)]}
VARS = ['final', 'uncapped', 'base+cap', 'bgfree', 'bgfree+cap', 'rw12', 'rw12+cap', 'rw12+bgfree', 'rw12+bgfree+cap', 'bgfree+v7b+cap']
rng = np.random.default_rng(3)


def load_frames(b):
    tabs = []
    for fn3 in sorted(glob.glob(f'{Q}/satrefit/out3/{b}_*_satrefit3.fits')):
        t3 = Table.read(fn3)
        fn2 = fn3.replace('/out3/', '/out2/').replace('_satrefit3.fits', '_satrefit.fits')
        fn1 = fn3.replace('/out3/', '/out/').replace('_satrefit3.fits', '_satrefit.fits')
        t2 = Table.read(fn2)
        t1 = Table.read(fn1) if os.path.exists(fn1) else Table()
        t = Table()
        for c in ('ra', 'dec', 'label', 'a_cat', 'a_raw', 'a_base', 'a_bgfree', 'a_rw12'):
            src = [x for x in (t3, t2, t1) if c in x.colnames][0]
            t[c] = np.asarray(src[c], float)
        for c in ('a_rw12+bgfree', 'a_bgfree+v7b'):
            src = [x for x in (t3, t2, t1) if c in x.colnames]
            t[c] = np.asarray(src[0][c], float)
        src = [x for x in (t3, t2, t1) if 'pk_bgfree+v7b' in x.colnames]
        t['pk'] = np.asarray(src[0]['pk_bgfree+v7b'], float)
        m = re.search(r'_(\d{5})_(nrcb[13])_', os.path.basename(fn3))
        t['exp'] = int(m.group(1))
        t['det'] = m.group(2)
        tabs.append(t)
    return tabs


def variants(t):
    acat, araw, base = t['a_cat'], t['a_raw'], t['a_base']
    cap = np.where(araw < 0.999 * acat, araw, np.inf)
    pk = np.where(np.isfinite(t['pk']), t['pk'], 1.0)
    with np.errstate(invalid='ignore', divide='ignore'):
        s_cap = -2.5 * np.log10(acat / araw)
        sh = lambda a: -2.5 * np.log10(a / base)
        d = {'final': np.zeros(len(t)), 'uncapped': s_cap, 'base+cap': s_cap + sh(np.minimum(base, cap)), 'bgfree': s_cap + sh(t['a_bgfree']),
             'bgfree+cap': s_cap + sh(np.minimum(t['a_bgfree'], cap)), 'rw12': s_cap + sh(t['a_rw12']), 'rw12+cap': s_cap + sh(np.minimum(t['a_rw12'], cap)),
             'rw12+bgfree': s_cap + sh(t['a_rw12+bgfree']), 'rw12+bgfree+cap': s_cap + sh(np.minimum(t['a_rw12+bgfree'], cap)),
             'bgfree+v7b+cap': s_cap + sh(np.minimum(t['a_bgfree+v7b'], cap * pk))}
    return d


def e(x):
    x = x[np.isfinite(x)]
    return 1.2533 * 1.4826 * np.median(np.abs(x - np.median(x))) / np.sqrt(len(x)) if len(x) > 2 else np.nan


def main():
    res, L = {}, ['# Round-3 satrefit variants split by detector\n',
                  'Star-level medians (median over the exposures of each star), dm = variant magnitude - dolphot - ZP; negative = ours bright. Error = 1.25 MAD/sqrt(N). Only the refit frames (nrcb1, nrcb3; F150W, F200W; 4 exposures each) are included.\n']
    for b, band in BANDS.items():
        z = np.load(f'{OUT}/rows_{band}.npz', allow_pickle=True)
        S = pd.DataFrame({k: z['S_' + k] for k in ('det', 'exp', 'ra', 'dec', 'final', 'ref', 'lab')})
        S = S[S.det.isin(['nrcb1', 'nrcb3'])].reset_index(drop=True)
        parts = []
        for t in load_frames(b):
            s = S[(S.det == t['det'][0]) & (S.exp == t['exp'][0])]
            sk = SkyCoord(s.ra.values * u.deg, s.dec.values * u.deg)
            tk = SkyCoord(t['ra'] * u.deg, t['dec'] * u.deg)
            i, d, _ = sk.match_to_catalog_sky(tk)
            ok = d.arcsec < 0.01
            v = variants(t)
            df = s[ok].copy()
            for k in VARS:
                df[k] = df['final'].values + v[k][i[ok]]
            df['label'] = np.asarray(t['label'])[i[ok]]
            parts.append(df)
        R = pd.concat(parts)
        R = R[R.label > 0]
        res[band] = {'n_rows': {d: int((R.det == d).sum()) for d in ('nrcb1', 'nrcb3')}}
        # require every variant finite
        R = R[np.isfinite(R[VARS].values).all(axis=1)]
        St = R.groupby(['lab', 'det']).median(numeric_only=True).reset_index()
        L.append(f'\n## F{b}: {len(R)} rows, {len(St)} star-detector medians (nrcb1 {int((St.det == "nrcb1").sum())}, nrcb3 {int((St.det == "nrcb3").sum())})\n')
        bins = BINS[b] + [(-99, 99)]
        hdr = '| variant | det | ' + ' | '.join('all' if lo < 0 else f'{lo}-{hi}' for lo, hi in bins) + ' |'
        L.append(hdr)
        L.append('|---|---|' + '---|' * len(bins))
        res[band]['table'] = {}
        for v in VARS:
            for dt in ('nrcb1', 'nrcb3', 'diff'):
                cells = []
                for lo, hi in bins:
                    if dt == 'diff':
                        a = St[(St.det == 'nrcb3') & (St.ref >= lo) & (St.ref < hi)][v].values
                        c = St[(St.det == 'nrcb1') & (St.ref >= lo) & (St.ref < hi)][v].values
                        if len(a) >= 5 and len(c) >= 5:
                            val, err = np.median(a) - np.median(c), np.hypot(e(a), e(c))
                            cells.append(f'{val:+.3f} +/- {err:.3f}')
                            res[band]['table'][f'{v}|diff|{lo}-{hi}'] = (float(val), float(err), len(a), len(c))
                        else:
                            cells.append('-')
                    else:
                        a = St[(St.det == dt) & (St.ref >= lo) & (St.ref < hi)][v].values
                        if len(a) >= 5:
                            cells.append(f'{np.median(a):+.3f} ({len(a)})')
                            res[band]['table'][f'{v}|{dt}|{lo}-{hi}'] = (float(np.median(a)), float(e(a)), len(a))
                        else:
                            cells.append('-')
                L.append(f'| {v} | {dt if dt != "diff" else "b3 - b1"} | ' + ' | '.join(cells) + ' |')
    # R(g0) curves
    L.append('\n# R(g0) curves per frame (satrefit/out3/*_rcurve.txt)\n')
    curves = {}
    for fn in sorted(glob.glob(f'{Q}/satrefit/out3/*_rcurve.txt')):
        m = re.match(r'(\d+W)_.*_(\d{5})_(nrcb[13])_', os.path.basename(fn))
        if m is None:
            continue
        a = np.loadtxt(fn)
        hdr = [l for l in open(fn) if l.startswith('# sig_low')][0]
        curves[(m.group(1), int(m.group(2)), m.group(3))] = (a, hdr.strip())
    grid = np.array([250, 400, 600, 900, 1300, 1800, 2500, 3300])
    res['rcurve'] = {}
    for b in BANDS:
        L.append(f'\n**F{b}: R at fixed g0 (DN), per frame; mean and nrcb3/nrcb1 ratio**\n')
        L.append('| frame | det | ' + ' | '.join(str(g) for g in grid) + ' | mean R (g0 400-2500) | intrinsic scatter (400-2500) | sig_low / rn0 / gain / s_flat |')
        L.append('|---|---|' + '---|' * (len(grid) + 3))
        mean = {}
        for (bb, ex, dt), (a, h) in sorted(curves.items()):
            if bb != b:
                continue
            g0, R_ = a[:, 0], a[:, 1]
            r_g = np.interp(grid, g0, R_, left=np.nan, right=np.nan)
            sel = (g0 >= 400) & (g0 <= 2500)
            mean[(ex, dt)] = r_g
            kv = dict(re.findall(r'(\w+)=([-\d.]+)', h))
            L.append(f'| {ex} | {dt} | ' + ' | '.join(f'{v:.4f}' for v in r_g) + f' | {np.mean(R_[sel]):.4f} | {np.mean(a[sel, 4]):.3f} | {float(kv["sig_low"]):.2f} / {float(kv["rn0"]):.2f} / {float(kv["gain"]):.2f} / {float(kv["s_flat"]):.3f} |')
            res['rcurve'][f'{b}_{ex}_{dt}'] = dict(R_grid=r_g.tolist(), mean=float(np.mean(R_[sel])), scatter=float(np.mean(a[sel, 4])))
        b1 = np.nanmean([mean[k] for k in mean if k[1] == 'nrcb1'], axis=0)
        b3 = np.nanmean([mean[k] for k in mean if k[1] == 'nrcb3'], axis=0)
        L.append(f'| mean of 4 | nrcb1 | ' + ' | '.join(f'{v:.4f}' for v in b1) + ' | | | |')
        L.append(f'| mean of 4 | nrcb3 | ' + ' | '.join(f'{v:.4f}' for v in b3) + ' | | | |')
        L.append(f'| ratio b3/b1 | | ' + ' | '.join(f'{v:.3f}' for v in b3 / b1) + ' | | | |')
        res['rcurve'][f'{b}_ratio'] = (b3 / b1).tolist()
    json.dump(res, open(f'{OUT}/det_variants.json', 'w'), indent=1)
    open(f'{OUT}/det_variants_tables.md', 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
