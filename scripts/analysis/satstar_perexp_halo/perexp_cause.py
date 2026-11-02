"""What drives the column-dependent per-exposure halo change (#1013 follow-up)?

    python perexp_cause.py pxh.npz <saturation reffile dir> <out dir>

1. Signal level: the NRCBLONG x~400 deficit and the a-vs-core-area slope, split
   by star brightness (star-visit mean halo amplitude A30).
2. Reference files: the CRDS SATURATION threshold map of each LW detector,
   column-median, against the halo change f(x).
3. The obs-061 target: its per-dither a, core area and detector x.
"""
import glob
import json
import os
import sys

import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perexp_figs import load  # noqa: E402

EDGES = np.arange(0, 2049, 128)
XC = 0.5 * (EDGES[1:] + EDGES[:-1])


def colprof(x, v, edges=EDGES, nmin=15):
    k = np.digitize(x, edges) - 1
    med = np.array([np.median(v[k == i]) if (k == i).sum() >= nmin else np.nan for i in range(len(edges) - 1)])
    err = np.array([1.2533 * np.std(v[k == i]) / np.sqrt((k == i).sum()) if (k == i).sum() >= nmin else np.nan
                    for i in range(len(edges) - 1)])
    return med, err


def satmaps(refdir):
    out = {}
    for fn in sorted(glob.glob(f'{refdir}/jwst_nircam_saturation_*.fits')):
        h = fits.getheader(fn)
        if h.get('DETECTOR') in ('NRCALONG', 'NRCBLONG'):
            s = fits.getdata(fn, 'SCI').astype(float)
            dq = fits.getdata(fn, 'DQ')
            s[(dq & 1) > 0] = np.nan
            out[h['DETECTOR']] = (os.path.basename(fn), s)
    return out


def main():
    fin, refdir, outd = sys.argv[1:4]
    os.makedirs(outd, exist_ok=True)
    t = load(fin, 0.02)
    res = {}
    # star brightness: star-visit mean A30 (scene-free halo amplitude at 30 px)
    A = np.array([np.mean(t['A30'][t['grp'] == g]) for g in t['grp']])
    fig, ax = plt.subplots(2, 3, figsize=(19, 9.5))
    for row, det in enumerate(('NRCBLONG', 'NRCALONG')):
        m = t['det'] == det
        q = np.quantile(A[m], [0, 1 / 3, 2 / 3, 1])
        res[det] = {}
        for j in range(3):
            mm = m & (A >= q[j]) & (A <= q[j + 1])
            med, err = colprof(t['xc'][mm], t['a'][mm])
            mda, _ = colprof(t['xc'][mm], t['area'][mm] - 1)
            lab = f'A30 {q[j]:.3g}–{q[j + 1]:.3g} (n={mm.sum()})'
            ax[row, 0].errorbar(XC + 8 * j, med, err, fmt='o-', ms=4, label=lab)
            ax[row, 1].plot(XC + 8 * j, mda, 's-', ms=4, label=lab)
            sl = np.polyfit(t['area'][mm] - 1, t['a'][mm], 1)[0]
            dip = (np.nanmean(med[(XC > 250) & (XC < 550)]) - np.nanmean(med[(XC < 150) | (XC > 750)])
                   if det == 'NRCBLONG' else np.nanmean(med[(XC < 300) | (XC > 1750)]) - np.nanmean(med[(XC > 700) & (XC < 1350)]))
            res[det][f'tercile{j}'] = dict(A30_range=[float(q[j]), float(q[j + 1])], n=int(mm.sum()),
                                           a_vs_area_slope=float(sl), column_feature=float(dip),
                                           a_prof=med.tolist(), area_prof=mda.tolist())
        ax[row, 0].set_title(f'{det}: per-exposure halo change a vs detector x, by brightness')
        ax[row, 1].set_title(f'{det}: core area − 1 vs detector x, by brightness')
        for a in ax[row, :2]:
            a.axhline(0, color='k', lw=0.5); a.set_xlabel('detector x [px]'); a.legend(fontsize=7)
        ax[row, 0].set_ylabel('median a (star-visit mean removed)')
    sm = satmaps(refdir)
    for row, det in enumerate(('NRCBLONG', 'NRCALONG')):
        if det not in sm:
            continue
        name, s = sm[det]
        cm = np.nanmedian(s[4:-4, :], axis=0)
        cm128 = np.array([np.nanmedian(s[4:-4, e0:e1]) for e0, e1 in zip(EDGES[:-1], EDGES[1:])])
        ax[row, 2].plot(np.arange(s.shape[1]), cm, lw=0.5, color='0.6', label='column median')
        ax[row, 2].plot(XC, cm128, 'ko-', ms=4, label='128-px bins')
        ax[row, 2].set_title(f'{det}: SATURATION reffile {name}')
        ax[row, 2].set_xlabel('detector x [px]'); ax[row, 2].set_ylabel('saturation threshold [DN]')
        ax[row, 2].legend(fontsize=7)
        allm = t['det'] == det
        a_all, _ = colprof(t['xc'][allm], t['a'][allm])
        ok = np.isfinite(a_all)
        res[det]['saturation_ref'] = dict(file=name, prof128=cm128.tolist(),
                                          corr_with_a=float(np.corrcoef(cm128[ok], a_all[ok])[0, 1]),
                                          rel_rms=float(np.nanstd(cm128) / np.nanmedian(cm128)))
    fig.tight_layout(); fig.savefig(f'{outd}/cause_signal_reffile.png', dpi=int(os.environ.get('FIG_DPI', 130)))
    json.dump(res, open(f'{outd}/cause.json', 'w'), indent=1)
    # saturated vs unsaturated stars on one axis (perexp_fieldstar_flux.py output)
    fsj = f'{outd}/fieldstar.json'
    if os.path.exists(fsj):
        fs = json.load(open(fsj))
        m = t['det'] == 'NRCBLONG'
        a_all, a_err = colprof(t['xc'][m], t['a'][m])
        ar_all, ar_err = colprof(t['xc'][m], t['area'][m] - 1)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.errorbar(XC, 1 + a_all, a_err, fmt='o-', label=f'saturated stars: halo 15–80 px (1 + a), {m.sum()} exposures')
        ax.errorbar(XC, 1 + ar_all, ar_err, fmt='s-', label='saturated stars: saturated-core area')
        ax.errorbar(XC, fs['median_ratio'], fs['err'], fmt='D-', color='k',
                    label=f'unsaturated field stars: aperture flux ({sum(fs["n"])} measurements)')
        ax.axhline(1, color='k', lw=0.5)
        ax.set_xlabel('NRCBLONG detector x [px]'); ax.set_ylabel('relative to the star-visit mean / median')
        ax.set_title('F480M, 10678: the x≈400 deficit is confined to saturated stars')
        ax.legend(fontsize=8); fig.tight_layout()
        fig.savefig(f'{outd}/cause_satvsunsat.png', dpi=int(os.environ.get('FIG_DPI', 130)))
    print(json.dumps({d: {k: {kk: vv for kk, vv in v.items() if 'prof' not in kk} for k, v in r.items()} for d, r in res.items()}, indent=1))


if __name__ == '__main__':
    main()
