"""Score the modified caps (capcore2_run.py).  Writes capcore2_tables.md and capcore2.png."""
import glob
import os
import sys
import numpy as np
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an  # noqa: E402
from astropy.table import Table, vstack  # noqa: E402
from astropy.coordinates import SkyCoord  # noqa: E402
import astropy.units as u  # noqa: E402

HERE = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit/capcore'
BANDS = ['150W', '200W', '250M', '300M']
LW = ('250M', '300M')
an.ZPWIN.update(an.zp_windows())
A = an.Arm('main2')


def mad(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def per_star(i, x, n):
    out = np.full(n, np.nan)
    m = np.isfinite(x)
    ii, xx = i[m], x[m]
    o = np.argsort(ii, kind='stable')
    ii, xx = ii[o], xx[o]
    uq, st = np.unique(ii, return_index=True)
    for k, s, e in zip(uq, st, list(st[1:]) + [len(ii)]):
        out[k] = np.median(xx[s:e])
    return out


def running(x, y, nwin):
    o = np.argsort(x)
    x, y = x[o], y[o]
    h = nwin // 2
    if len(x) < nwin:
        return [], []
    ks = range(h, len(x) - h, max(1, h // 2))
    return [np.median(x[k - h:k + h]) for k in ks], [np.median(y[k - h:k + h]) for k in ks]


L = []
res = {}
meta_lines = []
for band in BANDS:
    ok = A.matched & A.rep[band] & np.isfinite(A.ref[band])
    tgt = A.sky[A.idx[ok]]
    ref = A.ref[band][ok]
    dm_final = A.dm(band)[ok]
    n = ok.sum()
    tabs = []
    for fn in sorted(glob.glob(f'{HERE}/out2/{band}_*_cap2.fits')):
        t = Table.read(fn)
        meta_lines.append(f'| F{band} | {os.path.basename(fn).split("_")[3]} {os.path.basename(fn).split("_")[4]} | {t.meta["SA"]:.0f} | {t.meta["SB"]:.0f} | {t.meta["G0SAT99"]:.0f} |')
        tabs.append(t)
    T = vstack(tabs, metadata_conflicts='silent')
    sk = SkyCoord(np.asarray(T['ra']) * u.deg, np.asarray(T['dec']) * u.deg)
    i, j, _, _ = sk.search_around_sky(tgt, 0.1 * u.arcsec)
    g = lambda c: np.asarray(T[c], float)  # noqa: E731
    pre, raw = g('c_flux_fit_precap'), g('c_flux_fit_raw')
    lab = g('label') > 0
    tags = [c[4:] for c in T.colnames if c.startswith('cap_')]
    flux = {'final': raw, 'precap': pre}
    cap0 = g('cap0')
    for tg in tags:
        c = g('cap_' + tg)
        flux[tg] = np.where(np.isfinite(c), np.minimum(pre, c), pre)
    flux['cap0(check)'] = np.where(np.isfinite(cap0), np.minimum(pre, cap0), pre)
    dm = {}
    for k, f in flux.items():
        with np.errstate(invalid='ignore', divide='ignore'):
            s = np.where(lab, -2.5 * np.log10(f / raw), np.nan)
        dm[k] = dm_final + per_star(i, s[j], n) if k != 'final' else dm_final.copy()
    hb = np.isfinite(per_star(i, np.where(lab, 0.0, np.nan)[j], n))
    names = ['final', 'precap'] + [t for t in tags]
    lo0, hi0 = int(np.floor(ref[hb].min())), int(np.ceil(np.percentile(ref[hb], 99.5)))
    bins = [(b, b + 1) for b in range(lo0, hi0)]
    nl = lab.sum()
    chk = np.nanmax(np.abs(flux['a1.0_gm'][lab] / raw[lab] - 1))
    chk0 = np.nanmax(np.abs(flux['cap0(check)'][lab] / raw[lab] - 1))
    L.append(f'#### F{band}: {n} satstar-replaced matched stars, {hb.sum()} with rows.  Cells: median dm (MAD) [N].  Reproduction of flux_fit_raw: cap0 max |min(precap,cap0)/raw - 1| = {chk0:.1e}, threshold 1.0 (nothing excluded) = {chk:.1e}.')
    L.append('')
    L.append('| variant | ' + ' | '.join(f'{a}-{b}' for a, b in bins) + ' | all: med / MAD [N] | slope per mag | rows capped | cap evaluated -> skipped | rows with finite cap |')
    L.append('|---|' + '---|' * (len(bins) + 5))
    for k in names:
        d = dm[k]
        cells = []
        for a, b in bins:
            s = hb & np.isfinite(d) & (ref >= a) & (ref < b)
            cells.append(f'{np.median(d[s]):+.3f} ({mad(d[s]):.3f}) [{s.sum()}]' if s.sum() >= 3 else f'- [{s.sum()}]')
        s = hb & np.isfinite(d)
        ss = s & (np.abs(d - np.median(d[s])) < 0.5)
        sl = np.polyfit(ref[ss], d[ss], 1)[0] if ss.sum() > 20 else np.nan
        if k in ('final', 'precap'):
            capped = (raw < 0.999 * pre) if k == 'final' else np.zeros(len(pre), bool)
            sw = '-'
            fin = (np.isfinite(cap0) & lab).sum() if k == 'final' else 0
        else:
            c = g('cap_' + k) if k in tags else cap0
            capped = lab & (flux[k] < 0.999 * pre)
            sw = int((lab & np.isfinite(cap0) & ~np.isfinite(c)).sum())
            fin = int((lab & np.isfinite(c)).sum())
        L.append(f'| {k} | ' + ' | '.join(cells) + f' | {np.median(d[s]):+.3f} / {mad(d[s]):.3f} [{s.sum()}] | {sl:+.3f} | {(capped & lab).sum()}/{nl} = {(capped & lab).sum() / nl:.2f} | {sw} | {fin} |')
    L.append('')
    res[band] = dict(ref=ref, dm=dm, hb=hb)
open(f'{HERE}/capcore2_tables.md', 'w').write('\n'.join(L) + '\n')
open(f'{HERE}/capcore2_sff.md', 'w').write('| band | frame | S_ff (a): 99.9 pct of positive first frame (DN) | S_ff (b): max positive first frame (DN) | g0 99th pct at SATURATED px (DN) |\n|---|---|---|---|---|\n' + '\n'.join(meta_lines) + '\n')
print('\n'.join(L))
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
sty = {'final': ('k', '-'), 'precap': ('C0', '-'), 'a0.6_gm': ('C3', '-'), 'a0.6_go': ('C3', '--'), 'b0.6_gm': ('C2', '-'), 'b0.6_go': ('C2', '--'),
       'b0.8_go': ('C1', '--'), 'b0.8_gm': ('C1', '-')}
fig, axs = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
for ax, band in zip(axs.ravel(), BANDS):
    r = res[band]
    for k, (col, ls) in sty.items():
        s = r['hb'] & np.isfinite(r['dm'][k])
        x, y = running(r['ref'][s], r['dm'][k][s], 40 if s.sum() > 150 else 20)
        if len(x):
            ax.plot(x, y, color=col, ls=ls, label=k)
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_title(f'F{band}')
    ax.set_xlabel('dolphot mag')
    ax.set_ylim(-0.3, 0.3)
    ax.legend(fontsize=7)
axs[0, 0].set_ylabel('running median dm')
axs[1, 0].set_ylabel('running median dm')
fig.tight_layout()
fig.savefig(f'{HERE}/capcore2.png', dpi=110)
