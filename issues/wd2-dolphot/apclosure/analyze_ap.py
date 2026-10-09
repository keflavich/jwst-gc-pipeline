import sys, os, numpy as np
from astropy.table import Table
from scipy.stats import theilslopes
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'); import analyze as an
an.ZPWIN.update(an.zp_windows()); A = an.Arm('main2')
P = np.load(f'{an.Q}/photomver/areapred_main2.npz')
PT = np.load(f'{an.Q}/psfsum/predT_main2.npz')
HERE = os.path.dirname(os.path.abspath(__file__))
BANDS = ['150W', '164N', '187N', '200W', '212N', '182M', '250M', '300M', '410M', '466N', '115W', '162M', '277W', '335M', '323N', '405N']
RADII = [3, 5, 8]
DIFFS = {'a': 'psf_raw - ap_raw', 'b': 'ap_raw - ap_area', 'c': 'dolphot - ap_area', 'd': 'psf_raw - dolphot', 'e': 'psf_raw - ap_area'}


def rs(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def slope(p, d, clip=True):
    d = d - np.median(d)
    if clip:  # 5 sigma clip against outliers
        s = rs(d); k = np.abs(d) < 5 * s
        p, d = p[k], d[k]
    r = theilslopes(d, p, 0.95)
    return r[0], r[2], r[3], rs(d), len(p)


def fit2(d, p, pt, nboot=300, seed=1):
    """3-sigma clipped OLS of d on [pred, predT, 1]; bootstrap 95 % CI."""
    X = np.column_stack([p, pt, np.ones_like(p)]); keep = np.isfinite(d) & np.isfinite(pt)
    for _ in range(5):
        c = np.linalg.lstsq(X[keep], d[keep], rcond=None)[0]; r = d - X @ c
        keep = keep & (np.abs(r - np.median(r[keep])) < 3 * rs(r[keep]))
    rng = np.random.default_rng(seed); idx = np.flatnonzero(keep)
    bs = np.array([np.linalg.lstsq(X[s], d[s], rcond=None)[0] for s in (rng.choice(idx, len(idx)) for _ in range(nboot))])
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    return c, lo, hi, len(idx), np.corrcoef(p[keep], pt[keep])[0, 1]


def stars(band):
    fn = f'{HERE}/frames_{band}.ecsv'
    if not os.path.exists(fn): return None
    T = Table.read(fn)
    T = T[(T['src'] == 1) & (T['psf'] > 0)]   # common frames: PSF matched row present
    out = {r: [] for r in RADII}
    ids = np.unique(T['i'])
    rec = []
    for i in ids:
        s = T[T['i'] == i]
        if len(s) < 2: continue
        raw = {r: np.mean(s[f'raw{r}']) for r in RADII}; ar = {r: np.mean(s[f'area{r}']) for r in RADII}
        rec.append((i, np.mean(s['psf']), raw, ar, s['det'][0], len(s)))
    return rec


def diffs(band, rec, r):
    i = np.array([x[0] for x in rec])
    psf = np.array([x[1] for x in rec]); raw = np.array([x[2][r] for x in rec]); ar = np.array([x[3][r] for x in rec])
    good = (psf > 0) & (raw > 0) & (ar > 0)
    i, psf, raw, ar = i[good], psf[good], raw[good], ar[good]
    det = np.array([x[4] for x in rec])[good]
    m = lambda f: -2.5 * np.log10(f)
    dol = A.ref[band][i]; predT = PT[band][i]
    return dict(i=i, det=det, pred=P[band][i], predT=predT, mag=dol, a=m(psf) - m(raw), b=m(raw) - m(ar), c=dol - m(ar), d=m(psf) - dol, e=m(psf) - m(ar))


def main():
    L = []; res = {}; res2 = {}
    for band in BANDS:
        rec = stars(band)
        if rec is None or len(rec) < 30:
            L.append(f'F{band}: no/insufficient data'); continue
        L.append(f'\n## F{band}  (stars with >=2 PSF-matched frames: {len(rec)})')
        L.append('| diff | r | slope [95% CI] | robust std | N |')
        L.append('|---|---|---|---|---|')
        for r in RADII:
            D = diffs(band, rec, r)
            for k in 'abcde':
                s = slope(D['pred'], D[k]); res[(band, r, k)] = s
                L.append(f'| ({k}) {DIFFS[k]} | {r} | {s[0]:+.2f} [{s[1]:+.2f}, {s[2]:+.2f}] | {s[3]:.4f} | {s[4]} |')
        D = diffs(band, rec, 5)
        L.append(f'pred range (5/50/95%): {np.percentile(D["pred"], [5, 50, 95]).round(4)}; pred robust std {rs(D["pred"]):.4f}')
        L.append('two-regressor fit d = A*pred + B*predT + const (3-sigma clipped OLS, bootstrap 95% CI), r = 5 px:')
        for k in 'acde':
            c, lo, hi, n, rho = fit2(D[k], D['pred'], D['predT'])
            res2[(band, k)] = (c, lo, hi, n)
            L.append(f'  ({k}) {DIFFS[k]}: A = {c[0]:+.2f} [{lo[0]:+.2f}, {hi[0]:+.2f}]  B = {c[1]:+.2f} [{lo[1]:+.2f}, {hi[1]:+.2f}]  N={n}  corr(pred,predT)={rho:+.2f}')
        mid = np.median(D['mag'])
        for name, sel in (('brighter half', D['mag'] < mid), ('fainter half', D['mag'] >= mid)):
            t = ' ; '.join(f'({k}) {slope(D["pred"][sel], D[k][sel])[0]:+.2f} [{slope(D["pred"][sel], D[k][sel])[1]:+.2f},{slope(D["pred"][sel], D[k][sel])[2]:+.2f}]' for k in 'acd')
            L.append(f'r=5, {name} (N={sel.sum()}): {t}')
        dets = np.unique(D['det'])
        for dt in dets:
            sel = D['det'] == dt
            if sel.sum() >= 60:
                t = ' ; '.join(f'({k}) {slope(D["pred"][sel], D[k][sel])[0]:+.2f}' for k in 'acd')
                L.append(f'r=5, det {dt} (N={sel.sum()}, pred std {rs(D["pred"][sel]):.4f}): {t}')
    open(f'{HERE}/apclosure.txt', 'w').write('\n'.join(L) + '\n')
    # markdown table
    M = ['| band | (a) psf_raw-ap_raw | (c) dolphot-ap_area | (d) psf_raw-dolphot | (b) check | N |', '|---|---|---|---|---|---|']
    for band in BANDS:
        if (band, 5, 'a') not in res: continue
        f = lambda k: '%+.2f [%+.2f, %+.2f]' % res[(band, 5, k)][:3]
        M.append(f'| F{band} | {f("a")} | {f("c")} | {f("d")} | {res[(band, 5, "b")][0]:+.2f} | {res[(band, 5, "a")][4]} |')
    M += ['', 'Radius dependence (slope at r = 3 / 5 / 8 px):', '| band | (a) | (c) | (d) |', '|---|---|---|---|']
    for band in BANDS:
        if (band, 5, 'a') not in res: continue
        M.append(f'| F{band} | ' + ' | '.join(' / '.join(f'{res[(band, r, k)][0]:+.2f}' for r in RADII) for k in 'acd') + ' |')
    M += ['', 'Two-regressor fit at r = 5 px: coefficients A (pred), B (predT) [95% CI]', '| band | (a) psf_raw-ap_raw A | (a) B | (e) psf_raw-ap_area A | (e) B | (c) dolphot-ap_area A | (c) B | (d) psf_raw-dolphot A | (d) B |', '|---|---|---|---|---|---|---|---|---|']
    for band in BANDS:
        if (band, 'a') not in res2: continue
        cells = []
        for k in 'aecd':
            c, lo, hi, n = res2[(band, k)]
            cells += ['%+.2f [%+.2f, %+.2f]' % (c[j], lo[j], hi[j]) for j in (0, 1)]
        M.append(f'| F{band} | ' + ' | '.join(cells) + ' |')
    open(f'{HERE}/tables.md', 'w').write('\n'.join(M) + '\n')
    # figure
    show = [b for b in ('200W', '212N', '300M', '410M') if (b, 5, 'a') in res]
    fig, axs = plt.subplots(1, len(show), figsize=(5 * len(show), 4.5), sharey=True, squeeze=False)
    for ax, band in zip(axs[0], show):
        D = diffs(band, stars(band), 5)
        edges = np.quantile(D['pred'], np.linspace(0, 1, 9))
        xs = np.array([-0.04, 0.04])
        for k, c in (('a', 'C0'), ('c', 'C3'), ('d', 'C2')):
            d = D[k] - np.median(D[k])
            xm, ym, ye = [], [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                s = (D['pred'] >= lo) & (D['pred'] <= hi)
                xm.append(np.median(D['pred'][s])); ym.append(np.median(d[s])); ye.append(rs(d[s]) / np.sqrt(s.sum()) * 1.25)
            ax.errorbar(xm, ym, ye, fmt='o-', color=c, ms=4, label=f'({k}) {DIFFS[k]}  slope {res[(band, 5, k)][0]:+.2f}')
        ax.plot(xs, xs, 'k-', lw=1, label='slope 1'); ax.plot(xs, 0 * xs, 'k:', lw=1, label='slope 0')
        ax.set_title(f'F{band}, r = 5 px'); ax.set_xlabel('pred = 2.5 log10(pixel area / nominal) (mag)')
        ax.legend(fontsize=7, loc='upper left'); ax.set_xlim(-0.045, 0.045)
    axs[0][0].set_ylabel('binned median difference, zero-pointed (mag)')
    plt.tight_layout(); plt.savefig(f'{HERE}/apclosure.png', dpi=110)
    print(open(f'{HERE}/tables.md').read())


main()
