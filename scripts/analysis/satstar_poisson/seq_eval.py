"""Brightness sequence (README §5): LOO chi statistics for each star, from loo_crop.py outputs.

    python seq_eval.py <outdir> <tag-template> <prefix> [<prefix> ...]
e.g. python seq_eval.py figures bs_{p} tgt trn_116 trn_069 ...   (needs brightness.json;
     '<prefix>:<tag>' uses another LOO tag for that star)

For every star (prefix) and held-out dither k in <tag>_e<k>.npz:
    chi = (d - pred) / sqrt(VAR_POISSON + VAR_RNOISE + varQ)
over clean pixels (finite, not DO_NOT_USE, no SAT(2)/JUMP(4) flag, >= MINCOV training
dithers at the pattern node).  Per radial bin it reports the robust sigma(chi)
(1.4826 median|chi|), the non-Poisson excess relative to the noise, sqrt(sigma^2 - 1), and
that excess as a fraction of the star's local intensity,
    f_ex = sqrt(sigma^2 - 1) * median(sigma_tot) / median(pred - bkg),
bkg being the off-spike scene level at r = 350-450 px (brightness.py).
Writes <outdir>/fig13_brightness_sequence.png and ./brightness_sequence.json."""
import sys, os, glob, json, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exposure import Exposure
# nan-heavy medians on masked bins; other warnings stay visible
warnings.filterwarnings('ignore', category=RuntimeWarning)

BINS = [(30, 50), (50, 80), (80, 120), (120, 200), (200, 300)]
FINE = np.array([30, 40, 50, 65, 80, 100, 120, 150, 200, 250, 300])
IEDGES = np.logspace(0, 3.5, 15)          # local star intensity bins [MJy/sr]
MINCOV = int(os.environ.get('MINCOV', 3))
LABEL = {'tgt': 'obs 061 (target)'}


def rs(c):
    # uncentred robust sigma, 1.4826 median|chi| (as patternfit.py / loo_crop.py)
    return 1.4826*np.median(np.abs(c)) if len(c) > 50 else np.nan


def star_stats(tag, prefix, bkg):
    out = []
    J1 = np.load(sorted(glob.glob(f'{prefix}_e[1-6].npz'))[0])['J']
    for fn in sorted(glob.glob(f'{tag}_e[1-6].npz')):
        k = int(fn[-5])
        ex = Exposure(f'{prefix}_e{k}.npz', J1); z = np.load(fn)
        vt = ex.var+z['varQ']
        chi = (ex.d-z['pred'])/np.sqrt(vt)
        if 'cov' in z:
            cov = z['cov']
        else:
            warnings.warn(f'{fn} has no training-coverage map (older loo_crop.py output): the '
                          f'MINCOV={MINCOV} coverage cut is NOT applied for this dither', UserWarning)
            cov = np.full(chi.shape, np.inf)
        cl = ex.good & np.isfinite(chi) & (cov >= MINCOV) & ((ex.dq & 6) == 0)
        yy, xx = np.mgrid[:ex.n, :ex.n]; r = np.hypot(xx-ex.xt0, yy-ex.yt0)
        row = dict(dither=k, bins={})
        for lo, hi in BINS:
            m = cl & (r >= lo) & (r < hi)
            s = rs(chi[m]); I = np.median(z['pred'][m])-bkg; st = np.median(np.sqrt(vt[m]))
            ex_p = np.sqrt(max(s**2-1, 0))
            row['bins'][f'{lo}-{hi}'] = dict(sigma=float(s), excess_over_noise=float(ex_p), I_star=float(I),
                                             sigma_tot=float(st), f_excess=float(ex_p*st/I), npix=int(m.sum()))
        row['fine'] = [float(rs(chi[cl & (r >= a) & (r < b)])) for a, b in zip(FINE[:-1], FINE[1:])]
        # sigma(chi) against the star's local model intensity, r = 30-200 px
        I = z['pred']-bkg; m = cl & (r >= 30) & (r < 200)
        row['vs_I'] = [float(rs(chi[m & (I >= a) & (I < b)])) for a, b in zip(IEDGES[:-1], IEDGES[1:])]
        out.append(row)
    return out


if __name__ == '__main__':
    outdir, tmpl, pres = sys.argv[1], sys.argv[2], sys.argv[3:]
    B = json.load(open('brightness.json'))
    res = {}
    tags = {}
    for i, p in enumerate(pres):          # "<prefix>:<tag>" overrides the template for one star
        p, _, t = p.partition(':'); pres[i] = p; tags[p] = t or tmpl.format(p=p)
    for p in pres:
        st = star_stats(tags[p], p, B[p]['bkg'])
        if not st:
            print(p, 'no LOO outputs'); continue
        res[p] = dict(F_rel=B[p]['F_rel'], nsat_core=B[p]['nsat_core'], ndith=B[p]['ndith'], loo=st)
        for row in st:
            print(f"{p:8s} F/Ft={B[p]['F_rel']:.3f} e{row['dither']}: " +
                  ' '.join(f"{b}:{v['sigma']:.2f}" for b, v in row['bins'].items()) + '  | f_ex ' +
                  ' '.join(f"{b}:{100*v['f_excess']:.1f}%" for b, v in row['bins'].items()))
    json.dump(res, open('brightness_sequence.json', 'w'), indent=1)

    # ---- figure
    order = sorted(res, key=lambda p: -res[p]['F_rel'])
    cmap = plt.get_cmap('viridis'); lf = np.log10([res[p]['F_rel'] for p in order])
    col = {p: cmap(0.9*(lf[0]-l)/max(lf[0]-lf[-1], 1e-3)) for p, l in zip(order, lf)}
    mid = np.sqrt(FINE[1:]*FINE[:-1])
    fig, ax = plt.subplots(1, 4, figsize=(26, 5.6))
    for p in order:
        f = np.array([row['fine'] for row in res[p]['loo']])
        lab = f"{LABEL.get(p, 'obs '+p[-3:])}  F/F$_t$={res[p]['F_rel']:.2f}"
        ax[0].plot(mid, np.nanmean(f, 0), 'o-', color=col[p], label=lab, lw=2)
        if len(f) > 1:
            ax[0].fill_between(mid, np.nanmin(f, 0), np.nanmax(f, 0), color=col[p], alpha=0.15, lw=0)
    ax[0].axhline(1, color='k', lw=1); ax[0].set_xscale('log'); ax[0].set_yscale('log')
    ax[0].set_yticks([1, 1.5, 2, 3, 5, 8]); ax[0].set_yticklabels(['1', '1.5', '2', '3', '5', '8'])
    ax[0].set_xlabel('distance from saturated star [px] (1 px = 0.063")'); ax[0].set_ylabel(r'robust $\sigma(\chi)$, held-out dithers')
    ax[0].set_title('LOO (crop 300 px), clean pixels; band = range over held-out dithers'); ax[0].legend(fontsize=8)
    for b, mk in [('50-80', 'o'), ('80-120', 's')]:
        F = []; E = []
        for p in order:
            for row in res[p]['loo']:
                F.append(res[p]['F_rel']); E.append(row['bins'][b]['excess_over_noise'])
        ax[1].loglog(F, np.maximum(E, 0.05), mk, ms=8, label=f'r = {b} px')
        F = []; E = []
        for p in order:
            for row in res[p]['loo']:
                F.append(res[p]['F_rel']); E.append(100*row['bins'][b]['f_excess'])
        ax[2].semilogx(F, E, mk, ms=8, label=f'r = {b} px')
    ff = np.logspace(-1.4, 0.05, 20)
    e0 = np.median([row['bins']['80-120']['excess_over_noise'] for row in res['tgt']['loo']]) if 'tgt' in res else np.nan
    ax[1].loglog(ff, e0*np.sqrt(ff), 'k--', lw=1, label=r'constant fractional excess: $\propto\sqrt{F}$ (80-120)')
    ax[1].axhline(1, color='gray', lw=1)
    ax[1].axhline(np.sqrt(1.2**2-1), color='gray', lw=1, ls=':', label=r'far-field floor, $\sigma(\chi)$=1.2')
    ax[1].set_xlabel(r'$F/F_{\rm target}$'); ax[1].set_ylabel(r'non-Poisson excess / noise = $\sqrt{\sigma(\chi)^2-1}$'); ax[1].legend(fontsize=8)
    ax[1].set_title('excess relative to the noise')
    ax[2].set_xlabel(r'$F/F_{\rm target}$'); ax[2].set_ylabel('excess / local star intensity [%]'); ax[2].legend(fontsize=8)
    ax[2].set_title('excess as a fraction of the star\'s local intensity'); ax[2].set_ylim(0, None)
    imid = np.sqrt(IEDGES[1:]*IEDGES[:-1])
    for p in order:
        v = np.nanmean(np.array([row['vs_I'] for row in res[p]['loo']]), 0)
        ax[3].plot(imid, v, 'o-', color=col[p], lw=2)
    ax[3].axhline(1, color='k', lw=1); ax[3].set_xscale('log'); ax[3].set_yscale('log')
    ax[3].set_yticks([1, 1.5, 2, 3, 5, 8]); ax[3].set_yticklabels(['1', '1.5', '2', '3', '5', '8'])
    ax[3].set_xlabel('local star intensity, prediction - scene level [MJy/sr]'); ax[3].set_ylabel(r'robust $\sigma(\chi)$')
    ax[3].set_title(r'$\sigma(\chi)$ vs local intensity, r = 30-200 px')
    plt.tight_layout(); plt.savefig(f'{outdir}/fig13_brightness_sequence.png', dpi=75); plt.close()
    print('wrote', f'{outdir}/fig13_brightness_sequence.png')
