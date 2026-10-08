import warnings
import numpy as np
from astropy.table import Table
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.simplefilter('ignore')
D = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/radprof'
PIX = {'F250M': 0.063, 'F150W': 0.031}
SATBINS = {'F250M': [(13, 15), (15, 16), (16, 17)], 'F150W': [(14, 16), (16, 17), (17, 18), (18, 19)]}
RANGES_AS = [(0.05, 0.15), (0.15, 0.3), (0.3, 0.5), (0.5, 0.8), (0.8, 1.5)]
FW = {'F250M': 1.5, 'F150W': 1.7}
rng = np.random.default_rng(1)


def boot(v, n=300):
    v = v[np.isfinite(v)]
    if v.size < 5:
        return np.nan, np.nan, v.size
    b = [np.median(rng.choice(v, v.size)) for _ in range(n)]
    return np.median(v), np.std(b), v.size


def load(arm, band):
    t = Table.read(f'{D}/data/prof_{arm}_{band}.fits')
    edges = np.array([float(e) for e in t.meta['EDGES'].split(',')])
    return t, edges


def delta(t, final, minfrac=0.9):
    """per star, per bin: sum(data-bg-model)/sum(model); NaN where <minfrac pixels valid"""
    ntot = np.asarray(t['ntot'], float); nval = np.asarray(t['nval'], float)
    d = np.asarray(t['dsum']); m = np.asarray(t['msum'])
    sc = np.ones(len(t)) if final else np.asarray(t['ratio_precap'], float)
    sc = np.where(t['kind'] == 'sat', sc, 1.0)
    mp = m * sc[:, None]
    ok = (nval >= minfrac * np.maximum(ntot, 1)) & (mp > 0)
    return np.where(ok, (d - mp) / np.where(mp > 0, mp, 1), np.nan)


def rangeval(t, edges, band, lo, hi, final, field='dsum'):
    """per star: sum over bins with centre in [lo,hi) arcsec of (data-model) / sum model, valid bins only"""
    c = 0.5 * (edges[1:] + edges[:-1]) * PIX[band]
    sel = (c >= lo) & (c < hi)
    ntot = np.asarray(t['ntot'], float); nval = np.asarray(t['nval'], float)
    sc = np.ones(len(t)) if final else np.asarray(t['ratio_precap'], float)
    sc = np.where(t['kind'] == 'sat', sc, 1.0)
    mp = np.asarray(t['msum']) * sc[:, None]
    ok = (nval >= 0.9 * np.maximum(ntot, 1)) & (mp > 0)
    ok = ok[:, sel].all(axis=1)
    num = (np.asarray(t[field])[:, sel] - mp[:, sel]).sum(1)
    den = mp[:, sel].sum(1)
    return np.where(ok & (den > 0), num / np.where(den > 0, den, 1), np.nan)


def starparts(t, final, field='dsum'):
    ntot = np.asarray(t['ntot'], float); nval = np.asarray(t['nval'], float)
    sc = np.ones(len(t)) if final else np.asarray(t['ratio_precap'], float)
    sc = np.where(t['kind'] == 'sat', sc, 1.0)
    mp = np.asarray(t['msum']) * sc[:, None]
    ok = (nval >= 0.9 * np.maximum(ntot, 1)) & (mp > 0)
    return np.where(ok, np.asarray(t[field]) - mp, 0.0), np.where(ok, mp, 0.0), ok


def stack(t, edges, band, m, final, lo=None, hi=None, b=None, field='dsum', nboot=200, minn=5, dbar=None):
    """robust stacked Delta for stars in mask m, one bin b or radius range [lo,hi) arcsec (stars need all bins valid).
    y = sum(data-bg [-nbrs] - dbar*npix), x = sum(model), weights 1/npix, fit y = (1+S) x through the origin with 4-sigma
    clipping; returns S, bootstrap error over stars, N."""
    num, den, ok = starparts(t, final, field)
    nv = np.asarray(t['nval'], float)
    if b is not None:
        sel = np.zeros(num.shape[1], bool); sel[b] = True
    else:
        c = 0.5 * (edges[1:] + edges[:-1]) * PIX[band]
        sel = (c >= lo) & (c < hi)
    good = m & ok[:, sel].all(axis=1)
    n = int(good.sum())
    if n < minn:
        return np.nan, np.nan, n
    x = den[good][:, sel].sum(1)
    npx = nv[good][:, sel].sum(1)
    y = (num[good][:, sel] + den[good][:, sel]).sum(1)
    if dbar is not None:
        y = y - (np.asarray(dbar)[sel][None, :] * nv[good][:, sel]).sum(1)
    w = 1.0 / npx

    def solve(idx):
        keep = np.ones(idx.size, bool)
        for _ in range(3):
            xi, yi, wi = x[idx][keep], y[idx][keep], w[idx][keep]
            k1 = (wi * xi * yi).sum() / (wi * xi * xi).sum()
            r = (y[idx] - k1 * x[idx]) * np.sqrt(w[idx])
            sd = 1.4826 * np.median(np.abs(r - np.median(r)))
            keep = np.abs(r - np.median(r)) < 4 * sd
        return k1 - 1
    idx0 = np.arange(n)
    S = solve(idx0)
    bs = [solve(rng.integers(0, n, n)) for _ in range(nboot)]
    return S, float(np.std(bs)), n


def fit_offset(t, m, final, field='dsumb', nboot=100, minn=30):
    """per radial bin: weighted least squares  sum(data-bg) = (1+S) * model + dbar * npix  over stars in mask m.
    S = fractional shape residual (flux proportional), dbar = additive per-pixel offset (same units as the image).
    Weights 1/npix, 4-sigma clipping, bootstrap over stars."""
    num, den, ok = starparts(t, final, field)
    y_all = num + den  # = sum(data-bg-nbrs) where valid
    nv = np.asarray(t['nval'], float)
    nb = num.shape[1]
    S = np.full(nb, np.nan); dbar = np.full(nb, np.nan); eS = np.full(nb, np.nan); ed = np.full(nb, np.nan)
    for b in range(nb):
        g = m & ok[:, b]
        if g.sum() < minn:
            continue
        y = y_all[g, b]; x1 = den[g, b]; x2 = nv[g, b]; w = 1.0 / x2

        def solve(idx):
            keep = np.ones(idx.size, bool)
            for _ in range(3):
                A = np.c_[x1[idx][keep], x2[idx][keep]]
                W = w[idx][keep]
                coef = np.linalg.lstsq(A * np.sqrt(W)[:, None], y[idx][keep] * np.sqrt(W), rcond=None)[0]
                r = (y[idx] - np.c_[x1[idx], x2[idx]] @ coef) * np.sqrt(w[idx])
                sd = 1.4826 * np.median(np.abs(r - np.median(r)))
                keep = np.abs(r - np.median(r)) < 4 * sd
            return coef
        idx0 = np.arange(g.sum())
        c0 = solve(idx0)
        bs = np.array([solve(rng.integers(0, idx0.size, idx0.size)) for _ in range(nboot)])
        S[b] = c0[0] - 1; dbar[b] = c0[1]; eS[b] = bs[:, 0].std(); ed[b] = bs[:, 1].std()
    return S, dbar, eS, ed


def groups(t, band, sample):
    sel = t['iso'] == 1 if sample == 'iso' else t['strict'] == 1
    out = {}
    sat = (t['kind'] == 'sat') & sel
    for lo, hi in SATBINS[band]:
        out[f'sat {lo}-{hi}'] = sat & (t['mag'] >= lo) & (t['mag'] < hi)
    dao = (t['kind'] == 'dao') & sel
    out['unsat'] = dao
    e0 = float(np.min(t['mag'][t['kind'] == 'dao']))
    out[f'unsat {e0:.1f}-{e0 + 1:.1f}'] = dao & (t['mag'] < e0 + 1)
    out[f'unsat {e0 + 1:.1f}-{e0 + 2:.1f}'] = dao & (t['mag'] >= e0 + 1)
    return out


def main():
    md = [open(f'{D}/summary_head.md').read(), '# Radial residual profiles (radprof): tables\n']
    md.append('Delta(r) = S in the fit sum_bin(data - bg) = (1+S) sum_bin(model) over stars (weights 1/npix, 4-sigma clipping, error = bootstrap over stars), i.e. a robust flux-weighted version of sum(data-bg-model)/sum(model); the median of the per-star ratios is given in brackets in the aggregated tables. '
              'Bins with <90 % valid pixels are dropped for that star. Radii in arcsec of the bin centre. '
              '"precap" scales the star model by flux_fit_precap/flux_fit; "final" uses the model image as written. '
              'Unsaturated stars (daophot m7 flux, daophot model image, satstar-residual image) have no precap/final distinction.\n')
    md.append('Method: satstar data = SCI of the crf (== satstar residual + model where finite); invalid = non-finite SCI/residual/model or DQ SATURATED|DO_NOT_USE. '
              'Own model = satstar model image (sat) or daophot m7 model image (unsat), used only for stars with no other source >5 % of the target flux inside 1.0" LW / 0.6" SW ("iso"); '
              '"strict" additionally requires no source >1 % of the target flux within rmax+2 px (25 px LW, 50 px SW), which keeps the model image free of neighbour models. '
              'Background = sigma-clipped median of the satstar residual image at 1.8-2.4". Unsaturated sample = dolphot-matched daophot stars (flags 0/1) between the satstar 95th-percentile faint-edge magnitude and +2 mag.\n')
    for fname, sample, field in [('radprof.png', 'iso', 'dsumb_off'), ('radprof_B_raw.png', 'iso', 'dsumb'), ('radprof_A_strict.png', 'strict', 'dsum')]:
        fig, axs = plt.subplots(4, 2, figsize=(13, 16), sharex='col')
        for ci, band in enumerate(['F150W', 'F250M']):
            for ri, (arm, final) in enumerate([('main2kf', False), ('main2kf', True), ('main2', False), ('main2', True)]):
                t, edges = load(arm, band)
                c = 0.5 * (edges[1:] + edges[:-1]) * PIX[band]
                ax = axs[ri, ci]
                G = groups(t, band, sample)
                fld = 'dsumb' if field == 'dsumb_off' else field
                offs = {}
                if field == 'dsumb_off':
                    Gi = groups(t, band, 'iso')
                    offs['sat'] = fit_offset(t, (t['kind'] == 'sat') & (t['iso'] == 1), final)[1]
                    offs['unsat'] = fit_offset(t, (t['kind'] == 'dao') & (t['iso'] == 1), final)[1]
                for (name, m), col in zip(G.items(), plt.cm.viridis(np.linspace(0, 0.9, 10))):
                    if not name.startswith('sat') and name != 'unsat':
                        continue
                    med = np.full(len(c), np.nan); err = np.full(len(c), np.nan)
                    db = offs.get('sat' if name.startswith('sat') else 'unsat')
                    for b in range(len(c)):
                        med[b], err[b], n = stack(t, edges, band, m, final, b=b, field=fld, minn=8, dbar=db)
                    if name == 'unsat':
                        ax.errorbar(c, med, err, color='k', lw=2, label=f'unsat (N={int(m.sum())})')
                    else:
                        ax.errorbar(c, med, err, color=col, label=f'{name} (N={int(m.sum())})')
                ax.axhline(0, color='gray', lw=0.5)
                ax.set_xscale('log'); ax.set_ylim(-0.6, 1.0)
                ax.set_title(f'{band} {arm} {"final" if final else "precap"} ({sample}, {"neighbours removed, offset fitted" if field == "dsumb_off" else "neighbours removed" if field == "dsumb" else "raw data"})')
                if ri == 3:
                    ax.set_xlabel('r (arcsec)')
                ax.set_ylabel('Delta(r)')
                ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(f'{D}/{fname}', dpi=110); plt.close(fig)

    # tables
    for band in ['F150W', 'F250M']:
        for arm in ['main2kf', 'main2']:
            t, edges = load(arm, band)
            sat = t['kind'] == 'sat'
            md.append(f'\n## {band} {arm}\n')
            rm = np.asarray(t['rmask'][sat], float) * PIX[band]
            md.append(f'Masked core radius of satstars (outer edge of the inner bins with <50 % valid pixels): median {np.median(rm):.3f}", 5-95 pct {np.percentile(rm, 5):.3f}-{np.percentile(rm, 95):.3f}" '
                      f'({np.median(rm) / PIX[band]:.1f} px). Satstars {int(sat.sum())}, unsaturated {int((~sat).sum())} (all frames, pooled); iso: {int(((t["iso"] == 1) & sat).sum())} / {int(((t["iso"] == 1) & ~sat).sum())}; strict: {int(((t["strict"] == 1) & sat).sum())} / {int(((t["strict"] == 1) & ~sat).sum())}.\n')
            for sample in ['iso', 'strict']:
                G = groups(t, band, sample)
                for final in [False, True]:
                    lab = 'final' if final else 'precap'
                    if not final or True:
                        md.append(f'\n### {band} {arm} sample={sample}, satstars {lab}: Delta aggregated over radius ranges (arcsec), robust stacked ratio +- bootstrap (N) [median of per-star ratios]\n')
                        md.append('| group | ' + ' | '.join(f'{a}-{b}"' for a, b in RANGES_AS) + ' |')
                        md.append('|---|' + '---|' * len(RANGES_AS))
                        for name, m in G.items():
                            cells = []
                            for lo, hi in RANGES_AS:
                                v = rangeval(t, edges, band, lo, hi, final)[m]
                                med_ = boot(v)[0]
                                mm, ee, n = stack(t, edges, band, m, final, lo, hi)
                                cells.append(f'{mm:+.3f}+-{ee:.3f} ({n}) [med {med_:+.2f}]' if n >= 5 else 'n/a')
                            md.append(f'| {name} | ' + ' | '.join(cells) + ' |')
            md.append(f'\n### {band} {arm} all other modelled sources removed from the data (iso, satstars precap), aggregated Delta\n')
            md.append('| group | ' + ' | '.join(f'{a}-{b}"' for a, b in RANGES_AS) + ' |')
            md.append('|---|' + '---|' * len(RANGES_AS))
            for name, m in groups(t, band, 'iso').items():
                cells = []
                for lo, hi in RANGES_AS:
                    mm, ee, n = stack(t, edges, band, m, False, lo, hi, field='dsumb')
                    cells.append(f'{mm:+.3f}+-{ee:.3f} ({n})' if n >= 5 else 'n/a')
                md.append(f'| {name} | ' + ' | '.join(cells) + ' |')
            for final in [False, True]:
                lab = 'final' if final else 'precap'
                Sf = {k: fit_offset(t, (t['kind'] == kk) & (t['iso'] == 1), final) for k, kk in (('sat', 'sat'), ('unsat', 'dao'))}
                md.append(f'\n### {band} {arm} offset-corrected Delta (iso, neighbours removed, satstars {lab}): per-bin pooled fit  sum(data-bg) = (1+S) model + dbar npix\n')
                md.append('| r (") | sat S | sat dbar (MJy/sr) | unsat S | unsat dbar (MJy/sr) |')
                md.append('|---|---|---|---|---|')
                c = 0.5 * (edges[1:] + edges[:-1]) * PIX[band]
                for b in range(len(c)):
                    row = []
                    for k in ('sat', 'unsat'):
                        S_, d_, eS_, ed_ = Sf[k]
                        row += [f'{S_[b]:+.3f}+-{eS_[b]:.3f}' if np.isfinite(S_[b]) else 'n/a', f'{d_[b]:+.2e}+-{ed_[b]:.1e}' if np.isfinite(d_[b]) else 'n/a']
                    md.append(f'| {c[b]:.3f} | ' + ' | '.join(row) + ' |')
                md.append(f'\n### {band} {arm} offset-corrected aggregated Delta (iso, neighbours removed, satstars {lab}), stacked +- bootstrap (N)\n')
                md.append('| group | ' + ' | '.join(f'{a}-{b}"' for a, b in RANGES_AS) + ' |')
                md.append('|---|' + '---|' * len(RANGES_AS))
                for name, m in groups(t, band, 'iso').items():
                    db = Sf['sat' if name.startswith('sat') else 'unsat'][1]
                    cells = []
                    for lo, hi in RANGES_AS:
                        mm, ee, n = stack(t, edges, band, m, final, lo, hi, field='dsumb', dbar=db)
                        cells.append(f'{mm:+.3f}+-{ee:.3f} ({n})' if n >= 5 else 'n/a')
                    md.append(f'| {name} | ' + ' | '.join(cells) + ' |')
            # per-bin profile table for iso, precap
            G = groups(t, band, 'iso')
            c = 0.5 * (edges[1:] + edges[:-1]) * PIX[band]
            for final in [False, True]:
                D_ = delta(t, final)
                md.append(f'\n### {band} {arm} per-bin Delta(r) (neighbours removed), iso, satstars {"final" if final else "precap"} (median, N>=8)\n')
                names = [n for n in G if not n.startswith('unsat ') ]
                md.append('| r (") | ' + ' | '.join(names) + ' |')
                md.append('|---|' + '---|' * len(names))
                for b in range(len(c)):
                    cells = []
                    for n_ in names:
                        mm, ee, n = stack(t, edges, band, G[n_], final, b=b, field='dsumb', nboot=100, minn=8)
                        cells.append(f'{mm:+.3f}+-{ee:.3f}' if n >= 8 else 'n/a')
                    md.append(f'| {c[b]:.3f} | ' + ' | '.join(cells) + ' |')
            # EE table
            md.append(f'\n### {band} {arm} encircled energy (strict sample unless noted); FWHM = {FW[band]} px\n')
            md.append('| group | N | model(<1F)/model(<R) | model(<2F)/model(<R) | model(<3F)/model(<R) | model(<R)/flux_fit | data(<1F)/flux_fit | data(<2F)/flux_fit | data(<3F)/flux_fit | model(<1F)/flux_fit | model(<2F)/flux_fit | model(<3F)/flux_fit | data(<R)/flux_fit |')
            md.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
            for sample in ['strict', 'iso']:
                G = groups(t, band, sample)
                for name, m in G.items():
                    if m.sum() < 5:
                        continue
                    s = t[m]
                    f = np.asarray(s['flux'], float)
                    q = lambda col: np.asarray(s[col], float)
                    cells = [np.median(q(f'ee_mod{k}') / q('ee_modR')) for k in '123']
                    cells.append(np.median(q('ee_modR') / f))
                    if s['kind'][0] == 'dao':
                        cells += [np.median(q(f'ee_dat{k}') / f) for k in '123']
                    else:
                        cells += [np.nan] * 3
                    cells += [np.median(q(f'ee_mod{k}') / f) for k in '123']
                    cells.append(np.median(q('ee_datR') / f) if s['kind'][0] == 'dao' else np.nan)
                    md.append(f'| {name} [{sample}] | {len(s)} | ' + ' | '.join('n/a' if not np.isfinite(v) else f'{v:.3f}' for v in cells) + ' |')
            # magnitude dependence
            md.append(f'\n### {band} {arm} magnitude dependence (iso, precap, neighbours removed): Delta in 0.3-0.5" and 0.5-0.8" per mag bin\n')
            G = groups(t, band, 'iso')
            md.append('| group | median mag | Delta 0.3-0.5" | Delta 0.5-0.8" | Delta 0.8-1.5" |')
            md.append('|---|---|---|---|---|')
            for name, m in G.items():
                cells = []
                for lo, hi in [(0.3, 0.5), (0.5, 0.8), (0.8, 1.5)]:
                    mm, ee, n = stack(t, edges, band, m, False, lo, hi, field='dsumb')
                    cells.append(f'{mm:+.3f}+-{ee:.3f} ({n})' if n >= 5 else 'n/a')
                md.append(f'| {name} | {np.median(t["mag"][m]) if m.sum() else np.nan:.2f} | ' + ' | '.join(cells) + ' |')
    open(f'{D}/radprof.md', 'w').write('\n'.join(md) + '\n')


main()
