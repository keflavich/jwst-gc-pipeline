"""Narrow-band colour-continuity test of the satstar flux offset at the W-band saturation edge.
Usage: nice -19 python -u nbstep.py   (reads existing files only; writes nbstep_results.json, nbstep.png, nbstep_tables.md here)"""
import sys
import json
import numpy as np
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an
from astropy.table import Table
from astropy.coordinates import SkyCoord, search_around_sky
import astropy.units as u
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/nbstep'
ECSV = '/orange/adamginsburg/jwst/wd2/catalogs/wd2_nircam_wf_mf_nf.ecsv'
PAIRS = [('200W', '212N'), ('150W', '164N'), ('182M', '187N'), ('300M', '323N'), ('250M', '323N'), ('335M', '323N'),
         ('162M', '164N'), ('277W', '323N'), ('410M', '466N'), ('410M', '405N')]
WINS = (1.0, 1.5, 2.0)
ORDERS = (1, 2)
BIN = 0.5
B = 300
SEPMAX = 150.0   # mas
ERRMAX = 0.1
rng = np.random.default_rng(42)


def mag_ok(x):
    x = an.fl(x)
    x[(x > 40) | (x < 0)] = np.nan
    return x


def isolation(sky, mags, dm=2.0, rad=0.5):
    """True if no neighbour within rad arcsec is brighter than (own mag + dm) in any of the given mag arrays."""
    i1, i2, _, _ = search_around_sky(sky, sky, rad * u.arcsec)
    keep = i1 != i2
    i1, i2 = i1[keep], i2[keep]
    iso = np.ones(len(sky), bool)
    for m in mags:
        bad = np.isfinite(m[i1]) & np.isfinite(m[i2]) & (m[i2] < m[i1] + dm)
        iso[np.unique(i1[bad])] = False
    return iso


def edge(x, rep, thr=0.1, nmin=10, w=0.25):
    """Faintest x (bin upper edge) where the replaced fraction >= thr."""
    best = np.nan
    for lo in np.arange(10, 24, w):
        s = (x >= lo) & (x < lo + w)
        if s.sum() >= nmin and rep[s].mean() >= thr:
            best = lo + w
    return best


def fit_trend(x, c, order):
    p = None
    keep = np.ones(len(x), bool)
    for _ in range(4):
        if keep.sum() < order + 3:
            return None
        p = np.polyfit(x[keep], c[keep], order)
        r = c - np.polyval(p, x)
        s = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep])))
        keep = np.abs(r) < 3 * max(s, 0.01)
    return p


def steps(xf, cf, xs, cs, edges, order):
    """Step overall and per bin for sat stars (xs, cs) relative to polynomial trend fitted on (xf, cf)."""
    p = fit_trend(xf, cf, order)
    n = len(edges) - 1
    if p is None or len(xs) == 0:
        return np.nan, np.full(n, np.nan)
    r = cs - np.polyval(p, xs)
    per = np.array([np.median(r[(xs >= edges[i]) & (xs < edges[i + 1])]) if ((xs >= edges[i]) & (xs < edges[i + 1])).sum() >= 3 else np.nan for i in range(n)])
    return np.median(r), per


def boot_steps(xf, cfs, xs, css, edges, order, nb=B):
    """cfs, css: lists of colour arrays (ours, dolphot) sharing the same stars; paired bootstrap."""
    nc = len(cfs)
    res0 = [steps(xf, cfs[k], xs, css[k], edges, order) for k in range(nc)]
    nbin = len(edges) - 1
    ov = np.full((nb, nc), np.nan)
    pb = np.full((nb, nc, nbin), np.nan)
    for b in range(nb):
        i = rng.integers(0, len(xf), len(xf))
        j = rng.integers(0, len(xs), len(xs))
        for k in range(nc):
            o, pbk = steps(xf[i], cfs[k][i], xs[j], css[k][j], edges, order)
            ov[b, k] = o
            pb[b, k] = pbk
    return res0, ov, pb


def prep_matched():
    m = Table.read(an.PATH['main2'][1])
    ok = np.asarray(m['matched'], bool) & (an.fl(m['sep_mas']) <= SEPMAX)
    e = Table.read(ECSV)
    sm = SkyCoord(m['RA'] * u.deg, m['DEC'] * u.deg)
    se = SkyCoord(e['RA'] * u.deg, e['DEC'] * u.deg)
    idx, sep, _ = sm.match_to_catalog_sky(se)
    good = sep < 0.05 * u.arcsec
    return m, ok, e, se, idx, good


def pair_data_matched(W, N, m, ok, e, se, idx, good, iso_cache):
    d = {}
    for b in (W, N):
        d['ref_' + b] = mag_ok(m['ref_' + b])
        d['our_' + b] = mag_ok(m['our_' + b])
        d['rep_' + b] = an.fl(m['our_replaced_saturated_' + b]) == 1
        d['sat_' + b] = an.fl(m['our_is_saturated_' + b]) == 1
        fl_, fe = an.fl(m['our_flux_' + b]), an.fl(m['our_flux_err_' + b])
        d['oerr_' + b] = 1.0857 * fe / np.abs(fl_)
        er = np.full(len(m), np.nan)
        er[good] = an.fl(e['ERRMAG' + b])[idx[good]]
        d['derr_' + b] = er
    key = (W, N)
    if key not in iso_cache:
        mw, mn = mag_ok(e['MAG' + W]), mag_ok(e['MAG' + N])
        iso = isolation(se, [mw, mn])
        out = np.zeros(len(m), bool)
        out[good] = iso[idx[good]]
        iso_cache[key] = out
    d['iso'] = iso_cache[key]
    d['ok'] = ok & good
    return d


def analyse(W, N, d, label='matched'):
    ok = d['ok'] & np.isfinite(d['ref_' + W]) & np.isfinite(d['ref_' + N]) & np.isfinite(d['our_' + W]) & np.isfinite(d['our_' + N])
    x = d['ref_' + N]
    rW, rN = d['rep_' + W], d['rep_' + N]
    xe_W = edge(x[ok], rW[ok])
    xe_N = edge(x[ok], rN[ok])
    res = dict(pair=f'F{W}-F{N}', W=W, N=N, edgeW=xe_W, edgeN=xe_N)
    if not np.isfinite(xe_W) or not np.isfinite(xe_N):
        res['skip'] = f'no W-saturation edge found (W replaced fraction < 0.1 everywhere; N edge {xe_N})'
        return res, None
    lo = xe_N + 0.3
    res['sat_lo'], res['sat_hi'] = lo, xe_W
    if xe_W - lo < 0.5:
        res['skip'] = f'sat range [{lo:.2f},{xe_W:.2f}] narrower than 0.5 mag (N edge {xe_N:.2f} + 0.3 versus W edge {xe_W:.2f})'
        return res, None
    q = ok & d['iso'] & (d['oerr_' + N] < ERRMAX) & (d['derr_' + N] < ERRMAX)
    sat = q & rW & (x >= lo) & (x < xe_W)
    cO = d['our_' + W] - d['our_' + N]
    cD = d['ref_' + W] - d['ref_' + N]
    edges = np.arange(lo, xe_W + 1e-6, BIN)
    if edges[-1] < xe_W - 0.2:
        edges = np.append(edges, xe_W)
    else:
        edges[-1] = xe_W
    res['edges'] = edges.tolist()
    res['nsat'] = int(sat.sum())
    res['nsat_bin'] = [int(((x >= edges[i]) & (x < edges[i + 1]) & sat).sum()) for i in range(len(edges) - 1)]
    res['results'] = {}
    plot = dict(x=x, cO=cO, cD=cD, q=q, rW=rW, sat=sat, edges=edges, xeW=xe_W, lo=lo)
    for win in WINS:
        unsat = q & ~rW & ~d['sat_' + W] & (x >= xe_W) & (x < xe_W + win) & (d['oerr_' + W] < ERRMAX) & (d['derr_' + W] < ERRMAX)
        for order in ORDERS:
            if unsat.sum() < 30:
                continue
            r0, ov, pb = boot_steps(x[unsat], [cO[unsat], cD[unsat]], x[sat], [cO[sat], cD[sat]], edges, order)
            diff_ov = ov[:, 0] - ov[:, 1]
            diff_pb = pb[:, 0] - pb[:, 1]
            res['results'][f'w{win}_o{order}'] = dict(
                nfit=int(unsat.sum()),
                ours=[float(r0[0][0]), float(np.nanstd(ov[:, 0]))], dol=[float(r0[1][0]), float(np.nanstd(ov[:, 1]))],
                diff=[float(r0[0][0] - r0[1][0]), float(np.nanstd(diff_ov))],
                ours_bin=[[float(a), float(b)] for a, b in zip(r0[0][1], np.nanstd(pb[:, 0], axis=0))],
                dol_bin=[[float(a), float(b)] for a, b in zip(r0[1][1], np.nanstd(pb[:, 1], axis=0))],
                diff_bin=[[float(a - b), float(c)] for a, b, c in zip(r0[0][1], r0[1][1], np.nanstd(diff_pb, axis=0))])
            if win == 1.5:
                plot[f'p{order}'] = (fit_trend(x[unsat], cO[unsat], order), fit_trend(x[unsat], cD[unsat], order))
    # band-offset checks (ours - dolphot, per band) for W-saturated stars versus W-unsat stars in the 1.5 mag window
    unsat = q & ~rW & ~d['sat_' + W] & (x >= xe_W) & (x < xe_W + 1.5)
    chk = {}
    for b in (W, N):
        dd = d['our_' + b] - d['ref_' + b]
        chk[b] = dict(sat=[float(np.median(dd[sat])), int(sat.sum())], unsat=[float(np.median(dd[unsat])), int(unsat.sum())])
        bs = [np.median(dd[sat][rng.integers(0, sat.sum(), sat.sum())]) - np.median(dd[unsat][rng.integers(0, unsat.sum(), unsat.sum())]) for _ in range(B)]
        chk[b]['delta'] = [chk[b]['sat'][0] - chk[b]['unsat'][0], float(np.std(bs))]
    res['bandoffsets'] = chk
    return res, plot


def ours_only(W, N, cat):
    wl, nl = W.lower(), N.lower()
    x = mag_ok(cat['mag_vega_f' + nl])
    mw = mag_ok(cat['mag_vega_f' + wl])
    rW = an.fl(cat['replaced_saturated_f' + wl]) == 1
    rN = an.fl(cat['replaced_saturated_f' + nl]) == 1
    sW = an.fl(cat['is_saturated_f' + wl]) == 1
    sky = cat['skycoord_ref']
    if not isinstance(sky, SkyCoord):
        sky = SkyCoord(sky)
    ok = np.isfinite(x) & np.isfinite(mw)
    xe_W, xe_N = edge(x[ok], rW[ok]), edge(x[ok], rN[ok])
    res = dict(edgeW=xe_W, edgeN=xe_N)
    if not np.isfinite(xe_W) or not np.isfinite(xe_N) or xe_W - (xe_N + 0.3) < 0.5:
        res['skip'] = True
        return res
    lo = xe_N + 0.3
    eN = an.fl(cat['emag_ab_f' + nl])
    eW = an.fl(cat['emag_ab_f' + wl])
    iso = isolation(sky[ok], [mw[ok], x[ok]])
    isoa = np.zeros(len(cat), bool)
    isoa[np.where(ok)[0]] = iso
    q = ok & isoa & (eN < ERRMAX)
    c = mw - x
    sat = q & rW & (x >= lo) & (x < xe_W)
    edges = np.array([lo, xe_W])
    res['nsat'] = int(sat.sum())
    for win in WINS:
        unsat = q & ~rW & ~sW & (x >= xe_W) & (x < xe_W + win) & (eW < ERRMAX)
        for order in ORDERS:
            if unsat.sum() < 30:
                continue
            r0, ov, _ = boot_steps(x[unsat], [c[unsat]], x[sat], [c[sat]], edges, order, nb=150)
            res[f'w{win}_o{order}'] = [float(r0[0][0]), float(np.nanstd(ov[:, 0])), int(unsat.sum())]
    return res


def fmt(v, dig=3):
    return '-' if v is None or not np.isfinite(v) else f'{v:+.{dig}f}'


def main():
    m, ok, e, se, idx, good = prep_matched()
    iso_cache = {}
    allres, plots = [], {}
    for W, N in PAIRS:
        d = pair_data_matched(W, N, m, ok, e, se, idx, good, iso_cache)
        res, plot = analyse(W, N, d)
        allres.append(res)
        plots[(W, N)] = plot
        print(res['pair'], {k: v for k, v in res.items() if k in ('edgeW', 'edgeN', 'sat_lo', 'sat_hi', 'nsat', 'skip')}, flush=True)
    cat = Table.read(an.PATH['main2'][0])
    for r in allres:
        if 'skip' not in r:
            r['ours_unmatched_allowed'] = ours_only(r['W'], r['N'], cat)
            print(r['pair'], 'ours-only', r['ours_unmatched_allowed'], flush=True)
    json.dump(allres, open(f'{OUT}/nbstep_results.json', 'w'), indent=1)
    # tables
    L = []
    L.append('| pair | W edge (m_N) | N edge (m_N) | sat range | N sat (matched, quality) | status |')
    L.append('|---|---|---|---|---|---|')
    for r in allres:
        if 'skip' in r:
            L.append(f"| {r['pair']} | {fmt(r['edgeW'], 2)[1:] if np.isfinite(r['edgeW']) else '-'} | {r['edgeN']:.2f} | - | - | skipped: {r['skip']} |")
        else:
            L.append(f"| {r['pair']} | {r['edgeW']:.2f} | {r['edgeN']:.2f} | {r['sat_lo']:.2f}-{r['sat_hi']:.2f} | {r['nsat']} | used |")
    L.append('')
    L.append('Overall step (median of c - trend over W-saturated stars), mag; negative = W reads bright. Entries: value +- bootstrap sigma.')
    L.append('| pair | window | order | N fit | ours | dolphot | ours - dolphot |')
    L.append('|---|---|---|---|---|---|---|')
    for r in allres:
        if 'skip' in r:
            continue
        for k, v in r['results'].items():
            w, o = k.split('_')
            L.append(f"| {r['pair']} | {w[1:]} | {o[1:]} | {v['nfit']} | {v['ours'][0]:+.3f} +- {v['ours'][1]:.3f} | {v['dol'][0]:+.3f} +- {v['dol'][1]:.3f} | {v['diff'][0]:+.3f} +- {v['diff'][1]:.3f} |")
    L.append('')
    L.append('Per-bin steps (window 1.5 mag, linear and quadratic): m_N bin, N sat, ours, dolphot, ours - dolphot.')
    L.append('| pair | m_N bin | N | ours lin | dol lin | diff lin | ours quad | dol quad | diff quad |')
    L.append('|---|---|---|---|---|---|---|---|---|')
    for r in allres:
        if 'skip' in r:
            continue
        a, b = r['results'].get('w1.5_o1'), r['results'].get('w1.5_o2')
        for i in range(len(r['edges']) - 1):
            def g(v, key):
                return '-' if v is None or not np.isfinite(v[key][i][0]) else f'{v[key][i][0]:+.3f}+-{v[key][i][1]:.3f}'
            L.append(f"| {r['pair']} | {r['edges'][i]:.1f}-{r['edges'][i + 1]:.1f} | {r['nsat_bin'][i]} | {g(a, 'ours_bin')} | {g(a, 'dol_bin')} | {g(a, 'diff_bin')} | {g(b, 'ours_bin')} | {g(b, 'dol_bin')} | {g(b, 'diff_bin')} |")
    L.append('')
    L.append('Band offsets (ours - dolphot, raw, no zero point): median for W-saturated stars in the sat range, median for W-unsat stars in the 1.5 mag fit window, and the difference.')
    L.append('| pair | band | sat | N | unsat | N | delta |')
    L.append('|---|---|---|---|---|---|---|')
    for r in allres:
        if 'skip' in r:
            continue
        for b, v in r['bandoffsets'].items():
            L.append(f"| {r['pair']} | F{b} | {v['sat'][0]:+.3f} | {v['sat'][1]} | {v['unsat'][0]:+.3f} | {v['unsat'][1]} | {v['delta'][0]:+.3f} +- {v['delta'][1]:.3f} |")
    L.append('')
    L.append('Ours, unmatched allowed (all catalog rows, m_N from our catalog), steps +- sigma [N fit stars]:')
    L.append('| pair | edges W/N | N sat | w1.0 lin | w1.5 lin | w2.0 lin | w1.5 quad |')
    L.append('|---|---|---|---|---|---|---|')
    for r in allres:
        o = r.get('ours_unmatched_allowed')
        if not o or o.get('skip'):
            continue
        def h(k):
            return '-' if k not in o else f'{o[k][0]:+.3f}+-{o[k][1]:.3f} [{o[k][2]}]'
        L.append(f"| {r['pair']} | {o['edgeW']:.2f}/{o['edgeN']:.2f} | {o['nsat']} | {h('w1.0_o1')} | {h('w1.5_o1')} | {h('w2.0_o1')} | {h('w1.5_o2')} |")
    open(f'{OUT}/nbstep_tables.md', 'w').write('\n'.join(L))
    print('\n'.join(L))
    # figure
    used = [r for r in allres if 'skip' not in r]
    fig, axs = plt.subplots(len(used), 3, figsize=(15, 3.3 * len(used)), squeeze=False)
    for row, r in enumerate(used):
        W, N = r['W'], r['N']
        p = plots[(W, N)]
        for col, (nm, c) in enumerate((('ours', p['cO']), ('dolphot', p['cD']))):
            ax = axs[row, col]
            q = p['q']
            u_ = q & ~p['rW']
            s_ = q & p['rW']
            ax.plot(p['x'][u_], c[u_], '.', ms=2, color='tab:blue', alpha=0.4, rasterized=True)
            ax.plot(p['x'][s_], c[s_], '.', ms=3, color='tab:red', alpha=0.6, rasterized=True)
            xs = np.arange(p['lo'] - 0.5, p['xeW'] + 3.0, 0.25)
            for xc in xs:
                sel = q & (p['x'] >= xc) & (p['x'] < xc + 0.25)
                if sel.sum() >= 5:
                    ax.plot(xc + 0.125, np.median(c[sel]), 'ks', ms=4)
            xx = np.linspace(p['lo'], p['xeW'] + 1.5, 50)
            for o, ls in ((1, '-'), (2, '--')):
                pp = p.get(f'p{o}')
                if pp and pp[col] is not None:
                    ax.plot(xx, np.polyval(pp[col], xx), 'g', ls=ls, lw=1.5, label=f'trend order {o}')
            ax.axvline(p['xeW'], color='gray', ls=':')
            ax.axvline(r['edgeN'] + 0.3, color='gray', ls=':')
            med = np.median(c[q & ~p['rW'] & (p['x'] > p['xeW']) & (p['x'] < p['xeW'] + 1.5)])
            ax.set_ylim(med - 0.6, med + 0.6)
            ax.set_xlim(r['edgeN'] - 0.3, p['xeW'] + 3)
            ax.set_xlabel(f'F{N} (dolphot)')
            ax.set_ylabel(f'F{W} - F{N}')
            ax.set_title(f'{r["pair"]} {nm}')
            if row == 0 and col == 0:
                ax.legend(fontsize=7)
        ax = axs[row, 2]
        v = r['results'].get('w1.5_o1')
        if v:
            ed = np.array(r['edges'])
            xc = 0.5 * (ed[1:] + ed[:-1])
            for key, colr, off in (('ours_bin', 'tab:red', -0.03), ('dol_bin', 'tab:blue', 0.03), ('diff_bin', 'k', 0.0)):
                a = np.array(v[key])
                ax.errorbar(xc + off, a[:, 0], a[:, 1], fmt='o', color=colr, label=key.replace('_bin', ''), capsize=2)
            ax.axhline(0, color='gray', lw=0.5)
            ax.set_xlabel(f'F{N} (dolphot)')
            ax.set_ylabel('step (mag), 1.5 mag lin window')
            ax.set_title(f'{r["pair"]} step vs m_N')
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f'{OUT}/nbstep.png', dpi=80)


if __name__ == '__main__':
    main()
