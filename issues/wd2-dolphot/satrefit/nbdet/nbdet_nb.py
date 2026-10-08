"""Narrow-band colour-continuity step (nbstep) split by SW detector group (nrcb1 / nrcb3 / other SW).
Copies the selection and statistics of ../nbstep/nbstep.py (imported as nbstep_base, a verbatim copy) and adds the detector split.
Trend: (A) fitted once over all groups, applied per group; (B) fitted per group. Paired bootstrap (ours and dolphot share indices).
Usage: nice -19 python -u nbdet_nb.py   -> nbdet_nb_results.json, nbdet_nb_tables.md"""
import json
import numpy as np
import nbstep_base as nb
an = nb.an

PAIRS = [('150W', '164N'), ('162M', '164N'), ('182M', '187N')]
GROUPS = ['nrcb1', 'nrcb3', 'other']
B = 300
BIN = 0.5
WINS = (1.0, 1.5, 2.0)
rng = np.random.default_rng(7)
G = np.load('groups.npz')


def group_of(W):
    foot = G['foot_F' + W]
    g = np.full(len(foot), -1)
    g[foot == 'nrcb1'] = 0
    g[foot == 'nrcb3'] = 1
    g[(foot != '') & (foot != 'nrcb1') & (foot != 'nrcb3')] = 2
    return g


def binned(r, xs, edges):
    n = len(edges) - 1
    out = np.full(n, np.nan)
    for i in range(n):
        s = (xs >= edges[i]) & (xs < edges[i + 1])
        if s.sum() >= 3:
            out[i] = np.median(r[s])
    return out


def resid(p, xs, cs):
    return np.full(len(xs), np.nan) if p is None else cs - np.polyval(p, xs)


def stats(r, xs, edges):
    if len(r) == 0 or not np.isfinite(r).any():
        return np.nan, np.full(len(edges) - 1, np.nan)
    return np.median(r), binned(r, xs, edges)


def run_pair(W, N, m, ok0, e, se, idx, good, iso_cache):
    d = nb.pair_data_matched(W, N, m, ok0, e, se, idx, good, iso_cache)
    grp = group_of(W)
    ok = d['ok'] & np.isfinite(d['ref_' + W]) & np.isfinite(d['ref_' + N]) & np.isfinite(d['our_' + W]) & np.isfinite(d['our_' + N])
    x = d['ref_' + N]
    rW, rN = d['rep_' + W], d['rep_' + N]
    xeW, xeN = nb.edge(x[ok], rW[ok]), nb.edge(x[ok], rN[ok])   # edges from all stars (unchanged from nbstep)
    lo = xeN + 0.3
    q = ok & d['iso'] & (d['oerr_' + N] < nb.ERRMAX) & (d['derr_' + N] < nb.ERRMAX) & (grp >= 0)
    sat = q & rW & (x >= lo) & (x < xeW)
    cO = d['our_' + W] - d['our_' + N]
    cD = d['ref_' + W] - d['ref_' + N]
    edges = np.arange(lo, xeW + 1e-6, BIN)
    if edges[-1] < xeW - 0.2:
        edges = np.append(edges, xeW)
    else:
        edges[-1] = xeW
    res = dict(pair=f'F{W}-F{N}', W=W, N=N, edgeW=xeW, edgeN=xeN, lo=lo, edges=edges.tolist(), results={})
    res['nsat'] = [int((sat & (grp == k)).sum()) for k in range(3)]
    res['nsat_bin'] = [[int((sat & (grp == k) & (x >= edges[i]) & (x < edges[i + 1])).sum()) for i in range(len(edges) - 1)] for k in range(3)]
    plot = dict(x=x, cO=cO, cD=cD, q=q, rW=rW, sat=sat, grp=grp, edges=edges, xeW=xeW, lo=lo)
    for win in WINS:
        unsat = q & ~rW & ~d['sat_' + W] & (x >= xeW) & (x < xeW + win) & (d['oerr_' + W] < nb.ERRMAX) & (d['derr_' + W] < nb.ERRMAX)
        order = 1
        nu = [int((unsat & (grp == k)).sum()) for k in range(3)]
        iu = np.where(unsat)[0]
        isg = [np.where(sat & (grp == k))[0] for k in range(3)]
        iug = [np.where(unsat & (grp == k))[0] for k in range(3)]
        cats = (cO, cD)
        # point estimates
        def one(fit_all, sat_g, fit_g):
            """returns arrays [cat][mode]: mode A (global trend), mode B (group trend); each (overall[3], bins[3,nb])"""
            out = {}
            for c, cc in enumerate(cats):
                pA = nb.fit_trend(x[fit_all], cc[fit_all], order)
                for k in range(3):
                    s = sat_g[k]
                    out[(c, 'A', k)] = stats(resid(pA, x[s], cc[s]), x[s], edges)
                    pB = nb.fit_trend(x[fit_g[k]], cc[fit_g[k]], order) if len(fit_g[k]) >= 15 else None
                    out[(c, 'B', k)] = stats(resid(pB, x[s], cc[s]), x[s], edges)
            return out
        r0 = one(iu, isg, iug)
        bo = {key: [] for key in r0}
        for _ in range(B):
            fa = iu[rng.integers(0, len(iu), len(iu))]
            sg = [g[rng.integers(0, len(g), len(g))] if len(g) else g for g in isg]
            fg = [g[rng.integers(0, len(g), len(g))] if len(g) else g for g in iug]
            rb = one(fa, sg, fg)
            for key in bo:
                bo[key].append((rb[key][0], rb[key][1]))
        def pack(c, mode, k):
            ov = np.array([a for a, _ in bo[(c, mode, k)]])
            pb = np.array([b for _, b in bo[(c, mode, k)]])
            return ov, pb
        for mode in ('A', 'B'):
            for k in range(3):
                oO, bO = pack(0, mode, k)
                oD, bD = pack(1, mode, k)
                a0, a1 = r0[(0, mode, k)], r0[(1, mode, k)]
                res['results'][f'w{win}_{mode}_{GROUPS[k]}'] = dict(
                    nfit=nu[k] if mode == 'B' else int(unsat.sum()), nsat=len(isg[k]),
                    ours=[float(a0[0]), float(np.nanstd(oO))], dol=[float(a1[0]), float(np.nanstd(oD))],
                    diff=[float(a0[0] - a1[0]), float(np.nanstd(oO - oD))],
                    ours_bin=[[float(a), float(b)] for a, b in zip(a0[1], np.nanstd(bO, axis=0))],
                    dol_bin=[[float(a), float(b)] for a, b in zip(a1[1], np.nanstd(bD, axis=0))],
                    diff_bin=[[float(a - b), float(c)] for a, b, c in zip(a0[1], a1[1], np.nanstd(bO - bD, axis=0))])
        if win == 1.5:
            plot['unsat'] = unsat
            plot['pA'] = (nb.fit_trend(x[unsat], cO[unsat], 1), nb.fit_trend(x[unsat], cD[unsat], 1))
        # control: ours - dolphot per band, sat vs unsat window, per group
        if win == 1.5:
            chk = {}
            for k in range(3):
                for b in (W, N):
                    dd = d['our_' + b] - d['ref_' + b]
                    s_, u_ = dd[isg[k]], dd[iug[k]]
                    if len(s_) < 3 or len(u_) < 3:
                        continue
                    bs = [np.median(s_[rng.integers(0, len(s_), len(s_))]) for _ in range(B)]
                    bu = [np.median(u_[rng.integers(0, len(u_), len(u_))]) for _ in range(B)]
                    chk[f'{GROUPS[k]}_F{b}'] = dict(sat=[float(np.median(s_)), float(np.std(bs)), len(s_)], unsat=[float(np.median(u_)), float(np.std(bu)), len(u_)],
                                                    delta=[float(np.median(s_) - np.median(u_)), float(np.std(np.array(bs) - np.array(bu)))])
            res['control'] = chk
    return res, plot


def f(v, dig=3):
    return '-' if v is None or not np.isfinite(v) else f'{v:+.{dig}f}'


def main():
    m, ok0, e, se, idx, good = nb.prep_matched()
    iso_cache = {}
    allres, plots = [], {}
    for W, N in PAIRS:
        res, plot = run_pair(W, N, m, ok0, e, se, idx, good, iso_cache)
        allres.append(res)
        plots[(W, N)] = plot
        print(res['pair'], res['edgeW'], res['edgeN'], res['nsat'], flush=True)
    json.dump(allres, open('nbdet_nb_results.json', 'w'), indent=1)
    L = ['Edges and sample sizes (W edge, N edge in m_N; N sat per group nrcb1/nrcb3/other).', '', '| pair | W edge | N edge | sat range | N sat b1 | N sat b3 | N sat other |', '|---|---|---|---|---|---|---|']
    for r in allres:
        L.append(f"| {r['pair']} | {r['edgeW']:.2f} | {r['edgeN']:.2f} | {r['lo']:.2f}-{r['edgeW']:.2f} | {r['nsat'][0]} | {r['nsat'][1]} | {r['nsat'][2]} |")
    L += ['', 'Overall step (median of c - trend over W-saturated stars in the group), mag, +- paired-bootstrap sigma. Negative = W reads bright relative to the unsaturated trend. Trend A = linear fit over all groups; trend B = linear fit per group (N fit = unsaturated stars of that group).', '',
          '| pair | window | trend | group | N sat | N fit | ours | dolphot | ours - dolphot |', '|---|---|---|---|---|---|---|---|---|']
    for r in allres:
        for key, v in r['results'].items():
            w, mode, g = key.split('_')
            L.append(f"| {r['pair']} | {w[1:]} | {mode} | {g} | {v['nsat']} | {v['nfit']} | {f(v['ours'][0])} +- {v['ours'][1]:.3f} | {f(v['dol'][0])} +- {v['dol'][1]:.3f} | {f(v['diff'][0])} +- {v['diff'][1]:.3f} |")
    L += ['', 'Per-bin steps (window 1.5, trend A and B): m_N bin, N sat, ours, dolphot, ours - dolphot.', '',
          '| pair | group | m_N bin | N sat | ours A | dol A | diff A | ours B | dol B | diff B |', '|---|---|---|---|---|---|---|---|---|---|']
    for r in allres:
        for k, g in enumerate(GROUPS):
            A = r['results'][f'w1.5_A_{g}']
            Bm = r['results'][f'w1.5_B_{g}']
            for i in range(len(r['edges']) - 1):
                def h(v, key):
                    a = v[key][i]
                    return '-' if not np.isfinite(a[0]) else f'{a[0]:+.3f}+-{a[1]:.3f}'
                L.append(f"| {r['pair']} | {g} | {r['edges'][i]:.2f}-{r['edges'][i + 1]:.2f} | {r['nsat_bin'][k][i]} | {h(A, 'ours_bin')} | {h(A, 'dol_bin')} | {h(A, 'diff_bin')} | {h(Bm, 'ours_bin')} | {h(Bm, 'dol_bin')} | {h(Bm, 'diff_bin')} |")
    L += ['', 'Control: median (ours - dolphot) per band, for W-saturated stars in the sat range versus W-unsaturated stars in the 1.5 mag window, per group; delta = sat - unsat.', '',
          '| pair | group_band | sat (med +- err, N) | unsat (med +- err, N) | delta |', '|---|---|---|---|---|']
    for r in allres:
        for key, v in r['control'].items():
            L.append(f"| {r['pair']} | {key} | {f(v['sat'][0])} +- {v['sat'][1]:.3f} ({v['sat'][2]}) | {f(v['unsat'][0])} +- {v['unsat'][1]:.3f} ({v['unsat'][2]}) | {f(v['delta'][0])} +- {v['delta'][1]:.3f} |")
    open('nbdet_nb_tables.md', 'w').write('\n'.join(L))
    print('\n'.join(L))
    np.save('plots_nb.npy', np.array([plots, allres], dtype=object), allow_pickle=True)


if __name__ == '__main__':
    main()
