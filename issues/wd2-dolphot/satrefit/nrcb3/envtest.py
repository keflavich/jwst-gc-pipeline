"""Tasks A (fallback pair test) and B (environment) from the rows_<band>.npz tables built by build.py.
Usage: nice -19 python -u envtest.py -> env_results.json, env_tables.md"""
import json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

OUT = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/nrcb3'
BANDS = ['F150W', 'F162M', 'F182M', 'F200W']
rng = np.random.default_rng(7)
NB = 400
MINN = 3


def star_table(z, pre, key, dmcol):
    df = pd.DataFrame({'key': z[pre + key], 'det': z[pre + 'det'], 'dm': z[pre + dmcol], 'ref': z[pre + 'ref'], 'n1': z[pre + 'n1'], 'n2': z[pre + 'n2'],
                       'dbr': z[pre + 'dbr'], 'abkg': z[pre + 'abkg'], 'edge': z[pre + 'edge'], 'x': z[pre + 'x'], 'y': z[pre + 'y']})
    if pre == 'S_':
        df['lbkg'] = z['S_lbkg']
        df['ra'] = z['S_ra']
        df['dec'] = z['S_dec']
    g = df.groupby(['key', 'det'], sort=False)
    st = g.median().reset_index()
    st['nrow'] = g.size().values
    return st


def strat_diff(st, cols, nmin=MINN, boot=True):
    """weighted mean of (median dm nrcb3 - median dm nrcb1) over cells defined by `cols`; cluster bootstrap over stars."""
    s = st[st.det.isin(['nrcb1', 'nrcb3'])].dropna(subset=cols + ['dm'])

    def one(d):
        g = d.groupby(cols + ['det']).dm.agg(['median', 'size']).unstack('det')
        if ('median', 'nrcb1') not in g.columns or ('median', 'nrcb3') not in g.columns:
            return np.nan, 0, 0
        n1, n3 = g[('size', 'nrcb1')], g[('size', 'nrcb3')]
        ok = (n1 >= nmin) & (n3 >= nmin)
        if ok.sum() == 0:
            return np.nan, 0, 0
        w = 1.0 / (1.0 / n1[ok] + 1.0 / n3[ok])
        dif = g[('median', 'nrcb3')][ok] - g[('median', 'nrcb1')][ok]
        return float((w * dif).sum() / w.sum()), int(n1[ok].sum()), int(n3[ok].sum())
    est, n1, n3 = one(s)
    err = np.nan
    if boot and np.isfinite(est):
        b1, b3 = s[s.det == 'nrcb1'], s[s.det == 'nrcb3']
        vals = []
        for _ in range(NB):
            r = pd.concat([b1.iloc[rng.integers(0, len(b1), len(b1))], b3.iloc[rng.integers(0, len(b3), len(b3))]])
            vals.append(one(r)[0])
        err = float(np.nanstd(vals))
    return est, err, n1, n3


def qbins(v, nq):
    e = np.unique(np.nanquantile(v, np.linspace(0, 1, nq + 1)))
    e[0] -= 1e-9
    e[-1] += 1e-9
    return e


def regress(st, cols):
    """OLS dm ~ b3 + cols with star-cluster bootstrap on the b3 coefficient (cols standardised)."""
    s = st[st.det.isin(['nrcb1', 'nrcb3'])].dropna(subset=cols + ['dm']).copy()
    X = np.column_stack([np.ones(len(s)), (s.det == 'nrcb3').values.astype(float)] + [((s[c] - s[c].mean()) / s[c].std()).values for c in cols])
    y = s.dm.values

    def fit(idx):
        return np.linalg.lstsq(X[idx], y[idx], rcond=None)[0]
    b = fit(np.arange(len(s)))
    bs = np.array([fit(rng.integers(0, len(s), len(s))) for _ in range(NB)])
    return b, bs.std(axis=0), len(s)


def pair_test(st, rmax, dmag=0.25, envmatch=False):
    s = st[st.det.isin(['nrcb1', 'nrcb3'])].dropna(subset=['dm', 'ra', 'dec', 'ref']).reset_index(drop=True)
    ra0, dec0 = s.ra.median(), s.dec.median()
    xy = np.column_stack([(s.ra - ra0) * np.cos(np.deg2rad(dec0)) * 3600, (s.dec - dec0) * 3600])
    i1 = np.where(s.det == 'nrcb1')[0]
    i3 = np.where(s.det == 'nrcb3')[0]
    tree = cKDTree(xy[i1])
    diffs, sep = [], []
    for k in i3:
        cand = i1[tree.query_ball_point(xy[k], rmax)]
        if len(cand) == 0:
            continue
        ok = np.abs(s.ref.values[cand] - s.ref.values[k]) < dmag
        if envmatch:
            n2k = s.n2.values[k]
            ok &= np.abs(s.n2.values[cand] - n2k) <= max(2, 0.3 * n2k)
            ok &= np.abs(np.log10(s.abkg.values[cand] / s.abkg.values[k])) < 0.15
        cand = cand[ok]
        if len(cand) == 0:
            continue
        dd = np.hypot(*(xy[cand] - xy[k]).T)
        j = cand[np.argmin(dd)]
        diffs.append(s.dm.values[k] - s.dm.values[j])
        sep.append(dd.min())
    diffs = np.array(diffs)
    if len(diffs) < 3:
        return np.nan, np.nan, len(diffs), np.nan
    bs = [np.median(diffs[rng.integers(0, len(diffs), len(diffs))]) for _ in range(NB)]
    return float(np.median(diffs)), float(np.std(bs)), len(diffs), float(np.median(sep))


def main():
    res, L = {}, []
    L.append('# nrcb3 tables (full)\n\nStar-level medians (median over the rows of a star; each star lies on one detector in all of its exposures). dm = m_row + c_b - m_dolphot - ZP (final flux). Errors: bootstrap over stars.\n')
    for band in BANDS:
        z = np.load(f'{OUT}/rows_{band}.npz', allow_pickle=True)
        S = star_table(z, 'S_', 'lab', 'final')
        Dc = star_table(z, 'D_', 'starm', 'dm')
        r = {}
        # task A: count stars with rows on more than one detector
        nd = pd.DataFrame({'lab': z['S_lab'], 'det': z['S_det']}).groupby('lab').det.nunique()
        r['A_multi_det_stars'] = int((nd > 1).sum())
        r['A_n_stars'] = int(len(nd))
        sp = pd.DataFrame({'lab': z['S_lab'], 'ra': z['S_ra'], 'dec': z['S_dec']}).groupby('lab').agg(lambda v: np.ptp(v))
        r['A_max_dither_arcsec'] = float(np.nanmax(np.hypot(sp.ra * np.cos(np.deg2rad(-57.75)), sp.dec) * 3600))
        r['A_median_dither_arcsec'] = float(np.nanmedian(np.hypot(sp.ra * np.cos(np.deg2rad(-57.75)), sp.dec) * 3600))
        # fallback pair tests
        r['pairs'] = {}
        for rmax in (10, 20, 40):
            for env in (False, True):
                r['pairs'][f'R{rmax}_{"env" if env else "mag"}'] = pair_test(S, rmax, envmatch=env)
        # B: per detector medians
        st13 = S[S.det.isin(['nrcb1', 'nrcb3'])]
        r['star_median'] = {d: (float(st13[st13.det == d].dm.median()), int((st13.det == d).sum())) for d in ('nrcb1', 'nrcb3')}
        r['raw_diff'] = strat_diff(S.assign(one=1), ['one'])
        r['mag_strat'] = strat_diff(S.assign(mb=np.floor(S.ref)), ['mb'])
        # bins
        e_n2 = qbins(st13.n2, 4)
        e_n1 = qbins(st13.n1, 4)
        e_ab = qbins(st13.abkg, 4)
        e_lb = qbins(st13.lbkg, 4)
        e_ed = np.array([-1, 50, 150, 400, 3000])
        e_db = np.array([0, 0.3, 0.6, 1.0, 1.5, 3.0, 99])
        S = S.assign(mb=np.floor(S.ref), n2b=pd.cut(S.n2, e_n2, labels=False), n1b=pd.cut(S.n1, e_n1, labels=False), abb=pd.cut(S.abkg, e_ab, labels=False),
                     lbb=pd.cut(S.lbkg, e_lb, labels=False), edb=pd.cut(S.edge, e_ed, labels=False), dbb=pd.cut(S.dbr.fillna(99), e_db, labels=False))
        r['edges'] = dict(n2=e_n2.tolist(), n1=e_n1.tolist(), abkg=e_ab.tolist(), lbkg=e_lb.tolist(), edge=e_ed.tolist(), dbr=e_db.tolist())
        r['bins'] = {}
        L.append(f'\n## {band}\n')
        L.append(f"Stars with S rows on more than one detector: {r['A_multi_det_stars']} of {r['A_n_stars']}; per-star dither extent median {r['A_median_dither_arcsec']:.3f}\", max {r['A_max_dither_arcsec']:.2f}\".")
        L.append(f"nrcb1 median {r['star_median']['nrcb1'][0]:+.3f} (N {r['star_median']['nrcb1'][1]}), nrcb3 {r['star_median']['nrcb3'][0]:+.3f} (N {r['star_median']['nrcb3'][1]}); raw nrcb3-nrcb1 {r['raw_diff'][0]:+.3f} +/- {r['raw_diff'][1]:.3f}; matched in 1-mag bins {r['mag_strat'][0]:+.3f} +/- {r['mag_strat'][1]:.3f}\n")
        L.append('Nearest-neighbour pair test across the nrcb1/nrcb3 boundary (each nrcb3 star paired with nearest nrcb1 star of |dm_dolphot| < 0.25 mag): median(dm b3 - dm b1), error, N pairs, median separation (arcsec)\n')
        L.append('| max sep | mag-matched | mag + density + bkg matched |\n|---|---|---|')
        for rmax in (10, 20, 40):
            a, b = r['pairs'][f'R{rmax}_mag'], r['pairs'][f'R{rmax}_env']
            L.append(f'| {rmax}" | {a[0]:+.3f} +/- {a[1]:.3f} (N {a[2]}, sep {a[3]:.1f}) | {b[0]:+.3f} +/- {b[1]:.3f} (N {b[2]}, sep {b[3]:.1f}) |')
        for var, bcol, edges, ttl in (('n2', 'n2b', e_n2, 'neighbours within 2" with mag < own+3'), ('n1', 'n1b', e_n1, 'neighbours within 1" with mag < own+3'),
                                      ('abkg', 'abb', e_ab, 'annulus background 1.5-2.5" (MJy/sr)'), ('lbkg', 'lbb', e_lb, 'stored satstar local_bkg (MJy/sr)'),
                                      ('edge', 'edb', e_ed, 'distance to detector edge (px)'), ('dbr', 'dbb', e_db, 'nearest brighter neighbour (arcsec)')):
            L.append(f'\n**{band}: {ttl}**\n')
            L.append('| bin | nrcb1 median (N) | nrcb3 median (N) | raw b3-b1 | mag-matched b3-b1 (1-mag strata) |\n|---|---|---|---|---|')
            r['bins'][var] = []
            for k in range(len(edges) - 1):
                sub = S[S[bcol] == k]
                row = [f'{edges[k]:.3g}-{edges[k + 1]:.3g}']
                m1, m3 = sub[sub.det == 'nrcb1'], sub[sub.det == 'nrcb3']
                c1 = f'{m1.dm.median():+.3f} ({len(m1)})' if len(m1) >= 3 else f'- ({len(m1)})'
                c3 = f'{m3.dm.median():+.3f} ({len(m3)})' if len(m3) >= 3 else f'- ({len(m3)})'
                rd = (m3.dm.median() - m1.dm.median()) if len(m1) >= 3 and len(m3) >= 3 else np.nan
                ms = strat_diff(sub, ['mb'])
                r['bins'][var].append(dict(lo=float(edges[k]), hi=float(edges[k + 1]), n1=len(m1), n3=len(m3), m1=float(m1.dm.median()) if len(m1) else np.nan,
                                           m3=float(m3.dm.median()) if len(m3) else np.nan, raw=float(rd), matched=ms[0], matched_err=ms[1]))
                L.append(f'| {row[0]} | {c1} | {c3} | {rd:+.3f} | {ms[0]:+.3f} +/- {ms[1]:.3f} |')
        # stratified over combinations
        r['strat'] = {}
        for name, cols in (('mag+n2', ['mb', 'n2b']), ('mag+abkg', ['mb', 'abb']), ('mag+lbkg', ['mb', 'lbb']), ('mag+n2+abkg', ['mb', 'n2b', 'abb']),
                           ('mag+n2+abkg+edge', ['mb', 'n2b', 'abb', 'edb']), ('mag+n1+dbr', ['mb', 'n1b', 'dbb'])):
            r['strat'][name] = strat_diff(S, cols)
        L.append(f'\n**{band}: stratified nrcb3 - nrcb1 (weighted mean over cells with >= {MINN} stars per detector)**\n')
        L.append('| strata | b3 - b1 | N b1 / N b3 used |\n|---|---|---|')
        for k, v in r['strat'].items():
            L.append(f'| {k} | {v[0]:+.3f} +/- {v[1]:.3f} | {v[2]} / {v[3]} |')
        # regression
        r['reg'] = {}
        for name, cols in (('mag', ['ref']), ('mag+logn2', ['ref', 'logn2']), ('mag+logn2+logbkg', ['ref', 'logn2', 'logbkg']),
                           ('mag+logn2+logbkg+edge+dbr', ['ref', 'logn2', 'logbkg', 'edge', 'ldbr'])):
            Sx = S.assign(logn2=np.log10(1 + S.n2), logbkg=np.log10(S.abkg.clip(lower=1e-3)), ldbr=np.log10(S.dbr.fillna(30).clip(lower=0.03)))
            b, e, n = regress(Sx, cols)
            r['reg'][name] = dict(b3=float(b[1]), b3_err=float(e[1]), coefs=dict(zip(['const', 'b3'] + cols, b.tolist())), errs=dict(zip(['const', 'b3'] + cols, e.tolist())), n=n)
        L.append(f'\n**{band}: OLS on star medians, dm ~ b3 + standardised covariates**\n')
        L.append('| model | b3 coefficient | other coefficients (per 1 sd) | N |\n|---|---|---|---|')
        for k, v in r['reg'].items():
            oth = ', '.join(f'{c} {v["coefs"][c]:+.3f}+/-{v["errs"][c]:.3f}' for c in v['coefs'] if c not in ('const', 'b3'))
            L.append(f'| {k} | {v["b3"]:+.3f} +/- {v["b3_err"]:.3f} | {oth} | {v["n"]} |')
        # daophot control
        Dc = Dc.assign(mb=np.floor(Dc.ref))
        d13 = Dc[Dc.det.isin(['nrcb1', 'nrcb3'])]
        Dc = Dc.assign(n2b=pd.cut(Dc.n2, e_n2, labels=False), abb=pd.cut(Dc.abkg, e_ab, labels=False), edb=pd.cut(Dc.edge, e_ed, labels=False))
        rc = {'median': {d: (float(d13[d13.det == d].dm.median()), int((d13.det == d).sum())) for d in ('nrcb1', 'nrcb3')}}
        rc['raw_diff'] = strat_diff(Dc.assign(one=1), ['one'])
        rc['mag_strat'] = strat_diff(Dc, ['mb'])
        rc['strat'] = {name: strat_diff(Dc, cols) for name, cols in (('mag+n2', ['mb', 'n2b']), ('mag+abkg', ['mb', 'abb']), ('mag+n2+abkg', ['mb', 'n2b', 'abb']))}
        rc['bins'] = {}
        L.append(f'\n**{band}: daophot control (unsaturated stars in the ZP window, star medians over frames)**\n')
        L.append(f"nrcb1 {rc['median']['nrcb1'][0]:+.3f} (N {rc['median']['nrcb1'][1]}), nrcb3 {rc['median']['nrcb3'][0]:+.3f} (N {rc['median']['nrcb3'][1]}); raw b3-b1 {rc['raw_diff'][0]:+.3f} +/- {rc['raw_diff'][1]:.3f}; mag-matched {rc['mag_strat'][0]:+.3f} +/- {rc['mag_strat'][1]:.3f}; " +
                 '; '.join(f'{k} {v[0]:+.3f} +/- {v[1]:.3f}' for k, v in rc['strat'].items()) + '\n')
        for var, bcol, edges in (('n2', 'n2b', e_n2), ('abkg', 'abb', e_ab), ('edge', 'edb', e_ed)):
            L.append(f'| control {var} bin | nrcb1 median (N) | nrcb3 median (N) | mag-matched b3-b1 |\n|---|---|---|---|')
            rc['bins'][var] = []
            for k in range(len(edges) - 1):
                sub = Dc[Dc[bcol] == k]
                m1, m3 = sub[sub.det == 'nrcb1'], sub[sub.det == 'nrcb3']
                ms = strat_diff(sub, ['mb'])
                rc['bins'][var].append(dict(lo=float(edges[k]), hi=float(edges[k + 1]), n1=len(m1), n3=len(m3), m1=float(m1.dm.median()) if len(m1) else np.nan,
                                            m3=float(m3.dm.median()) if len(m3) else np.nan, matched=ms[0], matched_err=ms[1]))
                L.append(f'| {edges[k]:.3g}-{edges[k + 1]:.3g} | {m1.dm.median():+.3f} ({len(m1)}) | {m3.dm.median():+.3f} ({len(m3)}) | {ms[0]:+.3f} +/- {ms[1]:.3f} |')
            L.append('')
        r['control'] = rc
        res[band] = r
        print(band, r['star_median'], r['raw_diff'], r['mag_strat'], {k: v[:2] for k, v in r['strat'].items()}, flush=True)
    json.dump(res, open(f'{OUT}/env_results.json', 'w'), indent=1, default=lambda x: x.item() if hasattr(x, 'item') else str(x))
    open(f'{OUT}/env_tables.md', 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
