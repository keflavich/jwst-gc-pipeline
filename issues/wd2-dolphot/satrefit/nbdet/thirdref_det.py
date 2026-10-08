"""Ground JHKs (Ascenso 2007) step test of ../thirdref/thirdref.py, split by SW detector group (nrcb1 / nrcb3 / other).
Matching, isolation and selection copied from thirdref.py (default cuts). Colour term fitted on all unsaturated stars (A) or per group (B).
Step = median residual of saturated stars in the group - median residual of unsaturated stars (same trend) with ground mag in [edge, edge+w].
Bootstrap: unsaturated fit set and each group's saturated set resampled (ours and dolphot share indices)."""
import numpy as np
import json
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

ERRMAX, MATCH_R, NEIGH_R, GNEIGH_R, BLEND_FRAC = 0.1, 0.3, 0.8, 1.2, 0.05
B = 300
rng = np.random.default_rng(11)
GROUPS = ['nrcb1', 'nrcb3', 'other']
G = np.load('groups.npz')


def fl(x):
    a = np.ma.filled(x.astype(float), np.nan) if np.ma.isMaskedArray(x) else np.asarray(x, float)
    a = a.copy()
    a[np.abs(a) > 1e8] = np.nan
    return a


def mad(x):
    return 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) else np.nan


m = Table.read('/orange/adamginsburg/jwst/wd2/dolphot_benchmark/matched_Q_main2.fits')
b = Table.read('/orange/adamginsburg/jwst/wd2/catalogs/wd2_nircam_wf_mf_nf.ecsv')
g = Table.read('ascenso2007_w2phot.fits')
gc = SkyCoord(g['RAJ2000'], g['DEJ2000'], unit=(u.hourangle, u.deg))
sc = SkyCoord(np.array(m['RA']) * u.deg, np.array(m['DEC']) * u.deg)
bc = SkyCoord(np.array(b['RA']) * u.deg, np.array(b['DEC']) * u.deg)
off = (0.0, 0.0)
for it in range(3):
    s2 = SkyCoord(sc.ra + off[0] * u.arcsec / np.cos(sc.dec.rad), sc.dec + off[1] * u.arcsec)
    i, d, _ = s2.match_to_catalog_sky(gc)
    k = d < 1.0 * u.arcsec
    dra = ((gc[i].ra - s2.ra) * np.cos(sc.dec.rad)).arcsec[k]
    dde = (gc[i].dec - s2.dec).arcsec[k]
    off = (off[0] + np.median(dra), off[1] + np.median(dde))
s2 = SkyCoord(sc.ra + off[0] * u.arcsec / np.cos(sc.dec.rad), sc.dec + off[1] * u.arcsec)
i, d, _ = s2.match_to_catalog_sky(gc)
j, d2, _ = gc.match_to_catalog_sky(s2)
mutual = (j[i] == np.arange(len(sc))) & (d < MATCH_R * u.arcsec)
print('mutual matches', mutual.sum(), flush=True)
gi, gd, _ = gc.match_to_catalog_sky(gc, nthneighbor=2)
g_iso = (gd > GNEIGH_R * u.arcsec)[i]
f150 = 10 ** (-0.4 * fl(b['MAG150W']))
f200 = 10 ** (-0.4 * fl(b['MAG200W']))
idx2, idx1, sep, _ = sc.search_around_sky(bc, NEIGH_R * u.arcsec)
nf150 = np.zeros(len(sc))
nf200 = np.zeros(len(sc))
gd_ = sep.arcsec > 0.03
np.add.at(nf150, idx1[gd_], np.nan_to_num(f150[idx2[gd_]]))
np.add.at(nf200, idx1[gd_], np.nan_to_num(f200[idx2[gd_]]))
J, H, K = (fl(g[c])[i] for c in ('Jmag', 'Hmag', 'Ksmag'))
eJ, eH, eK = (fl(g[c])[i] for c in ('e_Jmag', 'e_Hmag', 'e_Ksmag'))


def fitcol(x, y):
    keep = np.ones(len(x), bool)
    co = None
    for _ in range(8):
        if keep.sum() < 8:
            return None
        A_ = np.vstack([np.ones(keep.sum()), x[keep]]).T
        co = np.linalg.lstsq(A_, y[keep], rcond=None)[0]
        r = y - (co[0] + co[1] * x)
        s = mad(r[keep])
        keep = np.abs(r) < 3 * s
    return co


def med(a):
    return np.median(a) if len(a) else np.nan


def run(band, gmag, egmag, color, ecol, tag, out):
    wband = 'F' + band
    grp = np.full(len(m), -1)
    foot = G['foot_' + wband]
    grp[foot == 'nrcb1'] = 0
    grp[foot == 'nrcb3'] = 1
    grp[(foot != '') & (foot != 'nrcb1') & (foot != 'nrcb3')] = 2
    sat = (fl(m['our_replaced_saturated_' + band]) == 1) | (fl(m['our_is_saturated_' + band]) == 1)
    nf = nf150 if band == '150W' else nf200
    fstar = 10 ** (-0.4 * fl(m['ref_' + band]))
    base = (mutual & g_iso & np.isfinite(gmag) & np.isfinite(color) & (egmag < ERRMAX) & (ecol < 1.5 * ERRMAX) & (nf < BLEND_FRAC * fstar)
            & np.isfinite(fl(m['ref_' + band])) & np.isfinite(fl(m['our_' + band])) & (grp >= 0))
    edge = np.percentile(gmag[base & sat], 95)
    uns = base & ~sat
    cats = {'ours': fl(m['our_' + band]) - gmag, 'dolphot': fl(m['ref_' + band]) - gmag}
    iu = np.where(uns)[0]
    ius = [np.where(uns & (grp == k))[0] for k in range(3)]
    isg = [np.where(base & sat & (grp == k))[0] for k in range(3)]
    rec = dict(tag=tag, edge=float(edge), nsat=[len(a) for a in isg], nuns_all=len(iu), nuns_grp=[len(a) for a in ius], res={})
    for w in (1.0, 1.5, 2.0):
        win = (gmag >= edge) & (gmag < edge + w)

        def one(fit_all, sat_g, fit_g):
            o = {}
            for cn, dm in cats.items():
                coA = fitcol(color[fit_all], dm[fit_all])
                for k in range(3):
                    s = sat_g[k]
                    for mode, co, fset in (('A', coA, fit_all), ('B', fitcol(color[fit_g[k]], dm[fit_g[k]]) if len(fit_g[k]) >= 12 else None, fit_g[k])):
                        if co is None or len(s) == 0:
                            o[(cn, mode, k)] = np.nan
                            continue
                        rs = dm[s] - (co[0] + co[1] * color[s])
                        fw = fset[win[fset]]
                        # unsaturated window residuals: for mode A use the group's own unsat window stars with the global trend
                        if mode == 'A':
                            fw = fit_g[k][win[fit_g[k]]]
                        ru = dm[fw] - (co[0] + co[1] * color[fw])
                        o[(cn, mode, k)] = med(rs) - med(ru) if len(ru) >= 3 else np.nan
            return o
        r0 = one(iu, isg, ius)
        bs = {k_: [] for k_ in r0}
        for _ in range(B):
            fa = iu[rng.integers(0, len(iu), len(iu))]
            sg = [a[rng.integers(0, len(a), len(a))] if len(a) else a for a in isg]
            # group unsat sets derived from the resampled global set
            fg = [fa[grp[fa] == k] for k in range(3)]
            rb = one(fa, sg, fg)
            for k_ in bs:
                bs[k_].append(rb[k_])
        for mode in ('A', 'B'):
            for k in range(3):
                oo = np.array(bs[('ours', mode, k)])
                dd = np.array(bs[('dolphot', mode, k)])
                rec['res'][f'w{w}_{mode}_{GROUPS[k]}'] = dict(
                    ours=[float(r0[('ours', mode, k)]), float(np.nanstd(oo))], dol=[float(r0[('dolphot', mode, k)]), float(np.nanstd(dd))],
                    diff=[float(r0[('ours', mode, k)] - r0[('dolphot', mode, k)]), float(np.nanstd(oo - dd))],
                    nwin=int((uns & (grp == k) & win).sum()))
    # binned residuals (global colour term) per group for the figure: bins of 0.5 mag in own-catalog mag
    coA = {cn: fitcol(color[iu], dm[iu]) for cn, dm in cats.items()}
    fig = {}
    for cn, dm in cats.items():
        res = dm - (coA[cn][0] + coA[cn][1] * color)
        mag = fl(m[('our_' if cn == 'ours' else 'ref_') + band])
        rows = []
        for k in range(3):
            for lo in np.arange(13, 22, 0.5):
                sel = base & (grp == k) & (mag >= lo) & (mag < lo + 0.5)
                s_, u_ = sel & sat, sel & ~sat
                rows.append((k, lo, int(s_.sum()), float(med(res[s_])) if s_.sum() >= 3 else np.nan, float(mad(res[s_]) / np.sqrt(max(s_.sum(), 1)) * 1.253) if s_.sum() >= 3 else np.nan,
                             int(u_.sum()), float(med(res[u_])) if u_.sum() >= 3 else np.nan))
        fig[cn] = rows
    rec['bins'] = fig
    # direct ours - dolphot per group (band)
    dAB = fl(m['our_' + band]) - fl(m['ref_' + band])
    rec['direct'] = {}
    for k in range(3):
        s_ = dAB[base & sat & (grp == k)]
        u_ = dAB[base & ~sat & (grp == k) & (gmag >= edge) & (gmag < edge + 1.5)]
        rec['direct'][GROUPS[k]] = dict(sat=[float(med(s_)), len(s_)], unsat=[float(med(u_)), len(u_)])
    out[tag] = rec
    print(tag, rec['edge'], rec['nsat'], rec['nuns_grp'], flush=True)


out = {}
run('150W', H, eH, J - H, np.hypot(eJ, eH), 'F150W-H (J-H)', out)
run('150W', H, eH, H - K, np.hypot(eH, eK), 'F150W-H (H-Ks)', out)
run('200W', K, eK, H - K, np.hypot(eH, eK), 'F200W-Ks (H-Ks)', out)
json.dump(out, open('thirdref_det_results.json', 'w'), indent=1)
L = ['Ground-reference step by detector group. Step = (median residual of saturated stars) - (median residual of unsaturated stars with ground mag in [edge, edge+w]); negative = JWST reads bright. Colour term A fitted over all unsaturated stars (window comparison uses the group\'s own unsaturated stars); B fitted per group.', '',
     '| test | edge | w | trend | group | N sat | N unsat in window | ours | dolphot | ours - dolphot |', '|---|---|---|---|---|---|---|---|---|---|']
for tag, r in out.items():
    for key, v in r['res'].items():
        w, mode, gname = key.split('_')
        k = GROUPS.index(gname)
        fm = lambda a: '-' if not np.isfinite(a[0]) else f'{a[0]:+.3f} +- {a[1]:.3f}'
        L.append(f"| {tag} | {r['edge']:.2f} | {w[1:]} | {mode} | {gname} | {r['nsat'][k]} | {v['nwin']} | {fm(v['ours'])} | {fm(v['dol'])} | {fm(v['diff'])} |")
L += ['', 'Direct ours - dolphot (band magnitude): saturated stars versus unsaturated stars in [edge, edge+1.5].', '', '| test | group | sat med (N) | unsat med (N) |', '|---|---|---|---|']
for tag, r in out.items():
    for gname, v in r['direct'].items():
        L.append(f"| {tag} | {gname} | {v['sat'][0]:+.3f} ({v['sat'][1]}) | {v['unsat'][0]:+.3f} ({v['unsat'][1]}) |")
L += ['', 'Binned residuals (global colour term A), 0.5 mag bins of the own-catalogue JWST magnitude: group, bin low edge, N sat, median sat, sem sat, N unsat, median unsat.', '']
for tag, r in out.items():
    for cn, rows in r['bins'].items():
        L.append(f'{tag} {cn}')
        L.append('| group | bin | N sat | sat med +- | N unsat | unsat med |')
        L.append('|---|---|---|---|---|---|')
        for k, lo, ns, ms, es, nu_, mu in rows:
            if ns >= 3 or nu_ >= 3:
                L.append(f"| {GROUPS[k]} | {lo:.1f} | {ns} | {ms:+.3f} +- {es:.3f} | {nu_} | {mu:+.3f} |")
        L.append('')
open('thirdref_det_tables.md', 'w').write('\n'.join(L))
