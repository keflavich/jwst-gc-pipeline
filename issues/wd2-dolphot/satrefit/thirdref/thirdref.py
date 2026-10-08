import numpy as np, json
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
TAGV = os.environ.get('TAGV', '')
ERRMAX = float(os.environ.get('ERRMAX', 0.1))
MATCH_R = float(os.environ.get('MATCH_R', 0.3))   # arcsec; ground-based seeing ~0.4-0.5"
NEIGH_R = 0.8   # arcsec JWST-side blend radius
GNEIGH_R = float(os.environ.get('GNEIGH_R', 1.2))  # arcsec ground-catalog neighbour exclusion
BLEND_FRAC = float(os.environ.get('BLEND_FRAC', 0.05))
out = open(f'thirdref_log{TAGV}.txt', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); out.write(s + '\n')

def fl(x):
    a = np.ma.filled(x.astype(float), np.nan) if np.ma.isMaskedArray(x) else np.asarray(x, float)
    a = a.copy(); a[np.abs(a) > 1e8] = np.nan; return a

m = Table.read('/orange/adamginsburg/jwst/wd2/dolphot_benchmark/matched_Q_main2.fits')
b = Table.read('/orange/adamginsburg/jwst/wd2/catalogs/wd2_nircam_wf_mf_nf.ecsv')
g = Table.read('ascenso2007_w2phot.fits')
gc = SkyCoord(g['RAJ2000'], g['DEJ2000'], unit=(u.hourangle, u.deg))
sc = SkyCoord(np.array(m['RA']) * u.deg, np.array(m['DEC']) * u.deg)
bc = SkyCoord(np.array(b['RA']) * u.deg, np.array(b['DEC']) * u.deg)

# global offset: iterate
off = (0.0, 0.0)
for it in range(3):
    s2 = SkyCoord(sc.ra + off[0] * u.arcsec / np.cos(sc.dec.rad), sc.dec + off[1] * u.arcsec)
    i, d, _ = s2.match_to_catalog_sky(gc)
    k = d < 1.0 * u.arcsec
    dra = ((gc[i].ra - s2.ra) * np.cos(sc.dec.rad)).arcsec[k]; dde = (gc[i].dec - s2.dec).arcsec[k]
    off = (off[0] + np.median(dra), off[1] + np.median(dde))
    P('iter', it, 'offset ground-minus-JWST (dRA*cos, dDec) arcsec', off, 'N', k.sum())
s2 = SkyCoord(sc.ra + off[0] * u.arcsec / np.cos(sc.dec.rad), sc.dec + off[1] * u.arcsec)
i, d, _ = s2.match_to_catalog_sky(gc)
# mutual nearest
j, d2, _ = gc.match_to_catalog_sky(s2)
mutual = (j[i] == np.arange(len(sc))) & (d < MATCH_R * u.arcsec)
P('mutual matches within', MATCH_R, 'arcsec:', mutual.sum(), ' median sep arcsec', np.median(d[mutual].arcsec))
# ground isolation
gi, gd, _ = gc.match_to_catalog_sky(gc, nthneighbor=2)
g_iso = (gd > GNEIGH_R * u.arcsec)[i]

# JWST neighbour blend fraction from full B catalog in F150W
ra_b = bc
f150 = 10 ** (-0.4 * fl(b['MAG150W']))
f200 = 10 ** (-0.4 * fl(b['MAG200W']))
# B frame: use unshifted sc for B search
idx2, idx1, sep, _ = sc.search_around_sky(bc, NEIGH_R * u.arcsec)  # returns (idx into bc, idx into sc)
from collections import defaultdict
nf150 = np.zeros(len(sc)); nf200 = np.zeros(len(sc))
self_b = np.array(m['RA'])  # self position equals bc entry; exclude sep<0.02"
for a_, b_, s_ in zip(idx1, idx2, sep.arcsec):
    pass
# vectorized
bcoord_idx_self = None
good = sep.arcsec > 0.03
a_ = idx1[good]; b_ = idx2[good]
np.add.at(nf150, a_, np.nan_to_num(f150[b_]))
np.add.at(nf200, a_, np.nan_to_num(f200[b_]))

res = {}
bands = {'150W': dict(gname='Hmag', ename='e_Hmag'), '200W': dict(gname='Ksmag', ename='e_Ksmag')}
J, H, K = (fl(g[c])[i] for c in ('Jmag', 'Hmag', 'Ksmag'))
eJ, eH, eK = (fl(g[c])[i] for c in ('e_Jmag', 'e_Hmag', 'e_Ksmag'))

def mad(x): return 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) else np.nan
def sem(x): return mad(x) / np.sqrt(len(x)) * 1.253 if len(x) > 1 else np.nan

def run(band, gmag, egmag, color, ecol, colname, tag, fig_ax=None):
    jr = {}
    sat = (fl(m['our_replaced_saturated_' + band]) == 1) | (fl(m['our_is_saturated_' + band]) == 1)
    nf = nf150 if band == '150W' else nf200
    fstar = 10 ** (-0.4 * fl(m['ref_' + band]))
    base = mutual & g_iso & np.isfinite(gmag) & np.isfinite(color) & (egmag < ERRMAX) & (ecol < 1.5 * ERRMAX) & (nf < BLEND_FRAC * fstar) & np.isfinite(fl(m['ref_' + band])) & np.isfinite(fl(m['our_' + band]))
    P(f'\n=== {tag}: base sample N={base.sum()}  (sat {np.sum(base&sat)}, unsat {np.sum(base&~sat)})')
    out_ = {}
    for cat, magcol in (('A', 'our_'), ('B', 'ref_')):
        mag = fl(m[magcol + band])
        dm = mag - gmag
        # edge: 95th percentile of ground mag of saturated stars in base
        edge = np.percentile(gmag[base & sat], 95)
        uns = base & ~sat
        # fit window: unsat stars, ground mag > edge-0.5 (excludes nothing bright of concern) -> all unsat
        fitm = uns.copy()
        # robust linear fit with sigma clipping
        x = color[fitm]; y = dm[fitm]; keep = np.ones(len(x), bool)
        for _ in range(8):
            A_ = np.vstack([np.ones(keep.sum()), x[keep]]).T
            co = np.linalg.lstsq(A_, y[keep], rcond=None)[0]
            r = y - (co[0] + co[1] * x)
            s = mad(r[keep]); keep = np.abs(r) < 3 * s
        resid = dm - (co[0] + co[1] * color)
        P(f'{cat}: color term a={co[0]:+.3f} b={co[1]:+.3f} (N fit {keep.sum()}, rms {s:.3f}); saturation edge (p95 ground {tag.split()[1] if False else ""}mag of sat stars) = {edge:.2f}')
        # bins in own-catalog mag
        P(f'{cat} residual (JWST - ground - colorterm) vs {cat} JWST mag, 1-mag bins: bin | unsat N med MAD | sat N med MAD')
        rows = []
        for lo in range(13, 23):
            sel = base & (mag >= lo) & (mag < lo + 1)
            su = sel & ~sat; ss = sel & sat
            rows.append((lo, su.sum(), np.median(resid[su]) if su.sum() else np.nan, mad(resid[su]) if su.sum() else np.nan,
                         ss.sum(), np.median(resid[ss]) if ss.sum() else np.nan, mad(resid[ss]) if ss.sum() else np.nan))
            P(f'  {lo}-{lo+1}: {rows[-1][1]:5d} {rows[-1][2]:+.3f} {rows[-1][3]:.3f} | {rows[-1][4]:5d} {rows[-1][5]:+.3f} {rows[-1][6]:.3f}')
        # step: sat stars vs unsat stars in ground-mag window [edge, edge+1.5] (just fainter than edge)
        for w in (1.0, 1.5, 2.0):
            ru = resid[uns & (gmag >= edge) & (gmag < edge + w)]
            rs = resid[base & sat]
            rs1 = resid[base & sat & (gmag >= edge - 1.5)]
            rs2 = resid[base & sat & (gmag < edge - 1.5)]
            st = np.median(rs) - np.median(ru)
            e = np.hypot(sem(rs), sem(ru))
            st1 = np.median(rs1) - np.median(ru); e1 = np.hypot(sem(rs1), sem(ru))
            P(f'  step[{cat}] unsat window ground in [{edge:.2f},{edge+w:.2f}] N={len(ru)} med={np.median(ru):+.3f}; sat all N={len(rs)} med={np.median(rs):+.3f} -> step {st:+.3f} +- {e:.3f}; sat within 1.5 mag of edge N={len(rs1)} med={np.median(rs1):+.3f} step {st1:+.3f} +- {e1:.3f}; sat brighter N={len(rs2)} med={np.median(rs2):+.3f}')
            out_[(cat, w)] = (len(ru), np.median(ru), len(rs), np.median(rs), st, e, len(rs1), np.median(rs1), st1, e1)
        out_[cat] = dict(mag=mag, resid=resid, co=co, edge=edge)
    # A-B direct
    dAB = fl(m['our_' + band]) - fl(m['ref_' + band])
    sel_s = base & sat; sel_u = base & ~sat & (gmag >= out_['A']['edge']) & (gmag < out_['A']['edge'] + 1.5)
    P(f'A-B direct: sat N={sel_s.sum()} med={np.median(dAB[sel_s]):+.3f}  unsat(edge..edge+1.5) N={sel_u.sum()} med={np.median(dAB[sel_u]):+.3f}  diff {np.median(dAB[sel_s])-np.median(dAB[sel_u]):+.3f}')
    for lo in range(13, 22):
        s_ = base & sat & (fl(m['ref_' + band]) >= lo) & (fl(m['ref_' + band]) < lo + 1)
        u_ = base & ~sat & (fl(m['ref_' + band]) >= lo) & (fl(m['ref_' + band]) < lo + 1)
        P(f'   A-B ref-mag {lo}-{lo+1}: sat N={s_.sum()} med={np.median(dAB[s_]) if s_.sum() else np.nan:+.3f} | unsat N={u_.sum()} med={np.median(dAB[u_]) if u_.sum() else np.nan:+.3f}')
    out_['base'] = base; out_['sat'] = sat; out_['dAB'] = dAB
    return out_

R = {}
R['F150W_JH'] = run('150W', H, eH, J - H, np.hypot(eJ, eH), 'J-H', 'F150W-H vs J-H')
R['F150W_HK'] = run('150W', H, eH, H - K, np.hypot(eH, eK), 'H-Ks', 'F150W-H vs H-Ks')
R['F200W_HK'] = run('200W', K, eK, H - K, np.hypot(eH, eK), 'H-Ks', 'F200W-Ks vs H-Ks')

# figure
fig, axs = plt.subplots(3, 2, figsize=(12, 13), sharex='row')
for r_, (key, band, lab) in enumerate([('F150W_JH', '150W', 'F150W - H  (color J-H)'), ('F150W_HK', '150W', 'F150W - H  (color H-Ks)'), ('F200W_HK', '200W', 'F200W - Ks  (color H-Ks)')]):
    O = R[key]
    for c_, cat in enumerate(('A', 'B')):
        ax = axs[r_, c_]; D = O[cat]; base = O['base']; sat = O['sat']
        mag = D['mag']; res = D['resid']
        ax.plot(mag[base & ~sat], res[base & ~sat], '.', ms=3, color='tab:blue', alpha=.4, label='JWST unsaturated')
        ax.plot(mag[base & sat], res[base & sat], '.', ms=4, color='tab:red', alpha=.6, label='JWST saturated')
        for sel, col in ((base & ~sat, 'navy'), (base & sat, 'darkred')):
            xs = np.arange(13, 23, 0.5); mm = []; xc = []
            for lo in xs:
                s_ = sel & (mag >= lo) & (mag < lo + 1)
                if s_.sum() >= 5: mm.append(np.median(res[s_])); xc.append(lo + .5)
            ax.plot(xc, mm, '-o', color=col, lw=2, ms=4)
        ax.axhline(0, color='k', lw=.5)
        ax.set_ylim(-0.6, 0.6); ax.set_xlim(13, 23)
        ax.set_title(f'{"A (ours, main2)" if cat=="A" else "B (dolphot)"}: {lab}', fontsize=10)
        ax.set_xlabel(f'{cat} JWST {band} mag'); ax.set_ylabel('residual after colour term [mag]')
        if r_ == 0 and c_ == 0: ax.legend(loc='upper left', fontsize=8)
plt.tight_layout(); plt.savefig(f'thirdref{TAGV}.png', dpi=110)

# save matched sample
base = R['F150W_JH']['base']
t = Table({'RA': np.array(m['RA']), 'DEC': np.array(m['DEC']), 'J': J, 'H': H, 'Ks': K, 'eH': eH,
           'A150': fl(m['our_150W']), 'B150': fl(m['ref_150W']), 'A200': fl(m['our_200W']), 'B200': fl(m['ref_200W']),
           'sat150': (fl(m['our_replaced_saturated_150W']) == 1) | (fl(m['our_is_saturated_150W']) == 1),
           'sat200': (fl(m['our_replaced_saturated_200W']) == 1) | (fl(m['our_is_saturated_200W']) == 1),
           'ground_match': mutual, 'ground_isolated': g_iso, 'nfrac150': nf150, 'nfrac200': nf200})
t[mutual].write(f'thirdref_matched{TAGV}.fits', overwrite=True)
out.close()
