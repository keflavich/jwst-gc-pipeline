"""External comparison of bright (9-15 mag) narrowband/broad photometry with 2MASS PSC (Vizier II/246).
usage: nice -19 python ext_2mass.py ARM [ARM ...]    (ARM: prod or a Q_integ arm name from analyze.PATH)
Outputs: fig/ext_dm_<band>.png, fig/ext_colour.png, ext_medians.md, ext_stats.txt  (in the cwd's nb_bright_ext dir)."""
import os
import sys
sys.path.insert(0, '..')
import numpy as np
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.vizier import Vizier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import analyze as an

arms = sys.argv[1:]
HERE = os.path.dirname(os.path.abspath(__file__))
# band: (2MASS recipe) ; frac = weight of Ks in H + frac (Ks - H), from effective wavelength (H 1.662, Ks 2.159 um)
FRAC = {'f162m': 0.0, 'f164n': 0.0, 'f182m': 0.37, 'f187n': 0.43, 'f200w': 0.9, 'f212n': 1.0}
BANDS = list(FRAC)
cache = f'{HERE}/2mass_wd2.ecsv'
if os.path.exists(cache):
    tm = Table.read(cache)
else:
    v = Vizier(columns=['RAJ2000', 'DEJ2000', 'Jmag', 'Hmag', 'Kmag', 'Qflg', 'Cflg', 'Bflg', 'Xflg'], row_limit=-1)
    tm = v.query_region(SkyCoord(155.957, -57.760, unit='deg'), radius=0.16 * u.deg, catalog='II/246')[0]
    tm.write(cache, format='ascii.ecsv')
print('2MASS rows', len(tm))
tm = tm[np.isfinite(np.asarray(np.ma.filled(tm['Hmag'], np.nan), float)) | np.isfinite(np.asarray(np.ma.filled(tm['Kmag'], np.nan), float))]
cm = SkyCoord(tm['RAJ2000'], tm['DEJ2000'], unit='deg')
H = np.asarray(np.ma.filled(tm['Hmag'], np.nan), float)
K = np.asarray(np.ma.filled(tm['Kmag'], np.nan), float)
Q = np.array([str(x) for x in tm['Qflg']])
C = np.array([str(x) for x in tm['Cflg']])
qh = np.array([q[1] in 'AB' if len(q) > 1 else False for q in Q])
qk = np.array([q[2] in 'AB' if len(q) > 2 else False for q in Q])
c0 = np.array([(c[1] == '0' and c[2] == '0') if len(c) > 2 else False for c in C])
# 2MASS isolation: nearest other 2MASS source within 3" brighter than own K + 3 (or any within 3" if K missing)
i1, d1, _ = cm.match_to_catalog_sky(cm, nthneighbor=2)
iso_2m = ~((d1 < 3 * u.arcsec) & (np.nan_to_num(K[i1], nan=99) < np.nan_to_num(K, nan=99) + 3))
sel = c0 & iso_2m
print('2MASS: clean-flag & isolated:', sel.sum(), 'with H+K AB', (sel & qh & qk).sum())
good_hk = sel & qh & qk & np.isfinite(H) & np.isfinite(K)


def twomass_pred(band):
    f = FRAC[band]
    return H + f * (K - H)


cm_s = cm
edges = np.arange(9, 15.01, 0.5)
mid = 0.5 * (edges[1:] + edges[:-1])
res = {}
cols = {'prod': 'k', 'ctrl': 'tab:gray', 'integ': 'tab:blue', 'integbg': 'c', 'integfixc': 'tab:purple', 'integfc': 'tab:green', 'integfcbg': 'tab:olive',
        'mainfcbg': 'tab:red', 'mainfcbgkf': 'tab:orange'}
fig_dm = {b: plt.subplots(1, 1, figsize=(7, 4.5)) for b in BANDS}
figc, axc = plt.subplots(1, 3, figsize=(15, 4.3))
out = open(f'{HERE}/ext_medians.md', 'w')
stats = open(f'{HERE}/ext_stats.txt', 'w')
PAIRS = [('f164n', 'f162m'), ('f187n', 'f182m'), ('f212n', 'f200w')]
# per 2MASS star: brightest of our sources within 1" per arm; common sample = matched and isolated in every arm
cm_all = cm
cand = np.where(good_hk)[0]
cc = cm[cand]
M = {}      # arm -> dict band -> mags array for cand (nan if none)
okarm = {}
for arm in arms:
    t = Table.read(an.PATH[arm][0])
    cs = SkyCoord(t['skycoord_ref'])
    mags = {b: an.fl(t[f'mag_vega_{b}']) for b in BANDS}
    ref = np.where(np.isfinite(mags['f200w']), mags['f200w'], mags['f212n'])
    idx_c, idx_s, d2, _ = cs.search_around_sky(cc, 2.5 * u.arcsec)
    sel_m = {b: np.full(len(cand), np.nan) for b in BANDS}
    ok = np.zeros(len(cand), bool)
    for k in range(len(cand)):
        w = idx_s[idx_c == k]
        if len(w) == 0:
            continue
        dd = cs[w].separation(cc[k]).arcsec
        r = np.where(np.isfinite(ref[w]), ref[w], 99)
        br = np.argmin(r)
        if dd[br] > 1.0 or r[br] > 90:
            continue
        others = np.delete(r, br)
        if len(others) and others.min() < r[br] + 3:
            continue
        ok[k] = True
        for b in BANDS:
            sel_m[b][k] = mags[b][w[br]]
    M[arm] = sel_m
    okarm[arm] = ok
    stats.write(f'{arm}: our rows {len(t)}, isolated+matched 2MASS stars {ok.sum()}\n')
common = np.all([okarm[a] for a in arms], axis=0)
stats.write(f'common sample (all arms): {common.sum()} stars of {len(cand)} 2MASS H+K AB clean-isolated candidates\n')
Hc, Kc = H[cand], K[cand]
for arm in arms:
    mags = M[arm]
    for b in BANDS:
        m = mags[b]
        pred = Hc + FRAC[b] * (Kc - Hc)
        dm = m - pred
        s = common & np.isfinite(m)
        meds = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            q = s & (m >= lo) & (m < hi)
            meds.append((q.sum(), np.median(dm[q]) if q.sum() >= 3 else np.nan, 1.253 * np.std(dm[q]) / np.sqrt(q.sum()) if q.sum() >= 3 else np.nan))
        res[(arm, b)] = meds
        # 1-mag bins vs 2MASS-predicted mag (same stars for each arm)
        e1 = [9, 10, 11, 12, 13, 14, 15]
        res[(arm, b + '_2m')] = [((s & (pred >= lo) & (pred < hi)).sum(), np.median(dm[s & (pred >= lo) & (pred < hi)]) if (s & (pred >= lo) & (pred < hi)).sum() >= 3 else np.nan) for lo, hi in zip(e1[:-1], e1[1:])]
        ax = fig_dm[b][1]
        md = np.array([x[1] for x in meds]); er = np.array([x[2] for x in meds])
        ax.errorbar(mid + 0.02 * list(cols).index(arm), md, er, color=cols.get(arm, 'm'), marker='o', ms=4, lw=1.2, label=arm)
        ax.scatter(m[s], dm[s], s=6, color=cols.get(arm, 'm'), alpha=0.25, zorder=0)
    for k, (nb, bb) in enumerate(PAIRS):
        col = mags[nb] - mags[bb]
        pred = (FRAC[nb] - FRAC[bb]) * (Kc - Hc)
        r = col - pred
        t = Table.read(an.PATH[arm][0])
        n_ = an.fl(t[f'mag_vega_{nb}']); b_ = an.fl(t[f'mag_vega_{bb}'])
        allref = np.isfinite(n_) & np.isfinite(b_) & (n_ >= 15) & (n_ < 18)
        c15 = np.median((n_ - b_)[allref])
        s = common & np.isfinite(r)
        meds = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            q = s & (mags[nb] >= lo) & (mags[nb] < hi)
            meds.append((q.sum(), np.median(r[q]) - c15 if q.sum() >= 3 else np.nan))
        res[(arm, f'col_{nb}')] = meds
        e1 = [9, 10, 11, 12, 13, 14, 15]
        predm = Hc + FRAC[nb] * (Kc - Hc)
        res[(arm, f'col_{nb}_2m')] = [((s & (predm >= lo) & (predm < hi)).sum(), np.median(r[s & (predm >= lo) & (predm < hi)]) - c15 if (s & (predm >= lo) & (predm < hi)).sum() >= 3 else np.nan) for lo, hi in zip(e1[:-1], e1[1:])]
        axc[k].plot(mid + 0.02 * list(cols).index(arm), [x[1] for x in meds], color=cols.get(arm, 'm'), marker='o', ms=4, label=arm)
for b, (f, ax) in fig_dm.items():
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel(f'our {b.upper()} Vega mag'); ax.set_ylabel(f'ours - 2MASS({"H" if FRAC[b] == 0 else "Ks" if FRAC[b] == 1 else "H/Ks interp, w=%.2f" % FRAC[b]})')
    ax.set_ylim(-1.5, 1.5); ax.set_xlim(9, 15); ax.legend(fontsize=7, ncol=2); ax.set_title(f'{b.upper()} vs 2MASS, isolated stars; grey = per-star ({arms[0]}), lines = 0.5 mag medians')
    f.tight_layout(); f.savefig(f'{HERE}/fig/ext_dm_{b}.png', dpi=110); plt.close(f)
for k, (nb, bb) in enumerate(PAIRS):
    axc[k].axhline(0, color='k', lw=0.5); axc[k].set_xlabel(f'our {nb.upper()} mag'); axc[k].set_ylabel(f'({nb.upper()}-{bb.upper()}) - 2MASS term - 15-18 mag median')
    axc[k].set_ylim(-0.6, 0.8); axc[k].legend(fontsize=7); axc[k].set_title(f'{nb.upper()}-{bb.upper()}')
figc.tight_layout(); figc.savefig(f'{HERE}/fig/ext_colour.png', dpi=110)
for key in [b for b in BANDS] + [f'col_{p[0]}' for p in PAIRS]:
    out.write(f'\n### {key}: median {"(ours - 2MASS)" if not key.startswith("col") else "(colour - 2MASS term) - 15-18 mag median"} per our-mag bin (N)\n')
    out.write('| arm | ' + ' | '.join(f'{lo:g}-{hi:g}' for lo, hi in zip(edges[:-1], edges[1:])) + ' |\n|---|' + '---|' * (len(edges) - 1) + '\n')
    for arm in arms:
        out.write(f'| {arm} | ' + ' | '.join((f'{x[1]:+.2f} ({x[0]})' if np.isfinite(x[1]) else f'- ({x[0]})') for x in res[(arm, key)]) + ' |\n')
for key in BANDS + [f'col_{p[0]}' for p in PAIRS]:
    out.write(f'\n### {key} vs 2MASS-predicted magnitude (same stars in every arm), 1 mag bins: median {"(ours - 2MASS)" if not key.startswith("col") else "(colour - 2MASS term) - 15-18 mag median"} (N)\n')
    out.write('| arm | 9-10 | 10-11 | 11-12 | 12-13 | 13-14 | 14-15 |\n|---|---|---|---|---|---|---|\n')
    for arm in arms:
        out.write(f'| {arm} | ' + ' | '.join((f'{x[1]:+.2f} ({x[0]})' if np.isfinite(x[1]) else f'- ({x[0]})') for x in res[(arm, key + '_2m')]) + ' |\n')
out.close(); stats.close()
print(open(f'{HERE}/ext_stats.txt').read())
