"""Trace the 327 F277W no-row dolphot stars through pk2 (tree_dbl2) and pk3 (tree_dbl3) m7 products.
Read-only.  Outputs: per_frame.ecsv, per_star.ecsv, stats.json in this directory."""
import json
import sys
import numpy as np
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an

G = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/f277w_gap'
OUT = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/pk_deblend_trace'
TREE = {'pk2': f'{G}/dbl/tree_dbl2', 'pk3': f'{G}/dbl/tree_dbl3'}
FRAMES = [(d, e) for d in 'ab' for e in (1, 2, 3, 4)]
R_HIT, R_NEAR = 0.08, 0.5

tr = Table.read(f'{G}/trace_mainfcbg_277W.ecsv')
star = SkyCoord(np.asarray(tr['ra'], float) * u.deg, np.asarray(tr['dec'], float) * u.deg)
refmag = np.asarray(tr['ref_mag'], float)
N = len(tr)


def merged(arm):
    t = Table.read(f'{TREE[arm]}/catalogs/f277w_merged_indivexp_merged_resbgsub_m7_dao_basic.fits')
    sk = t['skycoord']
    fl = np.asarray(t['flux'], float)
    ok = np.isfinite(sk.ra.deg) & (fl > 0)
    return t, ok


# zero point as in score_lw.py, from pk2 merged rows (not satstar replaced) at 18.6-21 mag dolphot stars
an.ZPWIN.update(an.zp_windows())
A = an.Arm('mainfcbg')
rs_all = SkyCoord(np.asarray(A.m['RA'], float) * u.deg, np.asarray(A.m['DEC'], float) * u.deg)
ref = A.ref['277W']
mid = np.where(np.isfinite(ref) & (ref >= 18.6) & (ref < 21))[0]
M = {}
for arm in TREE:
    t, ok = merged(arm)
    M[arm] = (t[ok], t[ok]['skycoord'], np.asarray(t[ok]['flux'], float))
t2, sk2, fl2 = M['pk2']
rep2 = np.asarray(t2['replaced_saturated'], bool)
j, d, _ = rs_all[mid].match_to_catalog_sky(sk2)
sel = (d.arcsec < 0.05) & ~rep2[j]
ZP = float(np.median(ref[mid][sel] + 2.5 * np.log10(fl2[j][sel])))
print('ZP', ZP, 'from', sel.sum(), flush=True)
dol_flux = 10 ** (-0.4 * (refmag - ZP))


def mag(f):
    return -2.5 * np.log10(np.where(f > 0, f, np.nan)) + ZP


def nearest(sk, tsk):
    if len(tsk) == 0:
        return np.full(N, -1), np.full(N, np.inf)
    j, d, _ = star.match_to_catalog_sky(tsk)
    return j, d.arcsec


wcs_cache = {}
rows = []
for arm, T in TREE.items():
    for det, e in FRAMES:
        stem = f'jw03523005001_10101_0000{e}_nrc{det}long_align_o005_crf'
        P = f'{T}/F277W/pipeline/{stem}_resbgsub_m7_satstar_'
        with fits.open(f'{T}/F277W/pipeline/{stem}.fits') as h:
            w = WCS(h['SCI'].header)
            shp = h['SCI'].data.shape
        x, y = w.world_to_pixel(star)
        cover = (x > 0) & (x < shp[1]) & (y > 0) & (y < shp[0])
        cat = Table.read(P + 'catalog.fits')
        rej = Table.read(P + 'rejected.fits')
        dao = Table.read(f'{T}/F277W/f277w_nrc{det}long_visit001_vgroup10101_exp0000{e}_resbgsub_m7_daophot_basic.fits')
        cs = cat['skycoord_fit']
        rs = rej['skycoord_fit']
        ds = dao['skycoord_centroid']
        jc, dc = nearest(star, cs)
        jr, dr = nearest(star, rs)
        jd, dd = nearest(star, ds)
        # nearest daophot row with finite flux
        fc = np.asarray(cat['flux_fit'], float)
        fd = np.asarray(dao['flux_fit'], float)
        cls = np.full(N, 5)
        cls[dd < R_NEAR] = 4  # placeholder; refined below
        cls = np.where(dd < R_HIT, 4, 5)
        cls = np.where(dr < R_NEAR, 3, cls)
        cls = np.where((dc >= R_HIT) & (dc < R_NEAR), 2, cls)
        cls = np.where(dc < R_HIT, 1, cls)
        # note: classes follow the requested priority 1,2,3,4,5; daophot within 0.08 only counts when 1-3 absent
        for i in np.where(cover)[0]:
            rows.append((arm, f'{det}{e}', i, cls[i], dc[i], fc[jc[i]] if np.isfinite(dc[i]) else np.nan,
                         dr[i], rej['reject_reason'][jr[i]] if dr[i] < R_NEAR else '',
                         np.asarray(rej['flux_fit'], float)[jr[i]] if dr[i] < R_NEAR else np.nan,
                         dd[i], fd[jd[i]], int(cat['flags'][jc[i]]) if dc[i] < R_NEAR else -1))
        print(arm, det, e, 'covered', cover.sum(), 'cat', len(cat), 'rej', len(rej), 'dao', len(dao), flush=True)
pf = Table(rows=rows, names=['arm', 'frame', 'idx', 'cls', 'sat_sep', 'sat_flux', 'rej_sep', 'rej_reason', 'rej_flux',
                             'dao_sep', 'dao_flux', 'sat_flags'])
pf['sat_sep'] = pf['sat_sep'] * 1.0
pf.write(f'{OUT}/per_frame.ecsv', overwrite=True)

# per-star summary + merged-catalog contrast
ps = Table()
ps['idx'] = np.arange(N)
ps['dolphot_idx'] = tr['dolphot_idx']
ps['ra'] = tr['ra']
ps['dec'] = tr['dec']
ps['ref_mag'] = refmag
for arm in TREE:
    s = pf[pf['arm'] == arm]
    best = np.full(N, 0)
    ncov = np.zeros(N, int)
    for i in range(N):
        m = s['idx'] == i
        ncov[i] = m.sum()
        best[i] = s['cls'][m].min() if m.any() else 0
    ps[f'{arm}_ncov'] = ncov
    ps[f'{arm}_best'] = best
    ps[f'{arm}_n1'] = [np.sum((s['idx'] == i) & (s['cls'] == 1)) for i in range(N)]
    t, sk, fl = M[arm]
    jm, dm_ = nearest(star, sk)
    ps[f'{arm}_msep'] = dm_
    ps[f'{arm}_mflux'] = fl[jm]
    ps[f'{arm}_mrep'] = np.asarray(t['replaced_saturated'], bool)[jm]
    ps[f'{arm}_mrow'] = jm
    ps[f'{arm}_mdm'] = mag(fl[jm]) - refmag
    ps[f'{arm}_mhit'] = dm_ < R_HIT
    # any-frame daophot hit within 0.08
    ps[f'{arm}_dao_hit'] = [bool(np.any((s['idx'] == i) & (s['dao_sep'] < R_HIT))) for i in range(N)]
ps['dol_flux'] = dol_flux
ps.write(f'{OUT}/per_star.ecsv', overwrite=True)
json.dump({'ZP': ZP, 'nZP': int(sel.sum())}, open(f'{OUT}/zp.json', 'w'))
