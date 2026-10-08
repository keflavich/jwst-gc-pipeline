"""Tabulate kf_step_diag outputs vs dolphot magnitude.  usage: python kf_step_tables.py BAND (e.g. 250M)"""
import sys
import numpy as np
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
import astropy.units as u

Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
OUT = f'{Q}/kf_lwmad'
b = sys.argv[1]
band = 'F' + b
st = Table.read(f'{OUT}/kf_step_diag_{band}_stars.fits')
px = Table.read(f'{OUT}/kf_step_diag_{band}_pix.fits')
mt = Table.read(f'{Q.rsplit("/", 1)[0]}/matched_Q_main2.fits')
mk = Table.read(f'{Q.rsplit("/", 1)[0]}/matched_Q_main2kf.fits')
ref = np.ma.filled(mt[f'ref_{b}'], np.nan).astype(float)
ok = np.isfinite(ref)
rc = SkyCoord(mt['RA'][ok] * u.deg, mt['DEC'][ok] * u.deg)
refmag = ref[ok]
# attach RA/Dec from the per-frame satstar catalogs
ras, decs = [], []
for fr in np.unique(np.char.decode(np.asarray(st['frame']).astype('S')) if st['frame'].dtype.kind=='S' else st['frame']):
    t = Table.read(f'{Q}/tree_main2/{band}/pipeline/{fr}_align_o005_crf_resbgsub_m7_satstar_catalog.fits')
    x = np.ma.filled(t['xcentroid'], np.nan).astype(float)
    y = np.ma.filled(t['ycentroid'], np.nan).astype(float)
    ra = np.ma.filled(t['sat_com_ra'], np.nan).astype(float)
    de = np.ma.filled(t['sat_com_dec'], np.nan).astype(float)
    s = st[np.asarray([str(q if not isinstance(q, bytes) else q.decode()) == fr for q in st['frame']])]
    idx = [int(np.argmin((x - xx) ** 2 + (y - yy) ** 2)) for xx, yy in zip(s['x'], s['y'])]
    ras += list(ra[idx]); decs += list(de[idx])
st['ra'] = ras; st['dec'] = decs
good = np.isfinite(st['ra']) & np.isfinite(st['dec'])
sc = SkyCoord(np.where(good, st['ra'], 0) * u.deg, np.where(good, st['dec'], 0) * u.deg)
i, d, _ = sc.match_to_catalog_sky(rc)
st['dmag_ref'] = np.where(good & (d < 0.3 * u.arcsec), refmag[i], np.nan)
st['ratio'] = st['f_on'] / st['f_off']
st['mag_ins'] = -2.5 * np.log10(st['f_off'])
st['dm_on_off'] = -2.5 * np.log10(st['ratio'])
st['frac_rew'] = st['nrew'] / np.maximum(st['nkeep'], 1)
st['crf_over_rew'] = st['sum_crf_rew'] / np.where(st['sum_rew_keep'] > 0, st['sum_rew_keep'], np.nan)
st.write(f'{OUT}/kf_step_stars_{band}_withref.fits', overwrite=False)
print(f'## {band}: {len(st)} frame-star rows, {np.isfinite(st["dmag_ref"]).sum()} with dolphot ref')
bins = [(10, 12), (12, 13), (13, 14), (14, 15), (15, 15.5), (15.5, 16), (16, 17), (17, 18.5)]
print('| dolphot mag | N | med on/off flux | dm=-2.5log(on/off) | MAD | med nsat px | med nkeep | med frac rewritten-by-off | med firstsat | med frac nuse<=1 | med crf/(R g0 rewrite) sum |')
print('|---|---|---|---|---|---|---|---|---|---|---|')
m = st['dmag_ref']
for lo, hi in bins:
    s = (m >= lo) & (m < hi) & np.isfinite(st['ratio'])
    if s.sum() == 0:
        continue
    r = st[s]
    mad = 1.4826 * np.median(np.abs(r['dm_on_off'] - np.median(r['dm_on_off'])))
    print(f'| {lo}-{hi} | {s.sum()} | {np.median(r["ratio"]):.3f} | {np.median(r["dm_on_off"]):+.3f} | {mad:.3f} | '
          f'{np.median(r["nsat"]):.0f} | {np.median(r["nkeep"]):.0f} | {np.nanmedian(r["frac_rew"]):.2f} | '
          f'{np.nanmedian(r["med_firstsat"]):.1f} | {np.nanmedian(r["frac_le1"]):.2f} | {np.nanmedian(r["crf_over_rew"]):.3f} |')
# pixel-level: crf / rewrite by number of usable groups
print(f'\n{band} pixel-level (pixels SATURATED, finite, no DNU, that the off arm rewrote): N={len(px)}')
print('| usable groups | N px | med crf/rewrite | 16-84% | frac group0 flagged |')
print('|---|---|---|---|---|')
rr = px['crf'] / px['rew']
for n in range(0, 8):
    s = px['nuse'] == n
    if s.sum() < 20:
        continue
    p16, p50, p84 = np.percentile(rr[s], [16, 50, 84])
    print(f'| {n} | {s.sum()} | {p50:.3f} | {p16:.3f}-{p84:.3f} | {np.mean(px["g0flag"][s]):.2f} |')
print('\n| rewrite value bin (MJy/sr) | N px | med crf/rewrite | med nuse |')
print('|---|---|---|---|')
edges = np.array([0, 100, 300, 1000, 3000, 1e4, 1e5, 1e9])
for a, c in zip(edges[:-1], edges[1:]):
    s = (px['rew'] >= a) & (px['rew'] < c)
    if s.sum() < 20:
        continue
    print(f'| {a:g}-{c:g} | {s.sum()} | {np.median(rr[s]):.3f} | {np.median(px["nuse"][s]):.1f} |')
