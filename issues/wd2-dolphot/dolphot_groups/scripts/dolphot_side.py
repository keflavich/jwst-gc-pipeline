"""Dolphot-side tests: candidates in 0.12" circle, per-detector centroid, flux ratio vs parent mag, detection fraction vs parent mag.
usage: nice -19 python dolphot_side.py <band> <dx> <dy>"""
import sys, glob, json
sys.path.insert(0, '/blue/adamginsburg/adamginsburg/tmp/claude-3663/s')
from an import *
from astropy.wcs import WCS
from astropy.io import fits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
band = sys.argv[1]; EX = np.array([float(sys.argv[2]), float(sys.argv[3])]); RC = 0.12
OUT = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/ghost_test'; D = '/orange/adamginsburg/jwst/wd2'
has, si, x, y, A, B = offs(band)
mref = fl(ma[f'ref_{band}'])[has]
pm = fl(ca[f'mag_vega_f{band.lower()}'])[si]
inc = np.hypot(x - EX[0], y - EX[1]) < RC
mir = np.hypot(x + EX[0], y + EX[1]) < RC
neither = ~A & ~B
print(band, 'circle', inc.sum(), 'mirror', mir.sum(), 'circle neither', (inc & neither).sum(), 'mirror neither', (mir & neither).sum())
# detector of parent in first-exposure frames
frames = {}
for f in sorted(glob.glob(f'{D}/F{band}/pipeline/jw03523*_00001_nrc*_align_o005_crf.fits')):
    h = fits.getheader(f, 'SCI'); frames[fits.getheader(f)['DETECTOR']] = WCS(h)
sc = SkyCoord(ca['skycoord_ref'][si])
det = np.full(len(has), '', dtype='U6')
for d, w in frames.items():
    px, py = w.world_to_pixel(sc)
    m = (px > 30) & (px < 2017) & (py > 30) & (py < 2017) & (det == '')
    det[m] = d
tab = Table(dict(RA=np.asarray(ma['RA'])[has][inc], DEC=np.asarray(ma['DEC'])[has][inc], dx_arcsec=x[inc], dy_arcsec=y[inc], sat_row=si[inc],
                 parent_mag=pm[inc], dolphot_mag=mref[inc], dmag=mref[inc] - pm[inc], matched_A=A[inc], matched_B=B[inc], parent_detector_dither1=det[inc]))
tab.write(f'{OUT}/ghost_candidates_{band}.ecsv', overwrite=True)
res = dict(n_circle=int(inc.sum()), n_mirror=int(mir.sum()), n_circle_neither=int((inc & neither).sum()), n_mirror_neither=int((mir & neither).sum()))
# per-detector centroid (neither-only) 
res['per_detector'] = {}
for d in sorted(set(det) - {''}):
    m = (det == d) & neither
    c, n = centroid(x[m], y[m], EX, 0.15, 6) if m.sum() > 3 else ([np.nan] * 2, 0)
    mm = np.hypot(x[m] - EX[0], y[m] - EX[1]) < RC; mi = np.hypot(x[m] + EX[0], y[m] + EX[1]) < RC
    sel = m & inc
    res['per_detector'][d] = dict(n_stars_sources=int(m.sum()), circle=int(mm.sum()), mirror=int(mi.sum()),
                                   centroid=[float(c[0]), float(c[1])],
                                   circle_mean=[float(x[sel].mean()), float(y[sel].mean())] if sel.sum() else None)
# module A vs B
for mod in 'AB':
    m = np.array([dd.startswith('NRC' + mod) for dd in det]) & neither
    c, n = centroid(x[m], y[m], EX, 0.15, 6)
    res[f'module_{mod}'] = dict(n_circle=int((m & inc).sum()), n_mirror=int((m & mir).sum()), centroid=[float(c[0]), float(c[1])])
# flux ratio vs parent mag
cn = inc & neither
res['dmag_circle_neither'] = dict(median=float(np.nanmedian((mref - pm)[cn])), std=float(np.nanstd((mref - pm)[cn])), n=int(cn.sum()))
bins = [0, 12, 13, 14, 15, 15.5, 16, 17, 18, 19, 30]
res['by_parent_mag'] = {}
# detection fraction: number of sat stars in bin vs circle sources/mirror sources
s = np.ma.filled(ca[f'replaced_saturated_f{band.lower()}'], 0).astype(bool)
allmag = fl(ca[f'mag_vega_f{band.lower()}'])[s]
for lo, hi in zip(bins[:-1], bins[1:]):
    nst = int(((allmag >= lo) & (allmag < hi)).sum())
    m = (pm >= lo) & (pm < hi)
    # sources counted per nearest saturated star: unique parents with a circle source
    npar = len(np.unique(si[cn & m])); nmir = len(np.unique(si[mir & neither & m]))
    res['by_parent_mag'][f'{lo}-{hi}'] = dict(n_sat_stars=nst, parents_with_circle_src=npar, parents_with_mirror_src=nmir,
                                               median_dmag=float(np.nanmedian((mref - pm)[cn & m])) if (cn & m).sum() else None, n=int((cn & m).sum()))
# also unmatched-in-both dolphot mag distribution vs parent mag slope
if cn.sum() > 5:
    p = np.polyfit(pm[cn & np.isfinite(pm) & np.isfinite(mref)], mref[cn & np.isfinite(pm) & np.isfinite(mref)], 1)
    res['mag_vs_parentmag_fit_slope_intercept'] = [float(p[0]), float(p[1])]
json.dump(res, open(f'{OUT}/dolphot_side_{band}.json', 'w'), indent=1)
print(json.dumps(res, indent=1))
fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
ax[0].hist2d(x, y, bins=np.arange(-1.2, 1.2, 0.04), cmap='gray_r'); ax[0].scatter(*EX, s=300, fc='none', ec='r'); ax[0].set_title('dolphot-only offsets rel. nearest sat star (all)')
ax[1].scatter(pm[cn], (mref - pm)[cn], s=6); ax[1].set_xlabel('parent mag'); ax[1].set_ylabel('dolphot mag - parent mag'); ax[1].set_title('circle sources, unmatched both')
ax[2].scatter(pm[cn], mref[cn], s=6); ax[2].set_xlabel('parent mag'); ax[2].set_ylabel('dolphot mag')
fig.savefig(f'{OUT}/fig/dolphot_side_{band}.png', dpi=90)
