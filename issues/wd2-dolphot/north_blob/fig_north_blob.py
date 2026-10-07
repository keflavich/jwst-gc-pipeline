"""Cutouts of saturated stars that have an unmatched dolphot source in the north blob (F150W, ~0.69" N).
Columns: F150W merged mosaic, then two individual-frame crf cutouts (different dithers/detectors).
red circles = all dolphot sources (blob source = thick yellow-edged circle); lime + = A-catalog (mainfcbg m8) rows;
cyan x = saturated star position.  Reads the scratch list north_blob_f150w_unmatched.ecsv made by the analysis script.
usage: nice -19 python fig_north_blob.py"""
import glob
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
import astropy.units as u
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = '/orange/adamginsburg/jwst/wd2'
K = f'{D}/dolphot_benchmark/Q_integ/kf_lost'
M8 = f'{D}/dolphot_benchmark/Q_integ/tree_mainfcbg/catalogs/basic_merged_indivexp_photometry_tables_merged_resbgsub_m8_dedup.fits'
MOS = f'{D}/F150W/pipeline/jw03523-o005_t001_nircam_clear-f150w-merged_i2d.fits'
HALF = 1.5  # arcsec half-width -> 3" cutout
NEX = 6

lst = Table.read(f'{K}/north_blob_f150w_unmatched.ecsv')
ca = Table.read(M8)
cat_all = SkyCoord(ca['skycoord_ref'])
mt = Table.read(f'{D}/dolphot_benchmark/matched_Q_mainfcbg.fits')
dol = SkyCoord(np.asarray(mt['RA']), np.asarray(mt['DEC']), unit='deg')
sat_pos_all = cat_all  # indexed by sat_row

# candidate saturated stars: one blob source, no other saturated star within 3.2" (isolation)
satmask = np.ma.filled(ca['replaced_saturated_f150w'], 0).astype(bool)
satc = cat_all[satmask]
rows, counts = np.unique(lst['sat_row'], return_counts=True)
cands = []
for r, n in zip(rows, counts):
    s = cat_all[r]
    sep = s.separation(satc).arcsec
    if n == 1 and (np.sort(sep)[1] > 3.2):
        cands.append(r)
cands = np.array(cands)
mag = np.ma.filled(ca['mag_vega_f150w'], np.nan).astype(float)
cm = mag[cands]
print('n candidates', len(cands))
ok = np.isfinite(cm) & (cm < 18.5)
cands, cm = cands[ok], cm[ok]
# 6 examples spread in saturated-star magnitude
order = np.argsort(cm)
pick = cands[order[np.linspace(0, len(order) - 1, NEX).astype(int)]]
print('picked sat rows', pick, mag[pick])

hm = fits.open(MOS)
wm = WCS(hm['SCI'].header)
crfs = sorted(glob.glob(f'{D}/F150W/pipeline/jw03523005001_*_nrc*_align_o005_crf.fits'))
crf_wcs = {}
for f in crfs:
    with fits.open(f) as h:
        crf_wcs[f] = WCS(h['SCI'].header)


def covering(sc, margin=60):
    out = []
    for f, w in crf_wcs.items():
        x, y = w.world_to_pixel(sc)
        if margin < x < 2048 - margin and margin < y < 2048 - margin:
            out.append(f)
    return out


def stretch(a):
    a = a[np.isfinite(a)]
    lo, hi = np.percentile(a, [0.5, 99.0])
    return lo, hi


def show(ax, img, lo, hi):
    s = (img - lo) / max(hi - lo, 1e-9)
    ax.imshow(np.arcsinh(np.clip(s, 0, None) * 30) / np.arcsinh(30), origin='lower', cmap='gray_r', vmin=0, vmax=1)


def marks(ax, w, sat, blob, off=(0, 0)):
    """w: WCS of the displayed (cut) image."""
    near_d = dol[dol.separation(sat) < 2.5 * u.arcsec]
    x, y = w.world_to_pixel(near_d)
    ax.scatter(x, y, s=70, facecolors='none', edgecolors='red', lw=0.9)
    x, y = w.world_to_pixel(blob)
    ax.scatter(x, y, s=170, facecolors='none', edgecolors='yellow', lw=1.8)
    near_c = cat_all[cat_all.separation(sat) < 2.5 * u.arcsec]
    x, y = w.world_to_pixel(near_c)
    ax.scatter(x, y, s=45, marker='+', c='lime', lw=1.0)
    x, y = w.world_to_pixel(sat)
    ax.scatter(x, y, s=60, marker='x', c='cyan', lw=1.2)


fig, axes = plt.subplots(NEX, 3, figsize=(10.5, 3.4 * NEX))
for i, r in enumerate(pick):
    sat = cat_all[r]
    lr = lst[lst['sat_row'] == r][0]
    blob = SkyCoord(lr['RA'], lr['DEC'], unit='deg')
    # mosaic
    cut = Cutout2D(hm['SCI'].data, sat, 2 * HALF * u.arcsec, wcs=wm, mode='partial', fill_value=np.nan)
    lo, hi = stretch(cut.data)
    ax = axes[i, 0]
    show(ax, cut.data, lo, hi)
    marks(ax, cut.wcs, sat, blob)
    ax.set_title(f'sat row {r}, F150W={mag[r]:.1f}: mosaic\nblob dx,dy=({lr["dx_arcsec"]:+.2f},{lr["dy_arcsec"]:+.2f})"', fontsize=8)
    # crfs: pick up to 2 frames from different exposures and detectors
    cov = covering(sat)
    chosen = []
    seen = set()
    for f in cov:
        key = f.split('_')[1] + f.split('_')[2]  # exposure id
        if key not in seen:
            seen.add(key)
            chosen.append(f)
    chosen = [chosen[0], chosen[-1]] if len(chosen) > 1 else chosen[:2]
    for j in range(2):
        ax = axes[i, 1 + j]
        if j >= len(chosen):
            ax.axis('off')
            continue
        f = chosen[j]
        with fits.open(f) as h:
            data = h['SCI'].data
        w = crf_wcs[f]
        c = Cutout2D(data, sat, 2 * HALF * u.arcsec, wcs=w, mode='partial', fill_value=np.nan)
        lo, hi = stretch(c.data)
        show(ax, c.data, lo, hi)
        marks(ax, c.wcs, sat, blob)
        # north arrow from WCS
        x0, y0 = c.wcs.world_to_pixel(sat)
        xn, yn = c.wcs.world_to_pixel(SkyCoord(sat.ra, sat.dec + 0.4 * u.arcsec))
        ax.annotate('N', xy=(x0 + 0.0, y0), xytext=(0, 0), alpha=0)
        ax.arrow(c.data.shape[1] * 0.12, c.data.shape[0] * 0.12, (xn - x0), (yn - y0), color='orange', width=0.3, head_width=2)
        name = f.split('/')[-1].replace('jw03523005001_', '').replace('_align_o005_crf.fits', '')
        ax.set_title(f'crf {name}', fontsize=8)
    for ax in axes[i]:
        ax.set_xticks([]); ax.set_yticks([])
fig.suptitle('Dolphot north-blob sources near saturated stars (F150W)\nred o = dolphot; yellow = blob source; '
             'lime + = our A catalog; cyan x = saturated star; orange arrow = north (0.4")', fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.985))
fig.savefig(f'{K}/fig/dolphot_north_blob.png', dpi=100)
