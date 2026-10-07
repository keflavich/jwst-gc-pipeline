"""Cutouts of the split pairs left after the one-to-one fix (mfs1 m7): F115W and F410M, 0.5".
usage: python fig_rest.py"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.table import Table
from astropy.coordinates import search_around_sky, SkyCoord
from astropy.visualization import simple_norm
import astropy.units as u
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an
Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
M7 = 'catalogs/basic_merged_indivexp_photometry_tables_merged_resbgsub_m7.fits'
an.PATH['mfs1'] = (f'{Q}/tree_mfs1/{M7}', f'{an.D}/matched_Q_s1t1_mfs1.fits')
an.ZPWIN.update(an.zp_windows())
A = an.Arm('mfs1')
cat = A.cat
bands = [b for b in an.BANDS if f'mag_vega_f{b.lower()}' in cat.colnames]
F = np.array([np.isfinite(np.asarray(an.fl(cat[f'mag_vega_f{b.lower()}']), float)) for b in bands]).T
mk = np.where(A.matched)[0]
mrow = A.idx[mk]
i1, i2, sep, _ = search_around_sky(A.sky[mrow], A.sky, 0.08 * u.arcsec)
k = mrow[i1] != i2
i1, i2, sep = i1[k], i2[k], sep[k]
d = ~(F[mrow[i1]] & F[i2]).any(axis=1)
pairs = {}
for a, p, s in zip(i1[d], i2[d], sep[d].to_value(u.mas)):
    pairs.setdefault(tuple(sorted((mrow[a], p))), (mk[a], s))
ref = Table.read('/orange/adamginsburg/jwst/wd2/catalogs/wd2_nircam_wf_mf_nf.ecsv')
rsc = SkyCoord(ref['RA'], ref['DEC'], unit='deg')
sw = np.isfinite(np.ma.filled(ref['MAG115W'], np.nan).astype(float)) | np.isfinite(np.ma.filled(ref['MAG200W'], np.nan).astype(float))
reff = np.asarray(cat['skycoord_ref_filtername']).astype(str)
D = '/orange/adamginsburg/jwst/wd2'
mos = {}
for band in ('f115w', 'f410m'):
    h = fits.open(f'{D}/{band.upper()}/pipeline/jw03523-o005_t001_nircam_clear-{band}-merged_i2d.fits', memmap=True)
    mos[band] = (h['SCI'].data, WCS(h['SCI'].header))
n = len(pairs)
fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 5.6))
for j, ((r1, r2), (kd, s)) in enumerate(pairs.items()):
    c0 = SkyCoord(*(A.sky[[r1, r2]].ra.deg.mean(), A.sky[[r1, r2]].dec.deg.mean()), unit='deg')
    for i, band in enumerate(('f115w', 'f410m')):
        ax = axes[i, j]
        data, w = mos[band]
        c = Cutout2D(data, c0, 0.5 * u.arcsec, wcs=w, mode='partial', fill_value=np.nan)
        fin = np.isfinite(c.data)
        ax.imshow(c.data, origin='lower', cmap='gray_r', norm=simple_norm(c.data[fin], 'asinh', percent=99.7))
        near = np.where(rsc.separation(c0) < 0.25 * u.arcsec)[0]
        for nn in near:
            x, y = c.wcs.world_to_pixel(rsc[nn])
            ax.plot(x, y, 'o' if sw[nn] else 'D', mfc='none', mec='red' if sw[nn] else 'orange', ms=11 if sw[nn] else 8, mew=1.3,
                    label=('dolphot (SW values)' if sw[nn] else 'dolphot (LW-only entry)'))
        for r, mkr, col, lab in ((r1, '+', 'lime', 'row 1'), (r2, 'x', 'cyan', 'row 2')):
            x, y = c.wcs.world_to_pixel(A.sky[r])
            ax.plot(x, y, mkr, color=col, ms=10, mew=1.6, label=lab)
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            ax.set_title(f'{s:.0f} mas; dolphot F200W {A.ref["200W"][kd]:.1f}\nrow 1 {reff[r1]}, row 2 {reff[r2]}', fontsize=8)
        ax.text(0.03, 0.03, band.upper(), transform=ax.transAxes, fontsize=8, color='k',
                bbox=dict(fc='w', ec='none', alpha=0.7))
        if i == 1 and j == 0:
            hh, ll = ax.get_legend_handles_labels()
            u_ = dict(zip(ll, hh))
            ax.legend(u_.values(), u_.keys(), fontsize=6, loc='upper left', framealpha=0.7)
fig.suptitle('Split pairs left after #1122 (mfs1 m7). Lime + / cyan x: the two catalog rows (creating band in title). '
             'Red o: dolphot star with SW values; orange diamond: dolphot LW-only entry. 0.5" cutouts.', fontsize=9)
out = f'{Q}/sat1to1/fig/split_rest_mfs1.png'
fig.savefig(out, dpi=110, bbox_inches='tight')
print('wrote', out)
