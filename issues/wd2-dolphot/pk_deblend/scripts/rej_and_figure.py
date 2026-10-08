"""(1) properties of the pk3 rejected (fit_quality_gate) fits sitting on the lost stars; (2) cutouts_pk3.png."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import simple_norm
import astropy.units as u
O = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/pk_deblend_trace'
G = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/f277w_gap/dbl'
ps = Table.read(f'{O}/per_star_ext.ecsv')
star = SkyCoord(ps['ra'] * u.deg, ps['dec'] * u.deg)
lost = np.asarray(ps['lost'], bool)
P = lambda tree, e, k: f'{G}/{tree}/F277W/pipeline/jw03523005001_10101_0000{e}_nrcblong_align_o005_crf_resbgsub_m7_satstar_{k}.fits'

# (1) rejected fits at lost stars, exp1
rej = Table.read(P('tree_dbl3', 1, 'rejected'))
j, d, _ = star[lost].match_to_catalog_sky(rej['skycoord_fit'])
d = d.arcsec
m = d < 0.08
r = rej[j][m]
imp = np.asarray(r['satstar_implied_peak'], float); obs = np.asarray(r['satstar_observed_peak'], float); fl = np.asarray(r['sat_severity_floor'], float)
print('pk3 exp1 rejected fits within 0.08" of lost stars:', m.sum(), 'of', lost.sum())
print('  qfit pctl 5,50,95', np.nanpercentile(r['qfit'], [5, 50, 95]).round(2), ' flux_fit/dolphot pctl 5,50,95', np.nanpercentile(np.asarray(r['flux_fit'], float) / ps['dol_flux'][lost][m], [5, 50, 95]).round(2))
print('  implied_peak/floor pctl 5,50,95', np.nanpercentile(imp / fl, [5, 50, 95]).round(2), ' observed_peak/floor', np.nanpercentile(obs / fl, [5, 50, 95]).round(2))
faint = (imp > 0) & (imp < 0.5 * fl) & ~(np.isfinite(obs) & (obs >= 0.5 * fl)) & (np.asarray(r['flux_fit'], float) > 0)
print('  meets faint-fit-quality hand-off criterion (implied < 0.5 floor, positive):', faint.sum(), 'of', m.sum())
print('  flags', np.unique(r['flags'], return_counts=True))

# (2) figure: pick 8 lost stars spread in ref_mag
idx = np.where(lost)[0]
order = idx[np.argsort(ps['ref_mag'][idx])]
pick = order[np.linspace(0, len(order) - 1, 8).astype(int)]
m2 = Table.read(f'{G}/tree_dbl2/catalogs/f277w_merged_indivexp_merged_resbgsub_m7_dao_basic.fits')
m3 = Table.read(f'{G}/tree_dbl3/catalogs/f277w_merged_indivexp_merged_resbgsub_m7_dao_basic.fits')
cat3 = Table.read(P('tree_dbl3', 1, 'catalog'))
with fits.open(f'{G}/tree_dbl3/F277W/pipeline/jw03523005001_10101_00001_nrcblong_align_o005_crf.fits') as h:
    w = WCS(h['SCI'].header)
imgs = {}
for key, tree, k in (('pk2 residual', 'tree_dbl2', 'residual'), ('pk3 residual', 'tree_dbl3', 'residual'), ('pk3 satstar model', 'tree_dbl3', 'model')):
    imgs[key] = fits.getdata(P(tree, 1, k)).astype(float)
half = 24  # px = 1.5"
fig, axs = plt.subplots(8, 3, figsize=(10, 26))
for r_, i in enumerate(pick):
    x, y = w.world_to_pixel(star[i])
    x0, y0 = int(round(float(x))), int(round(float(y)))
    for c_, (key, img) in enumerate(imgs.items()):
        ax = axs[r_, c_]
        cut = img[max(y0 - half, 0):y0 + half + 1, max(x0 - half, 0):x0 + half + 1]
        ox, oy = max(x0 - half, 0), max(y0 - half, 0)
        v = np.nan_to_num(cut, nan=0.0)
        vmax = np.nanpercentile(np.abs(v), 99.5) if np.any(v) else 1
        ax.imshow(v, origin='lower', cmap='gray_r', norm=simple_norm(v, 'asinh', vmin=np.percentile(v, 1), vmax=max(vmax, 1e-3), asinh_a=0.1))
        ax.plot(x - ox, y - oy, 'r+', ms=16, mew=1.5)
        for tbl, col, lab in ((m2, 'tab:blue', 'pk2 merged'), (m3, 'tab:orange', 'pk3 merged')):
            sk = tbl['skycoord']
            near = sk.separation(star[i]) < 2.2 * u.arcsec
            if near.any():
                xs, ys = w.world_to_pixel(sk[near])
                ax.plot(xs - ox, ys - oy, 'o', mfc='none', mec=col, ms=14, mew=1.2)
        sk = cat3['skycoord_fit']
        near = sk.separation(star[i]) < 2.2 * u.arcsec
        if near.any():
            xs, ys = w.world_to_pixel(sk[near])
            ax.plot(xs - ox, ys - oy, 's', mfc='none', mec='limegreen', ms=18, mew=1.5)
        ax.set_xlim(-0.5, cut.shape[1] - 0.5); ax.set_ylim(-0.5, cut.shape[0] - 0.5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{key}", fontsize=9)
        if c_ == 0:
            ax.set_ylabel(f"dolphot {ps['dolphot_idx'][i]}\nF277W {ps['ref_mag'][i]:.1f}", fontsize=8)
fig.suptitle('Lost stars (pk2 merged row, pk3 none), nrcblong exp1, 3x3 arcsec; panels are satstar-stage images\nRed +: dolphot; blue o: pk2 merged; orange o: pk3 merged; green square: pk3 accepted satstar', fontsize=8, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.985))
fig.savefig(f'{O}/cutouts_pk3.png', dpi=70)
print('picked', [int(ps['dolphot_idx'][i]) for i in pick])
