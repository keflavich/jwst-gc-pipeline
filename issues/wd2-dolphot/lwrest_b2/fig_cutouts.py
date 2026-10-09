"""b2c_cutouts.png: representative b2/c stars. Panels: (a,b) per-frame crf SCI cutouts in two frames, (c) merged m7 residual i2d."""
import glob, sys, warnings
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
import astropy.units as u
warnings.filterwarnings('ignore')
Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
sys.path.insert(0, Q)
import analyze as an
an.ZPWIN.update(an.zp_windows())
A = an.Arm('main2kfpk')
ZP = {'277W': 24.128923927404763, '250M': 24.315577424137363, '300M': 23.970356429490028}
picks = [('277W', 5943, 'R2 replace_saturated 2nd pass'), ('250M', 5218, 'R2 replace_saturated 2nd pass'), ('300M', 4980, 'R2 replace_saturated 2nd pass'),
         ('277W', 4803, 'S sigma-clip masks all frames'), ('300M', 4293, 'S sigma-clip masks all frames'), ('300M', 3187, 'P position 0.099" off (sat star)'),
         ('277W', 3555, 'c frames scattered (2 faint)'), ('277W', 4901, 'c too bright (neighbour wing)'), ('250M', 3648, 'c split star (1-frame fragment nearest)')]
HW = 24; PIXS = 0.063
fig, axs = plt.subplots(len(picks), 3, figsize=(11.5, 4.0*len(picks)))
cache = {}
for r, (b, idx, mech) in enumerate(picks):
    T = f'{Q}/tree_main2kfpk'
    if b not in cache:
        cache[b] = (Table.read(f'{T}/catalogs/f{b.lower()}_merged_indivexp_merged_resbgsub_m7_dao_basic.fits'), )
    mt = cache[b][0]
    sk = SkyCoord(A.m['RA'][idx]*u.deg, A.m['DEC'][idx]*u.deg); ref = float(A.ref[b][idx])
    msk = mt['skycoord']; mm = -2.5*np.log10(np.asarray(mt['flux'], float)) + ZP[b]
    sep = msk.separation(sk).arcsec
    near = np.where(sep < 0.08)[0]
    mtxt = f'{mm[near[np.argmin(sep[near])]]:.2f}' if len(near) else f'none <0.08" (nearest {sep.min():.2f}")'
    B = Table.read(f'basetrace_{b}.ecsv'); fr = B[B['dolphot_idx'] == idx]
    frames = list(dict.fromkeys(fr['frame']))
    if not frames:
        R = Table.read(f'rows_{b}.ecsv'); frames = list(dict.fromkeys(R[(R['dolphot_idx'] == idx) & (R['kind'] == 'frame')]['where']))
    frames = frames[:2] + frames[:1] * (2 - len(frames[:2]))
    for c, frname in enumerate(frames):
        det, exp = frname.split('_'); exp = int(exp)
        f = glob.glob(f'{T}/F{b}/pipeline/jw03523005001_*_{exp:05d}_{det}_align_o005_crf.fits')[0]
        sci = fits.getdata(f, 'SCI').astype(float); w = WCS(fits.getheader(f, 'SCI'))
        x0, y0 = [float(v) for v in w.world_to_pixel(sk)]
        pf = glob.glob(f'{T}/F{b}/f{b.lower()}_{det}_visit001_vgroup*_exp{exp:05d}_resbgsub_m7_daophot_basic.fits')[0]
        pft = Table.read(pf)
        acc = Table.read(f.replace('.fits', '_resbgsub_m7_satstar_catalog.fits'))
        ix, iy = int(round(x0)), int(round(y0))
        xs = slice(max(ix-HW, 0), ix+HW+1); ys = slice(max(iy-HW, 0), iy+HW+1)
        cut = sci[ys, xs]; ok = np.isfinite(cut)
        lo, hi = (np.nanpercentile(cut[ok], 2), np.nanpercentile(cut[ok], 99.7)) if ok.any() else (0, 1)
        ext = (xs.start-.5, xs.start+cut.shape[1]-.5, ys.start-.5, ys.start+cut.shape[0]-.5)
        ax = axs[r, c]
        ax.imshow(np.arcsinh((cut-lo)/max(hi-lo, 1e-6)*10), origin='lower', extent=ext, cmap='gray')
        ax.imshow(np.where(~ok, 1.0, np.nan), origin='lower', extent=ext, cmap='autumn_r', alpha=.6, vmin=0, vmax=1.5)
        ax.scatter([x0], [y0], s=220, facecolors='none', edgecolors='red', lw=1.4)
        ax.scatter(pft['x_fit'], pft['y_fit'], s=70, marker='+', c='cyan', lw=1.2)
        ax.scatter(acc['xcentroid'], acc['ycentroid'], s=70, marker='x', c='lime', lw=1.6)
        mx, my = w.world_to_pixel(msk[np.isfinite(msk.ra.deg)][:0]) if False else w.world_to_pixel(msk[(sep < 1.0)])
        ax.scatter(mx, my, s=70, marker='x', c='magenta', lw=1.4)
        for xx, yy, mg in zip(mx, my, mm[sep < 1.0]): ax.text(xx+0.7, yy+0.7, f'{mg:.1f}', color='magenta', fontsize=6)
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.tick_params(labelsize=6)
        ax.set_title(f'{det} exp{exp}', fontsize=7)
    ax = axs[r, 2]
    rf = f'{T}/F{b}/pipeline/jw03523-o005_t001_nircam_clear-f{b.lower()}-merged_resbgsub_m7_daophot_basic_mergedcat_residual_i2d.fits'
    with fits.open(rf) as hh:
        hd = hh['SCI'] if 'SCI' in hh else hh[1]
        c2 = Cutout2D(hd.data, sk, 3*u.arcsec, wcs=WCS(hd.header), mode='partial', fill_value=np.nan)
    v = c2.data; fin_ = np.isfinite(v)
    ax.imshow(np.arcsinh((v-np.nanpercentile(v, 2))/max(np.nanpercentile(v, 99.7)-np.nanpercentile(v, 2), 1e-6)*10), origin='lower', cmap='gray')
    xx, yy = c2.wcs.world_to_pixel(sk); ax.scatter([xx], [yy], s=220, facecolors='none', edgecolors='red', lw=1.4)
    sel = sep < 1.0
    mx, my = c2.wcs.world_to_pixel(msk[sel]); ax.scatter(mx, my, s=70, marker='x', c='magenta', lw=1.4)
    for x_, y_, mg in zip(mx, my, mm[sel]): ax.text(x_+0.7, y_+0.7, f'{mg:.1f}', color='magenta', fontsize=6)
    ax.set_xlim(0, v.shape[1]); ax.set_ylim(0, v.shape[0]); ax.tick_params(labelsize=6)
    ax.set_title('merged m7 residual i2d', fontsize=7)
    axs[r, 1].set_title(f'F{b} idx {idx} | dolphot {ref:.2f} | merged {mtxt} | {mech}\n'+axs[r,1].get_title(), fontsize=8)
axs[0, 0].text(0.02, 0.98, 'red o dolphot | cyan + frame m7 rows | magenta x merged rows (mag) | green x accepted satstar', transform=axs[0, 0].transAxes, fontsize=5, color='yellow', va='top')
plt.tight_layout(rect=(0, 0, 1, 1))
plt.savefig('b2c_cutouts.png', dpi=60)
print('saved')
