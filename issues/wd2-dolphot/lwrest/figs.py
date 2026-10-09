import glob, re, sys, warnings, io, contextlib
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
import astropy.units as u
from scipy import ndimage
warnings.filterwarnings('ignore')
Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
sys.path.insert(0, Q)
import analyze as an
REPO = '/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline-wd2main2kfpk'
sys.path.insert(0, REPO)
from jwst_gc_pipeline.photometry import cataloging as C
an.ZPWIN.update(an.zp_windows())
A = an.Arm('main2kfpk')
FW = {'277W': 1.48, '250M': 1.33, '300M': 1.58}
cl = Table.read('class_all.ecsv')
picks = [l.split() for l in open('picks.txt') if l.strip() and not l.startswith('#')]
HW = 24
fig, axs = plt.subplots(len(picks), 3, figsize=(12, 4 * len(picks)))
for r, (b, idx) in enumerate(picks):
    idx = int(idx)
    FWHM = FW[b]
    row = cl[(cl['band'] == b) & (cl['dolphot_idx'] == idx)][0]
    sf = Table.read(f'sf_{b}.ecsv'); sf = sf[sf['dolphot_idx'] == idx]
    # frame choice: handed-off frame if any, else on-sat frame, else first
    sel = sf[sf['d_cur'] <= max(1, .5 * FWHM)]
    if not len(sel): sel = sf[sf['on_sat']]
    if not len(sel): sel = sf
    fr = sel['frame'][int(np.argmin(sel['d_sat']))] if len(sel) else sf['frame'][0]
    det, exp = fr.split('_')
    exp = int(exp)
    P = f'{Q}/tree_main2kfpk/F{b}/pipeline'
    f = glob.glob(f'{P}/jw03523005001_*_{exp:05d}_{det}_align_o005_crf.fits')[0]
    sci = fits.getdata(f, 'SCI').astype(float); dq = fits.getdata(f, 'DQ'); err = fits.getdata(f, 'ERR').astype(float)
    w = WCS(fits.getheader(f, 'SCI'))
    sk = SkyCoord(A.m['RA'][idx] * u.deg, A.m['DEC'][idx] * u.deg)
    x0, y0 = [float(v) for v in w.world_to_pixel(sk)]
    acc = Table.read(f.replace('.fits', '_resbgsub_m7_satstar_catalog.fits'))
    rej = Table.read(f.replace('.fits', '_resbgsub_m7_satstar_rejected.fits'))
    pf = glob.glob(f'{Q}/tree_main2kfpk/F{b}/f{b.lower()}_{det}_visit001_vgroup*_exp{exp:05d}_resbgsub_m7_daophot_basic.fits')[0]
    pft = Table.read(pf)
    with contextlib.redirect_stdout(io.StringIO()):
        hxy = C._unaccepted_sat_component_xy(dq, acc, FWHM, sci=sci, data_floor=0.0, label='x', peak_min_area=50)
    hxy = np.empty((0, 2)) if hxy is None else hxy
    bad = ~np.isfinite(err) | (err == 0) | ~np.isfinite(sci) | (sci == 0)
    rest = C._handoff_restore_pixels(dq, sci, bad, hxy, acc, FWHM) if len(hxy) else np.zeros(sci.shape, bool)
    ix, iy = int(round(x0)), int(round(y0))
    xs = slice(max(ix - HW, 0), ix + HW + 1); ys = slice(max(iy - HW, 0), iy + HW + 1)
    cut = sci[ys, xs]
    ext = (xs.start - .5, xs.start + cut.shape[1] - .5, ys.start - .5, ys.start + cut.shape[0] - .5)
    ok = np.isfinite(cut)
    vmax = np.nanpercentile(cut[ok], 99.5) if ok.any() else 1
    vmin = np.nanpercentile(cut[ok], 1) if ok.any() else 0
    def marks(ax, nm=True):
        ax.scatter([x0], [y0], s=160, facecolors='none', edgecolors='red', lw=1.5, label='dolphot')
        if len(hxy):
            ax.scatter(hxy[:, 0], hxy[:, 1], s=90, marker='D', facecolors='none', edgecolors='magenta', lw=1.2, label='hand-off')
        ax.scatter(pft['x_fit'], pft['y_fit'], s=60, marker='+', c='cyan', lw=1.2, label='m7 rows')
        ax.scatter(acc['xcentroid'], acc['ycentroid'], s=60, marker='x', c='lime', lw=1.5, label='satstar')
        rx = np.asarray(rej['xcentroid'], float); ry = np.asarray(rej['ycentroid'], float)
        ax.scatter(rx, ry, s=60, marker='x', c='orange', lw=1.2, label='rejected sat')
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax = axs[r, 0]
    ax.imshow(np.arcsinh((cut - vmin) / max(vmax - vmin, 1e-6) * 10), origin='lower', extent=ext, cmap='gray')
    nanm = np.where(~ok, 1.0, np.nan)
    ax.imshow(nanm, origin='lower', extent=ext, cmap='autumn_r', alpha=.7, vmin=0, vmax=1.5)
    ax.contour(np.arange(xs.start, xs.start + cut.shape[1]), np.arange(ys.start, ys.start + cut.shape[0]), ((dq[ys, xs] & 2) != 0).astype(float), [.5], colors='white', linewidths=.6)
    marks(ax)
    ax.set_title(f'F{b} idx {idx} ref {row["ref_mag"]:.2f} {det} exp{exp}\n{row["cat"][:34]}', fontsize=8)
    if r == 0: ax.legend(fontsize=6, loc='upper right')
    ax = axs[r, 1]
    m = np.zeros(cut.shape) ; m[(dq[ys, xs] & 2) != 0] = 1; m[rest[ys, xs]] = 2; m[~ok] = 3
    ax.imshow(m, origin='lower', extent=ext, cmap=matplotlib.colors.ListedColormap(['k', '#4466aa', '#44cc88', 'gold']), vmin=0, vmax=3, interpolation='nearest')
    marks(ax)
    ax.set_title('SAT DQ (blue), restored (green), NaN (gold)\nSAT comp at star: %d px; d_acc %.1f px' % (row['max_comp'], row['min_dacc']), fontsize=8)
    ax = axs[r, 2]
    rf = f'{Q}/tree_main2kfpk/F{b}/pipeline/jw03523-o005_t001_nircam_clear-f{b.lower()}-merged_resbgsub_m7_daophot_basic_mergedcat_residual_i2d.fits'
    with fits.open(rf) as hh:
        hd = hh['SCI'] if 'SCI' in hh else hh[1]
        c2 = Cutout2D(hd.data, sk, 3 * u.arcsec, wcs=WCS(hd.header), mode='partial', fill_value=np.nan)
    v = c2.data
    ax.imshow(v, origin='lower', cmap='gray', vmin=np.nanpercentile(v, 1), vmax=np.nanpercentile(v, 99.5))
    xx, yy = c2.wcs.world_to_pixel(sk)
    ax.scatter([xx], [yy], s=160, facecolors='none', edgecolors='red', lw=1.5)
    mt = Table.read(f'{Q}/tree_main2kfpk/catalogs/f{b.lower()}_merged_indivexp_merged_resbgsub_m7_dao_basic.fits')
    mx, my = c2.wcs.world_to_pixel(mt['skycoord'])
    ax.scatter(mx, my, s=60, marker='+', c='cyan')
    ax.set_xlim(0, v.shape[1]); ax.set_ylim(0, v.shape[0])
    ax.set_title('merged m7 residual i2d (cyan + = merged rows)', fontsize=8)
for a in axs.ravel(): a.tick_params(labelsize=6)
plt.tight_layout()
plt.savefig('lwrest_cutouts.png', dpi=80)
