"""3x3 grid of per-pixel value histograms feeding the outlier-detection median.

For a 3x3 block of nrca1 pixels IN the rejected bright-star zone, show the actual
set of resampled per-exposure values at that sky position (the 24 group `_i2d`
drizzled images the median is taken over) -- a genuine histogram of the numbers,
with rejection-relevant summary stats per pixel.
"""
import glob, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
import jwst.datamodels as dm
from astropy.coordinates import SkyCoord
import astropy.units as u
from jwst.datamodels import dqflags

DIST = '/blue/adamginsburg/adamginsburg/tmp/outlier_dist'
BASE = '/orange/adamginsburg/jwst/brick/F200W/pipeline'
FRAME = 'jw01182004001_04101_00001_nrca1'
PRE = f'{BASE}/{FRAME}_destreak.fits'
CRF_DEF = sorted(glob.glob(f'/blue/adamginsburg/adamginsburg/tmp/outlier_validate/default/*{FRAME}_*outlierdetectionstep.fits'))[0]
STAR = (1362, 84)   # cx, cy in nrca1
OUT = '/blue/adamginsburg/adamginsburg/tmp/claude-3663/-orange-adamginsburg-jwst/81778923-7a53-4903-85de-4e1a21cfef0f/scratchpad/pixel_stack_hist.png'

P = dqflags.pixel

# nrca1 frame WCS + sci + rejection mask
m = dm.open(PRE)
wcs_in = m.meta.wcs
sci = np.asarray(m.data, float)
err = np.asarray(m.err, float)
dqd = fits.getdata(CRF_DEF, 'DQ').astype(np.uint32)
sat = (dqd & P['SATURATED']) != 0
rej = ((dqd & P['OUTLIER']) != 0) & ~sat

# blot (the reference the step compares against), for annotation
blot = fits.getdata(sorted(glob.glob(f'{DIST}/*{FRAME}_*blot.fits'))[0], 'SCI').astype(float)

# pick a 3x3 block of REJECTED pixels in the halo (r 45-90 from star)
cx, cy = STAR
yy, xx = np.mgrid[0:sci.shape[0], 0:sci.shape[1]]
r = np.hypot(yy - cy, xx - cx)
cand = rej & (r > 45) & (r < 90) & np.isfinite(sci)
# a 3x3 fully-rejected block: find a center whose 3x3 neighborhood is all rejected
from scipy.ndimage import uniform_filter
allrej = uniform_filter(rej.astype(float), size=3) > 0.999
pick = allrej & cand
ys, xs = np.where(pick)
cyc, cxc = ys[len(ys)//2], xs[len(xs)//2]     # a representative center
print(f'3x3 block center (x,y)=({cxc},{cyc}), r={np.hypot(cxc-cx,cyc-cy):.0f} px from star')

# median grid WCS (from median.fits) + the 24 group i2d files
med = dm.open(f'{DIST}/{FRAME.rsplit("_",1)[0]}_nrca1_destreak_o004_median.fits') \
    if os.path.exists(f'{DIST}/{FRAME}_destreak_o004_median.fits') else None
med_path = sorted(glob.glob(f'{DIST}/*median.fits'))[0]
medm = dm.open(med_path)
wcs_out = medm.meta.wcs
i2ds = sorted(glob.glob(f'{DIST}/*_i2d.fits'))
print(f'{len(i2ds)} group i2d (resample stack)')

# map the 9 pixels -> sky -> grid pixel; gather the 24 values at each
def grid_xy(ix, iy):
    sk = wcs_in.pixel_to_world(ix, iy)
    gx, gy = wcs_out.world_to_pixel(sk)
    return float(gx), float(gy)

offsets = [(-1,-1),(0,-1),(1,-1),(-1,0),(0,0),(1,0),(-1,1),(0,1),(1,1)]
# preload i2d SCI as memmap
i2d_data = [fits.open(f, memmap=True) for f in i2ds]
def stack_at(gx, gy):
    ix, iy = int(round(gx)), int(round(gy))
    vals = []
    for h in i2d_data:
        d = h['SCI'].data
        if 0 <= iy < d.shape[0] and 0 <= ix < d.shape[1]:
            v = float(d[iy, ix])
            if np.isfinite(v) and v != 0.0:
                vals.append(v)
    return np.array(vals)

fig, axes = plt.subplots(3, 3, figsize=(16, 13))
for ax, (ox, oy) in zip(axes.ravel(), offsets):
    px, py = cxc + ox, cyc + oy
    gx, gy = grid_xy(px, py)
    vals = stack_at(gx, gy)
    sciv = sci[py, px]; blotv = blot[py, px]; errv = err[py, px]
    if len(vals) >= 2:
        md = np.median(vals); mn = vals.mean(); sd = vals.std()
        mad = np.median(np.abs(vals - md)) * 1.4826
        rng = vals.max() - vals.min()
        z_std = (sciv - md) / sd if sd > 0 else np.nan
        z_mad = (sciv - md) / mad if mad > 0 else np.nan
        # bins: genuine integer-ish histogram of the N numbers
        ax.hist(vals, bins=max(5, len(vals)), color='0.6', edgecolor='k')
        ax.axvline(md, color='b', ls='-', lw=1.5, label=f'median={md:.1f}')
        ax.axvline(sciv, color='r', ls='--', lw=1.5, label=f'this-exp sci={sciv:.1f}')
        ax.axvline(blotv, color='g', ls=':', lw=1.5, label=f'blot={blotv:.1f}')
        stat = (f'N={len(vals)}  median={md:.1f}  mean={mn:.1f}\n'
                f'std={sd:.1f}  MAD*1.48={mad:.1f}  range={rng:.1f}\n'
                f'err={errv:.2f}  sci-med={sciv-md:.1f}\n'
                f'z_std={z_std:.1f}  z_MAD={z_mad:.1f}')
    else:
        ax.text(0.5, 0.5, f'N={len(vals)} (no stack)', ha='center')
        stat = f'N={len(vals)}'
    ax.set_title(f'px ({px},{py})', fontsize=9)
    ax.set_xlabel('resampled value (MJy/sr)'); ax.set_ylabel('count')
    ax.legend(fontsize=6, loc='upper right')
    ax.text(0.02, 0.98, stat, transform=ax.transAxes, fontsize=7.5, va='top',
            family='monospace', bbox=dict(fc='white', alpha=0.8, ec='0.7'))
fig.suptitle(f'Values feeding the median at a 3x3 REJECTED block — {FRAME}, halo of star {STAR}\n'
             f'(each panel: the {len(i2ds)} per-exposure resampled values at that sky pixel; '
             f'red=this exposure, blue=median, green=blot)', fontsize=12)
fig.tight_layout(rect=(0,0,1,0.95))
fig.savefig(OUT, dpi=120)
print('wrote', OUT)
