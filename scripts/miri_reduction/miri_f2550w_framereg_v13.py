#!/usr/bin/env python
"""
Brick F2550W v13: per-frame cross-correlation registration (2026-06-12).

User found a doubled star (17:46:08.63 -28:42:32.5 + ghost 4.2" away at
PA~165deg) in v11/v12 but not v2a/v1: a subset of frames carries a ~4"
WCS error (2024 tweakreg + hard-coded-shift era).  In v2a the displaced
copies were OUTLIER-rejected (part of the original overflagging); the
v4-v12 trim/skymatch fixes reduced rejection and let both positions
drizzle in.

Fix the frames: cross-correlate each frame's extended 25um structure
against the v2a mosaic (whose drizzle median followed the majority WCS),
measure (dx, dy) to sub-pixel by quadratic peak interpolation, and shift
each frame's FITS CRVAL.  Frames are then mutually consistent; the global
frame is anchored afterwards with the measured offset-histogram correction
as before.  Rebuild with the v12 recipe (robust tile constants, skymatch
skipped, E56 trim).
"""
import os
import json
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from scipy.ndimage import median_filter
from scipy.signal import fftconvolve
from reproject import reproject_interp
import warnings
warnings.filterwarnings('ignore')

basepath = '/orange/adamginsburg/jwst/brick'
pipedir = f'{basepath}/F2550W/pipeline'
outdir = f'{basepath}/F2550W/pipeline_v2'
os.environ["CRDS_PATH"] = f"{basepath}/crds/"
os.environ["CRDS_SERVER_URL"] = "https://jwst-crds.stsci.edu"

TRIM_EAST, TRIM_WEST, TRIM_ROWS = 56, 16, 12
FITMASK_EAST = 100
MAXSHIFT = 60  # pixels, search window

ref_fh = fits.open(f'{outdir}/jw02221-o002_t001_miri_f2550w_v2a_i2d.fits')
ref_hdr = ref_fh[1].header.copy()
if 'OLCRVAL1' in ref_hdr:  # v2a in pipeline_v2 is uncorrected, but be safe
    ref_hdr['CRVAL1'] = ref_hdr['OLCRVAL1']
    ref_hdr['CRVAL2'] = ref_hdr['OLCRVAL2']
ref = ref_fh[1].data
ref_wcs = WCS(ref_hdr)

asn_file = f'{outdir}/jw02221-o002_t001_miri_f2550w_v2a_asn.json'
with open(asn_file) as fh:
    asn_data = json.load(fh)
members = [m['expname'] if os.path.isabs(m['expname'])
           else os.path.join(pipedir, m['expname'])
           for m in asn_data['products'][0]['members']]


def xcorr_shift(img, refimg):
    """(dy, dx) that moves img onto refimg, sub-pixel."""
    def prep(a):
        a = a - np.nanmedian(a)
        a = np.nan_to_num(a)
        return a - median_filter(a, size=65)  # high-pass: structure, not pedestal
    a, b = prep(img), prep(refimg)
    corr = fftconvolve(b, a[::-1, ::-1], mode='same')
    cy, cx = np.array(corr.shape) // 2
    win = corr[cy - MAXSHIFT:cy + MAXSHIFT + 1, cx - MAXSHIFT:cx + MAXSHIFT + 1]
    py, px = np.unravel_index(np.argmax(win), win.shape)
    # quadratic sub-pixel refinement
    def subpix(v, i):
        if 0 < i < len(v) - 1 and (2 * v[i] - v[i-1] - v[i+1]) != 0:
            return i + 0.5 * (v[i+1] - v[i-1]) / (2 * v[i] - v[i-1] - v[i+1])
        return float(i)
    sy = subpix(win[:, px], py) - MAXSHIFT
    sx = subpix(win[py, :], px) - MAXSHIFT
    return sy, sx


shifts = {}
new_members = []
for fn in members:
    f2 = fits.open(fn)
    d = f2['SCI'].data
    dq = f2['DQ'].data
    ww = WCS(f2['SCI'].header)
    refproj, _ = reproject_interp((ref, ref_wcs), ww, shape_out=d.shape)
    valid = ((dq & 1) == 0) & np.isfinite(refproj) & np.isfinite(d)
    dmask = np.where(valid, d, np.nan)
    rmask = np.where(valid, refproj, np.nan)
    sy, sx = xcorr_shift(dmask, rmask)
    # convert pixel shift to CRVAL shift: moving the image by (sx, sy) pixels
    # equals shifting CRPIX by (-sx, -sy); implement via CRPIX (exact for any
    # rotation), FITS header only.
    shifts[os.path.basename(fn)] = (sx, sy)
    f2[1].header['CRPIX1'] = f2[1].header['CRPIX1'] + sx
    f2[1].header['CRPIX2'] = f2[1].header['CRPIX2'] + sy
    f2[1].header['XCORRDX'] = (sx, 'xcorr registration shift [pix] applied to CRPIX1')
    f2[1].header['XCORRDY'] = (sy, 'xcorr registration shift [pix] applied to CRPIX2')
    # NOTE: the embedded ASDF gwcs is NOT updated; resample reads the gwcs!
    # Therefore ALSO update the gwcs via tweakreg adjust_wcs (works on cal
    # frames, which these are).
    from jwst.datamodels import ImageModel
    from jwst.tweakreg.utils import adjust_wcs
    import astropy.units as u
    pixscale = np.sqrt(np.abs(np.linalg.det(ww.pixel_scale_matrix))) * 3600
    # pixel shift -> sky shift: use local WCS linearization
    sky0 = ww.pixel_to_world(d.shape[1] / 2, d.shape[0] / 2)
    sky1 = ww.pixel_to_world(d.shape[1] / 2 - sx, d.shape[0] / 2 - sy)
    dra = (sky1.ra - sky0.ra) * np.cos(sky0.dec)
    ddec = sky1.dec - sky0.dec
    fa = ImageModel(fn)
    ww_new = adjust_wcs(fa.meta.wcs, delta_ra=dra.to(u.arcsec), delta_dec=ddec.to(u.arcsec))
    fa.meta.wcs = ww_new
    outfn = fn.replace('_align.fits', '_xreg13.fits')
    fa.save(outfn, overwrite=True)
    # re-apply trim to the saved copy
    f3 = fits.open(outfn)
    dq3 = f3['DQ'].data
    colgood = ((dq3 & 1) == 0).any(axis=0)
    sci = np.where(colgood)[0]
    dq3[:, sci.max() - TRIM_EAST + 1:] |= 1
    dq3[:, :sci.min() + TRIM_WEST] |= 1
    dq3[:TRIM_ROWS, :] |= 1
    dq3[-TRIM_ROWS:, :] |= 1
    f3['DQ'].data = dq3
    f3[1].header['XCORRDX'] = (sx, 'xcorr registration dx [pix]')
    f3[1].header['XCORRDY'] = (sy, 'xcorr registration dy [pix]')
    f3.writeto(outfn, overwrite=True)
    new_members.append(outfn)
    print(f'{os.path.basename(fn)}: shift dx={sx:+.2f} dy={sy:+.2f} px '
          f'({sx*pixscale:+.2f}", {sy*pixscale:+.2f}")', flush=True)

print('\nshift summary by visit:')
vis = {}
for k, (sx, sy) in shifts.items():
    vis.setdefault(k[:19], []).append((sx, sy))
for v, ss in vis.items():
    ss = np.array(ss)
    print(f'  {v}: median dx={np.median(ss[:,0]):+.2f} dy={np.median(ss[:,1]):+.2f} px')

# tile sky constants on the registered frames (v12 recipe)
tiles = {}
for fn in new_members:
    tiles.setdefault(os.path.basename(fn)[:19], []).append(fn)
ny, nx = ref.shape
tile_meds = {}
for key, fns in tiles.items():
    stack = []
    for fn in fns:
        f2 = fits.open(fn)
        d = f2['SCI'].data.copy()
        dq = f2['DQ'].data
        bad = (dq & 1) > 0
        colgood = (~bad).any(axis=0)
        sci = np.where(colgood)[0]
        d[:, sci.max() - (FITMASK_EAST - TRIM_EAST) + 1:] = np.nan
        d[bad] = np.nan
        r, _ = reproject_interp((d, WCS(f2['SCI'].header)), ref_wcs, shape_out=(ny, nx))
        stack.append(r)
        f2.close()
    tile_meds[key] = np.nanmedian(np.array(stack, dtype='float32'), axis=0)
keys = list(tile_meds)
consts = {}
for k in keys:
    others = np.nanmedian(np.array([tile_meds[o] for o in keys if o != k], dtype='float32'), axis=0)
    diff = tile_meds[k] - others
    _, med, _ = sigma_clipped_stats(diff[np.isfinite(diff)], sigma=3, maxiters=5)
    consts[k] = med
meanc = np.mean(list(consts.values()))
print('tile constants:', {k: round(v - meanc, 1) for k, v in consts.items()})
for k, fns in tiles.items():
    corr = consts[k] - meanc
    for fn in fns:
        f2 = fits.open(fn)
        f2['SCI'].data = (f2['SCI'].data - corr).astype(f2['SCI'].data.dtype)
        f2[1].header['SKYCONST'] = (corr, 'tile sky constant subtracted [MJy/sr]')
        f2.writeto(fn, overwrite=True)

from jwst.pipeline import calwebb_image3
name = 'jw02221-o002_t001_miri_f2550w_v13'
asn_data['products'][0]['name'] = name
asn_data['products'][0]['members'] = [
    {'expname': fn, 'exptype': 'science'} for fn in new_members]
asn_out = f'{outdir}/{name}_asn.json'
with open(asn_out, 'w') as fh:
    json.dump(asn_data, fh)
os.chdir(pipedir)
calwebb_image3.Image3Pipeline.call(
    asn_out,
    steps={'tweakreg': {'skip': True},
           'skymatch': {'skip': True},
           'outlier_detection': {'snr': '30.0 25.0',
                                 'good_bits': 'SATURATED, JUMP_DET'}},
    output_dir=outdir,
    save_results=True)

# apply the global astrometric anchor (same measured offset; the xcorr step
# registered frames to v2a's frame, which carries the original error)
fh = fits.open(f'{outdir}/{name}_i2d.fits')
dra_as, ddec_as = -4.697, -2.469
cosd = np.cos(np.deg2rad(fh[1].header['CRVAL2']))
fh[1].header['OLCRVAL1'] = fh[1].header['CRVAL1']
fh[1].header['OLCRVAL2'] = fh[1].header['CRVAL2']
fh[1].header['CRVAL1'] = fh[1].header['CRVAL1'] - dra_as / 3600. / cosd
fh[1].header['CRVAL2'] = fh[1].header['CRVAL2'] - ddec_as / 3600.
fh[1].header['MIRIDRA'] = (-dra_as, 'applied RA correction [arcsec]')
fh[1].header['MIRIDDE'] = (-ddec_as, 'applied Dec correction [arcsec]')
fh[1].header['MIRIWCSN'] = ('FITS WCS corrected; ASDF gwcs NOT corrected',
                            'offset-histogram registration to NIRCam refcat')
fh.writeto(f'{outdir}/{name}_i2d.fits', overwrite=True)
print('applied WCS anchor to v13 product')
print('ALL DONE')
