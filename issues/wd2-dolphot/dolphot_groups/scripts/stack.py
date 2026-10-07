"""Pixel stack of isolated saturated stars from crf frames.
usage: nice -19 python stack.py <band e.g. 150W> [magmin magmax]
Saves npz with sky-aligned and detector-aligned stamps (annulus-normalised, background-subtracted)."""
import sys, glob
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import proj_plane_pixel_scales
from scipy.ndimage import map_coordinates

band = sys.argv[1]
mmin = float(sys.argv[2]) if len(sys.argv) > 2 else 15.5
mmax = float(sys.argv[3]) if len(sys.argv) > 3 else 18.0
D = '/orange/adamginsburg/jwst/wd2'
OUT = f'{D}/dolphot_benchmark/Q_integ/ghost_test'
M8 = f'{D}/dolphot_benchmark/Q_integ/tree_mainfcbg/catalogs/basic_merged_indivexp_photometry_tables_merged_resbgsub_m8_dedup.fits'
ca = Table.read(M8)
b = band.lower()
sat = np.ma.filled(ca[f'replaced_saturated_f{b}'], 0).astype(bool)
mag = np.ma.filled(ca[f'mag_vega_f{b}'], np.nan).astype(float)
sc_all = SkyCoord(ca['skycoord_ref'])
rows = np.where(sat)[0]
satc = sc_all[rows]
idx, sep, _ = satc.match_to_catalog_sky(satc, nthneighbor=2)
iso = sep.arcsec > 2.0
sel = rows[iso & (mag[rows] >= mmin) & (mag[rows] <= mmax)]
print('isolated sat stars', len(sel), flush=True)
sc = sc_all[sel]

HALF = 1.2
frames = sorted(glob.glob(f'{D}/F{band}/pipeline/jw03523*_*_nrc*_align_o005_crf.fits'))
print(len(frames), 'frames')
stamps_sky, stamps_det, meta = [], [], []
for f in frames:
    with fits.open(f) as h:
        data = h['SCI'].data.astype(np.float32)
        w = WCS(h['SCI'].header)
        pa = h['SCI'].header.get('PA_V3')
        det = h[0].header['DETECTOR']
        expid = f.split('/')[-1].split('_')[1] + f.split('/')[-1].split('_')[2]
    scale = proj_plane_pixel_scales(w)[0] * 3600
    n = int(round(HALF / 0.03)); g = (np.arange(-n, n + 1)) * 0.03
    X, Y = np.meshgrid(g, g)  # X: +RA (east) arcsec, Y: +Dec
    npx = 39 if scale < 0.045 else 19
    gp = np.arange(-npx, npx + 1)
    PX, PY = np.meshgrid(gp, gp)
    xs, ys = w.world_to_pixel(sc)
    ok = (xs > 40) & (xs < 2007) & (ys > 40) & (ys < 2007)
    for k in np.where(ok)[0]:
        # sky-aligned: offsets -> sky -> pixel
        cosd = np.cos(sc[k].dec.rad)
        ra = sc[k].ra.deg + (X / 3600) / cosd
        dec = sc[k].dec.deg + Y / 3600
        px, py = w.wcs_world2pix(ra, dec, 0)
        a = map_coordinates(data, [py, px], order=1, cval=np.nan)
        d = map_coordinates(data, [ys[k] + PY, xs[k] + PX], order=1, cval=np.nan)
        stamps_sky.append(a); stamps_det.append(d)
        meta.append((k, sel[k], mag[sel[k]], det, expid, pa, scale, xs[k], ys[k]))
S = np.array(stamps_sky, dtype=np.float32); Dt = np.array(stamps_det, dtype=np.float32)
print(S.shape, Dt.shape, flush=True)
np.savez_compressed(f'{OUT}/stamps_{band}.npz', sky=S, det=Dt, k=[m[0] for m in meta], row=[m[1] for m in meta],
                    mag=[m[2] for m in meta], detector=[m[3] for m in meta], exp=[m[4] for m in meta],
                    pa=[m[5] for m in meta], scale=[m[6] for m in meta], x=[m[7] for m in meta], y=[m[8] for m in meta],
                    g=g, gp=gp)
