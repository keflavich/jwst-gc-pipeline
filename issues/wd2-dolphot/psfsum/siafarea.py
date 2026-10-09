"""Compare three pixel-area estimates on NIRCam detectors:
(1) crf AREA extension (CRDS area ref), (2) the local pixel area from the
SIAF sci->idl Jacobian (pysiaf), (3) the per-position sum of the STPSF
distorted PSF grid (expected to go as 1/area).  Tests the orientation of
each against the others."""
import glob
import sys

import numpy as np
import pysiaf
from astropy.io import fits

T = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/tree_main2'
siaf = pysiaf.Siaf('NIRCam')


def siaf_area(det, x, y, h=0.5):
    ap = siaf[f'{det.upper()}_FULL']
    # x,y are 0-based numpy pixel indices; sci frame is 1-based
    xs, ys = np.asarray(x, float) + 1, np.asarray(y, float) + 1
    x1, y1 = ap.sci_to_idl(xs - h, ys)
    x2, y2 = ap.sci_to_idl(xs + h, ys)
    x3, y3 = ap.sci_to_idl(xs, ys - h)
    x4, y4 = ap.sci_to_idl(xs, ys + h)
    dxdx, dydx = (x2 - x1) / (2 * h), (y2 - y1) / (2 * h)
    dxdy, dydy = (x4 - x3) / (2 * h), (y4 - y3) / (2 * h)
    a = np.abs(dxdx * dydy - dxdy * dydx)
    return a


def frame_for(band, det):
    fs = sorted(glob.glob(f'{T}/{band.upper()}/pipeline/*{det.lower()}*_crf.fits'))
    if not fs and det.lower().endswith('5'):
        d2 = det.lower().replace('a5', 'along').replace('b5', 'blong')
        fs = sorted(glob.glob(f'{T}/{band.upper()}/pipeline/*{d2}*_crf.fits'))
    return fs[0] if fs else None


def corr(a, b):
    return np.corrcoef(np.log(a), np.log(b))[0, 1]


def slope(a, b):
    return np.polyfit(np.log(b), np.log(a), 1)[0]


def main():
  rows = []
  for band, det in [('F200W', 'nrca1'), ('F200W', 'nrca3'), ('F200W', 'nrcb3'),
                    ('F200W', 'nrcb4'), ('F150W', 'nrcb1'), ('F212N', 'nrca2'),
                    ('F300M', 'nrca5'), ('F300M', 'nrcb5'), ('F410M', 'nrca5')]:
      fr = frame_for(band, det)
      if fr is None:
          print(band, det, 'no frame'); continue
      area = fits.getdata(fr, 'AREA')
      ny, nx = area.shape
      # grid of test positions
      g = np.linspace(50, 1997, 12)
      gx, gy = np.meshgrid(g, g)
      gx, gy = gx.ravel(), gy.ravel()
      sa = siaf_area(det, gx, gy)
      a_xy = area[gy.astype(int), gx.astype(int)]   # AREA at (x, y)
      a_T = area[gx.astype(int), gy.astype(int)]    # AREA at (y, x)
      print(f'{band} {det} {fr.split("/")[-1][:40]}')
      print(f'   siaf-area vs crf AREA[y,x] (normal): corr {corr(sa, a_xy):+.3f} slope {slope(a_xy, sa):+.3f}')
      print(f'   siaf-area vs crf AREA[x,y] (transp): corr {corr(sa, a_T):+.3f} slope {slope(a_T, sa):+.3f}')
      # PSF grid sums
      pfn = glob.glob(f'{T}/psfs/nircam_{det}_{band.lower()}_fovp101_samp2_npsf16.fits')
      if pfn:
          h = fits.getheader(pfn[0]); d = fits.getdata(pfn[0])
          os2 = h['OVERSAMP'] ** 2
          s = d.sum(axis=(1, 2)) / os2
          yx = [eval(h[f'DET_YX{i}']) for i in range(len(d))]
          py = np.array([p[0] for p in yx]); px = np.array([p[1] for p in yx])
          sa_p = siaf_area(det, px, py)
          sa_pT = siaf_area(det, py, px)
          a_p = area[np.clip(py.astype(int), 0, ny-1), np.clip(px.astype(int), 0, nx-1)]
          a_pT = area[np.clip(px.astype(int), 0, ny-1), np.clip(py.astype(int), 0, nx-1)]
          print(f'   PSF sum (header as y,x) vs siaf-area at (x,y): corr {corr(s, sa_p):+.3f} slope {slope(s, sa_p):+.3f}')
          print(f'   PSF sum (header as y,x) vs siaf-area at (y,x): corr {corr(s, sa_pT):+.3f} slope {slope(s, sa_pT):+.3f}')
          print(f'   PSF sum (header as y,x) vs crf AREA normal   : corr {corr(s, a_p):+.3f} slope {slope(s, a_p):+.3f}')
          print(f'   PSF sum (header as y,x) vs crf AREA transp   : corr {corr(s, a_pT):+.3f} slope {slope(s, a_pT):+.3f}')
          print(f'   PSF sum range {s.min():.4f}-{s.max():.4f}')
      sys.stdout.flush()


if __name__ == "__main__":
    main()
