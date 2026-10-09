"""Star-scale structure of the m6 residual smoothed background at ZP-window stars.

For each band: sample the m6 residual_smoothed_bg mosaic at each main2 ZP-window star
(centre pixel) and the median over a 6-10 px ring.  dbg = centre - ring, scaled to a
magnitude bias of the m7 fit with dmag_bg = -2.5 log10(1 + dbg * NEFF / F), with F the
star's mean old-grid forced-fit flux (closure frames_<band>.ecsv, 5x5) and NEFF = 10 px.
Reports the slope of dmag_bg against pred - predT (the transposition term).
"""
import glob
import sys

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an  # noqa: E402

an.ZPWIN.update(an.zp_windows())
A = an.Arm('main2')
PT = np.load(f'{an.Q}/psfsum/predT_main2.npz')
P = np.load(f'{an.Q}/photomver/areapred_main2.npz')
NEFF = 10.0
yy, xx = np.mgrid[-10:11, -10:11]
rr = np.hypot(xx, yy)
ring = (rr >= 6) & (rr <= 10)
for b in sys.argv[1:]:
    fn = glob.glob(f'{an.Q}/tree_main2/F{b}/pipeline/*resbgsub_m6_daophot_basic_mergedcat_residual_smoothed_bg_i2d.fits')[0]
    with fits.open(fn) as fh:
        ext = 'SCI' if 'SCI' in fh else 0
        img = np.asarray(fh[ext].data, float)
        w = WCS(fh[ext].header)
    S = Table.read(f'{an.Q}/gridfix/stars_{b}.ecsv')
    F = Table.read(f'{an.Q}/gridfix/frames_{b}.ecsv')
    ids = np.asarray(S['i'])
    fmean = {int(i): np.nanmean(F['fold5'][F['i'] == i]) for i in ids}
    sky = SkyCoord(A.cat[f'skycoord_f{b.lower()}'])[A.idx[ids]]
    x, y = w.world_to_pixel(sky)
    out = []
    for k, i in enumerate(ids):
        ix, iy = int(round(x[k])), int(round(y[k]))
        if not (10 < ix < img.shape[1] - 11 and 10 < iy < img.shape[0] - 11):
            out.append(np.nan); continue
        cut = img[iy - 10:iy + 11, ix - 10:ix + 11]
        dbg = cut[10, 10] - np.nanmedian(cut[ring])
        out.append(-2.5 * np.log10(1 + dbg * NEFF / fmean[int(i)]))
    d = np.array(out)
    lA = P[b][ids] - PT[b][ids]
    ok = np.isfinite(d) & np.isfinite(lA)
    s = np.polyfit(lA[ok], d[ok], 1)[0]
    print(f'F{b}: N {ok.sum()}  rstd dmag_bg {an.mad(d[ok]):.4f}  median {np.median(d[ok]):+.4f}  '
          f'slope vs (pred - predT) {s:+.3f}  corr {np.corrcoef(lA[ok], d[ok])[0, 1]:+.3f}')
