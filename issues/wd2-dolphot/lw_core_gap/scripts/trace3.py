"""Per-frame DQ geometry for the trace.py no-row stars vs a same-mag control.
For every F277W frame (nrcblong + nrcalong) that covers the star:
  * distance from the star's frame pixel to the nearest SATURATED-DQ pixel
    (the m7 near-saturation filter drops fits with distance <= 1.0 px);
  * size of the SATURATED component holding (or nearest to) the star and the
    number of bright dolphot stars inside it;
  * distance from the star to that component's centre of mass, and to the
    nearest hand-off position (component COMs > 1.5 FWHM from an accepted
    satstar, plus gate rejects) -- the filter exempts fits within 1.0 px;
  * distance to the nearest accepted satstar (xcentroid/ycentroid).
usage: python trace3.py ARM BAND"""
import glob
import sys
from collections import Counter
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import astropy.units as u
from scipy import ndimage
from scipy.spatial import cKDTree
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an

SAT = 2
FWHM = {'277W': 1.48, '250M': 1.33, '300M': 1.58}
arm, b = sys.argv[1:3]
fwhm = FWHM[b]
T = f'{an.Q}/tree_{arm}'
P = f'{T}/F{b}/pipeline'
tr = Table.read(f'trace_{arm}_{b}.ecsv')
an.ZPWIN.update(an.zp_windows())
A = an.Arm(arm)
rs_all = SkyCoord(np.asarray(A.m['RA'], float) * u.deg, np.asarray(A.m['DEC'], float) * u.deg)
lo, hi = np.nanmin(tr['ref_mag']), np.nanmax(tr['ref_mag'])
ctrl = np.where(A.matched & np.isfinite(A.our[b]) & (A.ref[b] >= lo) & (A.ref[b] <= hi))[0]
noro = np.asarray(tr['dolphot_idx'])
# all bright dolphot stars (for counting stars per component)
bright_all = np.where(np.isfinite(A.ref[b]) & (A.ref[b] <= hi + 1))[0]
rb = rs_all[bright_all]

frames = sorted(glob.glob(f'{P}/jw03523005001_*_nrc?long_align_o005_crf.fits'))
rows = []
for f in frames:
    dq = fits.getdata(f, 'DQ')
    hdr = fits.getheader(f, 'SCI')
    w = WCS(hdr)
    sat = (dq & SAT) != 0
    lab, n = ndimage.label(sat)
    sizes = np.bincount(lab.ravel())
    com = np.asarray(ndimage.center_of_mass(sat, lab, np.arange(1, n + 1))).reshape(-1, 2)[:, ::-1]
    dist, (iy_n, ix_n) = ndimage.distance_transform_edt(~sat, return_indices=True)
    acc = Table.read(f.replace('.fits', '_resbgsub_m7_satstar_catalog.fits'))
    axy = np.column_stack([np.asarray(acc['xcentroid'], float), np.asarray(acc['ycentroid'], float)])
    axy = axy[np.isfinite(axy).all(1)]
    atree = cKDTree(axy) if len(axy) else None
    # hand-off component positions (COM > 1.5 FWHM of accepted)
    hand = com
    if atree is not None:
        d_ca, _ = atree.query(com)
        hand = com[d_ca > 1.5 * fwhm]
    rej = Table.read(f.replace('.fits', '_resbgsub_m7_satstar_rejected.fits'))
    if len(rej) and 'reject_reason' in rej.colnames:
        rr = np.asarray(rej['reject_reason']).astype(str)
        gx = np.column_stack([np.asarray(rej['xcentroid'], float), np.asarray(rej['ycentroid'], float)])[rr == 'implied_peak_gate']
        gx = gx[np.isfinite(gx).all(1)]
        if len(gx):
            hand = np.vstack([hand, gx])
    htree = cKDTree(hand)
    xb, yb = w.world_to_pixel(rb)
    for grp, idx in (('no-row', noro), ('control', ctrl)):
        x, y = w.world_to_pixel(rs_all[idx])
        ix, iy = np.rint(x).astype(int), np.rint(y).astype(int)
        inside = (ix >= 4) & (ix < dq.shape[1] - 4) & (iy >= 4) & (iy < dq.shape[0] - 4)
        for k in np.where(inside)[0]:
            d_sat = dist[iy[k], ix[k]]
            ny_, nx_ = iy_n[iy[k], ix[k]], ix_n[iy[k], ix[k]]
            L = lab[ny_, nx_]
            csize = sizes[L]
            cx, cy = com[L - 1]
            d_com = np.hypot(cx - x[k], cy - y[k])
            d_hand, _ = htree.query([x[k], y[k]])
            d_acc = atree.query([x[k], y[k]])[0] if atree is not None else np.inf
            inL = 0
            sel = (np.abs(xb - x[k]) < 40) & (np.abs(yb - y[k]) < 40)
            for xx, yy in zip(xb[sel], yb[sel]):
                jx, jy = int(np.rint(xx)), int(np.rint(yy))
                if 0 <= jx < dq.shape[1] and 0 <= jy < dq.shape[0] and lab[jy, jx] == L:
                    inL += 1
            rows.append((grp, int(idx[k]), f.split('/')[-1][:30], float(d_sat), int(csize),
                         float(d_com), float(d_hand), float(d_acc), inL))
t = Table(rows=rows, names=('grp', 'dolphot_idx', 'frame', 'd_sat', 'comp_size', 'd_com',
                            'd_hand', 'd_acc', 'n_bright_in_comp'))
t.write(f'trace3_{arm}_{b}.ecsv', overwrite=True)
out = []
for grp in ('no-row', 'control'):
    s = t[t['grp'] == grp]
    onsat = s['d_sat'] <= 1.0
    exempt = s['d_hand'] <= 1.0
    nearacc = s['d_acc'] <= 1.5 * fwhm
    out.append(f'### {arm} F{b} {grp}: {len(s)} star-frames, {len(set(s["dolphot_idx"]))} stars')
    out.append(f'  d_sat<=1 px (filter applies): {onsat.sum()} ({onsat.mean():.2f})')
    out.append(f'  of those: within 1 px of a hand-off position (exempt): {(onsat & exempt).sum()}; '
               f'within 1.5 FWHM of accepted satstar: {(onsat & nearacc).sum()}')
    so = s[onsat]
    if len(so):
        out.append(f'  comp size (px) of d_sat<=1: median {np.median(so["comp_size"]):.0f}, '
                   f'p90 {np.percentile(so["comp_size"], 90):.0f}')
        out.append(f'  bright dolphot stars per comp: {dict(sorted(Counter(np.minimum(so["n_bright_in_comp"], 6)).items()))} (6 = 6+)')
        out.append(f'  d_com (star->comp COM) px: median {np.median(so["d_com"]):.2f}, '
                   f'frac >1 px {(so["d_com"] > 1).mean():.2f}')
    # per-star: fraction of frames with filter applied and not exempt
    stars = sorted(set(s['dolphot_idx']))
    nf = np.array([np.sum((s['dolphot_idx'] == st) & (s['d_sat'] <= 1) & (s['d_hand'] > 1)) for st in stars])
    nt = np.array([np.sum(s['dolphot_idx'] == st) for st in stars])
    out.append(f'  stars with every covering frame "on-sat & not exempt": {(nf == nt).sum()} of {len(stars)}')
print('\n'.join(out))
open(f'trace3_{arm}_{b}.md', 'w').write('\n'.join(out) + '\n')
