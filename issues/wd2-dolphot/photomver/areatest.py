"""Does the pixel-area map explain the within-detector and between-detector dm of unsaturated stars?

Our catalogue calibrates flux_fit (a unit-sum ePSF fit on the MJy/sr crf) with one constant pixel area per frame,
wcs.proj_plane_pixel_area() (merge_catalogs.py; GDC_EXPERIMENT_REPORT.md section 12).  The true point-source flux is
sum(SB_i * PIXAR_SR * AREA_i).  So ours reads faint by
    pred = 2.5 log10(PIXAR_SR * AREA(x, y) / proj_plane_pixel_area)
at the star.  pred is averaged in flux over the exposures that contain the star (the merged catalogue averages them).
For each band: dm (unsat, ZP window, N-weighted) against pred; rms and per-detector medians and 4 x 4 cell gradients
before and after subtracting pred.
usage: python areatest.py [BAND ...] > areatest.txt"""
import sys
import glob
import json
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from scipy import stats
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an

an.ZPWIN.update(an.zp_windows())
A = an.Arm('main2')
tree = f'{an.Q}/tree_main2'
sky = A.sky[A.idx]
bands = sys.argv[1:] or an.BANDS
res = {}
for band in bands:
    frames = sorted(glob.glob(f'{tree}/F{band}/pipeline/jw03523005001_*_align_o005_crf.fits'))
    if not frames:
        continue
    dm = A.dm(band)
    lo, hi = an.ZPWIN.get(band, (0, 19))
    ok = A.matched & np.isfinite(dm) & ~A.rep[band] & ~A.sat[band] & (A.ref[band] >= lo) & (A.ref[band] < hi)
    sel = np.nonzero(ok)[0]
    fsum = np.zeros(len(sel))
    nexp = np.zeros(len(sel))
    detof = np.full(len(sel), '', dtype=object)
    xy1 = np.full((len(sel), 2), np.nan)
    ratio_cd = {}
    for fn in frames:
        with fits.open(fn, memmap=True) as fh:
            h0 = fh[0].header
            h = fh['SCI'].header
            area = np.asarray(fh['AREA'].data, float)
            w = WCS(h)
            ny, nx = area.shape
            x, y = w.world_to_pixel(sky[sel])
            ix, iy = np.round(x).astype(int), np.round(y).astype(int)
            inn = (x > 10) & (x < nx - 10) & (y > 10) & (y < ny - 10)
            cd = w.proj_plane_pixel_area().to(u.sr).value
            det = h0['DETECTOR'].lower()
            ratio_cd[det] = h['PIXAR_SR'] / cd
            a = np.full(len(sel), np.nan)
            a[inn] = area[iy[inn], ix[inn]] * h['PIXAR_SR'] / cd
            good = inn & np.isfinite(a) & (a > 0)
            fsum[good] += a[good]
            nexp[good] += 1
            detof[good & (detof == '')] = det
            if fn.split('/')[-1].split('_')[2] == '00001':
                xy1[good] = np.c_[x, y][good]
    have = nexp > 0
    pred = np.full(len(sel), np.nan)
    pred[have] = 2.5 * np.log10(fsum[have] / nexp[have])
    d = dm[sel]
    m = have & np.isfinite(d)
    slope, icpt, r, p, se = stats.linregress(pred[m], d[m])
    ts = stats.theilslopes(d[m], pred[m])
    print(f'\n### F{band} (N {int(m.sum())}): dm vs pred; OLS slope {slope:.2f} +- {se:.2f} (r {r:.2f}), Theil-Sen {ts[0]:.2f} [{ts[2]:.2f}, {ts[3]:.2f}]; '
          f'rms(pred) {np.std(pred[m]):.4f}; robust std dm {1.4826 * np.median(np.abs(d[m] - np.median(d[m]))):.4f} -> dm - pred '
          f'{1.4826 * np.median(np.abs((d - pred)[m] - np.median((d - pred)[m]))):.4f}')
    print('PIXAR_SR / proj_plane_pixel_area per detector: ' + ', '.join(f'{k} {v:.4f}' for k, v in sorted(ratio_cd.items())))
    print('| detector | N | median dm | median pred | median dm - pred | cell pk-pk dm | cell pk-pk dm - pred |')
    print('|---|---|---|---|---|---|---|')
    rows = []
    for det in sorted(set(detof[m])):
        s = m & (detof == det)
        if s.sum() < 20:
            continue
        cells, cells2 = [], []
        xx, yy = xy1[s, 0], xy1[s, 1]
        okxy = np.isfinite(xx)
        for i in range(4):
            for j in range(4):
                c = okxy & (xx >= i * 512) & (xx < (i + 1) * 512) & (yy >= j * 512) & (yy < (j + 1) * 512)
                if c.sum() >= 8:
                    cells.append(np.median(d[s][c]))
                    cells2.append(np.median((d - pred)[s][c]))
        pk = (max(cells) - min(cells)) if cells else np.nan
        pk2 = (max(cells2) - min(cells2)) if cells2 else np.nan
        row = dict(det=det, n=int(s.sum()), dm=float(np.median(d[s])), pred=float(np.median(pred[s])), res=float(np.median((d - pred)[s])), pk=float(pk), pk2=float(pk2))
        rows.append(row)
        print(f"| {det} | {row['n']} | {row['dm']:+.4f} | {row['pred']:+.4f} | {row['res']:+.4f} | {pk:.3f} | {pk2:.3f} |")
    res[band] = dict(slope=float(slope), se=float(se), r=float(r), ts=[float(t) for t in ts[:4]], rows=rows)
json.dump(res, open(f'{an.Q}/photomver/areatest.json', 'w'), indent=1)
