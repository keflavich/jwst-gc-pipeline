"""#1019 image test, 3-5 px bin: where around the brighter star do the
additions sit, in the reference image's pixel frame?

The reference i2d grids are rotated to within ~1 deg of the NIRCam detector
frame (PA_APER 90.5 Brick F200W, 89.5 Sgr B2 F182M; PC1_2 ~ 1), so the
angle of the source about its nearest brighter base-kept star in the
reference pixel grid is the detector-frame angle.  The JWST PSF has six
hexagon spikes (one pair along detector y) and two fainter strut spikes
along detector x.  The 60-deg rotated controls of img_realness.py map the
six hexagon spikes onto themselves; only the 180-deg control maps the strut
spikes.  A source on a strut spike that has a reference-image peak there
therefore reads above its chance.

Per 30-deg bin of the angle (0 = +y of the reference pixel grid, 90 = +x),
folded to 0-180 (the PSF is point-symmetric), prints the peak fraction, the
control (chance) fraction and image realness for the additions and the
base-kept stars of the same per-frame S/N and distance range, all of them
and those with data-i2d prominence >= 5 (the sky-clean tier's prominence
floor).

usage: python lsky_pa_diag.py <field> [variant=lsky] [snr_lo=0] [snr_hi=5] [d_lo=3] [d_hi=5]
writes lsky_pa_diag_<field>_<variant>_snr<lo>-<hi>_d<lo>-<hi>.json
"""
import json
import os
import sys

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from compare import FIELDS, sky  # noqa: E402
from added_gallery import CFG  # noqa: E402
from img_realness import SMOOTH_PX, R_PEAK, ROT_DEG  # noqa: E402

ANG_EDGES = np.arange(0, 181, 30)


def main(field, v='lsky', snr_lo=0, snr_hi=5, d_lo=3, d_hi=5):
    snr_lo, snr_hi, d_lo, d_hi = map(float, (snr_lo, snr_hi, d_lo, d_hi))
    band = FIELDS[field]['band']
    base = Table.read(f'{HERE}/out/{field}_{band}_seed.fits')
    var = Table.read(f'{HERE}/out/{field}_{band}_{v}.fits')
    assert np.array_equal(np.asarray(var['rowid']), np.asarray(base['rowid']))
    kb = np.asarray(base['kept'], bool)
    add = np.asarray(var['kept'], bool) & ~kb
    sc = sky(base)
    flux = np.asarray(base['flux'], float)
    snr = flux / np.asarray(base['flux_err'], float)
    with fits.open(CFG[field]['refimg']) as fh:
        hdu = fh['SCI'] if 'SCI' in [h.name for h in fh] else fh[0]
        img = np.asarray(hdu.data, np.float32)
        w = WCS(hdu.header)
        pa_aper = float(hdu.header.get('PA_APER', fh[0].header.get('PA_APER', np.nan)))
    sm = gaussian_filter(np.nan_to_num(img, nan=-1e30), SMOOTH_PX)
    ismax = (sm == maximum_filter(sm, size=3)) & np.isfinite(img)
    del sm
    ny, nx = img.shape
    pixas = float(np.sqrt(np.abs(np.linalg.det(w.pixel_scale_matrix))) * 3600)
    x, y = (np.asarray(a, float) for a in w.world_to_pixel(sc))
    good = np.isfinite(x) & np.isfinite(y)
    ik = np.flatnonzero(kb)
    tree = cKDTree(np.c_[x[ik], y[ik]])
    want = np.flatnonzero((add | kb) & good & (snr >= snr_lo) & (snr < snr_hi))
    dd, jj = tree.query(np.c_[x[want], y[want]], k=16, distance_upper_bound=20 * 0.0311 / pixas)
    d_b = np.full(len(base), np.inf)
    j_b = np.full(len(base), -1)
    for n, i in enumerate(want):
        for d, j in zip(dd[n], jj[n]):
            if not np.isfinite(d):
                break
            if ik[j] != i and flux[ik[j]] > flux[i]:
                d_b[i], j_b[i] = d * pixas / 0.0311, ik[j]
                break
    oy, ox = np.mgrid[-2:3, -2:3]

    def peak_at(px, py):
        out = np.full(px.shape, np.nan)
        cx, cy = np.rint(px).astype(int), np.rint(py).astype(int)
        ok = (cx >= 2) & (cx < nx - 2) & (cy >= 2) & (cy < ny - 2)
        for m in np.flatnonzero(ok):
            if not np.isfinite(img[cy[m] - 2:cy[m] + 3, cx[m] - 2:cx[m] + 3]).all():
                continue
            near = ((cx[m] + ox) - px[m]) ** 2 + ((cy[m] + oy) - py[m]) ** 2 <= R_PEAK ** 2
            out[m] = float(np.any(ismax[cy[m] - 2:cy[m] + 3, cx[m] - 2:cx[m] + 3] & near))
        return out

    sel = good & (snr >= snr_lo) & (snr < snr_hi) & (d_b >= d_lo) & (d_b < d_hi) & (j_b >= 0)
    res = dict(field=field, variant=v, pa_aper=pa_aper, snr=[snr_lo, snr_hi], d=[d_lo, d_hi], sets={})
    print(f'== {field} {v}: S/N [{snr_lo:g},{snr_hi:g}), {d_lo:g}-{d_hi:g} px from a brighter kept star; '
          f'reference PA_APER {pa_aper:.2f}; angle 0 = +y, 90 = +x of the reference grid (folded to 0-180)')
    rng = np.random.default_rng(0)
    prom = np.asarray(base['prominence'], float)
    # base-kept stars with the data-i2d prominence the sky-clean tier asks of its additions
    for name, m in (('added', sel & add), ('basekept', sel & kb),
                    ('basekept_prom5', sel & kb & (prom >= 5))):
        rows = np.flatnonzero(m)
        if name == 'basekept' and rows.size > 6000:
            rows = rng.choice(rows, 6000, replace=False)
        jr = j_b[rows]
        dx, dy = x[rows] - x[jr], y[rows] - y[jr]
        ang = np.degrees(np.arctan2(dx, dy)) % 180
        p = peak_at(x[rows], y[rows])
        ctl = {}
        for a in ROT_DEG:
            ca, sa = np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))
            ctl[a] = peak_at(x[jr] + ca * dx - sa * dy, y[jr] + sa * dx + ca * dy)
        ch = np.nanmean(np.vstack(list(ctl.values())), axis=0)
        ok = np.isfinite(p) & np.isfinite(ch)
        out = []
        print(f'-- {name} (n {int(ok.sum())}): angle bin: n peak chance(all 5) chance(180 only) img_rel')
        for lo, hi in zip(ANG_EDGES[:-1], ANG_EDGES[1:]):
            b = ok & (ang >= lo) & (ang < hi)
            if b.sum() < 5:
                out.append(dict(a_lo=int(lo), a_hi=int(hi), n=int(b.sum())))
                continue
            pk, c = float(np.mean(p[b])), float(np.mean(ch[b]))
            c180 = float(np.nanmean(ctl[180][b]))
            rel = (pk - c) / (1 - c) if c < 1 else np.nan
            out.append(dict(a_lo=int(lo), a_hi=int(hi), n=int(b.sum()), peak=pk, chance=c,
                            chance180=c180, img_rel=rel))
            print(f'   [{lo:3d},{hi:3d}) n {int(b.sum()):5d} {pk:.2f} {c:.2f} {c180:.2f} {rel:5.2f}')
        res['sets'][name] = out
    tag = f'{field}_{v}_snr{snr_lo:g}-{snr_hi:g}_d{d_lo:g}-{d_hi:g}'
    with open(f'{HERE}/lsky_pa_diag_{tag}.json', 'w') as fh:
        json.dump(res, fh, indent=1, default=float)


if __name__ == '__main__':
    main(*sys.argv[1:])
