"""Select bright, unsaturated, isolated stars in one NIRCam _cal frame and cut stamps.

For each frame:

1. background map: 30th percentile in 64x64-px blocks (crowded field: this is the
   confusion floor, not the true sky), bilinearly interpolated;
2. peaks: 3x3 local maxima more than 5 sigma above it (all of them -- they are the
   neighbour list), plus every connected SATURATED region as an infinitely bright
   neighbour;
3. candidates: peak S/N > ``snr_min``; NOT saturated in ANY group -- the peak pixel's
   accumulated signal at the LAST group, rate x TGROUP x NGROUPS in DN, must stay below
   ``SAT_FRACTION`` of the per-pixel CRDS saturation level (the cal DQ only flags a
   pixel that was saturated early enough to lose the ramp, so a star that saturates
   in groups 3-4 is unflagged but non-linear); no DQ SATURATED pixel anywhere within
   the stamp (+3 px); no DO_NOT_USE / NaN pixel within 3 px of the peak;
4. neighbours: a peak is a real neighbour (not a ring / spike of the candidate) if it
   exceeds 3x what the candidate's own PSF (STPSF, detector frame, peak-normalised)
   predicts at that offset.  Each real neighbour's light is predicted with the same 2-D
   template (spikes included); stamp pixels where it exceeds 5% of the candidate's own
   predicted signal AND 0.5 sigma are masked.  A candidate is rejected if any pixel
   within 2.5 px of its peak is masked, or more than 40% of the stamp is;
5. sky position of every accepted star from the GWCS (``meta.wcs``, never the SIP
   header), with the local 2x2 Jacobian so a refined sub-pixel centroid can be
   converted without the frame.

Output per frame (npz): star table + core stamps (and, for the brightest, larger
"wing" stamps), background-subtracted and divided by the peak amplitude, float16
(1e-3 relative precision, far below the per-pixel noise).
"""
import os

import numpy as np
from astropy.io import fits
from scipy import ndimage
from scipy.spatial import cKDTree
import stdatamodels.jwst.datamodels as dm

SAT_FRACTION = 0.5          # of the CRDS saturation level (which includes the bias)
DQ_DNU, DQ_SAT = 1, 2

CHANNEL = {
    'LW': dict(r_core=12, r_wing=30, snr_min=40., n_wing=80, snr_wing=150.),
    'SW': dict(r_core=12, r_wing=40, snr_min=40., n_wing=80, snr_wing=150.),
}


def channel_of(detector):
    return 'LW' if 'long' in detector.lower() else 'SW'


def background_map(v, bad, block=64, pct=30):
    ny, nx = v.shape
    w = np.where(bad, np.nan, v).reshape(ny // block, block, nx // block, block)
    b = np.nanpercentile(w.transpose(0, 2, 1, 3).reshape(ny // block, nx // block, -1), pct, axis=2)
    b = np.where(np.isfinite(b), b, np.nanmedian(b))
    yc = (np.arange(ny) + 0.5) / block - 0.5
    xc = (np.arange(nx) + 0.5) / block - 0.5
    yy, xx = np.meshgrid(yc, xc, indexing='ij')
    return ndimage.map_coordinates(b, [yy, xx], order=1, mode='nearest')


def box_sum(im):
    """Summed-area table with a zero row/col prepended."""
    s = np.zeros((im.shape[0] + 1, im.shape[1] + 1))
    s[1:, 1:] = im.cumsum(0).cumsum(1)
    return s


def box_count(sat, x0, y0, x1, y1):
    return sat[y1, x1] - sat[y0, x1] - sat[y1, x0] + sat[y0, x0]


class Template:
    """Peak-normalised detector-frame PSF (STPSF DET_DIST) looked up at integer offsets.

    Padded to half-size ``H`` so that any offset up to H is a plain index; beyond the
    STPSF footprint the azimuthal-mean profile is continued as r^-3.
    """

    def __init__(self, psf, H=160):
        p = psf / psf.max()
        h = psf.shape[0] // 2
        yy, xx = np.mgrid[-H:H + 1, -H:H + 1]
        r = np.hypot(yy, xx)
        edge = np.mean(p[np.abs(np.hypot(*np.mgrid[-h:h + 1, -h:h + 1]) - h) < 0.5])
        T = edge * (h / np.maximum(r, 1)) ** 3
        T[H - h:H + h + 1, H - h:H + h + 1] = p
        self.T, self.H = T, H

    def __call__(self, dy, dx):
        dy = np.clip(np.round(np.asarray(dy)).astype(int), -self.H, self.H)
        dx = np.clip(np.round(np.asarray(dx)).astype(int), -self.H, self.H)
        return self.T[dy + self.H, dx + self.H]


def neighbour_mask(ix, iy, ac, R, nb_xy, nb_a, errs, tmpl, yy, xx, max_masked=0.5, check_core=True):
    """Mask for a (2R+1)^2 stamp centred on integer (ix, iy).  Returns (mask, reject).

    A stamp pixel is masked where a real neighbour's predicted light (2-D STPSF
    template, spikes included) exceeds 3 sigma -- per neighbour, not summed: the sum of
    hundreds of distant r^-3 tails in this field is a smooth confusion floor that
    the fitted background absorbs, and summing it would mask every pixel.  The candidate is rejected if
    neighbours add more than 2% to any pixel within 2 px of its peak or more than 10%
    within 3 px, if any pixel within 3 px is masked, or if more than half the stamp is.
    Fainter contamination is left in: it is a quasi-uniform confusion floor absorbed
    by the fitted local background, and the clipped stack rejects the rest.
    """
    r = np.hypot(xx, yy)
    core = r <= 3
    mask = np.zeros(yy.shape, bool)
    if len(nb_a) == 0:
        return mask, False
    nb_xy = np.asarray(nb_xy)
    fin = nb_a > 0
    d = nb_xy[fin] - [ix, iy]
    amp = nb_a[fin][:, None]

    def contam_at(py, px):
        return (amp * tmpl(py[None] - d[:, 1, None], px[None] - d[:, 0, None])).max(0)

    if check_core:
        # cheap test on the r <= 3 core pixels first: most rejections happen here
        cy, cx = yy[core], xx[core]
        cc = ac * tmpl(cy, cx)
        for (dxn, dyn), area in zip(nb_xy[~fin] - [ix, iy], -nb_a[~fin]):
            if np.any(np.hypot(cx - dxn, cy - dyn) < 6 + 2 * np.sqrt(area)):
                return mask, True
        if fin.any():
            ct = contam_at(cy, cx)
            r2 = r[core] <= 2
            if (np.any(ct[r2] > 0.02 * cc[r2]) or np.any(ct > 0.1 * cc)
                    or np.any(ct > 3 * errs[core])):
                return mask, True
    for (dxn, dyn), area in zip(nb_xy[~fin] - [ix, iy], -nb_a[~fin]):   # saturated star: mask a disk
        mask |= np.hypot(xx - dxn, yy - dyn) < 6 + 2 * np.sqrt(area)
    if fin.any():
        mask |= contam_at(yy.ravel(), xx.ravel()).reshape(yy.shape) > 3 * errs
    reject = (check_core and bool(mask[core].any())) or mask.mean() > max_masked
    return mask, reject


def select_frame(fn, satref_fn, template_psf, outfn, channel=None):
    with fits.open(fn) as f:
        h0 = f[0].header
        h1 = f['SCI'].header
        sci = f['SCI'].data.astype(np.float64)
        err = f['ERR'].data.astype(np.float64)
        dq = f['DQ'].data.astype(np.uint32)
    det = h0['DETECTOR'].lower()
    ch = channel or channel_of(det)
    P = CHANNEL[ch]
    satref = fits.getdata(satref_fn, 'SCI').astype(np.float64)
    tgroup, ngroups = h0['TGROUP'], h0['NGROUPS']
    photmjsr = h1['PHOTMJSR']

    bad = ~np.isfinite(sci) | ((dq & DQ_DNU) > 0) | ~np.isfinite(err)
    sat = (dq & DQ_SAT) > 0
    v = np.where(bad, 0.0, sci)
    bkg = background_map(v, bad | sat)
    a = v - bkg
    e = np.where(bad | (err <= 0), np.inf, err)
    snr = a / e

    locmax = (ndimage.maximum_filter(v, size=3) == v) & (snr > 5) & ~bad & ~sat
    py, px = np.nonzero(locmax)
    pa = a[py, px]
    # saturated regions as infinitely bright neighbours
    lab, nlab = ndimage.label(ndimage.binary_dilation(sat, iterations=1))
    if nlab:
        com = np.array(ndimage.center_of_mass(sat, lab, np.arange(1, nlab + 1)))
        sy, sx = com[:, 0], com[:, 1]
        sarea = ndimage.sum(sat, lab, np.arange(1, nlab + 1))
    else:
        sy = sx = sarea = np.zeros(0)
    allx = np.concatenate([px, sx]); ally = np.concatenate([py, sy])
    alla = np.concatenate([pa, -np.maximum(sarea, 1)])   # < 0: saturated, -area
    tree = cKDTree(np.c_[allx, ally])

    dn_last = v / photmjsr * tgroup * ngroups
    satfrac = dn_last / np.where(np.isfinite(satref) & (satref > 0), satref, np.inf)
    bad_core = ndimage.binary_dilation(bad | sat, iterations=3)
    satsum = box_sum(sat.astype(np.int32))

    R, Rw = P['r_core'], P['r_wing']
    ny, nx = v.shape
    ci = np.nonzero((snr[py, px] > P['snr_min']) & (satfrac[py, px] < SAT_FRACTION)
                    & ~bad_core[py, px]
                    & (px >= R + 1) & (px < nx - R - 1) & (py >= R + 1) & (py < ny - R - 1))[0]
    ci = ci[np.argsort(-pa[ci])]
    tmpl = Template(template_psf)
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    yw, xw = np.mgrid[-Rw:Rw + 1, -Rw:Rw + 1]
    m = 3
    rows, core_s, core_e, core_m = [], [], [], []
    wing_rows, wing_s, wing_e, wing_m = [], [], [], []
    nrej = dict(sat=0, nbr=0)
    for i in ci:
        ix, iy, ac = px[i], py[i], pa[i]
        if box_count(satsum, max(ix - R - m, 0), max(iy - R - m, 0),
                     min(ix + R + m + 1, nx), min(iy + R + m + 1, ny)) > 0:
            nrej['sat'] += 1
            continue
        nb = [j for j in tree.query_ball_point([ix, iy], R * 1.415 + 6) if not (allx[j] == ix and ally[j] == iy)]
        nbx, nby, nba = allx[nb], ally[nb], alla[nb]
        # real neighbour = more than 3x what the candidate's own PSF predicts there
        pred = ac * tmpl(np.round(nby - iy).astype(int), np.round(nbx - ix).astype(int))
        real = (nba < 0) | (nba > 3 * pred + 5 * e[np.clip(nby.astype(int), 0, ny - 1), np.clip(nbx.astype(int), 0, nx - 1)])
        es = e[iy - R:iy + R + 1, ix - R:ix + R + 1]
        mask, rej = neighbour_mask(ix, iy, ac, R, np.c_[nbx[real], nby[real]], nba[real], es, tmpl, yy, xx)
        if rej:
            nrej['nbr'] += 1
            continue
        st = a[iy - R:iy + R + 1, ix - R:ix + R + 1]
        mk = mask | bad[iy - R:iy + R + 1, ix - R:ix + R + 1]
        c = v[iy - 1:iy + 2, ix - 1:ix + 2]
        den = c[1, 0] - 2 * c[1, 1] + c[1, 2]
        dx = 0.5 * (c[1, 0] - c[1, 2]) / den if den < 0 else 0.0
        den = c[0, 1] - 2 * c[1, 1] + c[2, 1]
        dy = 0.5 * (c[0, 1] - c[2, 1]) / den if den < 0 else 0.0
        rows.append((ix + np.clip(dx, -0.5, 0.5), iy + np.clip(dy, -0.5, 0.5), ix, iy, ac,
                     bkg[iy, ix], snr[iy, ix], satfrac[iy, ix], mk.mean()))
        core_s.append((st / ac).astype(np.float16))
        core_e.append(np.where(np.isfinite(es), es / ac, 6e4).astype(np.float16))
        core_m.append(mk)
        # wing stamp for the brightest
        if (len(wing_rows) < P['n_wing'] and snr[iy, ix] > P['snr_wing']
                and ix >= Rw and ix < nx - Rw and iy >= Rw and iy < ny - Rw):
            # saturated stars inside the wing stamp (beyond the SAT-free core stamp) are
            # masked with a disk of radius 6 + 2 sqrt(saturated area) px
            nbw = [j for j in tree.query_ball_point([ix, iy], Rw * 1.415 + 10)
                   if not (allx[j] == ix and ally[j] == iy)]
            wx, wy, wa = allx[nbw], ally[nbw], alla[nbw]
            predw = ac * tmpl(np.round(wy - iy).astype(int), np.round(wx - ix).astype(int))
            realw = (wa < 0) | (wa > 3 * predw + 5 * e[np.clip(wy.astype(int), 0, ny - 1), np.clip(wx.astype(int), 0, nx - 1)])
            esw = e[iy - Rw:iy + Rw + 1, ix - Rw:ix + Rw + 1]
            maskw, rejw = neighbour_mask(ix, iy, ac, Rw, np.c_[wx[realw], wy[realw]], wa[realw], esw, tmpl, yw, xw,
                                          max_masked=0.7, check_core=False)
            if not rejw:
                mkw = maskw | bad[iy - Rw:iy + Rw + 1, ix - Rw:ix + Rw + 1]
                wing_rows.append(len(rows) - 1)
                wing_s.append((a[iy - Rw:iy + Rw + 1, ix - Rw:ix + Rw + 1] / ac).astype(np.float16))
                wing_e.append(np.where(np.isfinite(esw), esw / ac, 6e4).astype(np.float16))
                wing_m.append(mkw)
    rows = np.array(rows, dtype=np.float64).reshape(-1, 9)

    # sky positions from the GWCS (ASTROMETRY RULE #2: never the SIP header)
    with dm.open(fn) as model:
        w = model.meta.wcs
        x0, y0 = rows[:, 0], rows[:, 1]
        ra, dec = w(x0, y0)
        rax, decx = w(x0 + 0.5, y0)
        ray, decy = w(x0, y0 + 0.5)
    cosd = np.cos(np.deg2rad(dec))
    jac = np.c_[(rax - ra) * 2 * cosd, (ray - ra) * 2 * cosd, (decx - dec) * 2, (decy - dec) * 2]  # deg/px

    np.savez_compressed(
        outfn,
        fname=os.path.basename(fn), detector=det, filter=h0['FILTER'], pupil=h0['PUPIL'],
        expstart=h0['EXPSTART'], date_obs=h0['DATE-OBS'] + 'T' + h0['TIME-OBS'],
        obs=h0['OBSERVTN'], visit=h0['VISIT'], expnum=h0['EXPOSURE'],
        photmjsr=photmjsr, tgroup=tgroup, ngroups=ngroups, nints=h0.get('NINTS', 1),
        x0=rows[:, 0], y0=rows[:, 1], ix=rows[:, 2].astype(np.int16), iy=rows[:, 3].astype(np.int16),
        amp=rows[:, 4], bkg0=rows[:, 5], snr=rows[:, 6], satfrac=rows[:, 7], maskfrac=rows[:, 8],
        ra0=ra, dec0=dec, jac=jac,
        core_s=np.array(core_s, dtype=np.float16).reshape(-1, 2 * R + 1, 2 * R + 1),
        core_e=np.array(core_e, dtype=np.float16).reshape(-1, 2 * R + 1, 2 * R + 1),
        core_m=np.packbits(np.array(core_m, dtype=bool).reshape(-1, (2 * R + 1) ** 2), axis=1),
        wing_idx=np.array(wing_rows, dtype=np.int32),
        wing_s=np.array(wing_s, dtype=np.float16).reshape(-1, 2 * Rw + 1, 2 * Rw + 1),
        wing_e=np.array(wing_e, dtype=np.float16).reshape(-1, 2 * Rw + 1, 2 * Rw + 1),
        wing_m=np.packbits(np.array(wing_m, dtype=bool).reshape(-1, (2 * Rw + 1) ** 2), axis=1),
        n_peaks=len(px), n_cand=len(ci), n_rej_sat=nrej['sat'], n_rej_nbr=nrej['nbr'],
        r_core=R, r_wing=Rw,
    )
    return len(rows), len(wing_rows), len(ci), nrej
