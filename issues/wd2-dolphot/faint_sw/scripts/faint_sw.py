"""Faint SW stars that move > 0.1 mag between arms A and B.  usage: python faint_sw.py A B
Writes faint_sw/frames_{A}_{B}.ecsv (per star-frame), stars_{A}_{B}.ecsv (per star), tables_{A}_{B}.md"""
import sys
import numpy as np
from astropy.table import Table, vstack
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from common import *

A_, B_ = sys.argv[1:3]
TA, TB = f'{Q}/tree_{A_}', f'{Q}/tree_{B_}'
an.ZPWIN.update(an.zp_windows())
A, B = an.Arm(A_), an.Arm(B_)
L = [f'## faint SW movers, {B_} vs {A_} (moved: both matched, both finite, |B-A| > 0.1 mag, dolphot mag > 20)']
star_tabs, frame_tabs = [], []
W = 3  # half-window 7x7 for PSF-weighted model difference

for band in BANDS4:
    b = band
    both = A.matched & B.matched & np.isfinite(A.our[b]) & np.isfinite(B.our[b]) & np.isfinite(A.ref[b]) & (A.ref[b] > 20)
    d = B.our[b] - A.our[b]
    mv = both & (np.abs(d) > 0.1)
    ctl = both & ~mv
    ids = np.where(mv | ctl)[0]
    ia, ib = A.idx[ids], B.idx[ids]
    pos = A.sky[ia]
    st = Table()
    st['band'] = [band] * len(ids); st['i'] = ids; st['moved'] = mv[ids]
    st['ref'] = A.ref[b][ids]; st['dmA'] = A.dm(b)[ids]; st['dmB'] = B.dm(b)[ids]; st['dBA'] = d[ids]
    st['RA'] = pos.ra.deg; st['DEC'] = pos.dec.deg
    lo = 'f' + band.lower()
    for nm, arm, ix in (('A', A, ia), ('B', B, ib)):
        c = arm.cat
        for col in ('nmatch', 'qfit', 'satstar_nframes', 'forced_filled', 'near_saturated_%s_%s' % (lo, lo), 'forced_refit_frac', 'flux_err'):
            full = f'{col}_{lo}' if not col.startswith('near_') else col
            if full in c.colnames:
                st[f'{col}_{nm}'] = an.fl(np.ma.asarray(c[full]).astype(float))[ix]
        st[f'spike_{nm}'] = arm.spike[ix] if arm.spike is not None else False
        st[f'rep_{nm}'] = arm.rep[b][ids]
    # crowding: m8 neighbours within 1" in A
    cs = A.sky
    i1, i2, s2, _ = cs.search_around_sky(pos, 1 * u.arcsec)
    st['nneigh1'] = np.bincount(i1, minlength=len(ids)) - 1
    # satstar positions pooled per arm over all frames of this band
    fa, fb = frame_files(TA, band), frame_files(TB, band)
    keys = [k for k in sorted(set(fa) & set(fb)) if pipe_stem(TA, band, *k) and pipe_stem(TB, band, *k)]
    print(band, 'frames', len(keys), 'of', len(set(fa) | set(fb)), flush=True)
    satpos = {'A': [], 'B': []}
    fsat = {}
    for k in keys:
        for nm, tree in (('A', TA), ('B', TB)):
            s = pipe_stem(tree, band, *k)
            t = Table.read(s + '_catalog.fits')
            c = sky_cols(t, 'skycoord_fit') if len(t) else None
            if c is not None:
                c = c[np.isfinite(c.ra.deg)]
            fsat[(nm,) + k] = (c, len(t))
            if c is not None:
                satpos[nm].append(c)
    def pool(l):
        return SkyCoord(np.concatenate([c.ra.deg for c in l]) * u.deg, np.concatenate([c.dec.deg for c in l]) * u.deg)
    sA, sB = pool(satpos['A']), pool(satpos['B'])
    st['dsatA'] = pos.match_to_catalog_sky(sA)[1].arcsec
    st['dsatB'] = pos.match_to_catalog_sky(sB)[1].arcsec
    st['dsat'] = np.minimum(st['dsatA'], st['dsatB'])
    # per frame
    sig = FWHM_PX[band] / 2.3548
    yy, xx = np.mgrid[-W:W + 1, -W:W + 1]
    P = np.exp(-(xx ** 2 + yy ** 2) / (2 * sig ** 2)); P /= P.sum()
    rows = []
    for k in keys:
        det, exp = k
        ta, tb = Table.read(fa[k]), Table.read(fb[k])
        ta, tb = ta[np.isfinite(sky_cols(ta, 'skycoord_centroid').ra.deg)], tb[np.isfinite(sky_cols(tb, 'skycoord_centroid').ra.deg)]
        ca, cb = sky_cols(ta, 'skycoord_centroid'), sky_cols(tb, 'skycoord_centroid')
        nA, nB = fsat[('A',) + k][1], fsat[('B',) + k][1]
        idx, sep, _ = pos.match_to_catalog_sky(ca)
        det_a = sep.arcsec < 0.1
        if not det_a.any():
            continue
        idxb, sepb, _ = pos.match_to_catalog_sky(cb)
        sA_, sB_ = A_stem = pipe_stem(TA, band, det, exp), pipe_stem(TB, band, det, exp)
        hdr = fits.getheader(sA_ + '_residual.fits')
        wcs = WCS(hdr)
        mA = fits.getdata(sA_ + '_model.fits'); mB = fits.getdata(sB_ + '_model.fits')
        D = (mB - mA).astype(float)
        sel = np.where(det_a)[0]
        x, y = wcs.world_to_pixel(pos[sel])
        xi, yi = np.round(x).astype(int), np.round(y).astype(int)
        ok = (xi >= W) & (yi >= W) & (xi < D.shape[1] - W) & (yi < D.shape[0] - W)
        # in-frame nearest satstar (A or B)
        cl = [fsat[(n,) + k][0] for n in 'AB' if fsat[(n,) + k][0] is not None]
        allsat = pool(cl) if cl else None
        for j, s in enumerate(sel):
            if not ok[j]:
                continue
            win = D[yi[j] - W:yi[j] + W + 1, xi[j] - W:xi[j] + W + 1]
            dpsf = float(np.nansum(win * P) / np.sum(P * P))
            d3 = float(np.nansum(D[yi[j] - 1:yi[j] + 2, xi[j] - 1:xi[j] + 2]))
            dm = float(mA[yi[j], xi[j]]); 
            fA = float(ta['flux_fit'][idx[s]])
            fB = float(tb['flux_fit'][idxb[s]]) if sepb[s].arcsec < 0.1 else np.nan
            ds = float(pos[s].separation(allsat).arcsec.min()) if allsat is not None else np.nan
            rows.append((band, int(s), det, exp, nA, nB, fA, fB, dpsf, d3, float(mA[yi[j], xi[j]]), ds))
        print(band, k, nA, nB, len(sel), flush=True)
    ft = Table(rows=rows, names=['band', 'k', 'det', 'exp', 'nsatA', 'nsatB', 'fA', 'fB', 'Dpsf', 'D3x3', 'Amodel_pix', 'dsat_frame'])
    ft['collapsed'] = ft['nsatB'] > 2 * np.maximum(ft['nsatA'], 1)
    ft['i'] = ids[ft['k']]
    frame_tabs.append(ft)
    star_tabs.append(st)

stars = vstack(star_tabs); frames = vstack(frame_tabs)
stars.write(f'{FS}/stars_{A_}_{B_}.ecsv', overwrite=True)
frames.write(f'{FS}/frames_{A_}_{B_}.ecsv', overwrite=True)
print('done')
