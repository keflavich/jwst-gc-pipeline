"""Radial residual profiles of satstars and bright unsaturated daophot stars (read-only on pipeline products).
Usage: python radprof_measure.py ARM BAND
Writes radprof/data/prof_<ARM>_<BAND>.fits : one row per (frame, star); array columns per radial bin.

Choices:
 * satstar data = SCI of the crf file (== satstar residual + model where finite); invalid pixels (non-finite SCI,
   DQ SATURATED|DO_NOT_USE, non-finite residual/model) are excluded.
 * satstar own model = PSF grid used by the satstar fitter (SW fovp512, LW nrcb5 fovp1024) evaluated at the fitted position
   with flux_fit (verified to reproduce the satstar model image for isolated stars).
   precap model = model * flux_fit_precap / flux_fit.
 * unsaturated star: data = satstar residual image; own model = daophot fovp101 PSF grid evaluated at the fitted position with flux_fit
   (the daophot model image is truncated to a small per-star stamp, so it is not used).
 * dsumb: data with all other modelled sources removed (sat: daophot model + satstar model minus own model; unsat: daophot model minus own 21x21 stamp).
 * background = sigma-clipped median of the satstar residual image in a 1.8-2.4 arcsec annulus.
"""
import os, sys, warnings
import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, match_coordinates_sky
import astropy.units as u
from scipy.spatial import cKDTree
from photutils.aperture import CircularAperture
from stpsf.utils import to_griddedpsfmodel

warnings.simplefilter('ignore')
Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
OUT = f'{Q}/radprof/data'
sys.path.insert(0, Q)
import analyze as an  # noqa: E402

CFG = {
    'F250M': dict(vg='04101', dets=['nrcblong'], pix=0.063, key='250M', iso_as=1.0, rmax=25, fwhm=1.5, satfov=1024,
                  edges=list(np.arange(0, 4.01, 0.5)) + [5, 6, 8, 10, 12, 15, 18, 22, 25]),
    'F150W': dict(vg='10101', dets=['nrcb3', 'nrcb1', 'nrcb2'], pix=0.031, key='150W', iso_as=0.6, rmax=50, fwhm=1.7, satfov=512,
                  edges=list(np.arange(0, 4.01, 0.5)) + [5, 6, 8, 10, 12, 15, 18, 22, 26, 30, 36, 43, 50]),
}
BG_AS = (1.8, 2.4)


def clipmed(v, n=6):
    v = v[np.isfinite(v)]
    if v.size < 50:
        return np.nan
    for _ in range(n):
        med = np.median(v)
        sd = 1.4826 * np.median(np.abs(v - med))
        if sd == 0:
            break
        k = np.abs(v - med) < 3 * sd
        if k.all():
            break
        v = v[k]
    return np.median(v)


def apsum(img, x, y, r):
    """exact circular aperture sum (NaN -> 0)"""
    ap = CircularAperture((x, y), r)
    m = ap.to_mask(method='exact')
    d = m.cutout(np.nan_to_num(img), fill_value=0.0)
    return float((d * m.data).sum())


def load_grid(arm, det, band, fov):
    g = to_griddedpsfmodel(f'{Q}/tree_{arm}/psfs/nircam_{det}_{band.lower()}_fovp{fov}_samp2_npsf16.fits')
    return g[0] if isinstance(g, list) else g


def ev_box(g, xx, yy, f, x, y, ix, iy, H, rmax):
    """PSF grid evaluated with flux f at (x, y) inside +-(rmax+2) px, zero elsewhere in the H-box"""
    out = np.zeros(xx.shape)
    h = rmax + 2
    sub = (slice(H - h, H + h + 1), slice(H - h, H + h + 1))
    out[sub] = g.evaluate(xx[sub], yy[sub], f, x, y)
    return out


def process_frame(arm, band, det, exp, ref, cfg):
    vg = cfg['vg']
    tree = f'{Q}/tree_{arm}/{band}'
    pl = f'{tree}/pipeline'
    stem = f'jw03523005001_{vg}_{exp:05d}_{det}_align_o005_crf'
    p = f'{pl}/{stem}'
    dstem = f'{pl}/jw03523-o005_t001_nircam_clear-{band.lower()}-{det}_visit001_vgroup{vg}_exp{exp:05d}_resbgsub_m7_daophot_basic'
    daof = f'{tree}/{band.lower()}_{det}_visit001_vgroup{vg}_exp{exp:05d}_resbgsub_m7_daophot_basic.fits'
    h = fits.open(p + '.fits')
    sci = h['SCI'].data.astype(float)
    dq = h['DQ'].data
    w = WCS(h['SCI'].header)
    mod = fits.getdata(p + '_resbgsub_m7_satstar_model.fits').astype(float)
    res = fits.getdata(p + '_resbgsub_m7_satstar_residual.fits').astype(float)
    dmod = fits.getdata(dstem + '_model.fits').astype(float)
    gdet = 'nrcb5' if det == 'nrcblong' else det
    gsat = load_grid(arm, gdet, band, cfg['satfov'])
    gdao = load_grid(arm, gdet, band, 101)
    bad = ((dq & 3) > 0) | ~np.isfinite(sci) | ~np.isfinite(res) | ~np.isfinite(mod)
    bgimg = np.where(~bad, res, np.nan)
    dbad = ~np.isfinite(res) | ((dq & 1) > 0)
    st = Table.read(p + '_resbgsub_m7_satstar_catalog.fits')
    da = Table.read(daof)
    xs, ys = w.world_to_pixel(st['skycoord_fit'])
    xd, yd = np.asarray(da['x_fit'], float), np.asarray(da['y_fit'], float)
    fs = np.asarray(st['flux_fit'], float)
    fd = np.asarray(da['flux_fit'], float)

    def dmatch(sky):
        idx, sep, _ = match_coordinates_sky(sky, ref['sky'])
        ok = sep < 0.1 * u.arcsec
        return np.where(ok, ref['mag'][idx], np.nan)
    mag_s = dmatch(st['skycoord_fit'])
    mag_d = dmatch(da['skycoord_centroid'])
    ax = np.concatenate([xs, xd]); ay = np.concatenate([ys, yd]); af = np.concatenate([fs, fd])
    g = np.isfinite(ax) & np.isfinite(ay) & np.isfinite(af) & (af > 0)
    ax, ay, af = ax[g], ay[g], af[g]
    tk = cKDTree(np.c_[ax, ay])
    pix = cfg['pix']
    bgr = (BG_AS[0] / pix, BG_AS[1] / pix)
    H = int(np.ceil(bgr[1])) + 2
    isopx = cfg['iso_as'] / pix
    rmax = cfg['rmax']
    edges = np.array(cfg['edges'], float)
    nb = len(edges) - 1
    ny, nx = sci.shape
    tg = [('sat', i, xs[i], ys[i], fs[i], mag_s[i]) for i in range(len(st))]
    for i in range(len(da)):
        if not (np.isfinite(mag_d[i]) and fd[i] > 0 and int(da['flags'][i]) in (0, 1)):
            continue
        if mag_d[i] < cfg['dao_mag'][0] or mag_d[i] > cfg['dao_mag'][1]:
            continue
        tg.append(('dao', i, xd[i], yd[i], fd[i], mag_d[i]))
    rows = []
    B = rmax + 2
    for kind, i, x, y, f, mg in tg:
        if not (np.isfinite(x) and np.isfinite(y) and f > 0 and np.isfinite(mg)):
            continue
        ix, iy = int(round(x)), int(round(y))
        if ix < H or iy < H or ix >= nx - H or iy >= ny - H:
            continue
        sl = (slice(iy - H, iy + H + 1), slice(ix - H, ix + H + 1))
        yy, xx = np.mgrid[iy - H:iy + H + 1, ix - H:ix + H + 1]
        d = np.hypot(xx - x, yy - y)
        bg = clipmed(bgimg[sl][(d >= bgr[0]) & (d < bgr[1])])
        if not np.isfinite(bg):
            continue
        nbr = tk.query_ball_point([x, y], max(isopx, B))
        dist = np.array([np.hypot(ax[j] - x, ay[j] - y) for j in nbr])
        fn = np.array([af[j] for j in nbr])
        m_self = dist > 0.3
        dist, fn = dist[m_self], fn[m_self]
        iso = int(not ((dist < isopx) & (fn > 0.05 * f)).any()) if dist.size else 1
        strict = int(not ((dist < B) & (fn > 0.01 * f)).any()) if dist.size else 1
        if kind == 'sat':
            data = sci[sl]; badm = bad[sl]
            own = ev_box(gsat, xx, yy, f, x, y, ix, iy, H, rmax)
            nbm = dmod[sl] + (mod[sl] - own)  # all other modelled sources (daophot + other satstars)
            ratio = float(st['flux_fit_precap'][i]) / f if 'flux_fit_precap' in st.colnames else np.nan
        else:
            data = res[sl]; badm = dbad[sl]
            own = ev_box(gdao, xx, yy, f, x, y, ix, iy, H, rmax)
            stamp = (np.abs(xx - ix) <= 10) & (np.abs(yy - iy) <= 10)  # daophot model stamp is 21x21 px (verified)
            nbm = dmod[sl] - own * stamp  # daophot models of all other stars (own stamp removed)
            ratio = 1.0
        k = d < rmax
        bi = np.digitize(d[k], edges) - 1
        valid = ~badm[k]
        dv = np.where(valid, data[k] - bg, 0.0)
        mv = np.where(valid, own[k], 0.0)
        ntot = np.bincount(bi, minlength=nb)[:nb]
        nval = np.bincount(bi, weights=valid.astype(float), minlength=nb)[:nb]
        if nbm is not None:
            dvb = np.where(valid & np.isfinite(nbm[k]), data[k] - bg - nbm[k], 0.0)
        else:
            dvb = dv
        dsumb = np.bincount(bi, weights=dvb, minlength=nb)[:nb]
        dsum = np.bincount(bi, weights=dv, minlength=nb)[:nb]
        msum = np.bincount(bi, weights=mv, minlength=nb)[:nb]
        mall = np.bincount(bi, weights=np.nan_to_num(own[k]), minlength=nb)[:nb]  # model over all pixels
        # masked radius: outer edge of the contiguous inner bins with < 50% valid pixels
        frac = nval / np.maximum(ntot, 1)
        okb = np.where(frac >= 0.5)[0]
        rmask = edges[okb[0]] if okb.size else rmax
        row = dict(kind=kind, exp=exp, det=det, idx=i, x=x, y=y, flux=f, bg=bg, mag=mg, iso=iso, strict=strict,
                   ratio_precap=ratio, rmask=rmask, nval=nval, ntot=ntot, dsum=dsum, dsumb=dsumb, msum=msum, mall=mall)
        if kind == 'sat':
            row['cap_psf_frac'] = float(st['cap_psf_frac'][i]) if 'cap_psf_frac' in st.colnames else np.nan
            row['flux_precap'] = float(st['flux_fit_precap'][i])
        # encircled energy (exact apertures) at 1,2,3 FWHM and at rmax
        for n_, rr in zip(('1', '2', '3', 'R'), (cfg['fwhm'], 2 * cfg['fwhm'], 3 * cfg['fwhm'], rmax)):
            lx, ly = x - (ix - H), y - (iy - H)
            row[f'ee_mod{n_}'] = apsum(own * (d < B), lx, ly, rr)
            row[f'ee_dat{n_}'] = apsum(np.where(badm, np.nan, data - bg), lx, ly, rr) if kind == 'dao' else np.nan
        rows.append(row)
    return Table(rows=rows)


def main(arm, band):
    cfg = dict(CFG[band])
    an.ZPWIN.update(an.zp_windows())
    m = Table.read('/orange/adamginsburg/jwst/wd2/dolphot_benchmark/matched_Q_main2kf.fits')
    key = cfg['key']
    mag = an.fl(m['ref_' + key])
    ok = np.isfinite(mag)
    ref = dict(sky=SkyCoord(np.asarray(m['RA'], float)[ok] * u.deg, np.asarray(m['DEC'], float)[ok] * u.deg), mag=mag[ok])
    # saturation faint edge from apclose star list
    t0 = Table.read(f'{Q}/apclose/data/stars_{arm}_{band}.fits')
    ms = np.asarray(t0['mag_dp'][t0['kind'] == 'sat'], float)
    edge = np.nanpercentile(ms, 95)
    cfg['dao_mag'] = (edge, edge + 2.0)
    print(arm, band, 'sat faint edge', edge, 'dao mag window', cfg['dao_mag'], flush=True)
    out = []
    for det in cfg['dets']:
        for exp in (1, 2, 3, 4):
            t = process_frame(arm, band, det, exp, ref, cfg)
            print(arm, band, det, exp, len(t), flush=True)
            out.append(t)
    T = vstack(out, metadata_conflicts='silent')
    T.meta['EDGES'] = ','.join(f'{e:g}' for e in cfg['edges'])
    T.meta['SATEDGE'] = float(edge)
    T.write(f'{OUT}/prof_{arm}_{band}.fits', overwrite=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
