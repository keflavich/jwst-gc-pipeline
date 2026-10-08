"""Aperture-closure measurement (read-only on pipeline products).

Usage: python apclose_measure.py ARM BAND   (ARM in main2kf/main2, BAND in F250M/F150W)
Writes apclose/data/stars_<ARM>_<BAND>.fits : one row per (frame, star) with annulus / circle sums.

Image the satstar fit ran on = SCI of <stem>_align_o005_crf.fits (verified: SCI == residual + model where SCI is finite).
Saturated cores of satstars hold truncated (DQ SATURATED) values, so closure is evaluated on annuli that
exclude the saturated region; the same annuli are used for unsaturated daophot stars.
Satstars use SCI; daophot stars use SCI - satstar model (= satstar residual image).
Background: sigma-clipped median of the satstar residual image in a distant annulus.
"""
import os
import sys
import warnings
import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, match_coordinates_sky
import astropy.units as u
from scipy.spatial import cKDTree
from photutils.aperture import CircularAperture, CircularAnnulus

warnings.simplefilter('ignore')
Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
OUT = f'{Q}/apclose/data'
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, Q)
import analyze as an  # noqa: E402

CFG = {
    'F250M': dict(vg='04101', dets=['nrcblong'], pix=0.063, key='250M', iso_as=1.0,
                  circles=[3, 5, 8], annuli=[(5, 8), (8, 12), (12, 18), (18, 26)], bg=(1.8, 2.4)),
    'F150W': dict(vg='10101', dets=['nrcb3', 'nrcb1', 'nrcb2'], pix=0.031, key='150W', iso_as=0.5,
                  circles=[4, 6, 10], annuli=[(8, 14), (14, 22), (22, 34), (34, 50)], bg=(1.8, 2.4)),
}
MAXMAG_PAD = 4.0  # daophot sample: dolphot mag < ZPWIN lower edge - 1 + MAXMAG_PAD
ISO_FRAC = 0.05


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


def mask_sum(shape_ap, img, dqbad, satbad, bg):
    """exact-overlap weighted sum of (img-bg); returns sum, n_nan, n_sat, n_dnu."""
    m = shape_ap.to_mask(method='exact')
    d = m.cutout(img, fill_value=np.nan)
    if d is None:
        return np.nan, 99, 99, 99
    w = m.data
    sel = w > 0
    dq = m.cutout(dqbad, fill_value=0)
    sq = m.cutout(satbad, fill_value=0)
    nn = int((~np.isfinite(d) & sel).sum())
    nsat = int(((sq > 0) & sel).sum())
    ndnu = int(((dq > 0) & sel).sum())
    s = np.nansum(w[sel] * (d[sel] - bg))
    return s, nn, nsat, ndnu


def process_frame(arm, band, det, exp, ref, cfg):
    vg = cfg['vg']
    tree = f'{Q}/tree_{arm}/{band}'
    stem = f'jw03523005001_{vg}_{exp:05d}_{det}_align_o005_crf'
    p = f'{tree}/pipeline/{stem}'
    daof = f'{tree}/{band.lower()}_{det}_visit001_vgroup{vg}_exp{exp:05d}_resbgsub_m7_daophot_basic.fits'
    h = fits.open(p + '.fits')
    sci = h['SCI'].data.astype(float)
    dq = h['DQ'].data
    w = WCS(h['SCI'].header)
    mod = fits.getdata(p + '_resbgsub_m7_satstar_model.fits')
    res = fits.getdata(p + '_resbgsub_m7_satstar_residual.fits')
    fin = np.isfinite(sci)
    chk = np.nanmax(np.abs((res + mod)[fin] - sci[fin]))
    bgimg = np.where(fin & ((dq & 1) == 0), res, np.nan)
    daoimg = np.where(fin, res, np.nan)
    dqbad = ((dq & 1) > 0).astype(np.uint8)
    satbad = ((dq & 2) > 0).astype(np.uint8)
    ny, nx = sci.shape
    st = Table.read(p + '_resbgsub_m7_satstar_catalog.fits')
    xs, ys = w.world_to_pixel(st['skycoord_fit'])
    da = Table.read(daof)
    xd, yd = np.asarray(da['x_fit'], float), np.asarray(da['y_fit'], float)
    sd = da['skycoord_centroid']
    fs = np.asarray(st['flux_fit'], float)
    fd = np.asarray(da['flux_fit'], float)
    # sky match to dolphot
    def dmatch(sky):
        idx, sep, _ = match_coordinates_sky(sky, ref['sky'])
        ok = sep < 0.1 * u.arcsec
        mag = np.where(ok, ref['mag'][idx], np.nan)
        return mag, sep.arcsec
    mag_s, sep_s = dmatch(st['skycoord_fit'])
    mag_d, sep_d = dmatch(sd)
    # neighbour tree: all sources, flux>0
    allx = np.concatenate([xs, xd])
    ally = np.concatenate([ys, yd])
    allf = np.concatenate([fs, fd])
    good = np.isfinite(allx) & np.isfinite(ally) & np.isfinite(allf) & (allf > 0)
    ax, ay, af = allx[good], ally[good], allf[good]
    tree_k = cKDTree(np.c_[ax, ay])
    pix = cfg['pix']
    bgr = (cfg['bg'][0] / pix, cfg['bg'][1] / pix)
    H = int(np.ceil(bgr[1])) + 2
    isopx = cfg['iso_as'] / pix
    magcut = ref['maglim'][cfg['key']]
    rows = []
    # targets
    tg = []
    for i in range(len(st)):
        tg.append(('sat', i, xs[i], ys[i], fs[i]))
    for i in range(len(da)):
        if not (np.isfinite(mag_d[i]) and mag_d[i] < magcut and fd[i] > 0):
            continue
        if int(da['flags'][i]) not in (0, 1):
            continue
        tg.append(('dao', i, xd[i], yd[i], fd[i]))
    ann = cfg['annuli']
    cir = cfg['circles']
    rmax_nb = max(isopx, max(a[1] for a in ann) + 2)
    for kind, i, x, y, f in tg:
        if not (np.isfinite(x) and np.isfinite(y) and f > 0):
            continue
        ix, iy = int(round(x)), int(round(y))
        if ix < H or iy < H or ix >= nx - H or iy >= ny - H:
            continue
        # background
        sl = (slice(iy - H, iy + H + 1), slice(ix - H, ix + H + 1))
        yy, xx = np.mgrid[iy - H:iy + H + 1, ix - H:ix + H + 1]
        d = np.hypot(xx - x, yy - y)
        bsel = (d >= bgr[0]) & (d < bgr[1])
        bg = clipmed(bgimg[sl][bsel])
        if not np.isfinite(bg):
            continue
        img = sci if kind == 'sat' else daoimg
        # neighbours
        nb = tree_k.query_ball_point([x, y], rmax_nb)
        dist = []
        fl_n = []
        for j in nb:
            dd = np.hypot(ax[j] - x, ay[j] - y)
            if dd < 0.3:  # itself
                continue
            dist.append(dd)
            fl_n.append(af[j])
        dist = np.array(dist)
        fl_n = np.array(fl_n)
        row = dict(kind=kind, exp=exp, det=det, idx=i, x=x, y=y, flux=f, bg=bg,
                   mag_dp=(mag_s if kind == 'sat' else mag_d)[i],
                   sep_dp=(sep_s if kind == 'sat' else sep_d)[i])
        if kind == 'sat':
            for c in ('flux_fit_precap', 'cap_psf_frac', 'wingcal_rmask', 'sat_area', 'qfit', 'flux_fit_raw'):
                row[c] = float(st[c][i]) if c in st.colnames else np.nan
        else:
            row['qfit'] = float(da['qfit'][i])
            row['flags'] = int(da['flags'][i])
        # equivalent saturated radius within 12 px
        sub = satbad[sl]
        row['rsat_eq'] = float(np.sqrt(((sub > 0) & (d < 12)).sum() / np.pi))
        for k, (r1, r2) in enumerate(ann):
            ap = CircularAnnulus((x, y), r1, r2)
            s, nn, ns, nd = mask_sum(ap, img, dqbad, satbad, bg)
            row[f'an{k}_sum'] = s
            row[f'an{k}_bad'] = nn + ns + nd
            row[f'an{k}_nnan'] = nn
            if kind == 'sat':
                sm, _, _, _ = mask_sum(ap, mod, dqbad * 0, satbad * 0, 0.0)
                row[f'an{k}_mod'] = sm
            R = max(isopx, r2 + 2)
            if dist.size:
                m = (dist < R) & (fl_n > ISO_FRAC * f)
                row[f'an{k}_iso'] = int(not m.any())
            else:
                row[f'an{k}_iso'] = 1
        for k, r in enumerate(cir):
            ap = CircularAperture((x, y), r)
            s, nn, ns, nd = mask_sum(ap, img, dqbad, satbad, bg)
            row[f'ci{k}_sum'] = s
            row[f'ci{k}_bad'] = nn + ns + nd
            row[f'ci{k}_nnan'] = nn
            R = isopx
            if dist.size:
                m = (dist < R) & (fl_n > ISO_FRAC * f)
                row[f'ci{k}_iso'] = int(not m.any())
            else:
                row[f'ci{k}_iso'] = 1
        rows.append(row)
    t = Table(rows=rows)
    t.meta['RSCHK'] = float(chk)
    return t, chk


def main(arm, band):
    cfg = dict(CFG[band])
    if os.environ.get('APBG'):
        cfg['bg'] = tuple(float(v) for v in os.environ['APBG'].split(','))
    if os.environ.get('APEXPS'):
        cfg['exps'] = [int(v) for v in os.environ['APEXPS'].split(',')]
    an.ZPWIN.update(an.zp_windows())
    m = Table.read('/orange/adamginsburg/jwst/wd2/dolphot_benchmark/matched_Q_main2kf.fits')
    key = cfg['key']
    mag = an.fl(m['ref_' + key])
    ok = np.isfinite(mag)
    ref = dict(sky=SkyCoord(np.asarray(m['RA'], float)[ok] * u.deg, np.asarray(m['DEC'], float)[ok] * u.deg),
               mag=mag[ok], maglim={key: an.ZPWIN[key][0] - 1 + MAXMAG_PAD})
    print(arm, band, 'dolphot rows', ok.sum(), 'mag limit for daophot sample', ref['maglim'][key], flush=True)
    out = []
    for det in cfg['dets']:
        for exp in cfg.get('exps', (1, 2, 3, 4)):
            t, chk = process_frame(arm, band, det, exp, ref, cfg)
            print(arm, band, det, exp, 'rows', len(t), 'nsat', int((t['kind'] == 'sat').sum()),
                  'max|SCI-(res+mod)|', chk, flush=True)
            out.append(t)
    T = vstack(out, metadata_conflicts='silent')
    T.meta['ZPLO'], T.meta['ZPHI'] = an.ZPWIN[key]
    T.write(f'{OUT}/stars_{arm}_{band}{os.environ.get("APTAG", "")}.fits', overwrite=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
