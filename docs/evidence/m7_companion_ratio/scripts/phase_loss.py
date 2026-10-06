"""Stars vetted in an earlier pass and missing from the final catalog.

Each phase's residual mosaic is the data minus the model of that phase's
VETTED catalog (``build_mergedcat_residuals`` is called with the vetted
path).  A star vetted at phase k that is not vetted at the final phase is
therefore subtracted in residual k and left in the final residual.

For one (catalog dir, pipeline dir, filter, module, obs token):

1. Union of the vetted catalogs m2..m6, keyed by the LAST phase that vetted
   each star (``last_phase``).
2. ``lost`` = union members with no final-phase vetted source within
   ``r_match``.
3. For each lost star, the phase after ``last_phase`` (``next_phase``):
   ``vetted_out`` when that phase's merged (pre-vetting) catalog has a source
   within ``r_match``, else ``not_fit``.  For ``not_fit`` the nearest source
   of the next phase's vetted catalog and the m7 seed (when present) are
   recorded.
4. PSF-matched residual flux and S/N at the lost position in residual
   ``last_phase`` (control: the star is subtracted there), residual
   ``next_phase`` and the final residual, plus the same in the data mosaic.
   ``frac_left`` = final-residual flux / data-mosaic flux at that position.

Positions are matched on the data mosaic's pixel grid (KD-tree), so the
match radius is in mosaic pixels.
"""
import argparse
import glob
import json
import os

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from scipy.spatial import cKDTree

PHASES = ['m2', 'm3', 'm4', 'resbgsub_m5', 'resbgsub_m6', 'resbgsub_m7']
FWHM_PIX = {'F182M': 1.99, 'F187N': 2.06, 'F210M': 2.304, 'F212N': 2.341,
            'F200W': 2.129, 'F115W': 1.29, 'F140M': 1.48, 'F162M': 1.71,
            'F150W': 1.63, 'F405N': 2.165, 'F410M': 2.15, 'F444W': 2.3,
            'F480M': 2.55, 'F470N': 2.5, 'F360M': 1.95, 'F335M': 1.85,
            'F300M': 1.65, 'F277W': 1.5, 'F356W': 1.85}


def _image_hdu(hdul):
    if 'SCI' in hdul:
        return hdul['SCI']
    for h in hdul:
        if h.data is not None and getattr(h.data, 'ndim', 0) == 2:
            return h
    raise ValueError('no 2-D image HDU')


def _one(pattern):
    hits = sorted(h for h in glob.glob(pattern) if 'badastrom' not in h)
    if len(hits) != 1:
        raise FileNotFoundError(f'{pattern}: {hits}')
    return hits[0]


def phase_files(catdir, pipedir, filt, module, obstok):
    f = filt.lower()
    out = {}
    for ph in PHASES:
        stem = f'{catdir}/{f}_{module}{obstok}_indivexp_merged_{ph}_dao_basic'
        out[ph] = dict(
            basic=stem + '.fits', vetted=stem + '_vetted.fits',
            resid=_one(f'{pipedir}/*-{f}-{module}_{ph}_daophot_basic_mergedcat_residual_i2d.fits'))
    out['data'] = _one(f'{pipedir}/*-{f}-{module}_data_i2d.fits')
    # m7 seed: the band seed + m6-residual detections (own-band on), else
    # the shared cross-band seed alone
    seeds = sorted(glob.glob(f'{catdir}/crossband_seed_manual*_{module}_{f}_i2dseed.fits'))
    band = sorted(glob.glob(f'{catdir}/crossband_seed_manual*_{module}_{f}_vetted.fits'))
    xb = sorted(h for h in glob.glob(f'{catdir}/crossband_seed_manual*.fits')
                if not h.endswith(('_vetted.fits', '_i2dseed.fits')))
    out['m7_band_seed'] = band[0] if band else None
    out['m7_xb_seed'] = xb[0] if len(xb) == 1 else None
    out['m7_seed'] = seeds[0] if seeds else out['m7_xb_seed']
    return out


class Mosaic:
    def __init__(self, path):
        self.hdul = fits.open(path, memmap=True)
        hdu = _image_hdu(self.hdul)
        self.wcs = WCS(hdu.header)
        self.data = hdu.data
        self.shape = hdu.data.shape
        from astropy.wcs.utils import proj_plane_pixel_scales
        self.pix_as = float(np.mean(proj_plane_pixel_scales(self.wcs.celestial)) * 3600)

    def xy(self, sc):
        x, y = self.wcs.world_to_pixel(sc)
        return np.asarray(x, float), np.asarray(y, float)

    def close(self):
        self.hdul.close()


def read_cat(path, mosaic):
    t = Table.read(path)
    sc = t['skycoord'] if 'skycoord' in t.colnames else t['skycoord_avg']
    sc = SkyCoord(sc)
    x, y = mosaic.xy(sc)
    t['_x'], t['_y'] = x, y
    return t


def stamp_metrics(mosaic, x, y, fwhm, half=7, r_in=4.5, r_out=7.0):
    """PSF-matched flux / S/N and compactness at (x, y) in a mosaic.

    Gaussian PSF of the filter FWHM, centred on the sub-pixel position,
    fitted over r <= 1.5 FWHM after subtracting the median of an
    r_in..r_out annulus; noise from the annulus MAD.
    """
    n = len(x)
    xi, yi = np.round(x).astype(int), np.round(y).astype(int)
    ny, nx = mosaic.shape
    ok = (xi - half >= 0) & (yi - half >= 0) & (xi + half < nx) & (yi + half < ny)
    d = np.arange(-half, half + 1)
    out = {k: np.full(n, np.nan) for k in
           ('flux', 'snr', 'sigma', 'bkg', 'peak_off', 'compact')}
    if not ok.any():
        return out
    idx = np.where(ok)[0]
    st = np.asarray(mosaic.data[(yi[idx, None, None] + d[None, :, None]),
                                (xi[idx, None, None] + d[None, None, :])], float)
    dx = d[None, None, :] - (x[idx] - xi[idx])[:, None, None]
    dy = d[None, :, None] - (y[idx] - yi[idx])[:, None, None]
    r = np.hypot(dx, dy)
    ann = (r >= r_in) & (r <= r_out)
    sann = np.where(ann & np.isfinite(st), st, np.nan)
    bkg = np.nanmedian(sann.reshape(len(idx), -1), axis=1)
    mad = np.nanmedian(np.abs(sann - bkg[:, None, None]).reshape(len(idx), -1), axis=1)
    sigma = 1.4826 * mad
    s = st - bkg[:, None, None]
    sig = fwhm / 2.3548
    p = np.exp(-0.5 * (r / sig) ** 2)
    core = r <= 1.5 * fwhm
    p = np.where(core, p, 0.0)
    p /= p.sum(axis=(1, 2))[:, None, None]
    good = np.isfinite(s) & core
    sp = np.where(good, s * p, 0.0).sum(axis=(1, 2))
    pp = np.where(good, p * p, 0.0).sum(axis=(1, 2))
    flux = sp / pp
    snr = flux * np.sqrt(pp) / sigma
    # peak offset: brightest pixel within r <= 2 px
    inner = np.where((r <= 2.0) & np.isfinite(s), s, -np.inf).reshape(len(idx), -1)
    k = np.argmax(inner, axis=1)
    pk_off = r.reshape(len(idx), -1)[np.arange(len(idx)), k]
    # compactness: aperture r<=1 FWHM over r<=2.5 FWHM (Gaussian: 0.94 / 1.0)
    a1 = np.where((r <= fwhm) & np.isfinite(s), s, 0.0).sum(axis=(1, 2))
    a2 = np.where((r <= 2.5 * fwhm) & np.isfinite(s), s, 0.0).sum(axis=(1, 2))
    for key, v in (('flux', flux), ('snr', snr), ('sigma', sigma), ('bkg', bkg),
                   ('peak_off', pk_off), ('compact', a1 / a2)):
        out[key][idx] = v
    return out


def _in_box(sc, inner):
    ra, dec, half = inner
    c = SkyCoord(ra * u.deg, dec * u.deg)
    dx, dy = c.spherical_offsets_to(sc)
    return (np.abs(dx.to_value(u.arcsec)) < half) & (np.abs(dy.to_value(u.arcsec)) < half)


def m7_seed_reason(lost, seed_dist, files, data, filt, r_match):
    """Why a star lost at m6 -> m7 is missing from the m7 fit.

    ``seeded``         in the m7 seed (fit, then dropped or merged away)
    ``companion_cut``  in the m6 vetted catalog, left out of the m7 band seed
                       by the own-band companion cut (a brighter cross-band or
                       own-band seed source within COMPFWHM FWHM)
    ``own_band_off``   no band seed (own-band off): m7 is seeded by the
                       cross-band seed alone, which does not hold the star
    ``other``          none of the above
    Only rows with ``next_phase`` = the final phase are classified.
    """
    out = np.full(len(lost), '', dtype='U16')
    m7 = np.asarray(lost['next_phase']) == PHASES[-1]
    out[m7] = 'other'
    out[m7 & (seed_dist <= r_match)] = 'seeded'
    todo = m7 & (seed_dist > r_match)
    if not todo.any():
        return out
    if files['m7_band_seed'] is None:
        out[todo] = 'own_band_off'
        return out
    bs = Table.read(files['m7_band_seed'])
    compfwhm = float(bs.meta.get('COMPFWHM', 0.0))
    origin = np.asarray(bs['seed_origin']).astype(str)
    bsc = SkyCoord(bs['skycoord'])
    own = Table.read(files['resbgsub_m6']['vetted'])
    osc = SkyCoord(own['skycoord'])
    of = np.asarray(own['flux'], float)
    pool = SkyCoord([bsc[origin == 'crossband'], osc])
    pf = np.concatenate([np.asarray(bs['flux'], float)[origin == 'crossband'], of])
    fw = FWHM_PIX[filt.upper()] * data.pix_as
    lx, ly = np.asarray(lost['x'])[todo], np.asarray(lost['y'])[todo]
    sc = data.wcs.pixel_to_world(lx, ly)
    # the lost star's own m6 vetted entry, on the mosaic pixel grid like
    # every other match in this module
    ox, oy = data.xy(osc)
    so, io = cKDTree(np.c_[ox, oy]).query(np.c_[lx, ly])
    rows = np.where(todo)[0]
    for k, r in enumerate(rows):
        if so[k] > r_match:
            continue
        sep = sc[k].separation(pool).to_value(u.arcsec)
        if np.any((sep > 0.005) & (sep < compfwhm * fw) & (pf > of[io[k]])):
            out[r] = 'companion_cut'
    return out


def _nearest(tree, x, y, k=1):
    d, i = tree.query(np.c_[x, y], k=k)
    return d, i


def run(catdir, pipedir, filt, module='merged', obstok='', r_match=1.0,
        out=None, label='', inner=None):
    """``inner`` = (ra, dec, half_arcsec): keep lost stars and controls
    inside that box only (the reference fields' evaluated inner box)."""
    files = phase_files(catdir, pipedir, filt, module, obstok)
    fwhm = FWHM_PIX[filt.upper()]
    data = Mosaic(files['data'])
    V = {ph: read_cat(files[ph]['vetted'], data) for ph in PHASES}
    B = {ph: read_cat(files[ph]['basic'], data) for ph in PHASES}
    final = PHASES[-1]
    vtree = {ph: cKDTree(np.c_[V[ph]['_x'], V[ph]['_y']]) for ph in PHASES}
    btree = {ph: cKDTree(np.c_[B[ph]['_x'], B[ph]['_y']]) for ph in PHASES}

    # union of m2..m6 vetted, keyed by the last phase that vetted the star
    ux, uy, ulast, uflux, usnr, uqfit, uiter = [], [], [], [], [], [], []
    taken = np.zeros((0, 2))
    for ph in PHASES[-2::-1]:
        t = V[ph]
        if len(taken):
            dd, _ = cKDTree(taken).query(np.c_[t['_x'], t['_y']])
            new = dd > r_match
        else:
            new = np.ones(len(t), bool)
        ux.append(t['_x'][new]); uy.append(t['_y'][new])
        ulast += [ph] * int(new.sum())
        uflux.append(np.asarray(t['flux'])[new])
        ferr = np.asarray(t['flux_err_prop'] if 'flux_err_prop' in t.colnames else t['flux_err'])
        usnr.append((np.asarray(t['flux']) / ferr)[new])
        uqfit.append(np.asarray(t['qfit'])[new])
        uiter.append(np.asarray(t['iter_found']).astype(str)[new]
                     if 'iter_found' in t.colnames else np.full(new.sum(), ''))
        taken = np.vstack([taken, np.c_[t['_x'][new], t['_y'][new]]])
    U = Table(dict(x=np.concatenate(ux), y=np.concatenate(uy), last_phase=np.array(ulast),
                   flux_last=np.concatenate(uflux), snr_last=np.concatenate(usnr),
                   qfit_last=np.concatenate(uqfit), iter_found=np.concatenate(uiter)))
    dfin, _ = vtree[final].query(np.c_[U['x'], U['y']])
    U['d_final'] = dfin
    lost = U[dfin > r_match]
    sc_lost = data.wcs.pixel_to_world(np.asarray(lost['x']), np.asarray(lost['y']))
    lost['ra'] = sc_lost.ra.deg
    lost['dec'] = sc_lost.dec.deg
    if inner is not None:
        lost = lost[_in_box(sc_lost, inner)]
        sc_lost = data.wcs.pixel_to_world(np.asarray(lost['x']), np.asarray(lost['y']))

    # next-phase classification
    nxt = np.array([PHASES[PHASES.index(p) + 1] for p in lost['last_phase']])
    lost['next_phase'] = nxt
    cat = np.full(len(lost), 'not_fit', dtype='U12')
    for col in ('nb_qfit', 'nb_snr', 'nb_flags', 'nb_flux', 'nn_dist', 'nn_flux', 'seed_dist'):
        lost[col] = np.nan
    for ph in set(nxt):
        m = nxt == ph
        db, ib = btree[ph].query(np.c_[lost['x'][m], lost['y'][m]])
        inb = db <= r_match
        cat_m = np.where(inb, 'vetted_out', 'not_fit')
        cat[m] = cat_m
        b = B[ph]
        ferr = np.asarray(b['flux_err_prop'] if 'flux_err_prop' in b.colnames else b['flux_err'])
        for col, v in (('nb_qfit', np.asarray(b['qfit'])), ('nb_flux', np.asarray(b['flux'])),
                       ('nb_snr', np.asarray(b['flux']) / ferr),
                       ('nb_flags', np.asarray(b['flags'], float))):
            arr = np.array(lost[col])
            arr[np.where(m)[0][inb]] = v[ib[inb]]
            lost[col] = arr
        dv, iv = vtree[ph].query(np.c_[lost['x'][m], lost['y'][m]])
        arr = np.array(lost['nn_dist']); arr[m] = dv; lost['nn_dist'] = arr
        arr = np.array(lost['nn_flux']); arr[m] = np.asarray(V[ph]['flux'])[iv]; lost['nn_flux'] = arr
    lost['category'] = cat
    lost['seed_reason'] = np.full(len(lost), '', dtype='U16')
    if files['m7_seed']:
        s = read_cat(files['m7_seed'], data)
        ds, _ = cKDTree(np.c_[s['_x'], s['_y']]).query(np.c_[lost['x'], lost['y']])
        lost['seed_dist'] = ds
        lost['seed_reason'] = m7_seed_reason(lost, ds, files, data, filt, r_match)

    # residual / data metrics at the lost positions
    def measure(path, xs, ys, prefix, tab, rows):
        mos = Mosaic(path)
        m = stamp_metrics(mos, xs, ys, fwhm)
        mos.close()
        for k, v in m.items():
            col = f'{prefix}_{k}'
            if col not in tab.colnames:
                tab[col] = np.nan
            arr = np.array(tab[col]); arr[rows] = v; tab[col] = arr

    allrows = np.arange(len(lost))
    dm = stamp_metrics(data, np.asarray(lost['x']), np.asarray(lost['y']), fwhm)
    for k, v in dm.items():
        lost[f'data_{k}'] = v
    for ph in PHASES:
        r_last = np.where(lost['last_phase'] == ph)[0]
        if len(r_last):
            measure(files[ph]['resid'], np.asarray(lost['x'])[r_last],
                    np.asarray(lost['y'])[r_last], 'res_last', lost, r_last)
        r_next = np.where(lost['next_phase'] == ph)[0]
        if len(r_next):
            measure(files[ph]['resid'], np.asarray(lost['x'])[r_next],
                    np.asarray(lost['y'])[r_next], 'res_next', lost, r_next)
    measure(files[final]['resid'], np.asarray(lost['x']), np.asarray(lost['y']),
            'res_final', lost, allrows)
    lost['frac_left'] = lost['res_final_flux'] / lost['data_flux']

    # control: kept final-vetted stars, final residual at their positions
    rng = np.random.default_rng(0)
    vf = V[final]
    if inner is not None:
        vf = vf[_in_box(SkyCoord(vf['skycoord']), inner)]
    pick = rng.choice(len(vf), size=min(len(vf), 5000), replace=False)
    fm = Mosaic(files[final]['resid'])
    ctrl = stamp_metrics(fm, np.asarray(vf['_x'])[pick], np.asarray(vf['_y'])[pick], fwhm)
    fm.close()
    data.close()

    summary = dict(label=label, filt=filt, n_vetted={ph: len(V[ph]) for ph in PHASES},
                   n_union_m2_m6=len(U), n_lost=len(lost),
                   ctrl_final_resid_snr_p50=float(np.nanmedian(ctrl['snr'])),
                   ctrl_final_resid_snr_p90=float(np.nanpercentile(ctrl['snr'], 90)),
                   ctrl_final_resid_snr_p99=float(np.nanpercentile(ctrl['snr'], 99)))
    sig = ((lost['res_final_snr'] >= 5) & (lost['res_final_peak_off'] <= 1.5)
           & (lost['frac_left'] >= 0.5))
    lost['starlike_left'] = sig
    by = {}
    for ph in PHASES[:-1]:
        for c in ('vetted_out', 'not_fit'):
            m = (lost['last_phase'] == ph) & (lost['category'] == c)
            by[f'{ph}->{c}'] = dict(n=int(m.sum()), starlike_left=int((m & sig).sum()))
    summary['by_last_phase'] = by
    reasons = {}
    for rsn in sorted(set(lost['seed_reason']) - {''}):
        m = np.asarray(lost['seed_reason']) == rsn
        reasons[rsn] = dict(n=int(m.sum()), starlike_left=int((m & sig).sum()))
    summary['m7_seed_reason'] = reasons
    summary['n_final_vetted_inner'] = int(len(vf))
    summary['n_starlike_left'] = int(sig.sum())
    if out:
        lost.write(out + '_lost.fits', overwrite=True)
        with open(out + '_summary.json', 'w') as fh:
            json.dump(summary, fh, indent=1)
    return lost, summary


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--catdir', required=True)
    p.add_argument('--pipedir', required=True)
    p.add_argument('--filter', required=True)
    p.add_argument('--module', default='merged')
    p.add_argument('--obstok', default='')
    p.add_argument('--r-match', type=float, default=1.0)
    p.add_argument('--out', required=True)
    p.add_argument('--label', default='')
    p.add_argument('--inner', type=float, nargs=3, default=None,
                   metavar=('RA', 'DEC', 'HALF_ARCSEC'))
    a = p.parse_args()
    lost, summ = run(a.catdir, a.pipedir, a.filter, a.module, a.obstok, a.r_match, a.out,
                     a.label, inner=a.inner)
    print(json.dumps(summ, indent=1))
