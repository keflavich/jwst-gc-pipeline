"""Per-star flat-field and photom change between our CRDS context (crf header) and an older context (default jwst_1298,
the pmap recorded in the dolphot input mosaics wd2_F*_AB_i2d.fits, cal 1.16.0).

Sign convention: crf = rate * PHOTMJSR / flat, so a star's flux in our crf relative to the old processing is
(PH_o / flat_o) / (PH_old / flat_old).  The magnitude change of the star (ours minus old; positive = fainter in ours) is
    fl = 2.5 log10(flat_o / flat_old)       (larger flat_o -> fainter)
    ph = -2.5 log10(PH_o / PH_old)          (larger PHOTMJSR_o -> brighter)
Flat ratios use 5x5 median-filtered flats with DQ DO_NOT_USE pixels masked, sampled at the star pixel in every exposure and
averaged in flux over exposures (as areapred.py).  Also reports whether the old AREA reference differs.
usage: python flatver_calc.py [old_pmap] -> flatver_<pmap>.npz, flatver_<pmap>_dets.json, flatver_<pmap>_map_<det>_<band>.npy
"""
import os
import sys
import glob
import json
import numpy as np
os.environ['CRDS_PATH'] = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/flatver/crds_cache'
os.environ['CRDS_SERVER_URL'] = 'https://jwst-crds.stsci.edu'
from crds.client import api
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import median_filter
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an

OLD = sys.argv[1] if len(sys.argv) > 1 else 'jwst_1298.pmap'
tag = OLD.replace('.pmap', '')
HERE = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/flatver'
SHARED = '/orange/adamginsburg/jwst/crds/references/jwst/nircam'
SCRATCH = f'{HERE}/crds_cache/references/jwst/nircam'
KEYS = {'META.INSTRUMENT.NAME': 'INSTRUME', 'META.INSTRUMENT.DETECTOR': 'DETECTOR', 'META.INSTRUMENT.FILTER': 'FILTER',
        'META.INSTRUMENT.PUPIL': 'PUPIL', 'META.EXPOSURE.TYPE': 'EXP_TYPE', 'META.EXPOSURE.READPATT': 'READPATT',
        'META.SUBARRAY.NAME': 'SUBARRAY', 'META.INSTRUMENT.CHANNEL': 'CHANNEL', 'META.INSTRUMENT.MODULE': 'MODULE',
        'META.OBSERVATION.DATE': 'DATE-OBS', 'META.OBSERVATION.TIME': 'TIME-OBS'}


def path(name):
    for d in (SHARED, SCRATCH):
        if os.path.exists(f'{d}/{name}'):
            return f'{d}/{name}'
    api.dump_references(OLD, [name])
    return f'{SCRATCH}/{name}'


def photmjsr(name, filt, pup):
    d = fits.getdata(path(name), 1)
    m = (np.char.strip(d['filter']) == filt) & (np.char.strip(d['pupil']) == pup)
    if 'subarray' in d.dtype.names:
        m &= np.char.strip(d['subarray']) == 'FULL'
    s = d[m]
    return float(s['photmjsr'][0])


def smooth(name):
    with fits.open(path(name)) as fh:
        f = np.asarray(fh['SCI'].data, float)
        dq = np.asarray(fh['DQ'].data)
    f[(dq & 1) > 0] = np.nan
    f[~(f > 0)] = np.nan
    # NaN-aware 5x5 median via filling with local nanmedian is slow; fill NaN with global median then restore mask
    bad = ~np.isfinite(f)
    g = np.where(bad, np.nanmedian(f), f)
    out = median_filter(g, size=5)
    out[bad] = np.nan
    return out


if __name__ == '__main__':
    an.ZPWIN.update(an.zp_windows())
    A = an.Arm('main2')
    sky = A.sky[A.idx]
    tree = f'{an.Q}/tree_main2'
    P = {}
    info = {}
    cache = {}
    for band in an.BANDS:
        frames = sorted(glob.glob(f'{tree}/F{band}/pipeline/jw03523005001_*_align_o005_crf.fits'))
        if not frames:
            continue
        sf = np.zeros(A.n); sp = np.zeros(A.n); n = np.zeros(A.n)
        for fn in frames:
            with fits.open(fn, memmap=True) as fh:
                h0 = fh[0].header
                h = fh['SCI'].header
                shape = fh['SCI'].data.shape
            det = h0['DETECTOR']
            hd = {k: str(h0[v]) for k, v in KEYS.items()}
            hd['META.EXPOSURE.START_TIME'] = 60505.2
            ref = api.get_best_references(OLD, hd, reftypes=['flat', 'photom', 'area'])
            fo = h0['R_FLAT'].split('/')[-1]; po = h0['R_PHOTOM'].split('/')[-1]; ao = h0['R_AREA'].split('/')[-1]
            fn_old, pn_old, an_old = (str(ref[k]) for k in ('flat', 'photom', 'area'))
            phot_o = h['PHOTMJSR']
            phot_old = photmjsr(pn_old, h0['FILTER'].strip(), h0['PUPIL'].strip())
            key = (fo, fn_old)
            if key not in cache:
                if fo == fn_old:
                    cache[key] = None
                else:
                    cache[key] = smooth(fo) / smooth(fn_old)
                    np.save(f'{HERE}/map_{tag}_{det}_{band}.npy', cache[key].astype(np.float32))
            rmap = cache[key]
            info.setdefault(f'{band}|{det}', dict(flat_ours=fo, flat_old=fn_old, photom_ours=po, photom_old=pn_old, area_ours=ao, area_old=an_old,
                                                   photmjsr_ours=phot_o, photmjsr_old=phot_old, ctx_ours=h0['CRDS_CTX']))
            w = WCS(h)
            x, y = w.world_to_pixel(sky)
            inn = A.matched & (x > 0) & (x < shape[1] - 1) & (y > 0) & (y < shape[0] - 1)
            ix, iy = np.round(x[inn]).astype(int), np.round(y[inn]).astype(int)
            r = np.ones(inn.sum()) if rmap is None else rmap[iy, ix]
            ok = np.isfinite(r)
            idx = np.nonzero(inn)[0][ok]
            sf[idx] += 1.0 / r[ok]                    # flux ratio from flat: flat_old/flat_o
            sp[idx] += phot_o / phot_old              # flux ratio from photom: PH_o/PH_old
            n[idx] += 1
        fl = np.full(A.n, np.nan); ph = np.full(A.n, np.nan)
        fl[n > 0] = -2.5 * np.log10(sf[n > 0] / n[n > 0])     # = 2.5 log10(flat_o/flat_old), flux-averaged
        ph[n > 0] = -2.5 * np.log10(sp[n > 0] / n[n > 0])     # = -2.5 log10(PH_o/PH_old)
        P[f'fl_{band}'] = fl
        P[f'ph_{band}'] = ph
        print(band, 'done', int((n > 0).sum()), flush=True)
    np.savez(f'{HERE}/flatver_{tag}.npz', **P)
    json.dump(info, open(f'{HERE}/flatver_{tag}_dets.json', 'w'), indent=1)
