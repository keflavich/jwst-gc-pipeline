"""Background holes left by m7 seeds that vetting drops.

Every m7 seed position is masked out of the m7 smoothed background.  With the
own-band seed on, each band's m7 seed adds the band's own m6 vetted sources
(``seed_origin == 'own_m6'``) and its m6-residual detections (``'i2d'``) to
the cross-band seed.  For one field and the two variants (own-band off / on,
same code otherwise), this script takes every added seed inside the evaluated
inner box and splits it by whether the final m7 vetted catalog has a source
within one pixel.  At each such position it reads the m7 smoothed background
of both variants and reports the difference (on - off), next to the same
difference at random inner-box positions at least 0.3" from every seed.
Differences are also given in units of the pixel scatter of the off
variant's m7 residual mosaic in the inner box (1.4826 MAD, ``sigma``).

usage: python bg_holes.py <field> <off_variant> <on_variant> <out.json>
"""
import glob
import json
import sys

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from jwst_gc_pipeline.photometry import reference_fields as RF

PIX_AS = 0.031


def _inner(sc, spec):
    c = SkyCoord(spec['ra'] * u.deg, spec['dec'] * u.deg)
    dx, dy = c.spherical_offsets_to(sc)
    half = spec['size_arcsec'] / 2 - spec['inner_margin_arcsec']
    return (np.abs(dx.to_value(u.arcsec)) < half) & (np.abs(dy.to_value(u.arcsec)) < half)


def _one(path_glob):
    hits = glob.glob(path_glob)
    if len(hits) != 1:
        raise FileNotFoundError(f'{path_glob}: {hits}')
    return hits[0]


def _bg_at(path, sc):
    with fits.open(path) as hdul:
        hdu = hdul['SCI'] if 'SCI' in hdul else hdul[0]
        x, y = WCS(hdu.header).world_to_pixel(sc)
        xi, yi = np.round(x).astype(int), np.round(y).astype(int)
        ok = (xi >= 0) & (yi >= 0) & (xi < hdu.data.shape[1]) & (yi < hdu.data.shape[0])
        out = np.full(len(sc), np.nan)
        out[ok] = hdu.data[yi[ok], xi[ok]]
    return out


def run_pair(spec, off, on, seed, rng):
    filt = spec['filters'][0].lower()
    d_on, d_off = RF.run_dir(spec, on, seed), RF.run_dir(spec, off, seed)
    t = Table.read(_one(f'{d_on}/catalogs/crossband_seed_manual_merged_{filt}_i2dseed.fits'))
    origin = np.asarray(t['seed_origin']).astype(str)
    allseed = SkyCoord(t['skycoord'])
    added = allseed[origin != 'crossband']
    added = added[_inner(added, spec)]
    final = SkyCoord(Table.read(_one(f'{d_on}/catalogs/{filt}_*_m7_dao_basic_vetted.fits'))['skycoord'])
    kept = np.zeros(len(added), bool)
    if len(added) and len(final):
        _, sep, _ = added.match_to_catalog_sky(final)
        kept = sep.to_value(u.arcsec) < PIX_AS
    bgname = '*_m7_daophot_basic_mergedcat_residual_smoothed_bg_i2d.fits'
    bg_on = _one(f'{d_on}/{filt.upper()}/pipeline/{bgname}')
    bg_off = _one(f'{d_off}/{filt.upper()}/pipeline/{bgname}')
    diff = _bg_at(bg_on, added) - _bg_at(bg_off, added)
    resid = _one(f'{d_off}/{filt.upper()}/pipeline/*_m7_daophot_basic_mergedcat_residual_i2d.fits')
    with fits.open(resid) as hdul:
        hdu = hdul['SCI'] if 'SCI' in hdul else hdul[0]
        wcs, data = WCS(hdu.header), hdu.data
    yy, xx = np.mgrid[0:data.shape[0], 0:data.shape[1]]
    ins = _inner(wcs.pixel_to_world(xx.ravel(), yy.ravel()), spec).reshape(data.shape)
    v = data[ins & np.isfinite(data)]
    sigma = float(1.4826 * np.median(np.abs(v - np.median(v))))
    # control: random inner-box positions away from every seed
    c = SkyCoord(spec['ra'] * u.deg, spec['dec'] * u.deg)
    half = spec['size_arcsec'] / 2 - spec['inner_margin_arcsec']
    off_xy = rng.uniform(-half, half, size=(400, 2))
    ctrl = c.spherical_offsets_by(off_xy[:, 0] * u.arcsec, off_xy[:, 1] * u.arcsec)
    _, sep, _ = ctrl.match_to_catalog_sky(allseed)
    ctrl = ctrl[sep.to_value(u.arcsec) > 0.3]
    cdiff = _bg_at(bg_on, ctrl) - _bg_at(bg_off, ctrl)
    return dict(seed=seed, n_added=int(len(added)), n_kept=int(kept.sum()), sigma=sigma,
                diff_dropped=diff[~kept].tolist(), diff_kept=diff[kept].tolist(),
                diff_control=cdiff[np.isfinite(cdiff)].tolist())


def main(field, off, on, out):
    _, fields = RF.load_config()
    spec = fields[field]
    rng = np.random.default_rng(0)
    rows = [run_pair(spec, off, on, s, rng) for s in [0] + list(spec['seeds'])]
    dropped = np.concatenate([r['diff_dropped'] for r in rows])
    kept = np.concatenate([r['diff_kept'] for r in rows])
    ctrl = np.concatenate([r['diff_control'] for r in rows])
    sig = np.concatenate([[r['sigma']] * len(r['diff_dropped']) for r in rows])
    print(f'{field}: {off} -> {on}, {len(rows)} runs')
    print(f"added m7 seeds in the inner box: {sum(r['n_added'] for r in rows)} "
          f"({np.mean([r['n_added'] for r in rows]):.1f} per run), "
          f"kept in the final catalog: {sum(r['n_kept'] for r in rows)}")
    for name, v in (('dropped seeds', dropped), ('kept seeds', kept), ('control', ctrl)):
        if len(v):
            print(f'  smoothed bg on - off at {name}: n={len(v)} median {np.nanmedian(v):+.1f} '
                  f'p10 {np.nanpercentile(v, 10):+.1f} p90 {np.nanpercentile(v, 90):+.1f}')
    print(f"  residual sigma per run: median {np.median([r['sigma'] for r in rows]):.1f}; "
          f"dropped-seed difference / sigma: median {np.nanmedian(dropped / sig):+.2f} "
          f"p10 {np.nanpercentile(dropped / sig, 10):+.2f}")
    with open(out, 'w') as fh:
        json.dump(dict(field=field, off=off, on=on, runs=rows), fh, indent=1)


if __name__ == '__main__':
    main(*sys.argv[1:])
