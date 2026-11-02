"""Build the frozen injection tables of the reference fields.

Run once per field (the tables are committed; regenerating them changes the
truth every threshold was calibrated on)::

    python -m jwst_gc_pipeline.photometry.reference_fields.make_injections superdense

For each seed the table holds ``n_inject`` stars at random sky positions in
the field's inner box (the cutout minus ``inner_margin_arcsec`` on every
side), at least ``min_sep_pix`` apart and ``satstar_avoid_arcsec`` from any
saturated star.  Their primary-band brightness is drawn log-uniform in the
ERR-based S/N

    snr_true = F * sqrt(sum P^2) / sigma

over ``snr_range``, where ``sigma`` is the production data i2d ERR at the
position and ``sum P^2`` comes from the filter's WebbPSF fitting grid at
detector sampling (averaged over the four half-pixel phases).  It is the S/N
of an isolated PSF fit with known position on a smooth background; crowding
and structured emission push the achievable S/N below it, which is what the
completeness thresholds measure.  Every other band gets the field's median
real-star flux ratio, measured from the production m6 vetted catalogs within
``color_radius_arcsec`` of the field centre, so injected stars have the colour
of the field's real stars and the m7 cross-band seed sees them as it sees
those.
"""
import argparse
import glob
import os

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from jwst_gc_pipeline.photometry import reference_fields as RF
from jwst_gc_pipeline.photometry.injection import flux_column


def production_path(spec, filt, kind):
    """A production product of ``spec``'s target for ``filt``.

    ``kind`` is ``'data_i2d'``, ``'vetted'`` (m6 vetted catalog) or
    ``'satstar'``.  The pattern is ``spec['production'][kind]`` with
    ``{filt}``/``{FILT}`` filled in.
    """
    pattern = spec['production'][kind].format(filt=filt.lower(), FILT=filt.upper())
    hits = sorted(glob.glob(os.path.join(RF.basepath(spec), pattern)))
    if not hits:
        raise FileNotFoundError(f"{spec['name']}: no {kind} for {filt} "
                                f"({os.path.join(RF.basepath(spec), pattern)})")
    return hits[0]


def psf_sum_p2(spec, filt):
    """``sum P^2`` of a unit-flux PSF at detector sampling.

    Read from the target's cached WebbPSF fitting grid
    (``psfs/nircam_<det>_<filt>_fovp101_samp2_npsf16.fits``, the grid the
    fit uses), averaged over the grid's PSFs and the four half-pixel
    sampling phases.
    """
    hits = sorted(glob.glob(os.path.join(
        RF.basepath(spec), 'psfs', f'nircam_*_{filt.lower()}_fovp101_samp2_npsf16.fits')))
    if not hits:
        raise FileNotFoundError(f"{spec['name']}: no fovp101 PSF grid for {filt}")
    cube = fits.getdata(hits[0]).astype(float)
    vals = []
    for psf in cube:
        psf = psf / psf.sum()
        for dy in (0, 1):
            for dx in (0, 1):
                p = psf[dy:, dx:]
                ny, nx = (p.shape[0] // 2) * 2, (p.shape[1] // 2) * 2
                det = p[:ny, :nx].reshape(ny // 2, 2, nx // 2, 2).sum(axis=(1, 3))
                vals.append(np.sum(det ** 2))
    return float(np.mean(vals))


def field_color_ratios(spec, radius_arcsec=None):
    """Median real-star flux ratio band/primary (Jy/Jy) near the field."""
    radius = (radius_arcsec or spec.get('color_radius_arcsec', 30.0)) * u.arcsec
    centre = SkyCoord(spec['ra'] * u.deg, spec['dec'] * u.deg)
    filters = spec['filters']
    cats = {}
    for f in filters:
        t = Table.read(production_path(spec, f, 'vetted'))
        sc = SkyCoord(t['skycoord'])
        near = sc.separation(centre) < radius
        t = t[near]
        pixar = float(fits.getheader(production_path(spec, f, 'data_i2d'), 'SCI')['PIXAR_SR'])
        snr = np.asarray(t['flux'], float) / np.asarray(t['flux_err'], float)
        good = np.isfinite(snr) & (snr > 20)
        cats[f] = (SkyCoord(t['skycoord'][good]),
                   np.asarray(t['flux'][good], float) * pixar * 1e6)
    p_sc, p_fl = cats[filters[0]]
    ratios = {filters[0]: 1.0}
    for f in filters[1:]:
        sc, fl = cats[f]
        idx, sep, _ = p_sc.match_to_catalog_sky(sc)
        ok = sep < 0.1 * u.arcsec
        if ok.sum() < 10:
            raise ValueError(f"{spec['name']}: only {ok.sum()} {filters[0]}x{f} "
                             f"matches within {radius}; widen color_radius_arcsec")
        ratios[f] = float(np.median(fl[idx[ok]] / p_fl[ok]))
    return ratios


def draw_positions(spec, rng, wcs, err, sat_sc):
    """Random sky positions in the inner box, ``min_sep_pix`` apart."""
    centre = SkyCoord(spec['ra'] * u.deg, spec['dec'] * u.deg)
    cx, cy = wcs.world_to_pixel(centre)
    pixas = np.sqrt(abs(np.linalg.det(wcs.pixel_scale_matrix))) * 3600
    half = (spec['size_arcsec'] / 2 - spec['inner_margin_arcsec']) / pixas
    min_sep = spec['min_sep_pix']
    xs, ys = [], []
    tries = 0
    while len(xs) < spec['n_inject']:
        tries += 1
        if tries > 100000:
            raise RuntimeError(f"{spec['name']}: could not place {spec['n_inject']} stars")
        x = cx + rng.uniform(-half, half)
        y = cy + rng.uniform(-half, half)
        ix, iy = int(round(x)), int(round(y))
        if not np.isfinite(err[iy, ix]) or err[iy, ix] <= 0:
            continue
        if xs and np.min(np.hypot(np.array(xs) - x, np.array(ys) - y)) < min_sep:
            continue
        if sat_sc is not None and len(sat_sc):
            sep = wcs.pixel_to_world(x, y).separation(sat_sc)
            if np.min(sep.arcsec) < spec['satstar_avoid_arcsec']:
                continue
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def make_table(spec, seed):
    """The injection table of ``spec`` for ``seed``."""
    filters = spec['filters']
    prim = filters[0]
    rng = np.random.default_rng(int(seed) * 7919 + sum(map(ord, spec['name'])))
    i2d = production_path(spec, prim, 'data_i2d')
    with fits.open(i2d) as h:
        wcs = WCS(h['SCI'].header)
        err = h['ERR'].data
        pixar = float(h['SCI'].header['PIXAR_SR'])
        try:
            sat = Table.read(production_path(spec, prim, 'satstar'))
            sat_sc = SkyCoord(sat['skycoord_fit'] if 'skycoord_fit' in sat.colnames
                              else sat['skycoord'])
        except FileNotFoundError:
            sat_sc = None
        xs, ys = draw_positions(spec, rng, wcs, err, sat_sc)
        sigma = np.array([np.nanmedian(err[int(round(y)) - 2:int(round(y)) + 3,
                                           int(round(x)) - 2:int(round(x)) + 3])
                          for x, y in zip(xs, ys)])
    lo, hi = spec['snr_range']
    snr = np.exp(rng.uniform(np.log(lo), np.log(hi), len(xs)))
    sp2 = psf_sum_p2(spec, prim)
    flux_img = snr * sigma / np.sqrt(sp2)
    flux_jy = flux_img * pixar * 1e6
    sky = wcs.pixel_to_world(xs, ys)
    tbl = Table()
    tbl['id'] = np.arange(len(xs))
    tbl['ra'] = sky.ra.deg
    tbl['dec'] = sky.dec.deg
    tbl[f'snr_true_{prim}'] = snr
    tbl[flux_column(prim)] = flux_jy
    ratios = field_color_ratios(spec)
    for f in filters[1:]:
        tbl[flux_column(f)] = flux_jy * ratios[f]
    tbl.meta.update(field=spec['name'], seed=int(seed), primary=prim,
                    sum_p2=sp2, i2d=os.path.basename(i2d),
                    color_ratios={f: float(r) for f, r in ratios.items()},
                    snr_definition='F*sqrt(sum P^2)/ERR(production data i2d)')
    return tbl


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('fields', nargs='*', help='field names (default: all)')
    p.add_argument('--overwrite', action='store_true',
                   help='replace existing (committed) tables')
    a = p.parse_args(argv)
    _, fields = RF.load_config()
    os.makedirs(RF.INJECTION_DIR, exist_ok=True)
    for name in a.fields or list(fields):
        spec = fields[name]
        for seed in spec['seeds']:
            out = RF.injection_table_path(name, seed)
            if os.path.exists(out) and not a.overwrite:
                print(f'{out} exists; --overwrite to replace')
                continue
            tbl = make_table(spec, seed)
            tbl.write(out, overwrite=True)
            print(f'wrote {out} ({len(tbl)} stars, color ratios '
                  f'{tbl.meta["color_ratios"]})')


if __name__ == '__main__':
    main()
