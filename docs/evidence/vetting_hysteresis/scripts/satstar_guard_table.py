"""Distance to the nearest satstar of the sources only variant B keeps, and
what a satstar guard of a given radius would remove from them.

usage: python satstar_guard_table.py <varA> <varB> <field:band> [<field:band> ...]

Sources: the ab_gallery_m7.py candidates (m7 vetted in B, no m7 vetted
source of A within 1 px, inner box), pooled over all runs of the field.
Classes, from the PSF-matched residual S/N at the source:
  good      A residual >= 5 (A leaves it whole) and B residual > -3
  over-sub  B residual <= -3
Two sets of guard centres, both from variant B at m7:
  rows          the is_saturated rows of the vetted catalog (after the
                merge's in-field duplicate collapse)
  consolidated  every fit of the band's consolidated satstar catalog; the
                merge places its satstar rows at these positions before the
                collapse (less the position-only rows and the
                faint-replacement vetoes)
Distances are in arcsec and in units of the band's FWHM (packaged
fwhm_table).
"""
import os
import sys

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ab_gallery_m7 as G  # noqa: E402

FLAT_ARCSEC = (0.3, 0.4, 0.5, 0.6, 0.7)
FWHM_MULT = (3.0, 4.0, 4.5, 5.0, 6.0)


def _fwhm(band):
    from jwst_gc_pipeline.reduction.fwhm import fwhm_table_path
    t = Table.read(fwhm_table_path(None, 'NIRCAM'))
    row = t[np.char.upper(np.asarray(t['Filter']).astype(str)) == band.upper()]
    return float(row['PSF FWHM (arcsec)'][0])


def _rng(v):
    v = v[np.isfinite(v)]
    return f'{v.min():.2f}–{v.max():.2f}' if len(v) else '–'


def _row_dist(spec, vb, seed, band, x, y):
    """Distance (arcsec) from (x, y) to the nearest is_saturated row of B's
    m7 vetted catalog."""
    fb = G._files(spec, vb, seed, band)
    v7 = Table.read(fb[G.M7]['vetted'])
    if not len(x) or 'is_saturated' not in v7.colnames:
        return np.full(len(x), np.inf)
    iss = v7['is_saturated']
    iss = np.asarray(iss.filled(False) if hasattr(iss, 'filled') else iss, bool)
    if not iss.any():
        return np.full(len(x), np.inf)
    data = G.Mosaic(fb['data'])
    p = data.wcs.pixel_to_world(x, y)
    _, d, _ = p.match_to_catalog_sky(SkyCoord(v7['skycoord'])[iss])
    return d.arcsec


def main(va, vb, *specs):
    _, fields = G.RF.load_config()
    rows = []
    for s in specs:
        name, band = s.split(':')
        band = band.upper()
        spec = fields[name]
        r = []
        for seed in [0] + list(spec['seeds']):
            c = G.candidates(spec, name, band, seed, va, vb)
            drow = _row_dist(spec, vb, seed, band, c['x'], c['y'])
            r += list(zip(c['snr_a'], c['snr_b'], drow, c['sat_dist_as']))
        r = np.array(r, float).reshape(-1, 4)
        rows.append((name, band, _fwhm(band), r))

    print(f'sources only {vb} keeps (vs {va}), all runs; good = {va} resid >= 5 and '
          f'{vb} resid > -3; over-sub = {vb} resid <= -3')
    print()
    print('| field / band | centres | n | good | over-sub | good dist ″ (FWHM) | over-sub dist ″ (FWHM) |')
    print('|---|---|---|---|---|---|---|')
    for name, band, fw, r in rows:
        a, b = r[:, 0], r[:, 1]
        good, bad = (a >= 5) & (b > -3), b <= -3
        for lab, d in (('rows', r[:, 2]), ('consolidated', r[:, 3])):
            print(f'| {name} {band} | {lab} | {len(r)} | {good.sum()} | {bad.sum()} | '
                  f'{_rng(d[good])} ({_rng(d[good] / fw)}) | {_rng(d[bad])} ({_rng(d[bad] / fw)}) |')
    print()
    print('removed by a guard of this radius: good / over-sub')
    print()
    cols = [f'{x:g}″' for x in FLAT_ARCSEC] + [f'{k:g} FWHM' for k in FWHM_MULT]
    print('| field / band | centres | ' + ' | '.join(cols) + ' |')
    print('|---' * (len(cols) + 2) + '|')
    for name, band, fw, r in rows:
        a, b = r[:, 0], r[:, 1]
        good, bad = (a >= 5) & (b > -3), b <= -3
        for lab, d in (('rows', r[:, 2]), ('consolidated', r[:, 3])):
            cells = []
            for rad in list(FLAT_ARCSEC) + [k * fw for k in FWHM_MULT]:
                near = np.isfinite(d) & (d <= rad)
                cells.append(f'{np.sum(good & near)} / {np.sum(bad & near)}')
            print(f'| {name} {band} | {lab} | ' + ' | '.join(cells) + ' |')


if __name__ == '__main__':
    main(*sys.argv[1:])
