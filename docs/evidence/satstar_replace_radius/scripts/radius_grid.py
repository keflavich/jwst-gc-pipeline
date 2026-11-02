"""Second-pass radius grid on merged catalogs that used the old fixed 0.5".

For each catalog, the replace_saturated rows paired by the mutual-nearest
second pass (satstar_match_sep beyond the tight first-pass radius, 0.05"
short-wave / 0.1" long-wave) are split by lr = log10(flux_init / flux) (see
swap_census.py): 'self' (|lr| < 0.2, the satstar's own clipped
row) and 'nbr' (lr < -0.5, a row seeded > 3x fainter: a neighbour star).
For k in KS, a radius of k x FWHM keeps the self rows with sep <= k FWHM and
protects the nbr rows with sep > k FWHM.  The FWHM is the band's
``PSF FWHM (arcsec)`` from the table ``_satstar_replace_radius`` reads for
the catalog's field tree (<field>/catalogs/<catalog> -> <field>).

usage: python radius_grid.py <catalog.fits> [<catalog.fits> ...]
"""
import os
import re
import sys

import numpy as np
from astropy.table import Table

from jwst_gc_pipeline.photometry.naming import _instrument_override
from jwst_gc_pipeline.reduction.fwhm import fwhm_table_path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from swap_census import tight_radius  # noqa: E402

KS = (1.0, 1.25, 1.5, 2.0, 2.5)


def _fwhm(basepath, filt):
    tbl = Table.read(fwhm_table_path(basepath, _instrument_override()))
    row = tbl[np.char.upper(np.asarray(tbl['Filter']).astype(str)) == filt]
    return float(row['PSF FWHM (arcsec)'][0])


def main(*paths):
    print('| catalog | FWHM (") | self rows | nbr rows | '
          + ' | '.join(f'k={k}: self kept / nbr protected' for k in KS) + ' |')
    print('|---' * (4 + len(KS)) + '|')
    for p in paths:
        filt = re.match(r'(f\d+[wmn])_', os.path.basename(p)).group(1).upper()
        m = Table.read(p)
        s = np.asarray(m['satstar_match_sep'], float)
        rep = np.asarray(m['replaced_saturated'], bool) & np.isfinite(s) & (s > tight_radius(filt))
        with np.errstate(divide='ignore', invalid='ignore'):
            lr = np.log10(np.asarray(m['flux_init'], float) / np.asarray(m['flux'], float))
        sf, nb = rep & (np.abs(lr) < 0.2), rep & (lr < -0.5)
        fw = _fwhm(os.path.dirname(os.path.dirname(os.path.abspath(p))), filt)
        x = s / fw
        cells = [f'{np.sum(sf & (x <= k))}/{sf.sum()} / {np.sum(nb & (x > k))}/{nb.sum()}' for k in KS]
        name = os.path.relpath(os.path.abspath(p), os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(p)))))
        print(f'| {name} | {fw:.3f} | {sf.sum()} | {nb.sum()} | ' + ' | '.join(cells) + ' |')


if __name__ == '__main__':
    main(*sys.argv[1:])
