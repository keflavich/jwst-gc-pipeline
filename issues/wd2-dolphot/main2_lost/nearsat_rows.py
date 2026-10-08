"""Rows of arm X near saturated rows: total, matched to dolphot, unmatched.  usage: nearsat_rows.py X [X ...]"""
import sys
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
D = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark'
M8 = 'catalogs/basic_merged_indivexp_photometry_tables_merged_resbgsub_m8_dedup.fits'
for X in sys.argv[1:]:
    c = Table.read(f'{D}/Q_integ/tree_{X}/{M8}')
    m = Table.read(f'{D}/matched_Q_{X}.fits')
    sat = np.zeros(len(c), bool)
    for k in c.colnames:
        if k.startswith('replaced_saturated_'):
            sat |= np.ma.filled(c[k], 0).astype(bool)
    sk = SkyCoord(c['skycoord_ref'])
    _, d, _ = sk.match_to_catalog_sky(sk[sat], nthneighbor=1)
    d = d.arcsec
    d[sat] = 0
    used = np.zeros(len(c), bool)
    used[np.asarray(m['our_idx'])[np.asarray(m['matched'])]] = True
    for lab, lo, hi in (('<1"', 1e-9, 1), ('1-3"', 1, 3), ('>=3"', 3, 1e9)):
        s = ~sat & (d >= lo) & (d < hi)
        print(f'{X:10s} dsat {lab:5s} rows {s.sum():6d} matched {np.sum(s & used):6d} unmatched {np.sum(s & ~used):6d}')
    print(f'{X:10s} total rows {len(c)} saturated {sat.sum()} matched {used.sum()}')
