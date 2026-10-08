"""Flux deficit per mag bin, expressed as the magnitude of the missing flux:
   dF = F_ref * (10**(-0.4 dm) - 1);  m_def = -2.5 log10(-dF / K) with K the mag-0 flux.
   Constant m_def across bins -> additive offset; m_def tracking m_ref -> multiplicative."""
import sys
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

tag = sys.argv[1] if len(sys.argv) > 1 else 'mainfcbg'
t = Table.read(f'/orange/adamginsburg/jwst/wd2/dolphot_benchmark/matched_Q_{tag}.fits')
ref = Table.read('/orange/adamginsburg/jwst/wd2/catalogs/wd2_nircam_wf_mf_nf.ecsv')
cr = SkyCoord(ref['RA'], ref['DEC'], unit='deg')
ct = SkyCoord(t['RA'], t['DEC'], unit='deg')
_, i2, _, _ = ct.search_around_sky(cr, 1 * u.arcsec)
dens = np.bincount(i2, minlength=len(t)) - 1
edges = np.arange(15, 24, 1.0)
bands = sys.argv[2].split(',') if len(sys.argv) > 2 else ['200W', '212N', '277W', '300M', '335M', '410M', '405N', '466N']
print(f'tag={tag}; cells = median dm / m_def (mag of missing flux)')
for dlab, dsel in (('all', dens >= 0), ('dens<=1', dens <= 1), ('dens>=4', dens >= 4)):
    print(f'-- {dlab}')
    print('band  ' + ' '.join(f'{e:>13.0f}' for e in edges[:-1]))
    for b in bands:
        r = np.asarray(t[f'ref_{b}'], float)
        o = np.asarray(t[f'our_{b}'], float)
        sat = np.asarray(t[f'our_is_saturated_{b}']).astype(bool) | np.asarray(t[f'our_replaced_saturated_{b}']).astype(bool)
        ok = np.isfinite(r) & np.isfinite(o) & ~sat & (r < 90) & (o < 90) & dsel
        cells = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            s = ok & (r >= lo) & (r < hi)
            if s.sum() < 20:
                cells.append(f'{"":>13}')
                continue
            dm = np.median(o[s] - r[s])
            mr = np.median(r[s])
            frac = 10 ** (-0.4 * dm) - 1
            mdef = mr - 2.5 * np.log10(-frac) if frac < 0 else np.nan
            cells.append(f'{dm:+.3f}/{mdef:5.2f}')
        print(f'{b:5s} ' + ' '.join(cells))
