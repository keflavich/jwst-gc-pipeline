"""Bright-star (10-14 mag) per-band magnitude shift between two arms, for the narrowband / adjacent-band pairs
of color_check.py: which band carries the colour change.  Rows matched within 0.05".
usage: python bright_band_shift.py ARM_A ARM_B"""
import sys
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
import analyze as an

PAIRS = [('f164n', 'f162m'), ('f187n', 'f182m'), ('f212n', 'f200w'), ('f405n', 'f410m'), ('f466n', 'f410m')]
BINS = [(8, 10), (10, 11), (11, 12), (12, 13), (13, 14), (15, 18)]
a, b = sys.argv[1:3]
ta, tb = Table.read(an.PATH[a][0]), Table.read(an.PATH[b][0])
ca, cb = SkyCoord(ta['skycoord_ref']), SkyCoord(tb['skycoord_ref'])
i, d, _ = ca.match_to_catalog_sky(cb)
ok = d < 0.05 * u.arcsec
print(f'{a} -> {b}: {ok.sum()} of {len(ta)} rows matched within 0.05"')
for nb, bb in PAIRS:
    print(f'\n### {nb.upper()} / {bb.upper()}: per {nb.upper()} ({a}) mag bin: N / median d{nb} / median d{bb} / median d(colour)  (d = {b} - {a}); '
          f'replaced_saturated fraction in {a} / {b} for {nb}, {bb}')
    m1a, m2a = an.fl(ta[f'mag_vega_{nb}']), an.fl(ta[f'mag_vega_{bb}'])
    m1b, m2b = an.fl(tb[f'mag_vega_{nb}'])[i], an.fl(tb[f'mag_vega_{bb}'])[i]
    rs = {}
    for t, nm, idx in ((ta, a, slice(None)), (tb, b, i)):
        for band in (nb, bb):
            c = f'replaced_saturated_{band}'
            rs[(nm, band)] = (np.ma.filled(t[c], 0).astype(bool)[idx] if c in t.colnames else np.zeros(len(ta), bool))
    good = ok & np.isfinite(m1a) & np.isfinite(m2a) & np.isfinite(m1b) & np.isfinite(m2b)
    for lo, hi in BINS:
        s = good & (m1a >= lo) & (m1a < hi)
        if not s.sum():
            continue
        d1, d2 = m1b[s] - m1a[s], m2b[s] - m2a[s]
        print(f'  {lo}-{hi}: {s.sum()} / {np.median(d1):+.3f} / {np.median(d2):+.3f} / {np.median(d1 - d2):+.3f}   '
              f'sat {nb} {rs[(a, nb)][s].mean():.2f}/{rs[(b, nb)][s].mean():.2f}, {bb} {rs[(a, bb)][s].mean():.2f}/{rs[(b, bb)][s].mean():.2f}')
