"""Additive vs multiplicative: fit our_flux_Jy = a * ref_flux_Jy + b per band (ref mags -> Jy via our own ZP),
in stellar-density terciles. Density = dolphot stars within 1 arcsec."""
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
i1, i2, _, _ = ct.search_around_sky(cr, 1 * u.arcsec)
dens = np.bincount(i2, minlength=len(t)) - 1
q = np.percentile(dens, [33.3, 66.7])
print('density terciles (dolphot stars within 1"):', q)
bands = ['200W', '212N', '277W', '300M', '335M', '410M', '405N', '466N']
for b in bands:
    r = np.asarray(t[f'ref_{b}'], float)
    o = np.asarray(t[f'our_{b}'], float)
    fo = np.asarray(t[f'our_flux_{b}'], float)
    lb = np.asarray(t[f'our_local_bkg_{b}'], float)
    sat = np.asarray(t[f'our_is_saturated_{b}']).astype(bool) | np.asarray(t[f'our_replaced_saturated_{b}']).astype(bool)
    ok = np.isfinite(r) & np.isfinite(o) & ~sat & (r < 90) & (o < 90) & np.isfinite(fo) & (fo > 0)
    # our flux units -> per-mag: fo = K * 10**(-0.4 o); K from bright rows
    K = np.median(fo[ok] * 10 ** (0.4 * o[ok]))
    fr = K * 10 ** (-0.4 * r)
    line = [f'{b:5s}']
    for lab, sel in (('lo', dens <= q[0]), ('mid', (dens > q[0]) & (dens <= q[1])), ('hi', dens > q[1])):
        s = ok & sel
        # robust fit on faint half: our - ref flux vs ref flux
        x = fr[s]
        y = fo[s] - fr[s]
        # median offset in bins of ref flux; additive term = offset at faint end
        fb = np.percentile(x, [0, 25, 50, 75, 100])
        meds = [np.median(y[(x >= lo) & (x < hi)]) for lo, hi in zip(fb[:-1], fb[1:])]
        mx = [np.median(x[(x >= lo) & (x < hi)]) for lo, hi in zip(fb[:-1], fb[1:])]
        # fit y = (a-1) x + b over the 4 quartile medians
        A = np.vstack([mx, np.ones(4)]).T
        (am1, bb), *_ = np.linalg.lstsq(A, meds, rcond=None)
        line.append(f'{lab}: a-1={am1:+.3f} b={bb:+.3g} (b/f_q1={bb/mx[0]:+.3f}) lbkg_med={np.nanmedian(lb[s]):.3g} N={s.sum()}')
    print(' | '.join(line))
