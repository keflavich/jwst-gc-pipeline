"""Disentangle: is the injected-star recovery bias a real masked-core deficit, or
the wing-selfcal OVER-correcting injected stars?

The wing-selfcal ratio is calibrated on the frame's REAL unsaturated stars (real
wings are broader than the STPSF model), and divides satstar flux DOWN. But
injected stars carry STPSF wings (injected with the same grid), so they do NOT
have the real-wing excess -- applying the real-star correction to them would
over-divide and manufacture a faint bias. Test: recover the SAME injected frame
with SATSTAR_WINGCAL=1 vs =0 and compare bias vs saturation depth.

  wingcal OFF bias ~ 0  -> the +mag bias was wing-selfcal over-correcting the
      STPSF-winged injected stars = harness artifact; need EMPIRICAL wings (#2).
  wingcal OFF bias still large -> real masked-core under-recovery.

Usage: test_wingcal_onoff.py <injected_fits> <inj_recovery_ecsv> <band> <psfs>
"""
import os
import sys

os.environ.setdefault('CRDS_PATH', '/orange/adamginsburg/jwst/crds')
os.environ.setdefault('STPSF_PATH', '/orange/adamginsburg/jwst/stpsf-data/')

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

from jwst_gc_pipeline.photometry import artificial_stars as art

INJ = sys.argv[1]
ECSV = sys.argv[2]
BAND = sys.argv[3]
PSFS = sys.argv[4]
REGIMES = [(1, 30, 'mild'), (30, 150, 'moderate'), (150, 500, 'deep'), (500, 1e9, 'hard')]


def recover(wingcal):
    os.environ['SATSTAR_WINGCAL'] = '1' if wingcal else '0'
    from jwst_gc_pipeline.reduction import saturated_star_finding as ssf
    from jwst_gc_pipeline.frame_wcs import frame_wcs
    fh = fits.open(INJ)
    pixar_sr = float(fh['SCI'].header.get('PIXAR_SR', fh[0].header.get('PIXAR_SR', 2.29e-14)))
    cat = ssf.get_saturated_stars(fits.open(INJ), path_prefix=PSFS, plot=False,
                                  use_merged_psf_for_merged=False)
    inj = Table.read(ECSV)
    w = frame_wcs(fh).gwcs
    ra, dec = w(np.asarray(inj['x'], float), np.asarray(inj['y'], float))
    sc_i = SkyCoord(np.asarray(ra) * u.deg, np.asarray(dec) * u.deg)
    sc_r = SkyCoord(cat['skycoord_fit'])
    fin = np.isfinite(sc_r.ra.deg) & np.isfinite(sc_r.dec.deg)
    idx, sep, _ = sc_i.match_to_catalog_sky(sc_r[fin])
    f = np.asarray(cat[fin]['flux_fit'][idx], float)
    rec_mag = art.imflux_to_mag(np.where(sep.arcsec < 0.15, f, np.nan), BAND, pixar_sr)
    dmag = rec_mag - np.asarray(inj['mag_inj'], float)
    return np.asarray(inj['sat_area'], float), dmag


def main():
    print(f"[wctest] {os.path.basename(INJ)}")
    sa_on, d_on = recover(True)
    sa_off, d_off = recover(False)
    print(f"\n  {'regime':9} {'N_on':>5} {'bias_ON':>8} {'N_off':>5} {'bias_OFF':>9}  (mmag, +ve=faint)")
    for lo, hi, lab in REGIMES:
        m_on = (sa_on >= lo) & (sa_on < hi) & np.isfinite(d_on)
        m_off = (sa_off >= lo) & (sa_off < hi) & np.isfinite(d_off)
        b_on = np.median(d_on[m_on]) * 1000 if m_on.sum() else np.nan
        b_off = np.median(d_off[m_off]) * 1000 if m_off.sum() else np.nan
        print(f"  {lab:9} {int(m_on.sum()):5d} {b_on:+8.0f} {int(m_off.sum()):5d} {b_off:+9.0f}")
    print("\n  interpretation: if bias_OFF ~ 0 while bias_ON large -> the bias is "
          "wing-selfcal over-correcting STPSF-winged injected stars (harness "
          "artifact -> need empirical wings). If bias_OFF still large -> real "
          "masked-core under-recovery.")


if __name__ == '__main__':
    main()
