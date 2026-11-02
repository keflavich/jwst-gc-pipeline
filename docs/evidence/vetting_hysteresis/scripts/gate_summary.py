"""Summarise the vet_gates.py replay of the vetting-flicker losses.

For every star vetted at phase K and vetted out at phase K+1 (category
``vetted_out`` of the phase-loss tables), report which keep branches it
passed at K, and how many of them the previous-phase keep would re-admit at
K+1: the K+1 merged row lies within 0.5 FWHM of the star's phase-K position
and has qfit < 0.6 and S/N >= 10.  FWHM per band from the pipeline's
fwhm_table (jwst_gc_pipeline.photometry.crowdsource_catalogs_long).

usage: python gate_summary.py <vet_gates out.fits> ...
"""
import sys
from collections import Counter

import numpy as np
from astropy.table import Table, vstack

from jwst_gc_pipeline.photometry import crowdsource_catalogs_long as L

QMAX, SNR_MIN, RADIUS_FWHM = 0.6, 10.0, 0.5


def main(*paths):
    t = vstack([Table.read(p) for p in paths])
    ftab = Table.read(L.fwhm_table_path())
    fwhm = {str(f): float(v) for f, v in zip(ftab['Filter'], ftab['PSF FWHM (arcsec)'])}
    filt = np.asarray(t['filt']).astype(str)
    field = np.asarray(t['field']).astype(str)
    star = np.asarray(t['starlike_left'], bool)
    r_mas = np.array([RADIUS_FWHM * fwhm[f] * 1e3 for f in filt])
    with np.errstate(invalid='ignore'):
        keep = ((np.asarray(t['next_sep_mas'], float) <= r_mas)
                & (np.asarray(t['next_qfit'], float) < QMAX)
                & (np.asarray(t['next_snr'], float) >= SNR_MIN))
    print(f'vetted at K, vetted out at K+1: {len(t)}; star-like residual left: {star.sum()}')
    print('keep branches passed at K (prev_kept_by):')
    for k, n in Counter(np.asarray(t['prev_kept_by']).astype(str)).most_common():
        print(f'  {k:28s} {n}')
    bi = (np.asarray(t['prev_kept_by']).astype(str) == 'bright_iso') & (field == 'superdense') & (filt == 'F212N')
    print(f'superdense F212N, bright_iso only: n={bi.sum()}, median qfit at K '
          f'{np.median(t["prev_qfit"][bi]):.2f}, at K+1 {np.median(t["next_qfit"][bi]):.2f}')
    print(f'previous-phase keep (r <= {RADIUS_FWHM} FWHM, qfit < {QMAX}, S/N >= {SNR_MIN}) '
          f're-admits {keep.sum()} of {len(t)}; star-like {np.sum(keep & star)}; '
          f'other {np.sum(keep & ~star)} (median S/N {np.median(t["next_snr"][keep & ~star]):.0f})')
    print('| field / band | vetted_out | star-like | re-admitted | re-admitted star-like |')
    print('|---|---|---|---|---|')
    for fl, fb in dict.fromkeys(zip(field, filt)):
        s = (field == fl) & (filt == fb)
        print(f'| {fl} {fb} | {s.sum()} | {np.sum(s & star)} | {np.sum(s & keep)} | {np.sum(s & keep & star)} |')


if __name__ == '__main__':
    main(*sys.argv[1:])
