#!/usr/bin/env python
"""Generate a production offsets-table for merge_catalogs.merge_daophot from the
per-frame VIRAC2 alignment solution, using the verified convention:

  final = skycoord_centroid - RAOFFSET_meta + dra_table   (dra is Δα RA-coordinate)
  want  = skycoord_centroid + correction (on-sky)
  =>  dra_table  = RAOFFSET_meta + (corr_dRA_onsky)/cos(dec)
      ddec_table = DEOFFSET_meta + corr_dDec

corr = my measured per-frame correction to VIRAC2 (Gaia-tied frame), on-sky, mas
(f115w_virac2_perframe_shifts.ecsv: correction_d{RA,Dec}_mas). Validates the table
by running the actual shift_individual_catalog and checking frames land on VIRAC2.

Everything runs from ``main()`` -- importing this module has no side effects
(it used to write the production CSV at import time).
"""
import glob
import os
import re
import warnings

import numpy as np
import astropy.units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord

from jwst_gc_pipeline.photometry.merge_catalogs import shift_individual_catalog
from jwst_gc_pipeline.astrometry_utils import farr, prop
from jwst_gc_pipeline.photometry.measure_offsets import assert_sparse_reference_for_nn_median

BASE = '/orange/adamginsburg/jwst/brick'
EPOCH = 2022.70
SHIFTS = f'{BASE}/astrometry_diag/f115w_virac2_perframe_shifts.ecsv'
OUT = f'{BASE}/offsets/Offsets_JWST_Brick1182_F115W_VIRAC2frame.csv'


def main():
    warnings.simplefilter('ignore')

    sol = Table.read(SHIFTS)
    # key: (Module, Visit int, Exposure int) -> (corr_dRA_mas, corr_dDec_mas)
    corr = {(r['Module'], int(r['Visit']), int(r['Exposure'])):
            (float(r['correction_dRA_mas']), float(r['correction_dDec_mas'])) for r in sol}

    frames = sorted(glob.glob(f'{BASE}/F115W/f115w_*_visit*_exp*_daophot_basic.fits'))
    frames = [f for f in frames if not any(t in f for t in ('_m1', '_m2', '_m3', '_m4', 'iter', 'resbgsub', '_group'))]

    rows = []
    for fn in frames:
        t = Table.read(fn)
        mr = re.search(r'(nrc[ab]\d)_visit(\d+)_vgroup\d+_exp(\d+)', fn)
        mod, vis, exp = mr.group(1), int(mr.group(2)), int(mr.group(3))
        key = (mod, vis, exp)
        if key not in corr:
            continue
        RA0 = float(t.meta['RAOFFSET']); DE0 = float(t.meta['DEOFFSET'])
        dec0 = float(np.nanmedian(SkyCoord(t['skycoord_centroid']).icrs.dec.deg))
        cdra, cddec = corr[key]   # on-sky mas
        dra_table = RA0 + (cdra / 1000.0) / np.cos(np.radians(dec0))   # arcsec, Δα coordinate
        ddec_table = DE0 + (cddec / 1000.0)
        rows.append(dict(Filter='F115W', Module=mod, Visit=f'jw01182004{vis:03d}', Exposure=exp,
                         dra=dra_table, ddec=ddec_table,
                         RAOFFSET_meta=RA0, DEOFFSET_meta=DE0, corr_dRA_mas=cdra, corr_dDec_mas=cddec))

    ot = Table(rows)
    os.makedirs(f'{BASE}/offsets', exist_ok=True)
    ot.write(OUT, overwrite=True)
    print(f"Wrote {OUT}: {len(ot)} frames")

    # ---- validate: run shift_individual_catalog on a sample, check landing on VIRAC2 ----
    v = Table.read(f'{BASE}/astrometry_diag/refcache/virac2.fits')
    virac = SkyCoord(*prop(farr(v['RAJ2000']), farr(v['DEJ2000']), farr(v['pmRA']), farr(v['pmDE']), EPOCH - 2014.0), unit='deg')  # VIRAC2 ref epoch 2014.0 (II/387 paper value)

    def voff(sc):
        # FORBIDDEN-METHOD GUARD: matching to the DENSE VIRAC2 catalog with a
        # nearest-neighbour median is exactly the "got fooled twice" validation that
        # reported a spurious ~0 offset and let brick-1182 ship ~1.9" off.  Refuse it;
        # validate with offset-histogram stacking (or a Gaia-only subset) instead.
        assert_sparse_reference_for_nn_median(
            virac, 0.25 * u.arcsec,
            context="generate_offsets_table VIRAC2 validation (voff)")
        ci, s, _ = sc.match_to_catalog_sky(virac); ri, _, _ = virac.match_to_catalog_sky(sc)
        k = (ri[ci] == np.arange(len(ci))) & (s <= 0.25 * u.arcsec)
        if k.sum() < 8: return None
        mc, mr = sc[np.where(k)[0]], virac[ci[k]]
        dra = (mc.ra.deg - mr.ra.deg) * np.cos(np.radians(mc.dec.deg)) * 3.6e6
        dd = (mc.dec.deg - mr.dec.deg) * 3.6e6
        m, md = np.nanmedian(dra), np.nanmedian(dd); cl = np.hypot(dra - m, dd - md) <= 120
        return round(float(np.nanmedian(dra[cl])), 1), round(float(np.nanmedian(dd[cl])), 1), int(cl.sum())

    print("\nValidation: per-frame offset to VIRAC2 BEFORE vs AFTER applying the table (sample):")
    sample = frames[::40]
    for fn in sample:
        t = Table.read(fn)
        mr = re.search(r'(nrc[ab]\d)_visit(\d+)_vgroup\d+_exp(\d+)', fn)
        mod, vis, exp = mr.group(1), int(mr.group(2)), int(mr.group(3))
        sc_before = SkyCoord(t['skycoord_centroid']).icrs
        b = voff(sc_before)
        out = shift_individual_catalog(t.copy(), ot, verbose=False)
        a = voff(SkyCoord(out['skycoord_centroid']).icrs)
        print(f"  {mod} v{vis} e{exp}: before {b} -> after {a} mas")


if __name__ == '__main__':
    main()
