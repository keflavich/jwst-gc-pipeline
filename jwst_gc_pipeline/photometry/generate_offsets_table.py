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
"""
import sys, glob, re, warnings
sys.path.insert(0, '/orange/adamginsburg/repos/jwst-gc-pipeline')
import numpy as np
import astropy.units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import stats
from jwst_gc_pipeline.photometry.merge_catalogs import shift_individual_catalog

warnings.simplefilter('ignore')
BASE = '/orange/adamginsburg/jwst/brick'
EPOCH = 2022.70
SHIFTS = f'{BASE}/astrometry_diag/f115w_virac2_perframe_shifts.ecsv'
OUT = f'{BASE}/offsets/Offsets_JWST_Brick1182_F115W_VIRAC2frame.csv'


def farr(x):
    return np.asarray(np.ma.filled(np.ma.masked_invalid(np.asarray(x, float)), np.nan), float)


def prop(ra, dec, pmra, pmde, dt):
    pmra = np.where(np.isfinite(pmra), pmra, 0.); pmde = np.where(np.isfinite(pmde), pmde, 0.)
    return ra + (pmra * dt / 3.6e6) / np.cos(np.radians(dec)), dec + (pmde * dt / 3.6e6)


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
import os
os.makedirs(f'{BASE}/offsets', exist_ok=True)
ot.write(OUT, overwrite=True)
print(f"Wrote {OUT}: {len(ot)} frames")

# ---- validate: run shift_individual_catalog on a sample, check landing on VIRAC2 ----
v = Table.read(f'{BASE}/astrometry_diag/refcache/virac2.fits')
virac = SkyCoord(*prop(farr(v['RAJ2000']), farr(v['DEJ2000']), farr(v['pmRA']), farr(v['pmDE']), EPOCH - 2016.0), unit='deg')


def voff(sc):
    ci, s, _ = sc.match_to_catalog_sky(virac); ri, _, _ = virac.match_to_catalog_sky(sc)
    k = (ri[ci] == np.arange(len(ci))) & (s <= 0.25 * u.arcsec)
    if k.sum() < 8: return None
    mc, mr = sc[np.where(k)[0]], virac[ci[k]]
    dra = (mc.ra.deg - mr.ra.deg) * np.cos(np.radians(mc.dec.deg)) * 3.6e6
    dd = (mc.dec.deg - mr.dec.deg) * 3.6e6
    m, md = np.nanmedian(dra), np.nanmedian(dd); cl = np.hypot(dra - m, dd - md) <= 120
    return round(float(np.nanmedian(dra[cl])), 1), round(float(np.nanmedian(dd[cl])), 1), int(cl.sum())

print("\nValidation: per-frame offset to VIRAC2 BEFORE vs AFTER applying the table (sample):")
import random
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
