"""Prominence + continuum-counterpart table for a band's m6 merged catalog.

usage: python build_pl.py <pipeline_worktree> <field> <band> <cont_band> <out.fits>
                          [band_stem cont_stem i2d_glob]
band_stem / cont_stem default to '<band>_merged'; i2d_glob (relative to the
field's <BAND>/pipeline) defaults to the o001 merged data i2d.
Same columns as pl_w51*.fits: prominence on the merged data i2d (computed by
_filter_extended_emission, which writes it onto the input table), and
cont_match = a continuum (S/N >= 3) m6 source within 60 mas.
"""
import glob
import sys
WT, FLD, BAND, CONT, OUTF = sys.argv[1:6]
BSTEM = sys.argv[6] if len(sys.argv) > 6 else f'{BAND.lower()}_merged'
CSTEM = sys.argv[7] if len(sys.argv) > 7 else f'{CONT.lower()}_merged'
IGLOB = sys.argv[8] if len(sys.argv) > 8 else f'jw*-o001_t001_nircam_clear-{BAND.lower()}-merged_data_i2d.fits'
sys.path.insert(0, WT)
import numpy as np                                       # noqa: E402
import astropy.units as u                                # noqa: E402
from astropy.coordinates import SkyCoord                 # noqa: E402
from astropy.io import fits                              # noqa: E402
from astropy.table import Table                          # noqa: E402
from astropy import wcs                                  # noqa: E402
from jwst_gc_pipeline.photometry import cataloging as C  # noqa: E402

R = '/orange/adamginsburg/jwst'
b, c = BAND.lower(), CONT.lower()
t = Table.read(f'{R}/{FLD}/catalogs/{BSTEM}_indivexp_merged_resbgsub_m6_dao_basic.fits')
dp = glob.glob(f'{R}/{FLD}/{BAND.upper()}/pipeline/{IGLOB}')
assert len(dp) == 1, dp
with fits.open(dp[0]) as dh:
    d = dh['SCI'].data.astype(float)
    ww = wcs.WCS(dh['SCI'].header)
C._filter_extended_emission(t, data_i2d_image=d, ww_i2d=ww, sky_clean_keep=False, label='pl')
assert 'prominence' in t.colnames
ct = Table.read(f'{R}/{FLD}/catalogs/{CSTEM}_indivexp_merged_resbgsub_m6_dao_basic.fits')
cs = np.asarray(ct['flux'] / ct['flux_err'], float)
cc = SkyCoord(ct['skycoord'][np.isfinite(cs) & (cs >= 3)])
sep = SkyCoord(t['skycoord']).match_to_catalog_sky(cc)[1].to_value(u.mas)
keep = ['skycoord', 'flux', 'flux_err', 'qfit', 'local_bkg', 'prominence', 'prominence_robust',
        'peak_sb', 'flags', 'group_size', 'nmatch']
o = t[[k for k in keep if k in t.colnames]]
o['cont_sep_mas'] = sep
o['cont_match'] = sep < 60
o.meta = {'FIELD': FLD, 'BAND': BAND, 'CONT': CONT, 'BSTEM': BSTEM, 'CSTEM': CSTEM, 'DATAI2D': dp[0]}
o.write(OUTF, overwrite=True)
print(len(o), 'rows ->', OUTF)
