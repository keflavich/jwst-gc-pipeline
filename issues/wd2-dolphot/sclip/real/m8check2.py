"""Rescued good rows (fix-only combine rows within 0.08" and 0.3 mag of dolphot): what the main2kfpk m8 catalog
gives the same dolphot star in that band (value or not, forced_filled, dm), vs the rescued row's dm."""
import sys, warnings
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.stats import mad_std
import astropy.units as u
warnings.filterwarnings('ignore')
Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
sys.path.insert(0, Q)
import analyze as an
IN = '/blue/adamginsburg/adamginsburg/tmp/claude-3663/sclip_real'
an.ZPWIN.update(an.zp_windows())
A = an.Arm('main2kfpk')
m8 = Table.read(f'{Q}/tree_main2kfpk/catalogs/basic_merged_indivexp_photometry_tables_merged_resbgsub_m8_dedup.fits')
rs = SkyCoord(np.asarray(A.m['RA'], float)*u.deg, np.asarray(A.m['DEC'], float)*u.deg)
oi = np.asarray(A.m['our_idx']); mt = np.asarray(A.matched, bool)
for b, zp in (('277W', 24.1296), ('250M', 24.3157), ('300M', 23.9704)):
    ref = A.ref[b]; okr = np.flatnonzero(np.isfinite(ref))
    d = np.load(f'{IN}/comb_fix_{b}.npz'); new = np.load(f'{IN}/rescued_{b}.npy')
    c = SkyCoord(d['ra'][new]*u.deg, d['dec'][new]*u.deg)
    j, sep, _ = c.match_to_catalog_sky(rs[okr])
    dmr = -2.5*np.log10(d['flux'][new]) + zp - ref[okr][j]
    g = (sep.arcsec < 0.08) & (np.abs(dmr) < 0.3)
    idx = okr[j[g]]; dmr = dmr[g]
    dm8 = A.dm(b)[idx]
    has = np.isfinite(dm8)
    ff = np.zeros(len(idx), bool)
    sel = mt[idx] & (oi[idx] >= 0)
    ff[sel] = np.ma.filled(m8[f'forced_filled_f{b.lower()}'][oi[idx][sel]], False).astype(bool)
    good8 = has & (np.abs(dm8) < 0.3)
    print(f'F{b}: rescued good {len(idx)}; dolphot star matched in m8 {mt[idx].sum()}; m8 band value {has.sum()} '
          f'(forced_filled {(ff & has).sum()}); m8 value good {good8.sum()}; '
          f'=> stars that become good with the fix: {len(idx) - good8.sum()}')
    both = has
    print(f'   where m8 has a value: |dm| rescued median {np.median(np.abs(dmr[both])):.3f}, m8 median {np.median(np.abs(dm8[both])):.3f}; '
          f'rescued closer in {(np.abs(dmr[both]) < np.abs(dm8[both])).sum()} / {both.sum()}; '
          f'robust std dm rescued {mad_std(dmr[both]):.3f} m8 {mad_std(dm8[both]):.3f}')
    print(f'   not-good-in-m8 stars: dolphot mags {np.round(ref[idx][~good8], 2).tolist()}, m8 dm {np.round(dm8[~good8], 2).tolist()}')
