"""Purity proxy for the rescued rows: P(pass vetting | flux bin) from the main combine rows (pass = a row of
the production main2kfpk m7 vetted band catalog within 0.02"), applied to the rescued rows by flux bin.
Reports rescued rows near / not near dolphot and the expected number that would reach the vetted catalog."""
import sys, warnings
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
warnings.filterwarnings('ignore')
Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
IN = '/blue/adamginsburg/adamginsburg/tmp/claude-3663/sclip_real'
sys.path.insert(0, Q)
import analyze as an
an.ZPWIN.update(an.zp_windows())
A = an.Arm('main2kfpk')
rs = SkyCoord(np.asarray(A.m['RA'], float)*u.deg, np.asarray(A.m['DEC'], float)*u.deg)
BANDS = sys.argv[1:] or ['115W', '150W', '162M', '182M', '200W', '250M', '277W', '300M', '335M', '410M',
                         '164N', '187N', '212N', '323N', '405N', '466N']
print('| band | main rows matching a production band row (<0.02″) | main rows passing vetting | rescued: near dolphot / not near | '
      'expected to pass vetting: near / not near |')
print('|---|---|---|---|---|')
for b in BANDS:
    M = np.load(f'{IN}/comb_main_{b}.npz'); F = np.load(f'{IN}/comb_fix_{b}.npz'); new = np.flatnonzero(np.isfinite(F['ra']) & (F['flux'] > 0) & ~(np.isfinite(M['ra']) & (M['flux'] > 0)))
    T = f'{Q}/tree_main2kfpk/catalogs/f{b.lower()}_merged_indivexp_merged_resbgsub_m7_dao_basic'
    band = Table.read(T + '.fits'); vet = Table.read(T + '_vetted.fits')
    ok = np.isfinite(M['ra']) & (M['flux'] > 0)
    cm = SkyCoord(M['ra'][ok]*u.deg, M['dec'][ok]*u.deg)
    _, sb, _ = cm.match_to_catalog_sky(SkyCoord(band['skycoord']))
    _, sv, _ = cm.match_to_catalog_sky(SkyCoord(vet['skycoord']))
    inband = sb.arcsec < 0.02; passv = sv.arcsec < 0.02
    lf = np.log10(M['flux'][ok])
    edges = np.quantile(lf, np.linspace(0, 1, 41)); edges[-1] += 1e-6
    k = np.clip(np.digitize(lf, edges) - 1, 0, 39)
    p = np.array([passv[k == i].mean() if (k == i).any() else 0 for i in range(40)])
    fl = F['flux'][new]
    kn = np.clip(np.digitize(np.log10(np.clip(fl, 1e-30, None)), edges) - 1, 0, 39)
    cn = SkyCoord(F['ra'][new]*u.deg, F['dec'][new]*u.deg)
    ok_ref = np.isfinite(A.ref[b])
    _, sd, _ = cn.match_to_catalog_sky(rs[ok_ref])
    near = sd.arcsec < 0.08
    pn = p[kn] * (fl > 0)
    print(f'| F{b} | {inband.mean():.3f} | {passv.sum()} / {ok.sum()} | {near.sum()} / {(~near).sum()} | '
          f'{pn[near].sum():.0f} / {pn[~near].sum():.0f} |')
