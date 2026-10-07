"""Remaining complementary split pairs in the mfs1 m7 catalog: for each, the dolphot stars nearby and every vetted
per-band row within 0.15" (offset from the dolphot star, flux, replaced).  usage: python split_rest.py"""
import sys
import numpy as np
from astropy.table import Table
from astropy.coordinates import search_around_sky, SkyCoord
import astropy.units as u
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an
Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
M7 = 'catalogs/basic_merged_indivexp_photometry_tables_merged_resbgsub_m7.fits'
an.PATH['mfs1'] = (f'{Q}/tree_mfs1/{M7}', f'{an.D}/matched_Q_s1t1_mfs1.fits')
an.ZPWIN.update(an.zp_windows())
A = an.Arm('mfs1')
cat = A.cat
bands = [b for b in an.BANDS if f'mag_vega_f{b.lower()}' in cat.colnames]
F = np.array([np.isfinite(np.asarray(an.fl(cat[f'mag_vega_f{b.lower()}']), float)) for b in bands]).T
mk = np.where(A.matched)[0]
mrow = A.idx[mk]
i1, i2, sep, _ = search_around_sky(A.sky[mrow], A.sky, 0.08 * u.arcsec)
k = mrow[i1] != i2
i1, i2, sep = i1[k], i2[k], sep[k]
d = ~(F[mrow[i1]] & F[i2]).any(axis=1)
pairs = {}
for a, p, s in zip(i1[d], i2[d], sep[d].to_value(u.mas)):
    key = tuple(sorted((mrow[a], p)))
    pairs.setdefault(key, (mk[a], s))
ref = Table.read('/orange/adamginsburg/jwst/wd2/catalogs/wd2_nircam_wf_mf_nf.ecsv')
rsc = SkyCoord(ref['RA'], ref['DEC'], unit='deg')
V = {}
for b in bands:
    t = Table.read(f'{Q}/tree_mfs1/catalogs/f{b.lower()}_merged_indivexp_merged_resbgsub_m7_dao_basic_vetted.fits')
    V[b] = t
reff = np.asarray(cat['skycoord_ref_filtername']).astype(str)
for (r1, r2), (kd, s) in pairs.items():
    dsc = SkyCoord(an.fl(A.m['RA'])[kd], an.fl(A.m['DEC'])[kd], unit='deg')
    print(f'\n### rows {r1} ({reff[r1]}) / {r2} ({reff[r2]}), {s:.1f} mas apart; dolphot F200W {A.ref["200W"][kd]:.2f}, '
          f'F115W {A.ref["115W"][kd]:.2f}, F410M {A.ref["410M"][kd]:.2f}')
    for r in (r1, r2):
        off = dsc.spherical_offsets_to(A.sky[r])
        print(f'  row {r}: offset from dolphot ({off[0].to_value(u.mas):+.0f}, {off[1].to_value(u.mas):+.0f}) mas; '
              f'bands {",".join(b for j, b in enumerate(bands) if F[r, j])}')
    near = np.where(rsc.separation(dsc) < 0.2 * u.arcsec)[0]
    for n in near:
        off = dsc.spherical_offsets_to(rsc[n])
        mags = ' '.join(f'{b}={float(ref['MAG' + b][n]):.2f}' for b in ('115W', '200W', '410M'))
        print(f'  dolphot star at ({off[0].to_value(u.mas):+.0f}, {off[1].to_value(u.mas):+.0f}) mas  {mags}')
    for b in bands:
        t = V[b]
        sc = t['skycoord']
        w = np.where(sc.separation(dsc) < 0.15 * u.arcsec)[0]
        for i in w:
            off = dsc.spherical_offsets_to(sc[i])
            rep = bool(np.ma.filled(t['replaced_saturated'][i], False))
            print(f'    F{b}: ({off[0].to_value(u.mas):+.0f}, {off[1].to_value(u.mas):+.0f}) mas  flux {float(np.ma.filled(t["flux"][i], np.nan)):.3g}'
                  f'  rep={rep}')
