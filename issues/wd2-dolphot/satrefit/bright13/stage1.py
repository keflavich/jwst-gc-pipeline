"""Stage 1: map 10-13 mag LW score stars to per-frame rows; join dolphot ecsv, our catalog columns."""
import glob, os, pickle, sys
import numpy as np
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
import astropy.units as u
Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
an.ZPWIN.update(an.zp_windows())
A = an.Arm('main2')
ecsv = Table.read('/orange/adamginsburg/jwst/wd2/catalogs/wd2_nircam_wf_mf_nf.ecsv')
esk = SkyCoord(np.asarray(ecsv['RA'], float) * u.deg, np.asarray(ecsv['DEC'], float) * u.deg)
out = {}
for band in ('250M', '300M'):
    pk = pickle.load(open(f'{Q}/satrefit/out7/score7_{band}.pkl', 'rb'))
    ok = A.matched & A.rep[band] & np.isfinite(A.ref[band])
    tgt = A.sky[A.idx[ok]]
    ref = np.asarray(A.ref[band][ok], float)
    assert np.allclose(ref, pk['ref'])
    sel = pk['have'] & (pk['ref'] >= 10) & (pk['ref'] < 13)
    oi = np.where(ok)[0]            # matched-file row index
    # frame tables
    tabs = []
    for fn2 in sorted(glob.glob(f'{Q}/satrefit/out2/{band}_*_satrefit.fits')):
        t2 = Table.read(fn2); t5 = Table.read(fn2.replace('/out2/', '/out5/').replace('_satrefit.fits', '_satrefit5.fits'))
        t7 = Table.read(fn2.replace('/out2/', '/out7/').replace('_satrefit.fits', '_satrefit7.fits'))
        tt = Table()
        for c in ('idx', 'label', 'a_cat', 'a_raw', 'ra', 'dec'):
            tt[c] = t2[c] if c in t2.colnames else np.arange(len(t2))
        for c in ('a_base', 'cap_base', 'nfit', 'nrim_fit', 'nrw_h'):
            tt[c] = t5[c]
        for c in ('a_H', 'cap_H', 'a_H+bgfree', 'a_H+h0'):
            tt[c] = t7[c]
        tt['pixfile'] = os.path.basename(fn2).replace(f'{band}_', '').replace('_satrefit.fits', '')
        tt['row'] = np.arange(len(t2))
        tabs.append(tt)
    T = vstack(tabs, metadata_conflicts='silent')
    sk = SkyCoord(np.asarray(T['ra']) * u.deg, np.asarray(T['dec']) * u.deg)
    i, j, _, _ = sk.search_around_sky(tgt, 0.1 * u.arcsec)
    good = np.asarray(T['label']) > 0
    rows = []
    for ii, jj in zip(i, j):
        if good[jj] and sel[ii]:
            rows.append((ii, jj))
    # dolphot ecsv
    d2, = [None]
    idx, d2d, _ = tgt.match_to_catalog_sky(esk)
    star = Table()
    star['istar'] = np.where(sel)[0]
    star['ref'] = ref[sel]
    star['mi'] = oi[sel]
    star['ecsv_idx'] = idx[sel]
    star['ecsv_sep_mas'] = d2d[sel].to(u.mas).value
    for k in ('final', 'H+cap', 'H+bgfree+cap', 'H+h0+bgfree+cap', 'uncapped'):
        star['dm_' + k] = pk['dm'][k][sel]
    star['ecsv_mag'] = np.asarray(ecsv['MAG' + band], float)[idx[sel]]
    star['ecsv_err'] = np.asarray(ecsv['ERRMAG' + band], float)[idx[sel]]
    star['ra'] = tgt.ra.deg[sel]; star['dec'] = tgt.dec.deg[sel]
    star['our_qfit'] = A.qf[band][ok][sel]
    star['our_sat_flag'] = A.sat[band][ok][sel]
    for b in ('115W', '150W', '200W', '277W', '335M', '410M', '300M', '250M'):
        star['dm_' + b] = np.where(A.matched & np.isfinite(A.ref[b]), A.dm(b), np.nan)[ok][sel]
        star['ref_' + b] = A.ref[b][ok][sel]
        star['rep_' + b] = A.rep[b][ok][sel]
    star['nm'] = np.zeros(len(star), int)
    sset = {int(s): k for k, s in enumerate(star['istar'])}
    rt = Table(rows=[dict(istar=int(a), **{c: T[c][b] for c in T.colnames}) for a, b in rows])
    star['nrows'] = [int((rt['istar'] == s).sum()) for s in star['istar']]
    out[band] = (star, rt)
    star.write(f'star_{band}.fits', overwrite=True); rt.write(f'rows_{band}.fits', overwrite=True)
    print(band, len(star), len(rt), 'median sep mas', np.median(star['ecsv_sep_mas']))
