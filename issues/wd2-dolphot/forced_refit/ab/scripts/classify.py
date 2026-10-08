"""Cross-match fr0/fr1 m7 merged catalogs; classify fr1 forced rows (frac>0).
a: matched fr0 row (<=SEP) forced there; b: matched, not forced there; c: no fr0 row.
usage: python classify.py [BAND ...]  -> classify_<band>.npz + printed table"""
import sys
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an
H = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/forced_refit/ab'
SEP = 0.04
an.ZPWIN.update(an.zp_windows())
A = an.Arm('mainfcbg')
dsk = SkyCoord(np.asarray(A.m['RA'], float) * u.deg, np.asarray(A.m['DEC'], float) * u.deg)


def load(arm, b):
    t = Table.read(f'{H}/tree_{arm}/catalogs/f{b.lower()}_merged_indivexp_merged_resbgsub_m7_dao_basic.fits')
    sk = t['skycoord']
    fl = np.asarray(t['flux'], float)
    ok = np.isfinite(sk.ra.deg) & (fl > 0)
    return t[ok], sk[ok], fl[ok]


def zp_of(sk, fl, rep, ref):
    mi = -2.5 * np.log10(fl)
    mid = np.where(np.isfinite(ref) & (ref >= 18.6) & (ref < 21))[0]
    j, d, _ = dsk[mid].match_to_catalog_sky(sk)
    sel = (d.arcsec < 0.05) & ~rep[j]
    return np.median(ref[mid][sel] - mi[j][sel])


if __name__ == '__main__':
    bands = [x.upper().lstrip('F') for x in sys.argv[1:]] or ['150W', '187N', '200W', '277W']
    print('| band | fr0 rows | fr1 rows | fr0 forced | fr1 forced | fr1 forced: (a) fr0 forced | (b) fr0 not forced | (c) absent in fr0 |')
    print('|---|---|---|---|---|---|---|---|')
    for b in bands:
        t0, s0, f0 = load('fr0', b)
        t1, s1, f1 = load('fr1', b)
        ref = A.ref[b]
        zp0 = zp_of(s0, f0, np.asarray(t0['replaced_saturated'], bool), ref)
        zp1 = zp_of(s1, f1, np.asarray(t1['replaced_saturated'], bool), ref)
        ff0 = np.asarray(t0['forced_refit_frac'], float)
        ff1 = np.asarray(t1['forced_refit_frac'], float)
        m1 = -2.5 * np.log10(f1) + zp1
        m0 = -2.5 * np.log10(f0) + zp0
        j, d, _ = s1.match_to_catalog_sky(s0)
        has = d.arcsec <= SEP
        cls = np.where(~has, 'c', np.where(ff0[j] > 0, 'a', 'b'))
        cls = np.where(ff1 > 0, cls, '-')
        # reverse: fr0 rows with no fr1 counterpart
        jr, dr, _ = s0.match_to_catalog_sky(s1)
        # dolphot
        jd, dd, _ = s1.match_to_catalog_sky(dsk)
        # nearest brighter row (within same arm)
        from scipy.spatial import cKDTree
        ra = s1.ra.deg; de = s1.dec.deg
        x = np.cos(np.deg2rad(de)) * ra * 3600; y = de * 3600
        tree = cKDTree(np.c_[x, y])
        nnb = np.full(len(t1), np.nan)
        idx = np.where(ff1 > 0)[0]
        dist, nb = tree.query(np.c_[x[idx], y[idx]], k=6)
        for k, i in enumerate(idx):
            br = [dist[k, q] for q in range(1, 6) if m1[nb[k, q]] < m1[i]]
            nnb[k and i or i] = br[0] if br else np.nan
        np.savez(f'classify_{b}.npz', cls=cls, m1=m1, m0match=np.where(has, m0[j], np.nan), ff1=ff1,
                 ff0match=np.where(has, ff0[j], np.nan), dsep=d.arcsec, ddolphot=dd.arcsec, nnb=nnb,
                 ra=ra, dec=de, zp0=zp0, zp1=zp1, nfr1=np.asarray(t1['forced_refit_nframes'], float),
                 nmatch=np.asarray(t1['nmatch'], float))
        nf = ff1 > 0
        print(f'| F{b} | {len(t0)} | {len(t1)} | {(ff0 > 0).sum()} | {nf.sum()} | {(cls == "a").sum()} | {(cls == "b").sum()} | {(cls == "c").sum()} |', flush=True)
        print(f'#   F{b}: fr0 rows with no fr1 row within {SEP}": {(dr.arcsec > SEP).sum()}; fr1 rows total absent from fr0: {(~has).sum()}; ZP {zp0:.3f}/{zp1:.3f}', flush=True)
