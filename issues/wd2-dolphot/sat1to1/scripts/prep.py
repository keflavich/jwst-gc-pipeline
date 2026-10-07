"""Build scratch trees for the one-row-per-satstar test of the m7 cross-band merge.
tree_mfs0: tree_mainfcbg's per-band _vetted tables unchanged (baseline).
tree_mfs1: the same tables with each group of satstar-replaced rows within 1 mas reduced to the row with the
           smallest satstar_match_sep (what the fixed replace_saturated keeps at the per-band merge).
Both trees get <BAND>/pipeline/ directories holding symlinks to tree_mainfcbg's satstar and crf files."""
import glob
import os
import numpy as np
from astropy.table import Table
from astropy.coordinates import search_around_sky
import astropy.units as u

Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
SRC = f'{Q}/tree_mainfcbg'
BANDS = ['f115w', 'f150w', 'f162m', 'f164n', 'f182m', 'f187n', 'f200w', 'f212n', 'f250m', 'f277w', 'f300m',
         'f323n', 'f335m', 'f405n', 'f410m', 'f466n']
for arm in ('mfs0', 'mfs1'):
    T = f'{Q}/tree_{arm}'
    os.makedirs(f'{T}/catalogs', exist_ok=True)
    for b in BANDS:
        pdir = f'{T}/{b.upper()}/pipeline'
        os.makedirs(pdir, exist_ok=True)
        srcs = (glob.glob(f'{SRC}/{b.upper()}/pipeline/*satstar*.fits')
                + glob.glob(f'{SRC}/{b.upper()}/pipeline/*_crf.fits'))
        for s in srcs:
            d = f'{pdir}/{os.path.basename(s)}'
            if not os.path.lexists(d):
                os.symlink(os.path.realpath(s), d)
        fn = f'{b}_merged_indivexp_merged_resbgsub_m7_dao_basic_vetted.fits'
        out = f'{T}/catalogs/{fn}'
        if os.path.exists(out):
            continue
        t = Table.read(f'{SRC}/catalogs/{fn}')
        n0 = len(t)
        if arm == 'mfs1':
            c = t['skycoord']
            i1, i2, _, _ = search_around_sky(c, c, 1 * u.mas)
            k = (i1 != i2)
            rep = np.asarray(t['replaced_saturated'], bool)
            k &= rep[i1] & rep[i2]
            i1, i2 = i1[k], i2[k]
            msep = np.asarray(t['satstar_match_sep'], float)
            msep = np.where(np.isfinite(msep), msep, np.inf)
            # drop i1 when its partner is nearer to the fit (ties: keep the lower index)
            worse = (msep[i1] > msep[i2]) | ((msep[i1] == msep[i2]) & (i1 > i2))
            drop = np.unique(i1[worse])
            t.remove_rows(drop)
            print(f'{arm} {b}: {n0} -> {len(t)} rows (dropped {len(drop)})', flush=True)
        t.write(out)
    print(arm, 'done', flush=True)
