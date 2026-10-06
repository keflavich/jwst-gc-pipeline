"""m7 vetted sources missing from the m8 cross-band table (catalog level).

For each band of one target/observation: the m7 vetted (merged module)
sources, matched by position (5 mas) to the band's skycoord_<f> column of
the m8 table and of the m8 dedup table.  Missing sources: their S/N and the
separation to the nearest other m7 vetted source of the same band and to
the nearest m8 row's band position.

usage: python m8check.py <target_dir> <obstok> <out.json>
"""
import glob
import json
import os
import re
import sys

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

base, obstok, outpath = sys.argv[1], sys.argv[2], sys.argv[3]
cat = f'{base}/catalogs'
res = {}
for kind in ('m8', 'm8_dedup'):
    p = f'{cat}/basic_merged_indivexp_photometry_tables_merged_resbgsub_{kind}{obstok}.fits'
    if not os.path.exists(p):
        print('missing', p)
        continue
    m8 = Table.read(p)
    bands = sorted({re.match(r'skycoord_(f\w+)\.ra', c).group(1) if re.match(r'skycoord_(f\w+)\.ra', c) else None
                    for c in m8.colnames} - {None})
    if not bands:
        bands = sorted({c.split('_', 1)[1] for c in m8.colnames
                        if c.startswith('skycoord_f') and not c.endswith(('.ra', '.dec'))})
    for b in bands:
        vp = glob.glob(f'{cat}/{b}_merged{obstok}_indivexp_merged_resbgsub_m7_dao_basic_vetted.fits')
        if len(vp) != 1:
            continue
        v = Table.read(vp[0])
        vsc = v['skycoord'] if isinstance(v['skycoord'], SkyCoord) else SkyCoord(v['skycoord'])
        if f'skycoord_{b}' in m8.colnames:
            msc = m8[f'skycoord_{b}']
            msc = msc if isinstance(msc, SkyCoord) else SkyCoord(msc)
        else:
            msc = SkyCoord(m8[f'skycoord_{b}.ra'], m8[f'skycoord_{b}.dec'], unit='deg')
        ok = np.isfinite(msc.ra.deg) & np.isfinite(np.asarray(m8[f'flux_{b}'], float))
        if f'forced_filled_{b}' in m8.colnames:
            ok &= ~np.asarray(m8[f'forced_filled_{b}'], bool)
        msc = msc[ok]
        _, sep, _ = vsc.match_to_catalog_sky(msc)
        missing = sep.to_value(u.mas) > 5
        fe = np.asarray(v['flux_err_prop'] if 'flux_err_prop' in v.colnames else v['flux_err'], float)
        snr = np.asarray(v['flux'], float) / fe
        _, nn, _ = vsc.match_to_catalog_sky(vsc, nthneighbor=2)
        nn = nn.to_value(u.mas)
        r = dict(n_m7_vetted=len(v), n_band_rows=int(ok.sum()), n_missing=int(missing.sum()),
                 missing_snr_pcts=np.nanpercentile(snr[missing], [10, 50, 90]).tolist() if missing.any() else None,
                 n_missing_snr_gt10=int((missing & (snr > 10)).sum()),
                 missing_nn_mas_pcts=np.nanpercentile(nn[missing], [10, 50, 90]).tolist() if missing.any() else None,
                 kept_nn_mas_pcts=np.nanpercentile(nn[~missing], [10, 50, 90]).tolist(),
                 n_missing_nn_lt_100mas=int((missing & (nn < 100)).sum()),
                 n_missing_dup_band_pos=int((missing & (sep.to_value(u.mas) < 60)).sum()))
        res[f'{kind}:{b}'] = r
        print(kind, b, json.dumps(r), flush=True)
json.dump(res, open(outpath, 'w'), indent=1)
