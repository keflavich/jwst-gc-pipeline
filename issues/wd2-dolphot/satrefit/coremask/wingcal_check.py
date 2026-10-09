"""Production wingcal columns (catalog) per band and mag bin, for comparison with the unsaturated-star R(r).  Writes wingcal_check.txt"""
import pickle, re, sys
import numpy as np
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit')
import run_frames5 as R5
from astropy.table import Table
CB = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit/capbind'
BINS = {'150W': [(14, 15), (15, 16), (16, 17)], '200W': [(13, 14), (14, 15), (15, 16)],
        '250M': [(12.3, 13), (13, 13.5), (13.5, 14), (14, 15)], '300M': [(12.3, 13), (13, 13.5), (13.5, 14), (14, 15)]}
out = []
for band in BINS:
    m = pickle.load(open(f'{CB}/map_{band}.pkl', 'rb'))
    rr, wr, lab, ar, ac = [], [], [], [], []
    for fnm in m['files']:
        mm = re.match(r'(\w+?)_jw03523005001_(\d+)_(\d+)_(\w+?)_align', fnm)
        b_, vg, e_, det = mm.groups()
        cat = Table.read(R5.path_of(band, det, int(e_)).replace('.fits', '') + '_resbgsub_m7_satstar_catalog.fits')
        if band == '150W' and not out:
            out.append('catalog columns: ' + ' '.join(cat.colnames))
        rr.append(np.asarray(cat['wingcal_rmask'], float))
        wr.append(np.asarray(cat['wingcal_ratio'], float) if 'wingcal_ratio' in cat.colnames else np.full(len(cat), np.nan))
        ar.append(np.asarray(cat['flux_fit_raw'], float)); ac.append(np.asarray(cat['flux_fit'], float))
    rr, wr, ar, ac = map(np.concatenate, (rr, wr, ar, ac))
    ref = np.asarray(m['ref'], float); i, j = m['i'], m['j']
    for lo, hi in BINS[band]:
        s = (ref[i] >= lo) & (ref[i] < hi)
        jj = j[s]
        out.append(f'{band} {lo}-{hi} N={len(jj)} wingcal_rmask med {np.nanmedian(rr[jj]):.2f} wingcal_ratio med {np.nanmedian(wr[jj]):.3f} (16-84: {np.nanpercentile(wr[jj],16):.3f}-{np.nanpercentile(wr[jj],84):.3f}) flux_fit/flux_fit_raw med {np.nanmedian(ac[jj]/ar[jj]):.3f}')
open('wingcal_check.txt', 'w').write('\n'.join(out))
print('\n'.join(out))
