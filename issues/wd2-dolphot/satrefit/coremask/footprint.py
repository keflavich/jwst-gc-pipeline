"""Typical masked footprint of saturated stars per band / dolphot-magnitude bin (round-7 out7 tables + satstar catalogs).
Writes footprint.pkl.  Mask-area estimate A = pi (sqrt(sat_area/pi) + buf)^2 with buf = compute_adaptive_mask_buffer(sat_area);
also reports nfit, nrim_fit and the unmasked-pixel deficit 81*81 - nfit."""
import pickle, re, sys
import numpy as np
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit')
import satrefit_core as C
import run_frames5 as R5
from satrefit_core import S, Table
CB = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit/capbind'
BINS = {'150W': [(14, 15), (15, 16), (16, 17)], '200W': [(13, 14), (14, 15), (15, 16)],
        '250M': [(12.3, 13), (13, 13.5), (13.5, 14), (14, 15)], '300M': [(12.3, 13), (13, 13.5), (13.5, 14), (14, 15)]}
out = {}
for band in BINS:
    m = pickle.load(open(f'{CB}/map_{band}.pkl', 'rb'))
    rows = dict(sat_area=[], nfit=[], nrim=[], buf=[], area=[])
    for fnm in m['files']:
        t7 = Table.read(f"{C.Q}/satrefit/out7/{fnm.replace('_satrefit.fits', '_satrefit7.fits')}")
        mm = re.match(r'(\w+?)_jw03523005001_(\d+)_(\d+)_(\w+?)_align', fnm)
        b_, vg, e_, det = mm.groups()
        cat = Table.read(R5.path_of(band, det, int(e_)).replace('.fits', '') + '_resbgsub_m7_satstar_catalog.fits')
        assert len(cat) == len(t7)
        sa = np.asarray(cat['sat_area'], float)
        rows['sat_area'].append(sa)
        rows['nfit'].append(np.asarray(t7['nfit'], float) if 'nfit' in t7.colnames else np.full(len(t7), np.nan))
        rows['nrim'].append(np.asarray(t7['nrim_fit'], float) if 'nrim_fit' in t7.colnames else np.full(len(t7), np.nan))
        rows['label'] = rows.get('label', []) + [np.asarray(t7['label'])]
    for k in ('sat_area', 'nfit', 'nrim', 'label'):
        rows[k] = np.concatenate(rows[k])
    buf = np.array([S.compute_adaptive_mask_buffer(int(s)) if s > 0 else 0 for s in rows['sat_area']], float)
    area = np.pi * (np.sqrt(np.maximum(rows['sat_area'], 0) / np.pi) + buf) ** 2
    ref = np.asarray(m['ref'], float)
    i, j = m['i'], m['j']
    sel = rows['label'][j] > 0
    res = {}
    for lo, hi in BINS[band]:
        s = sel & (ref[i] >= lo) & (ref[i] < hi)
        jj = j[s]
        res[(lo, hi)] = dict(n=int(s.sum()), sat_area=float(np.median(rows['sat_area'][jj])), buf=float(np.median(buf[jj])),
                             area_est=float(np.median(area[jj])), nfit=float(np.nanmedian(rows['nfit'][jj])),
                             deficit=float(np.nanmedian(81 * 81 - rows['nfit'][jj])), nrim=float(np.nanmedian(rows['nrim'][jj])),
                             sat_area_q=[float(x) for x in np.percentile(rows['sat_area'][jj], [16, 84])])
        print(band, lo, hi, res[(lo, hi)])
    out[band] = res
pickle.dump(out, open('footprint.pkl', 'wb'))
