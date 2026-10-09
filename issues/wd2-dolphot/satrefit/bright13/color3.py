import numpy as np, pickle
from astropy.table import Table
s250 = Table.read('allstar_250M.fits'); s300 = Table.read('allstar_300M.fits')
Q='/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit/out7/'
p250 = pickle.load(open(Q+'score7_250M.pkl','rb')); p300 = pickle.load(open(Q+'score7_300M.pkl','rb'))
d250 = dict(zip(s250['mi'], range(len(s250)))); d300 = dict(zip(s300['mi'], range(len(s300))))
common = sorted(set(d250) & set(d300))
print('stars with satstar refit in both F250M and F300M:', len(common))
i250 = np.array([d250[k] for k in common]); i300 = np.array([d300[k] for k in common])
r250 = np.array(s250['ref'])[i250]; r300 = np.array(s300['ref'])[i300]
col_dol = r250 - r300
def med(x): return np.median(x)
print('\nref(F250M) bin | N | dolphot F250M-F300M | ours(H+cap) F250M-F300M | ours(uncapped) | ours(final) | dm250-dm300 (H+cap) | (uncapped)')
for lo, hi in [(12.3, 13), (13, 13.5), (13.5, 14), (14, 15), (15, 16), (16, 17)]:
    s = (r250 >= lo) & (r250 < hi)
    if s.sum() < 4: continue
    out = [f'{lo}-{hi}', s.sum(), f'{med(col_dol[s]):+.3f}']
    dd = {}
    for k in ('H+cap', 'uncapped', 'final'):
        a = np.array(s250['dm_' + k])[i250][s]; b = np.array(s300['dm_' + k])[i300][s]
        dd[k] = med(a - b)
        out.append(f'{med(col_dol[s] + a - b):+.3f}')
    out += [f'{dd["H+cap"]:+.3f}', f'{dd["uncapped"]:+.3f}']
    print(' | '.join(str(x) for x in out))
# dolphot colour vs mag trend for ALL dolphot stars with both bands (not only satstar-refit)
e = Table.read('/orange/adamginsburg/jwst/wd2/catalogs/wd2_nircam_wf_mf_nf.ecsv')
a = np.asarray(e['MAG250M'].filled(np.nan), float); b = np.asarray(e['MAG300M'].filled(np.nan), float)
print('\nall dolphot stars with F250M and F300M: F250M bin | N | median F250M-F300M | MAD')
for lo, hi in [(12.3, 13), (13, 13.5), (13.5, 14), (14, 14.5), (14.5, 15), (15, 16), (16, 17)]:
    s = (a >= lo) & (a < hi) & np.isfinite(b) & (np.abs(b) < 50)
    if s.sum() >= 4:
        c = (a - b)[s]; print(f'{lo}-{hi} | {s.sum()} | {np.median(c):+.3f} | {1.4826*np.median(abs(c-np.median(c))):.3f}')
