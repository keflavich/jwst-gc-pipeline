"""usage: python score.py BAND  -> out/score_<band>.npz and out/tables_<band>.md"""
import glob, sys, pickle
import numpy as np
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
import astropy.units as u

Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
band = sys.argv[1]
BINS = {'150W': [(13, 14), (14, 15), (15, 16), (16, 17), (17, 18), (18, 19)],
        '200W': [(12, 13), (13, 14), (14, 15), (15, 16), (16, 17), (17, 18)],
        '250M': [(10, 13), (13, 14), (14, 15), (15, 16), (16, 18)],
        '300M': [(10, 13), (13, 14), (14, 15), (15, 16), (16, 18)]}[band]
an.ZPWIN.update(an.zp_windows())
A = an.Arm('main2')
ok = A.matched & A.rep[band] & np.isfinite(A.ref[band])
tgt = A.sky[A.idx[ok]]
ref = A.ref[band][ok]
dm_final = A.dm(band)[ok]
tabs = []
for fn in sorted(glob.glob(f'{Q}/satrefit/out/{band}_*_satrefit.fits')):
    t = Table.read(fn)
    t['frame'] = fn.split('/')[-1]
    tabs.append(t)
T = vstack(tabs, metadata_conflicts='silent')
good = np.asarray(T['label']) > 0
sk = SkyCoord(np.asarray(T['ra']) * u.deg, np.asarray(T['dec']) * u.deg)
i, j, _, _ = sk.search_around_sky(tgt, 0.1 * u.arcsec)
vnames = [c[2:] for c in T.colnames if c.startswith('a_') and c not in ('a_cat', 'a_raw')]
a = {v: np.asarray(T['a_' + v], float) for v in vnames}
acat, araw = np.asarray(T['a_cat'], float), np.asarray(T['a_raw'], float)
n = ok.sum()


def med_per_star(x):
    out = np.full(n, np.nan)
    m = good[j] & np.isfinite(x[j])
    ii, xx = i[m], x[j][m]
    order = np.argsort(ii, kind='stable')
    ii, xx = ii[order], xx[order]
    u_, st = np.unique(ii, return_index=True)
    for k, s, e in zip(u_, st, list(st[1:]) + [len(ii)]):
        out[k] = np.median(xx[s:e])
    return out


s_cap = med_per_star(-2.5 * np.log10(acat / araw))
dm_unc = dm_final + s_cap
shift = {v: med_per_star(-2.5 * np.log10(a[v] / a['base'])) for v in vnames}
dm = {'final': dm_final, 'uncapped': dm_unc}
for v in vnames:
    dm[v] = dm_unc + shift[v]
have = np.isfinite(s_cap)
for v in vnames:
    have &= np.isfinite(shift[v])
have_base = np.isfinite(s_cap)
# unsaturated reference
zpwin_ok = A.matched & ~A.rep[band] & np.isfinite(A.dm(band)) & np.isfinite(A.ref[band])
fb_hi = BINS[-1][1]
unsat = zpwin_ok & (A.ref[band] >= fb_hi) & (A.ref[band] < fb_hi + 1)
unsat_bright = zpwin_ok & (A.ref[band] >= BINS[0][0] - 1) & (A.ref[band] < BINS[0][0])
pickle.dump(dict(ref=ref, dm=dm, have=have, have_base=have_base, shift=shift, vnames=vnames,
                 unsat_dm=A.dm(band)[unsat], unsat_ref=A.ref[band][unsat], bins=BINS), open(f'{Q}/satrefit/out/score_{band}.pkl', 'wb'))
L = []
L.append(f'#### F{band}: {n} satstar-replaced matched stars; {have_base.sum()} with rows in the refit frames; {have.sum()} with every variant finite\n')
hdr = '| variant | ' + ' | '.join(f'{lo}-{hi}' for lo, hi in BINS) + ' | all (med / MAD) |'
L.append(hdr)
L.append('|---|' + '---|' * (len(BINS) + 1))
order = ['final', 'uncapped'] + [v for v in vnames if v != 'base'] 
ncount = [(have & (ref >= lo) & (ref < hi)).sum() for lo, hi in BINS]
for v in order:
    d = dm[v]
    cells = []
    for lo, hi in BINS:
        s = have & (ref >= lo) & (ref < hi) & np.isfinite(d)
        cells.append(f'{np.median(d[s]):+.3f} ({an.mad(d[s]):.3f})' if s.sum() >= 3 else '-')
    s = have & np.isfinite(d)
    L.append(f'| {v} | ' + ' | '.join(cells) + f' | {np.median(d[s]):+.3f} / {an.mad(d[s]):.3f} |')
L.append('')
L.append('N per bin: ' + ', '.join(map(str, ncount)))
L.append('')
# trend: slope of dm vs mag (median per bin difference between faintest and brightest bins)
L.append('Unsaturated matched stars (not replaced), ' + f'{fb_hi}-{fb_hi+1} mag: median dm {np.nanmedian(A.dm(band)[unsat]):+.3f}, MAD {an.mad(A.dm(band)[unsat]):.3f}, N={unsat.sum()}')
L.append('Unsaturated matched stars, ' + f'{BINS[0][0]-1}-{BINS[0][0]} mag: median dm {np.nanmedian(A.dm(band)[unsat_bright]):+.3f}, N={unsat_bright.sum()}')
open(f'{Q}/satrefit/out/tables_{band}.md', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
