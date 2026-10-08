"""Diagnose the module-B floor: influence of single stars, pm/parallax/quality subsets, no-pm propagation.
Uses gaia_fit_<TAG>.pkl from datascale_gaia2.py.  Usage: python diag_floor.py TAG band [band ...]"""
import sys, pickle
import numpy as np
TAG = sys.argv[1]
BANDS_ = sys.argv[2:] or ['f212n', 'f164n']
sys.argv = [sys.argv[0]]
import datascale_gaia2 as D
P = pickle.load(open(f'gaia_fit_{TAG}.pkl', 'rb'))
data, G = P['data'], P['G']
DT = 2024.55 - 2016.0   # epoch gap used for the no-pm test (frames are within ~0.01 yr of each other)
gmag, pmra, pmde = (np.asarray(G[c], float) for c in ('phot_g_mean_mag', 'pmra', 'pmdec'))
epm = np.hypot(np.asarray(G['pmra_error'], float), np.asarray(G['pmdec_error'], float))
plx = np.asarray(G['parallax'], float)
ruwe = np.asarray(G['ruwe'], float)


def mfit(d, sel=None, nopm=False):
    sel = np.ones(len(d['x']), bool) if sel is None else sel
    if sel.sum() < 15:
        return np.nan, 0
    rx, ry = d['rx'].copy(), d['ry'].copy()
    if nopm:
        rx = rx + pmra[d['gid']] * DT
        ry = ry + pmde[d['gid']] * DT
    fr = d['fr'][sel]
    _u, fr = np.unique(fr, return_inverse=True)
    arr = (fr, d['gid'][sel], d['x'][sel], d['y'][sel], rx[sel], ry[sel])
    mask, s = D.clipped(arr)
    J = D.fit(*arr, np.ones(len(mask)), mask)[0]
    return D.inv(J)[2], len(np.unique(arr[1][mask]))


out = []
for band in BANDS_:
    for det in ['nrca2', 'nrca3', 'nrcb1', 'nrcb2', 'nrcb3', 'nrcb4']:
        d = data.get((band, det))
        if d is None:
            continue
        g = d['gid']
        base = mfit(d)
        # leave-one-star-out
        ug = np.unique(g)
        loo = []
        for s_ in ug:
            m, _ = mfit(d, g != s_)
            loo.append((m, s_))
        loo = np.array(loo)
        dm = np.abs(loo[:, 0] - base[0])
        top = np.argsort(-dm)[:3]
        row = dict(
            base=base, loo_range=(np.nanmin(loo[:, 0]), np.nanmax(loo[:, 0])),
            top=[(int(loo[i, 1]), round(float(gmag[int(loo[i, 1])]), 1), round(float(np.hypot(pmra[int(loo[i, 1])], pmde[int(loo[i, 1])])), 1),
                  round(float(epm[int(loo[i, 1])]), 2), round(float(loo[i, 0]), 1)) for i in top],
            pmerr_lt_0p15=mfit(d, epm[g] < 0.15), pmerr_lt_0p3=mfit(d, epm[g] < 0.3), pm_lt_4=mfit(d, np.hypot(pmra[g], pmde[g]) < 4),
            ruwe_lt_1p2=mfit(d, ruwe[g] < 1.2), G_gt_16=mfit(d, gmag[g] > 16), G_gt_18=mfit(d, gmag[g] > 18), G_lt_18=mfit(d, gmag[g] < 18),
            inner=mfit(d, (np.abs(d['x']) < 850) & (np.abs(d['y']) < 850)),
            nopm=mfit(d, nopm=True), plx_med=float(np.nanmedian(plx[ug])), plx_max=float(np.nanmax(np.abs(plx[ug]))),
            pm_med=float(np.nanmedian(np.hypot(pmra[ug], pmde[ug]))), epm_med=float(np.nanmedian(epm[ug])))
        out.append((band, det, row))
        print(f'\n{band} {det}: base m={base[0]:.1f} (N={base[1]}); leave-one-star-out m range {row["loo_range"][0]:.1f}-{row["loo_range"][1]:.1f}')
        print('  most influential (source idx, G, |pm| mas/yr, pm_err, m without it):', row['top'])
        print('  subsets m (N):  ' + '  '.join(f'{k}={row[k][0]:.1f}({row[k][1]})' for k in
              ['pmerr_lt_0p15', 'pmerr_lt_0p3', 'pm_lt_4', 'ruwe_lt_1p2', 'G_gt_16', 'G_gt_18', 'G_lt_18', 'inner', 'nopm']))
        print(f'  median pm={row["pm_med"]:.1f} mas/yr, median pm_err={row["epm_med"]:.2f} mas/yr (x{DT:.1f} yr = {row["epm_med"]*DT:.1f} mas), parallax median {row["plx_med"]:.2f} max |{row["plx_max"]:.1f}| mas')
pickle.dump(out, open(f'diag_floor_{TAG}.pkl', 'wb'))
