"""main2kfh vs main2kf: per-frame satstar flux ratio and binned satstar dm by detector."""
import glob
import re
import sys
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an

Q = an.Q
an.PATH['main2kfh'] = (f'{Q}/tree_main2kfh/{an.M8}', f'{an.D}/matched_Q_main2kfh.fits')
an.ZPWIN.update(an.zp_windows())
ARMS = ['main2', 'main2kf', 'main2kfh']
A = {n: an.Arm(n) for n in ARMS}
fig, ax = plt.subplots(1, 5, figsize=(24, 4.6))
L = []
a = ax[0]
for b, ls in (('150W', '-'), ('200W', '--')):
    for d, c in (('nrcb1', 'C0'), ('nrcb3', 'C3')):
        r = []
        for f in sorted(glob.glob(f'{Q}/tree_main2kfh/F{b}/pipeline/*_{d}_*_m7_satstar_catalog.fits')):
            B = Table.read(f)
            K = Table.read(f.replace('tree_main2kfh', 'tree_main2kf'))
            assert len(B) == len(K)
            r.append(np.asarray(B['flux_fit'], float) / np.asarray(K['flux_fit'], float))
        r = np.concatenate(r)
        r = r[np.isfinite(r)]
        dm = -2.5 * np.log10(r)
        L.append(f'F{b} {d}: per-frame rows {len(r)}, changed (|dm|>1e-4) {(np.abs(dm) > 1e-4).sum()}, |dm|>0.005 {(np.abs(dm) > 0.005).sum()}, '
                 f'p1/p50/p99 dm {np.percentile(dm, 1):+.4f}/{np.percentile(dm, 50):+.4f}/{np.percentile(dm, 99):+.4f}')
        a.hist(np.clip(dm, -0.02, 0.02), bins=np.linspace(-0.02, 0.02, 81), histtype='step', color=c, ls=ls, label=f'F{b} {d}', log=True)
a.set_xlabel('per-frame satstar dm, main2kfh - main2kf (mag)')
a.set_ylabel('rows')
a.set_title('(a) m7 satstar flux_fit, every per-frame row', fontsize=9)
a.legend(fontsize=7)
BINS = {'150W': np.arange(13, 19.01, 1.0), '200W': np.arange(12, 18.01, 1.0)}
for k, (b, d) in enumerate((('150W', 'nrcb1'), ('150W', 'nrcb3'), ('200W', 'nrcb1'), ('200W', 'nrcb3'))):
    a = ax[k + 1]
    fr = sorted(glob.glob(f'{Q}/tree_main2kfh/F{b}/pipeline/*_{d}_*_m7_satstar_catalog.fits'))
    sk = SkyCoord(np.concatenate([Table.read(f)['skycoord_fit'].ra.deg for f in fr]) * u.deg,
                  np.concatenate([Table.read(f)['skycoord_fit'].dec.deg for f in fr]) * u.deg)
    sk = sk[np.isfinite(sk.ra.deg)]
    a0 = A['main2kfh']
    sel = a0.matched & a0.rep[b] & np.isfinite(a0.ref[b])
    for n in ARMS:
        sel &= A[n].matched & np.isfinite(A[n].dm(b))
    _, dist, _ = a0.sky[a0.idx[sel]].match_to_catalog_sky(sk)
    on = np.zeros(len(sel), bool)
    on[np.where(sel)[0][dist < 0.1 * u.arcsec]] = True
    ref = a0.ref[b]
    e = BINS[b]
    xc = 0.5 * (e[1:] + e[:-1])
    for n, c, ls, mk in (('main2', 'k', '-', 'o'), ('main2kf', 'C0', '-', 's'), ('main2kfh', 'C3', '--', 'x')):
        dmv = A[n].dm(b)
        y = [np.median(dmv[on & (ref >= lo) & (ref < hi)]) if (on & (ref >= lo) & (ref < hi)).sum() >= 5 else np.nan for lo, hi in zip(e[:-1], e[1:])]
        a.plot(xc, y, color=c, ls=ls, marker=mk, ms=5, label=f'{n} (all {np.median(dmv[on]):+.3f})')
    a.axhline(0, color='gray', lw=0.7)
    a.set_ylim(-0.15, 0.08)
    a.set_xlabel(f'dolphot F{b} (mag)')
    a.set_ylabel('median dm, satstar rows')
    a.set_title(f'({"bcde"[k]}) F{b} {d}: N = {on.sum()}', fontsize=9)
    a.legend(fontsize=7)
fig.tight_layout()
fig.savefig(f'{Q}/kfhdet/kfh_ab.png', dpi=90)
open(f'{Q}/kfhdet/fig_kfh.txt', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
