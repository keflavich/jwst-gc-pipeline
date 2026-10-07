"""Dolphot reference stars around saturated stars, stacked in (sep/lambda, PA) per band, split by match status
in arms A and B.  Bead zone: sep/lambda within 0.035 "/um of n*S (n=1..4, S fitted spacing) and PA mod 60 within
6 deg of the spike PA; control zone: same radii, PA rotated by 30 deg.  Excess of dolphot stars in the bead zone
over the control zone counts spike beads in the reference catalog.
usage: beadstack.py A B [band ...]   -> fig/beadstack_A_B.png, beadstack_A_B.txt"""
import sys
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark'
Q = f'{D}/Q_integ'
M8 = 'catalogs/basic_merged_indivexp_photometry_tables_merged_resbgsub_m8_dedup.fits'
A, B = sys.argv[1:3]
BANDS = sys.argv[3:] or ['164N', '250M', '150W', '162M', '182M', '187N', '200W', '212N']
S, PA0, RMAX = 0.3047, 20.6, 4.0
ma, mb = Table.read(f'{D}/matched_Q_{A}.fits'), Table.read(f'{D}/matched_Q_{B}.fits')
assert np.allclose(ma['RA'], mb['RA']) and np.allclose(ma['DEC'], mb['DEC'])
ca = Table.read(f'{Q}/tree_{A}/{M8}')
ref = SkyCoord(ma['RA'], ma['DEC'], unit='deg')


def fl(x):
    return np.ma.filled(x, np.nan).astype(float) if np.ma.isMaskedArray(x) else np.asarray(x, float)


def zone(x, pa, off):
    n = np.round(x / S)
    return (n >= 1) & (n <= 4) & (np.abs(x - n * S) < 0.035) & (np.abs(((pa - PA0 - off) + 30) % 60 - 30) < 6)


lines = []
fig, axes = plt.subplots(2, (len(BANDS) + 1) // 2, figsize=(4.2 * ((len(BANDS) + 1) // 2), 8.6))
for ax, b in zip(axes.flat, BANDS):
    col = f'replaced_saturated_f{b.lower()}'
    sat = SkyCoord(ca['skycoord_ref'][np.ma.filled(ca[col], 0).astype(bool)])
    has = np.isfinite(fl(ma[f'ref_{b}']))
    ia, js, sep, _ = ref[has].search_around_sky(sat, RMAX * u.arcsec)
    keep = sep.arcsec > 0.15
    ia, js, sep = ia[keep], js[keep], sep[keep]
    idx = np.where(has)[0][js]
    lam = int(b[:3]) / 100
    x = sep.arcsec / lam
    pa = sat[ia].position_angle(ref[idx]).deg % 360
    inA = np.asarray(ma['matched'])[idx] & np.isfinite(fl(ma[f'our_{b}']))[idx]
    inB = np.asarray(mb['matched'])[idx] & np.isfinite(fl(mb[f'our_{b}']))[idx]
    cat = {'both': inA & inB, f'{A} only': inA & ~inB, f'{B} only': ~inA & inB, 'neither': ~inA & ~inB}
    bz, cz = zone(x, pa, 0), zone(x, pa, 30)
    lines.append(f'F{b}: sat {len(sat)}, dolphot within {RMAX}" {len(idx)}; bead zone {bz.sum()} vs control {cz.sum()}; '
                 + ', '.join(f'{k} bead/control {(v & bz).sum()}/{(v & cz).sum()}' for k, v in cat.items()))
    spk = np.round(((pa - PA0) % 360) / 60).astype(int) % 6
    lines.append('   bead-zone stars by spike (PA0+60k, k=0..5): ' + str(np.bincount(spk[bz], minlength=6))
                 + f'  {A}-only by spike: ' + str(np.bincount(spk[bz & cat[f"{A} only"]], minlength=6)))
    th = np.deg2rad(pa)
    for (k, v), c, s_ in zip(cat.items(), ['0.6', 'tab:red', 'tab:blue', 'k'], [3, 10, 10, 2]):
        ax.scatter(x[v] * np.sin(th[v]), x[v] * np.cos(th[v]), s=s_, c=c, label=f'{k} ({v.sum()})', lw=0)
    for k in range(6):
        t = np.deg2rad(PA0 + 60 * k)
        ax.plot([0, 1.35 * np.sin(t)], [0, 1.35 * np.cos(t)], color='orange', lw=0.5, alpha=0.6)
    for n in range(1, 5):
        ax.add_patch(plt.Circle((0, 0), n * S, fill=False, ls=':', lw=0.5, ec='orange'))
    lim = RMAX / lam
    lim = min(lim, 1.4)
    ax.set_xlim(lim, -lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
    ax.set_title(f'F{b}: {len(sat)} saturated stars', fontsize=9)
    ax.set_xlabel('E offset / lambda ["/um]', fontsize=8); ax.set_ylabel('N offset / lambda ["/um]', fontsize=8)
    ax.legend(fontsize=6, loc='lower left', markerscale=1.5)
for ax in list(axes.flat)[len(BANDS):]:
    ax.axis('off')
fig.suptitle(f'Dolphot stars around saturated stars (A={A}, B={B}); orange: spike directions and n x {S} "/um rings', fontsize=10)
fig.tight_layout()
fig.savefig(f'{Q}/kf_lost/fig/beadstack_{A}_{B}.png', dpi=100)
open(f'{Q}/kf_lost/beadstack_{A}_{B}.txt', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
