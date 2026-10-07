"""Excess-density maps of dolphot stars around saturated stars, per band: 2D histogram of offsets (in "/um and in
arcsec) divided by its azimuthal mean at each radius.  Overlays: stars matched in A only (red), in neither arm (black
contours of their excess).  Also counts the excess in a 0.08 "/um circle at the strongest off-centre peak.
usage: beadmap.py A B [band ...]   -> fig/beadmap_A_B.png, beadmap_A_B.txt"""
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
BANDS = sys.argv[3:] or ['164N', '250M', '150W', '162M', '182M', '200W']
ma, mb = Table.read(f'{D}/matched_Q_{A}.fits'), Table.read(f'{D}/matched_Q_{B}.fits')
ca = Table.read(f'{Q}/tree_{A}/{M8}')
ref = SkyCoord(ma['RA'], ma['DEC'], unit='deg')
NB, LIM = 71, 1.4


def fl(x):
    return np.ma.filled(x, np.nan).astype(float) if np.ma.isMaskedArray(x) else np.asarray(x, float)


def excess(dx, dy, lim):
    e = np.linspace(-lim, lim, NB + 1)
    h, _, _ = np.histogram2d(dx, dy, bins=[e, e])
    c = 0.5 * (e[1:] + e[:-1])
    R = np.hypot(*np.meshgrid(c, c, indexing='ij'))
    rb = np.digitize(R, np.linspace(0, lim * 1.5, 40))
    mean = np.zeros_like(h)
    for k in np.unique(rb):
        mean[rb == k] = h[rb == k].mean()
    return h, np.where(mean > 0, h / np.where(mean > 0, mean, 1), np.nan), e


lines = []
fig, axes = plt.subplots(2, len(BANDS), figsize=(3.6 * len(BANDS), 7.4))
for j, b in enumerate(BANDS):
    col = f'replaced_saturated_f{b.lower()}'
    sat = SkyCoord(ca['skycoord_ref'][np.ma.filled(ca[col], 0).astype(bool)])
    has = np.isfinite(fl(ma[f'ref_{b}']))
    lam = int(b[:3]) / 100
    ia, js, sep, _ = ref[has].search_around_sky(sat, LIM * lam * 1.5 * u.arcsec)
    idx = np.where(has)[0][js]
    dra, ddec = sat[ia].spherical_offsets_to(ref[idx])
    dx, dy = dra.arcsec, ddec.arcsec
    inA = np.asarray(ma['matched'])[idx] & np.isfinite(fl(ma[f'our_{b}']))[idx]
    inB = np.asarray(mb['matched'])[idx] & np.isfinite(fl(mb[f'our_{b}']))[idx]
    for row, (scale, unit) in enumerate([(lam, '"/um'), (1.0, 'arcsec')]):
        ax = axes[row, j]
        lim = LIM if scale != 1.0 else LIM * 1.6
        h, ex, e = excess(dx / scale, dy / scale, lim)
        ax.imshow(ex.T, origin='lower', extent=(e[0], e[-1], e[0], e[-1]), cmap='magma', vmin=0.5, vmax=2.5)
        nn = ~inA & ~inB
        _, exn, _ = excess(dx[nn] / scale, dy[nn] / scale, lim)
        c = 0.5 * (e[1:] + e[:-1])
        ax.contour(c, c, np.nan_to_num(exn.T), levels=[3], colors='cyan', linewidths=0.6)
        ao = inA & ~inB
        ax.scatter(dx[ao] / scale, dy[ao] / scale, s=6, c='lime', lw=0)
        ax.set_xlim(lim, -lim); ax.set_ylim(-lim, lim)
        ax.set_title(f'F{b} ({len(sat)} sat), offsets in {unit}', fontsize=8)
        ax.tick_params(labelsize=7)
        if row == 0:
            # strongest off-centre excess cell beyond 0.2 "/um
            cc = np.meshgrid(c, c, indexing='ij')
            m = (np.hypot(*cc) > 0.2) & (h > 20)
            k = np.nanargmax(np.where(m, exn, -1))
            px, py = cc[0].flat[k], cc[1].flat[k]
            r = np.hypot(dx / scale - px, dy / scale - py) < 0.08
            rr = np.hypot(dx / scale + px, dy / scale + py) < 0.08  # point-mirrored control
            lines.append(f'F{b}: peak neither-excess at ({px:+.2f}, {py:+.2f}) "/um, r={np.hypot(px, py):.2f}, PA={np.degrees(np.arctan2(px, py)) % 360:.0f}: '
                         f'dolphot in 0.08 circle {r.sum()} (mirror {rr.sum()}); neither {(r & nn).sum()} (mirror {(rr & nn).sum()}); '
                         f'A-only {(r & ao).sum()}, both {(r & inA & inB).sum()}')
fig.suptitle(f'Dolphot star density around saturated stars / azimuthal mean (top: offsets / lambda; bottom: arcsec). '
             f'lime = matched in {A} only; cyan contour = 3x excess of stars matched in neither arm', fontsize=9)
fig.tight_layout()
fig.savefig(f'{Q}/kf_lost/fig/beadmap_{A}_{B}.png', dpi=100)
open(f'{Q}/kf_lost/beadmap_{A}_{B}.txt', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
