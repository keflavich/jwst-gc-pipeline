"""Cutouts of the worst faint SW movers (largest increase in abs(dm) with B).  usage: fig_movers.py A B [N=8]
Columns: m7 input (A residual + A model), A m7 satstar residual, B residual, B model - A model."""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import AsinhNorm, SymLogNorm
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from common import *

A_, B_ = sys.argv[1:3]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 8
TA, TB = f'{Q}/tree_{A_}', f'{Q}/tree_{B_}'
S = Table.read(f'{FS}/stars_{A_}_{B_}.ecsv'); F = Table.read(f'{FS}/frames_{A_}_{B_}.ecsv')
S['worse'] = np.abs(S['dmB']) - np.abs(S['dmA'])
m = S[S['moved']]
m = m[np.argsort(-m['worse'])]
# two worst per band first (band coverage), then fill with the next worst overall
pick, per = [], {}
for r in m:
    if per.get(r['band'], 0) < N // 4 and (F['band'] == r['band']).any() and ((F['band'] == r['band']) & (F['i'] == r['i'])).any():
        pick.append(r); per[r['band']] = per.get(r['band'], 0) + 1
pick = sorted(pick, key=lambda r: -r['worse'])[:N]
fig, axs = plt.subplots(len(pick), 4, figsize=(15, 3.9 * len(pick)))
titles = ['m7 input (A resid + A model)', 'A m7 satstar residual', 'B m7 satstar residual', 'B model - A model']
for irow, r in enumerate(pick):
    band = r['band']; star = SkyCoord(r['RA'] * u.deg, r['DEC'] * u.deg)
    g = F[(F['band'] == band) & (F['i'] == r['i']) & (F['fA'] > 0)]
    g = g[np.argsort(-np.abs(g['Dpsf'] / g['fA']))][0]
    det, exp = str(g['det']), int(g['exp'])
    fa, fb = frame_files(TA, band), frame_files(TB, band)
    ta, tb = Table.read(fa[(det, exp)]), Table.read(fb[(det, exp)])
    ca, cb = sky_cols(ta, 'skycoord_centroid'), sky_cols(tb, 'skycoord_centroid')
    ca, cb = ca[np.isfinite(ca.ra.deg)], cb[np.isfinite(cb.ra.deg)]
    stems = {'A': pipe_stem(TA, band, det, exp), 'B': pipe_stem(TB, band, det, exp)}
    data = {}
    for arm in 'AB':
        s = stems[arm]
        data[arm + 'res'] = fits.getdata(s + '_residual.fits'); data[arm + 'mod'] = fits.getdata(s + '_model.fits')
        t = Table.read(s + '_catalog.fits')
        data[arm + 'cat'] = t
    wcs = WCS(fits.getheader(stems['A'] + '_residual.fits'))
    inp = data['Ares'] + data['Amod']; diff = data['Bmod'] - data['Amod']
    x0, y0 = [float(v) for v in wcs.world_to_pixel(star)]
    def pix(c):
        x, y = wcs.world_to_pixel(c); return np.atleast_1d(x), np.atleast_1d(y)
    sat = {}
    for k in 'AB':
        t = data[k + 'cat']
        c = sky_cols(t, 'skycoord_fit') if len(t) else None
        sat[k] = pix(c[np.isfinite(c.ra.deg)]) if c is not None else (np.array([]), np.array([]))
    pa, pb = pix(ca), pix(cb)
    pscale = np.sqrt(abs(np.linalg.det(wcs.pixel_scale_matrix))) * 3600
    allx = np.concatenate([sat['A'][0], sat['B'][0]]); ally = np.concatenate([sat['A'][1], sat['B'][1]])
    dall = np.hypot(allx - x0, ally - y0); k0 = int(np.argmin(dall))
    sx, sy = allx[k0], ally[k0]; dist = dall[k0] * pscale
    dB = np.hypot(sat['B'][0] - sx, sat['B'][1] - sy); dA = np.hypot(sat['A'][0] - sx, sat['A'][1] - sy)
    inA = len(dA) > 0 and dA.min() <= 0.5; inB = len(dB) > 0 and dB.min() <= 0.5
    if inA and inB:
        ratio_s = f'{float(data["Bcat"]["flux_fit"][int(np.argmin(dB))]) / float(data["Acat"]["flux_fit"][int(np.argmin(dA))]):.2f}'
    else:
        ratio_s = 'sat only in B' if inB else 'sat only in A'
    cx, cy = 0.5 * (x0 + sx), 0.5 * (y0 + sy)
    half = int(min(max(np.hypot(x0 - sx, y0 - sy) / 2 + 14, 20), 60))
    cxi, cyi = int(round(cx)), int(round(cy))
    sl = (slice(max(cyi - half, 0), cyi + half + 1), slice(max(cxi - half, 0), cxi + half + 1))
    xo, yo = sl[1].start, sl[0].start
    ext = (xo - 0.5, sl[1].stop - 0.5, yo - 0.5, sl[0].stop - 0.5)
    satc = wcs.pixel_to_world(sx, sy)
    spikes = []
    for kk in range(6):
        xs, ys = wcs.world_to_pixel(satc.directional_offset_by((20 + 60 * kk) * u.deg, 3 * u.arcsec)); spikes.append((float(xs), float(ys)))
    vmax = np.nanpercentile(inp[sl], 99.5); vmin = np.nanpercentile(inp[sl], 1)
    rmax = np.nanpercentile(data['Ares'][sl], 99.5)
    dlim = np.nanmax(np.abs(diff[sl])); dlim = dlim if dlim > 0 else 1
    panels = [(inp, AsinhNorm(linear_width=max(abs(vmax) * 0.02, 1e-3), vmin=vmin, vmax=vmax), 'gray_r'),
              (data['Ares'], AsinhNorm(linear_width=max(abs(rmax) * 0.02, 1e-3), vmin=vmin, vmax=rmax), 'gray_r'),
              (data['Bres'], AsinhNorm(linear_width=max(abs(rmax) * 0.02, 1e-3), vmin=vmin, vmax=rmax), 'gray_r'),
              (diff, SymLogNorm(linthresh=dlim * 0.01, vmin=-dlim, vmax=dlim), 'RdBu_r')]
    for k, (arr, norm, cm) in enumerate(panels):
        ax = axs[irow, k]
        ax.imshow(arr[sl], origin='lower', extent=ext, norm=norm, cmap=cm, interpolation='nearest')
        ax.plot(x0, y0, 'o', mfc='none', mec='red', ms=14, mew=1.5)
        ax.plot(*pa, '+', color='lime', ms=10, mew=1.5); ax.plot(*pb, 'x', color='cyan', ms=9, mew=1.5)
        ax.plot(*sat['A'], 's', mfc='none', mec='magenta', ms=15, mew=1.3); ax.plot(*sat['B'], 'D', mfc='none', mec='orange', ms=12, mew=1.3)
        for xs, ys in spikes:
            ax.plot([sx, xs], [sy, ys], '-', color='yellow', lw=0.6, alpha=0.45)
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_xticks([]); ax.set_yticks([])
        if irow == 0:
            ax.set_title(titles[k], fontsize=10)
    axs[irow, 0].set_ylabel(f'F{band} {det} exp{exp}  ref #{int(r["i"])}\ndolphot {r["ref"]:.1f}  A dm {r["dmA"]:+.2f}  B dm {r["dmB"]:+.2f}\n'
                            f'nearest satstar {dist:.2f}"  (B/A flux_fit {ratio_s})\nframe nsat A/B {int(g["nsatA"])}/{int(g["nsatB"])}  Dpsf/fA {g["Dpsf"] / g["fA"]:+.2f}', fontsize=8.5)
    print(f'F{band} {det} exp{exp} #{int(r["i"])} ref {r["ref"]:.1f} dmA {r["dmA"]:+.2f} dmB {r["dmB"]:+.2f} dsat {dist:.2f} sat {ratio_s} nsat {int(g["nsatA"])}/{int(g["nsatB"])} Dpsf/fA {g["Dpsf"] / g["fA"]:+.2f}', flush=True)
handles = [plt.Line2D([], [], marker='o', mfc='none', mec='red', ls='', label='dolphot'),
           plt.Line2D([], [], marker='+', color='lime', ls='', label='A daophot'),
           plt.Line2D([], [], marker='x', color='cyan', ls='', label='B daophot'),
           plt.Line2D([], [], marker='s', mfc='none', mec='magenta', ls='', label='A satstar'),
           plt.Line2D([], [], marker='D', mfc='none', mec='orange', ls='', label='B satstar')]
fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=10)
fig.suptitle(f'Worst faint SW movers ({B_} vs {A_}); yellow lines = 6 nominal spike directions through nearest satstar (PA offset 20 deg, not fitted)', fontsize=11)
fig.tight_layout(rect=(0, 0.02, 1, 0.98))
fig.savefig(f'{FS}/fig/faint_movers_{A_}_{B_}.png', dpi=100)
