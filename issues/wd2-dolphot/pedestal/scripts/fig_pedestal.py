"""Frame-lineage pedestal in wd2 m7: per-band pedestal and per-frame row
growth m6 -> m7, plus F200W / F150W cutouts with m6 and m7 detections.
usage: python fig_pedestal.py"""
import glob
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

warnings.filterwarnings('ignore')
R = '/orange/adamginsburg/jwst/wd2'
D = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark'
ALL16 = ['F115W', 'F150W', 'F162M', 'F164N', 'F182M', 'F187N', 'F200W', 'F212N',
         'F250M', 'F277W', 'F300M', 'F323N', 'F335M', 'F405N', 'F410M', 'F466N']
# dolphot -> pipeline offset (pipeline - dolphot), mas (RA*cos, Dec); #1128
OFF = {'F150W': (-28.1, -12.7)}
OFF_DEFAULT = (-42.4, 5.4)


def med(a):
    a = a[np.isfinite(a)]
    return np.median(a)


rows = []
for b in ALL16:
    bl = b.lower()
    det = 'nrcalong' if int(b[1:4]) >= 250 else 'nrca1'
    r = sorted(glob.glob(f'{R}/{b}/pipeline/jw03523-o005_t001_nircam_clear-{bl}-{det}_visit001_vgroup*_exp00001_resbgsub_m6_daophot_basic_residual.fits'))[0]
    vg = r.split('vgroup')[1][:5]
    rm = fits.getdata(r, 'SCI') + fits.getdata(r.replace('_residual.fits', '_model.fits'), 'SCI')
    ped = med(fits.getdata(f'{R}/{b}/pipeline/jw03523005001_{vg}_00001_{det}_align_o005_crf.fits', 'SCI') - rm)
    n = {}
    for m in ('m6', 'm7'):
        fs = glob.glob(f'{R}/{b}/{bl}_nrc*_visit001_vgroup{vg}_exp0000?_resbgsub_{m}_daophot_basic.fits')
        n[m] = (sum(fits.getheader(f, 1)['NAXIS2'] for f in fs), len(fs))
    lin = 'destreak' if abs(ped) > 0.05 else 'align'
    rows.append((b, det, ped, n['m6'][0], n['m7'][0], n['m6'][1], n['m7'][1], lin))
with open('pedestal_table.tsv', 'w') as o:
    o.write('band\tdet\tpedestal_MJy_sr\tm6_rows\tm7_rows\tm6_frames\tm7_frames\tm6_lineage\n')
    for r_ in rows:
        o.write('\t'.join(str(x) if not isinstance(x, float) else f'{x:.4f}' for x in r_) + '\n')
print(open('pedestal_table.tsv').read())

dm = Table.read(f'{D}/matched_Q_mainfcbg.fits')
dsk = SkyCoord(np.asarray(dm['RA'], float) * u.deg, np.asarray(dm['DEC'], float) * u.deg)


def cut(ax, b, vg, cx, cy, half=50, title=''):
    bl = b.lower()
    p = f'{R}/{b}/pipeline/jw03523005001_{vg}_00001_{DET}_align_o005_crf.fits'
    img = fits.getdata(p, 'SCI')
    w = WCS(fits.getheader(p, 'SCI'))
    sl = (slice(cy - half, cy + half), slice(cx - half, cx + half))
    sub = img[sl]
    lo, hi = np.nanpercentile(sub, [5, 99.5])
    ax.imshow(np.arcsinh((sub - lo) / (0.05 * (hi - lo))), origin='lower', cmap='gray_r',
              extent=(cx - half - 0.5, cx + half - 0.5, cy - half - 0.5, cy + half - 0.5))
    for m, mk, c, s in (('m7', 'x', 'tab:red', 22), ('m6', 'o', 'tab:blue', 70)):
        t = Table.read(f'{R}/{b}/{bl}_{DET}_visit001_vgroup{vg}_exp00001_resbgsub_{m}_daophot_basic.fits')
        x, y = np.asarray(t['x_fit'], float), np.asarray(t['y_fit'], float)
        k = (abs(x - cx) < half) & (abs(y - cy) < half)
        if mk == 'o':
            ax.scatter(x[k], y[k], s=s, facecolors='none', edgecolors=c, lw=1.0, label=f'{m} rows ({k.sum()})')
        else:
            ax.scatter(x[k], y[k], s=s, c=c, marker=mk, lw=0.9, label=f'{m} rows ({k.sum()})')
    ox, oy = OFF.get(b, OFF_DEFAULT)
    dsh = SkyCoord(dsk.ra + (ox * u.mas) / np.cos(dsk.dec.rad), dsk.dec + oy * u.mas)
    xd, yd = w.world_to_pixel(dsh)
    k = (abs(xd - cx) < half) & (abs(yd - cy) < half)
    ax.scatter(xd[k], yd[k], s=90, marker='+', c='tab:green', lw=1.0, label=f'dolphot ({k.sum()})')
    ax.set_xlim(cx - half - 0.5, cx + half - 0.5)
    ax.set_ylim(cy - half - 0.5, cy + half - 0.5)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc='upper right', framealpha=0.85)
    ax.set_xticks([])
    ax.set_yticks([])


fig = plt.figure(figsize=(13, 9))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.25])
ax = fig.add_subplot(gs[0, 0])
xs = np.arange(len(rows))
peds = [r_[2] for r_ in rows]
col = ['tab:red' if r_[7] == 'destreak' else 'tab:gray' for r_ in rows]
ax.bar(xs, peds, color=col)
ax.set_xticks(xs)
ax.set_xticklabels([r_[0] for r_ in rows], rotation=60, fontsize=8)
ax.set_ylabel('m7 pedestal: median(align crf - m6 res - m6 model)\n[MJy/sr], exp00001 nrca1 / nrcalong', fontsize=8)
ax.set_title('m6 fit read destreak_o005_crf (red) or align_o005_crf (gray)', fontsize=9)
ax.set_yscale('symlog', linthresh=0.1)
ax = fig.add_subplot(gs[0, 1])
ratio = [r_[4] / r_[3] for r_ in rows]
ax.bar(xs, ratio, color=col)
ax.axhline(1, color='k', lw=0.8)
ax.set_xticks(xs)
ax.set_xticklabels([r_[0] for r_ in rows], rotation=60, fontsize=8)
ax.set_yscale('log')
ax.set_ylabel('per-frame rows m7 / m6 (all frames)', fontsize=8)
ax.set_title('production per-frame catalogs', fontsize=9)
for i, r_ in enumerate(rows):
    ax.text(i, ratio[i] * 1.08, f'{r_[4]/1e3:.0f}k', ha='center', fontsize=6)
# faint field on exp00001 nrca1 (few stars, so noise-peak rows stand out)
DET = 'nrca1'
cx, cy = 1200, 700
cut(fig.add_subplot(gs[1, 0]), 'F200W', '12101', cx, cy,
    title=f'F200W exp00001 {DET} align crf, 100 px; m6 bg from destreak frames')
cut(fig.add_subplot(gs[1, 1]), 'F150W', '10101', cx, cy,
    title=f'F150W exp00001 {DET} align crf, 100 px; m6 bg from align frames')
fig.suptitle('wd2 m7: SW m1-m6 fit destreak frames, m7 fits align frames -> pedestal -> noise-peak rows', fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig('fig_pedestal.png', dpi=110)
print('wrote fig_pedestal.png')
