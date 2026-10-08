"""Mosaic cutouts of dolphot stars matched in arm A but not in arm B (from lost.py).
Left panel: the band of the A row's reference position; right panel: F200W.
usage: lostcut_ab.py A B [nstars]"""
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
import astropy.units as u

ROOT = '/orange/adamginsburg/jwst/wd2'
D = f'{ROOT}/dolphot_benchmark'
KL = f'{D}/Q_integ/kf_lost'
M8 = 'catalogs/basic_merged_indivexp_photometry_tables_merged_resbgsub_m8_dedup.fits'
A, B = sys.argv[1:3]
NSTAR = int(sys.argv[3]) if len(sys.argv) > 3 else 12
OUT = f'{KL}/fig/lostcut_{A}_{B}.png'


def fl(c):
    return np.array(c.filled(np.nan), float) if hasattr(c, 'filled') else np.array(c, float)


ma = Table.read(f'{D}/matched_Q_{A}.fits')
ca, cb = Table.read(f'{D}/Q_integ/tree_{A}/{M8}'), Table.read(f'{D}/Q_integ/tree_{B}/{M8}')
lost = Table.read(f'{KL}/lost_{A}_{B}.ecsv')
rbands = [c[4:] for c in ma.colnames if c.startswith('ref_')]
ref = SkyCoord(ma['RA'] * u.deg, ma['DEC'] * u.deg)
sa, sb = SkyCoord(ca['skycoord_ref']), SkyCoord(cb['skycoord_ref'])
ok = np.asarray(ma['matched'])
ia = np.asarray(ma['our_idx'])
dra = np.median((sa[ia[ok]].ra - ref[ok].ra).to(u.mas) * np.cos(ref[ok].dec))
ddec = np.median((sa[ia[ok]].dec - ref[ok].dec).to(u.mas))
refs = ref.spherical_offsets_by(dra, ddec)
nb = np.zeros(len(ma), int)
for b in rbands:
    r = fl(ma[f'ref_{b}'])
    nb += (np.isfinite(r) & (r < 90)).astype(int)


def satmask(cat):
    s = np.zeros(len(cat), bool)
    for c in cat.colnames:
        if c.startswith('replaced_saturated_'):
            s |= np.asarray(cat[c]).astype(bool)
    return s


satB = satmask(cb)
cache = {}


def mosaic(band):
    if band not in cache:
        f = glob.glob(f'{ROOT}/{band.upper()}/pipeline/jw03523-o005_t001_nircam_clear-{band}-merged_i2d.fits')[0]
        h = fits.open(f, memmap=True)
        cache[band] = (h['SCI'].data, WCS(h['SCI'].header))
    return cache[band]


# sample across distance to the saturated row
lost.sort('dsatB')
pick = np.unique(np.linspace(0, len(lost) - 1, NSTAR).round().astype(int))
rows = lost[pick]
fig, axs = plt.subplots(len(rows), 2, figsize=(7.6, 3.7 * len(rows)))
axs = np.atleast_2d(axs)
for k, r in enumerate(rows):
    i, arow = int(r['i']), int(r['Arow'])
    star = refs[i]
    fband = ca['skycoord_ref_filtername'][arow]
    fband = (fband.decode() if isinstance(fband, bytes) else str(fband)).strip()
    half = float(np.clip(r['dsatB'] + 0.7, 1.0, 3.0)) * u.arcsec
    dol = [f'{b}={fl(ma["ref_" + b])[i]:.2f}' for b in rbands
           if np.isfinite(fl(ma['ref_' + b])[i]) and fl(ma['ref_' + b])[i] < 90]
    for j, band in enumerate([fband, 'f200w']):
        ax = axs[k, j]
        data, w = mosaic(band)
        cut = Cutout2D(data, star, 2 * half, wcs=w, mode='partial', fill_value=np.nan)
        d = cut.data
        lo, hi = np.nanpercentile(d, [5, 99.7])
        ax.imshow(np.arcsinh((d - lo) / max((hi - lo) / 200, 1e-3)), origin='lower', cmap='gray_r',
                  interpolation='nearest')

        def plot(sc, sel, *args, **kw):
            near = sel & (sc.separation(star) < half * 1.42)
            if near.any():
                x, y = cut.wcs.world_to_pixel(sc[near])
                ax.plot(x, y, *args, **kw)
        allt = np.ones(len(refs), bool)
        plot(refs, allt, '.', color='red', ms=5)
        mag_a, mag_b = fl(ca[f'mag_vega_{band}']), fl(cb[f'mag_vega_{band}'])
        plot(sa, np.ones(len(ca), bool), '+', color='lime', ms=11, mew=1.4)
        plot(sb, np.ones(len(cb), bool), 'x', color='cyan', ms=9, mew=1.4)
        plot(sb, satB, 'D', mfc='none', mec='orange', ms=16, mew=1.4)
        x0, y0 = cut.wcs.world_to_pixel(star)
        ax.plot(x0, y0, 'o', mfc='none', mec='red', ms=16, mew=1.6)
        xa, ya = cut.wcs.world_to_pixel(sa[arow])
        ax.plot(xa, ya, 's', mfc='none', mec='lime', ms=20, mew=1.4)
        ax.set_xlim(-0.5, d.shape[1] - 0.5); ax.set_ylim(-0.5, d.shape[0] - 0.5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'{band.upper()}  A row {mag_a[arow]:.2f}', fontsize=8)
    nbA = int(np.sum([np.isfinite(fl(ca[c])[arow]) for c in ca.colnames if c.startswith('mag_vega_')]))
    axs[k, 0].set_ylabel(f'ref #{i}  dsat={float(r["dsatB"]):.2f}"\n'
                         f'dolphot {nb[i]} bands; A row {nbA} bands\n'
                         + '\n'.join(dol[:4]), fontsize=7.5)
handles = [plt.Line2D([], [], marker='o', mfc='none', mec='red', ls='', label='lost dolphot star'),
           plt.Line2D([], [], marker='.', color='red', ls='', label='other dolphot'),
           plt.Line2D([], [], marker='s', mfc='none', mec='lime', ls='', label=f'{A} matched row'),
           plt.Line2D([], [], marker='+', color='lime', ls='', label=f'{A} rows'),
           plt.Line2D([], [], marker='x', color='cyan', ls='', label=f'{B} rows'),
           plt.Line2D([], [], marker='D', mfc='none', mec='orange', ls='', label=f'{B} saturated')]
fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=8)
fig.suptitle(f'dolphot stars matched in {A} but not {B} (dolphot shifted {dra.value:.1f},{ddec.value:.1f} mas)',
             fontsize=9)
fig.tight_layout(rect=(0, 0.03, 1, 0.985))
fig.savefig(OUT, dpi=100)
print('wrote', OUT, 'rows', list(rows['i']))
