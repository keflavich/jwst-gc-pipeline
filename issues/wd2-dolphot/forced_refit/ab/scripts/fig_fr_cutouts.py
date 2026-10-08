"""Cutouts for the forced-refit A/B.  Per band: 4 dolphot stars forced in both arms (frac>=0.5), dolphot 17-21 mag,
|dm_fr0|>0.3 and |dm_fr1|<0.1; 3 class-(c) rows (fr1 forced, no fr0 row, frac>=0.5, pipeline mag in the band's
typical range).  Panels per object: AB mosaic (asinh), then per-frame m7 residual of fr0 and of fr1 (same frame,
linear, +-vmax).  Markers: red circle = dolphot, cyan x = fr0 merged rows, lime + = fr1 merged rows.
usage: python fig_fr_cutouts.py"""
import sys, glob, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy.visualization import simple_norm
import astropy.units as u
sys.path.insert(0, '.')
from classify import load, dsk, A, H
HALF = 0.6  # arcsec half-width
rng = np.random.default_rng(11)
BANDS = ['187N', '200W', '277W']
NA, NC = {'187N': 4, '200W': 4, '277W': 4}, {'187N': 3, '200W': 3, '277W': 3}


def frame_files(arm, b):
    d = f'{H}/tree_{arm}/F{b}/pipeline'
    out = {}
    for c in sorted(glob.glob(f'{H}/tree_{arm}/F{b}/*_m7_daophot_basic.fits')):
        stem = os.path.basename(c).replace('_resbgsub_m7_daophot_basic.fits', '')
        bl, det, rest = stem.split('_', 2)
        res = f'{d}/jw03523-o005_t001_nircam_clear-{bl}-{det}_{rest}_resbgsub_m7_daophot_basic_residual.fits'
        out[stem] = (c, res)
    return out


def pick(b):
    t0, s0, f0 = load('fr0', b); t1, s1, f1 = load('fr1', b)
    z = np.load(f'classify_{b}.npz')
    ref = A.ref[b]
    ff0 = np.asarray(t0['forced_refit_frac'], float); ff1 = np.asarray(t1['forced_refit_frac'], float)
    m0 = -2.5 * np.log10(f0) + z['zp0']; m1 = z['m1']
    have = np.where(np.isfinite(ref) & (ref >= 17) & (ref < 21))[0]
    j0, d0, _ = dsk[have].match_to_catalog_sky(s0); j1, d1, _ = dsk[have].match_to_catalog_sky(s1)
    ok = (d0.arcsec < 0.05) & (d1.arcsec < 0.05) & (ff0[j0] >= 0.5) & (ff1[j1] >= 0.5)
    dm0 = m0[j0] - ref[have]; dm1 = m1[j1] - ref[have]
    ok &= (np.abs(dm0) > 0.3) & (np.abs(dm1) < 0.1)
    cand = np.where(ok)[0]
    print(f'F{b}: {len(cand)} class-a candidates')
    # spread over dolphot mag: sort by mag and take evenly spaced
    cand = cand[np.argsort(ref[have][cand])]
    pk = cand[np.linspace(0, len(cand) - 1, NA[b]).astype(int)] if len(cand) > NA[b] else cand
    A_ = [dict(kind='a', ra=s1.ra.deg[j1[i]], dec=s1.dec.deg[j1[i]], ref=ref[have][i], m0=m0[j0[i]], m1=m1[j1[i]],
               dm0=dm0[i], dm1=dm1[i], ff1=ff1[j1[i]], ff0=ff0[j0[i]]) for i in pk]
    # class c
    lo, hi = {'187N': (22.5, 25), '200W': (23, 26.5), '277W': (21, 25)}[b]
    cc = np.where((z['cls'] == 'c') & (ff1 >= 0.5) & (m1 >= lo) & (m1 < hi))[0]
    print(f'F{b}: {len(cc)} class-c candidates in mag {lo}-{hi}')
    pc = rng.choice(cc, min(NC[b], len(cc)), replace=False)
    C_ = [dict(kind='c', ra=s1.ra.deg[i], dec=s1.dec.deg[i], ref=np.nan, m0=np.nan, m1=m1[i], dm0=np.nan, dm1=np.nan,
               ff1=ff1[i], ff0=np.nan, dol=float(z['ddolphot'][i])) for i in pc]
    LIM = {'187N': 24.5, '200W': 26.0, '277W': 23.0}[b]
    return A_, C_, (s0[m0 < LIM], s1[m1 < LIM])


def frame_cutout(b, obj, ffs, frame_cats):
    """first fr1 frame whose per-frame catalog has a forced row within 0.04" of the object and covers it"""
    pos = SkyCoord(obj['ra'] * u.deg, obj['dec'] * u.deg)
    for stem, (cat, res) in ffs.items():
        if not os.path.exists(res):
            continue
        key = stem
        if key not in frame_cats:
            t = Table.read(cat)
            frame_cats[key] = (t, np.asarray(t['forced_refit'], bool),
                               SkyCoord(np.asarray(t['skycoord_centroid'].ra.deg if hasattr(t['skycoord_centroid'], 'ra') else t['skycoord_centroid'], float) * u.deg,
                                        np.asarray(t['skycoord_centroid'].dec.deg, float) * u.deg))
        t, fo, sk = frame_cats[key]
        if not fo.any():
            continue
        sep = pos.separation(sk[fo]).arcsec
        if len(sep) and sep.min() < 0.04:
            return stem, res
    return None, None


def show(ax, im, norm=None, cmap='gray_r', **kw):
    ax.imshow(im, origin='lower', norm=norm, cmap=cmap, interpolation='nearest', **kw)
    ax.set_xticks([]); ax.set_yticks([])


def mark(ax, wcs, pos, kind, size, s0, s1, ref_pos):
    for sk, st, col in ((s0, 'x', 'cyan'), (s1, '+', 'lime')):
        sel = sk.separation(pos).arcsec < HALF * 1.5
        if sel.any():
            xy = np.array(wcs.world_to_pixel(sk[sel]))
            ax.plot(xy[0], xy[1], st, color=col, ms=6, mew=1.0, ls='none')
    if ref_pos is not None:
        sel = ref_pos.separation(pos).arcsec < HALF * 1.5
        if sel.any():
            xy = np.array(wcs.world_to_pixel(ref_pos[sel]))
            ax.plot(xy[0], xy[1], 'o', mfc='none', mec='red', ms=16, mew=1.2, ls='none')


nb = len(BANDS)
NR = max(NA.values())
fig, axs = plt.subplots(nb * NR, 6 * 1, figsize=(6 * 1.9, nb * NR * 2.1))
for a in np.ravel(axs):
    a.axis('off')
frame_cats_all = {}
for ib, b in enumerate(BANDS):
    A_, C_, (s0, s1) = pick(b)
    mos = fits.open(f'/orange/adamginsburg/jwst/wd2/wd2_F{b}_AB_i2d.fits', memmap=False)['SCI']
    mw = WCS(mos.header); mimg = mos.data
    ffs = {arm: frame_files(arm, b) for arm in ('fr0', 'fr1')}
    fcats = {}
    for blk, objs in enumerate((A_, C_)):
        for ir, obj in enumerate(objs):
            row = ib * NR + ir
            pos = SkyCoord(obj['ra'] * u.deg, obj['dec'] * u.deg)
            c = Cutout2D(mimg, pos, (2 * HALF * u.arcsec, 2 * HALF * u.arcsec), wcs=mw, mode='partial', fill_value=np.nan)
            cols = [blk * 3 + k for k in range(3)]
            ax = axs[row, cols[0]]; ax.axis('on')
            d = c.data
            vmin, vmax = np.nanpercentile(d, [2, 99.8])
            show(ax, d, simple_norm(d, 'asinh', vmin=vmin, vmax=max(vmax, vmin + 1e-6)))
            mark(ax, c.wcs, pos, obj['kind'], d.shape, s0, s1, dsk)
            peak = np.nanmax(d) - np.nanmedian(d); sig = 1.4826 * np.nanmedian(np.abs(d - np.nanmedian(d)))
            vm = max(0.5 * peak, 5 * sig)
            if obj['kind'] == 'a':
                ttl = f"F{b} dol {obj['ref']:.2f}\nfr0 {obj['m0']:.2f} ({obj['dm0']:+.2f}) fr1 {obj['m1']:.2f} ({obj['dm1']:+.2f})"
            else:
                ttl = f"F{b} new (c): fr1 {obj['m1']:.2f}\nfrac {obj['ff1']:.2f}, dolphot {obj['dol']:.2f}\" away"
            ax.set_title(ttl, fontsize=6.5)
            stem, res1 = frame_cutout(b, obj, ffs['fr1'], fcats)
            for k, arm in ((1, 'fr0'), (2, 'fr1')):
                ax = axs[row, cols[k]]; ax.axis('on')
                if stem is None:
                    show(ax, np.zeros((4, 4))); ax.set_title(f'{arm} residual: no frame', fontsize=6.5); continue
                rpath = ffs[arm][stem][1]
                hh = fits.open(rpath, memmap=True)
                fw = WCS(hh['SCI'].header)
                fc = Cutout2D(hh['SCI'].data, pos, (2 * HALF * u.arcsec, 2 * HALF * u.arcsec), wcs=fw, mode='partial', fill_value=np.nan)
                show(ax, fc.data - np.nanmedian(fc.data), None, cmap='RdBu_r', vmin=-vm, vmax=vm)
                mark(ax, fc.wcs, pos, obj['kind'], fc.data.shape, s0, s1, dsk)
                ax.set_title(f'{arm} resid, {stem.split("_",1)[1][:9]}{stem.split("_exp")[1]}', fontsize=6.5)
                hh.close()
fig.suptitle('Left block: dolphot stars forced in both arms (frac>=0.5), dolphot 17-21 mag, |dm_fr0|>0.3, |dm_fr1|<0.1.  '
             'Right block: fr1-only forced rows (class c).\nPer object: AB mosaic | fr0 frame residual | fr1 frame residual (same frame).  '
             'red o dolphot, cyan x fr0 rows, lime + fr1 rows', fontsize=8)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig('fig_fr_cutouts.png', dpi=110)
print('saved')
