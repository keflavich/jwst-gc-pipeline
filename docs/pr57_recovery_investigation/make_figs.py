#!/usr/bin/env python3
"""Generate ~2-arcsec zoom figures (data + catalog overlays + residuals) for the
PR #57 recovery investigation writeup.  Reads the on-disk cutout products."""
import warnings, numpy as np, glob, re
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
from astropy.visualization import simple_norm
from astropy.stats import sigma_clipped_stats
import astropy.units as u

FIG = '/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline-roundness/docs/pr57_recovery_investigation/figs'


def load(cutdir, nrc, kind='data'):
    """Return (image, wcs) for a cutout's data or m6 residual i2d (module nrc)."""
    if kind == 'data':
        g = glob.glob(f'{cutdir}/*/pipeline/*-{nrc}_data_i2d.fits')
    else:
        g = [x for x in glob.glob(f'{cutdir}/*/pipeline/*-{nrc}_*m6*residual_i2d.fits') if 'resbgsub' in x]
    if not g:
        return None, None
    with fits.open(sorted(g)[-1]) as h:
        return np.asarray(h['SCI'].data, float), WCS(h['SCI'].header)


def cat(cutdir, patt='*resbgsub_m6_dao_basic_vetted.fits'):
    cats = sorted(glob.glob(f'{cutdir}/catalogs/{patt}'))
    if not cats:
        return None
    T = [Table.read(c) for c in cats]
    t = vstack(T) if len(T) > 1 else T[0]
    return SkyCoord(t['skycoord'].ra, t['skycoord'].dec), t


def zoom(ax, img, ww, cen, half_as, pixscale, cmap='inferno', resid=False):
    x, y = ww.world_to_pixel(cen); x, y = float(x), float(y)
    r = half_as / pixscale
    if resid:
        _, _, s = sigma_clipped_stats(img[np.isfinite(img)], sigma=3)
        ax.imshow(img, origin='lower', cmap='RdBu_r', vmin=-4 * s, vmax=4 * s)
    else:
        ax.imshow(img, origin='lower', cmap=cmap,
                  norm=simple_norm(img, 'asinh', min_percent=30, max_percent=99.7))
    ax.set_xlim(x - r, x + r); ax.set_ylim(y - r, y + r); ax.set_xticks([]); ax.set_yticks([])
    return x, y


# ---- FIG 2: roundness lever on the Arches clump (baseline vs loosened) ----
def fig_roundness():
    A = '/orange/adamginsburg/jwst/arches/cutouts'
    cen = SkyCoord('17:45:52.30 -28:51:38.0', unit=(u.hourangle, u.deg))
    ps = 0.031
    hz = SkyCoord(['17:45:52.378 -28:51:38.42', '17:45:52.363 -28:51:38.23', '17:45:52.381 -28:51:37.87',
                   '17:45:52.260 -28:51:38.14', '17:45:52.266 -28:51:39.41', '17:45:52.259 -28:51:39.05',
                   '17:45:52.270 -28:51:38.82', '17:45:52.269 -28:51:38.68', '17:45:52.219 -28:51:40.63'],
                  unit=(u.hourangle, u.deg))
    fig, ax = plt.subplots(2, 2, figsize=(11, 11))
    for col, (lab, tag) in enumerate([('groupA_baseline', 'baseline (round +-0.3)'),
                                      ('groupA_bothloose', 'loosened (round +-1.0 + seed)')]):
        d, ww = load(f'{A}/{lab}', 'nrca', 'data'); r, _ = load(f'{A}/{lab}', 'nrca', 'resid')
        oc, _ = cat(f'{A}/{lab}')
        x, y = zoom(ax[0, col], d, ww, cen, 1.5, ps)
        ox, oy = ww.world_to_pixel(oc); tx, ty = ww.world_to_pixel(hz)
        ax[0, col].scatter(ox, oy, s=45, facecolors='none', edgecolors='lime', lw=0.9)
        ax[0, col].scatter(tx, ty, s=130, facecolors='none', edgecolors='red', lw=1.4)
        ax[0, col].set_title(f'{tag}\ndata + our catalog (green) + Hosek clump (red)', fontsize=9)
        zoom(ax[1, col], r, ww, cen, 1.5, ps, resid=True)
        ax[1, col].scatter(tx, ty, s=130, facecolors='none', edgecolors='lime', lw=1.2)
        ax[1, col].set_title('m6 residual (blue=oversub)', fontsize=9)
    fig.suptitle('H2: looser residual roundness recovers blended companions (Arches F212N clump, ~3" FOV)', fontsize=12)
    fig.tight_layout(); fig.savefig(f'{FIG}/fig2_roundness_arches.png', dpi=130); plt.close(fig)
    print('fig2 done')


# ---- FIG 3: emission safety (W51 F187N dark filament) ----
def fig_emission():
    W = '/orange/adamginsburg/jwst/w51/cutouts'
    d0, ww = load(f'{W}/w51df187_base', 'nrcb', 'data')
    cy, cx = d0.shape[0] // 2, d0.shape[1] // 2
    cen = SkyCoord(*ww.pixel_to_world_values(cx, cy), unit='deg')
    ps = 0.031
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.6))
    zoom(ax[0], d0, ww, cen, 2.0, ps); ax[0].set_title('data (dark filament + stars)', fontsize=10)
    for k, (lab, tag) in enumerate([('w51df187_base', 'baseline residual'),
                                    ('w51df187_loose', 'loosened-roundness residual')]):
        r, wr = load(f'{W}/{lab}', 'nrcb', 'resid')
        zoom(ax[k + 1], r, wr, cen, 2.0, ps, resid=True)
        ax[k + 1].set_title(f'{tag}\n(emission stays RED = preserved)', fontsize=10)
    fig.suptitle('H3: looser per-frame roundness is emission-safe -- extended emission stays in the residual (W51 F187N, ~4" FOV)', fontsize=11)
    fig.tight_layout(); fig.savefig(f'{FIG}/fig3_emission_w51.png', dpi=130); plt.close(fig)
    print('fig3 done')


# ---- FIG 5: survival diagnosis + tiered keep on the Brick F182M clump ----
def fig_brick():
    B = '/orange/adamginsburg/jwst/brick/cutouts'
    ra, dec = [], []
    for line in open('/orange/adamginsburg/jwst/brick/regions_/jwst_cat_f182m_selected_stars.reg'):
        m = re.match(r'\s*(?:point|box)\(([-0-9.]+),\s*([-0-9.]+)', line)
        if m: ra.append(float(m.group(1))); dec.append(float(m.group(2)))
    sel = SkyCoord(ra[2:] * u.deg, dec[2:] * u.deg)
    cen = SkyCoord(266.54292, -28.71272, unit='deg'); ps = 0.031
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.6))
    d, ww = load(f'{B}/f182m_kt_baseline', 'nrca', 'data')
    tx, ty = ww.world_to_pixel(sel)
    for k, (lab, tag) in enumerate([('f182m_kt_baseline', 'baseline keep (21/34)'),
                                    ('f182m_kt_tiered', 'tiered keep (28/34)')]):
        oc, ot = cat(f'{B}/{lab}')
        di, wwi = load(f'{B}/{lab}', 'nrca', 'data')
        zoom(ax[k], di, wwi, cen, 1.6, ps)
        ox, oy = wwi.world_to_pixel(oc); txx, tyy = wwi.world_to_pixel(sel)
        lfq = np.asarray(ot['low_fit_quality'], bool) if 'low_fit_quality' in ot.colnames else np.zeros(len(ot), bool)
        ax[k].scatter(ox[~lfq], oy[~lfq], s=40, facecolors='none', edgecolors='lime', lw=0.8)
        ax[k].scatter(ox[lfq], oy[lfq], s=40, facecolors='none', edgecolors='orange', lw=0.9)
        ax[k].scatter(txx, tyy, s=150, facecolors='none', edgecolors='red', lw=1.3)
        ax[k].set_title(f'{tag}\ngreen=kept, orange=low_fit_quality, red=target', fontsize=9)
    r, wr = load(f'{B}/f182m_kt_tiered', 'nrca', 'resid')
    zoom(ax[2], r, wr, cen, 1.6, ps, resid=True)
    txx, tyy = wr.world_to_pixel(sel); ax[2].scatter(txx, tyy, s=150, facecolors='none', edgecolors='lime', lw=1.2)
    ax[2].set_title('tiered m6 residual\n(emission preserved, no shredding)', fontsize=9)
    fig.suptitle('H6/H7: the loss is SURVIVAL not detection -- tiered multi-frame keep recovers 21->28/34 (Brick F182M clump, ~3" FOV)', fontsize=11)
    fig.tight_layout(); fig.savefig(f'{FIG}/fig5_keeptier_brick.png', dpi=130); plt.close(fig)
    print('fig5 done')


fig_roundness()
fig_emission()
fig_brick()
print('ALL FIGS DONE')
