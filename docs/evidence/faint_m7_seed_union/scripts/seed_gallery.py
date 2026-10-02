"""#1015: density-controlled realness and a stamp gallery of the restored seeds.

(a) The 'own_m6 NOT in production m7' group (the sources #1015 restores to the
    m7 seed) vs 'own_m6 in production m7', matched in F182M flux AND local
    density (number of m7 seeds within 1"), against Brick 1182/o004 F200W m7
    vetted (independent visit).  Checks that the low flux-matched realness
    (seed_realness.py) is not a crowding artifact.
(b) Gallery: random restored seeds in the reference footprint, 2" stamps of
    F182M data, m6 residual, production m7 residual, F200W o004 data.
    Cyan dots = production m7 vetted sources (current catalog); orange
    circles = other restored seeds (what #1015 adds to the seed); orange ticks
    = the drawn seed.  Label green/red = F200W o004 counterpart within 60 mas.

usage: python seed_gallery.py <out_prefix>
"""
import json
import os
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                      # noqa: E402
from astropy.io import fits                          # noqa: E402
from astropy.table import Table                      # noqa: E402
from astropy.coordinates import SkyCoord             # noqa: E402
from astropy.nddata import Cutout2D                  # noqa: E402
from astropy import wcs                              # noqa: E402
import astropy.units as u                            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from realness import match_fraction, in_footprint    # noqa: E402
from seed_realness import REF, EDGES                 # noqa: E402

warnings.simplefilter('ignore', wcs.FITSFixedWarning)
P = '/orange/adamginsburg/jwst/brick/F182M/pipeline/jw02221-o001_t001_nircam_clear-f182m-merged'
IMGS = [('F182M data', f'{P}_data_i2d.fits'),
        ('m6 residual', f'{P}_resbgsub_m6_daophot_basic_mergedcat_residual_i2d.fits'),
        ('production m7 residual', f'{P}_resbgsub_m7_daophot_basic_mergedcat_residual_i2d.fits'),
        ('F200W o004 (independent visit)',
         '/orange/adamginsburg/jwst/brick/F200W/pipeline/jw01182-o004_t001_nircam_clear-f200w-merged_data_i2d.fits')]
HALF = 1.0


def load(path):
    h = fits.open(path, memmap=True)
    ext = 'SCI' if 'SCI' in h else 0
    return h[ext].data, wcs.WCS(h[ext].header)


def density_rel(sc, lf, dens, tgt, refgrp, ref):
    """Realness of tgt vs refgrp, matched in log-flux x density cells."""
    dedges = np.quantile(dens[tgt | refgrp], [0, 1 / 3, 2 / 3, 1])
    dedges[-1] += 1
    rows, num, den = [], 0.0, 0.0
    for di in range(3):
        dsel = (dens >= dedges[di]) & (dens < dedges[di + 1])
        dn = dd = 0.0
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            fsel = dsel & (lf >= lo) & (lf < hi)
            k, kr = tgt & fsel, refgrp & fsel
            if k.sum() < 20 or kr.sum() < 20:
                continue
            kr_idx = np.flatnonzero(kr)
            if kr_idx.size > 3000:
                kr_idx = np.random.default_rng(0).choice(kr_idx, 3000, replace=False)
            mr, cr = match_fraction(sc[kr_idx], ref)
            if not mr > cr:
                continue
            mk, ck = match_fraction(sc[k], ref)
            dn += k.sum() * (mk - ck)
            dd += k.sum() * (mr - cr)
        rows.append((dedges[di], dedges[di + 1], int((tgt & dsel).sum()), dn / dd if dd > 0 else np.nan))
        num += dn
        den += dd
    return rows, (num / den if den > 0 else np.nan)


def main(prefix):
    t = Table.read(f'{HERE}/brick/seed_f182m.fits')
    sc = t['skycoord'] if isinstance(t['skycoord'], SkyCoord) else SkyCoord(t['skycoord'])
    ref = Table.read(REF)['skycoord']
    ref = ref if isinstance(ref, SkyCoord) else SkyCoord(ref)
    fp = in_footprint(sc, ref)
    org = np.asarray(t['seed_origin']).astype(str)
    inprod = np.asarray(t['sep_prod_m7_mas'], float) < 60
    lf = np.log10(np.clip(np.asarray(t['flux'], float), 1e-30, None))
    i1, _, _, _ = sc.search_around_sky(sc, 1.0 * u.arcsec)
    dens = np.bincount(i1, minlength=len(t)) - 1
    tgt = (org == 'own_m6') & ~inprod & fp
    refgrp = (org == 'own_m6') & inprod & fp
    rows, rel = density_rel(sc, lf, dens, tgt, refgrp, ref)
    lines = ['== Brick F182M restored seeds (own_m6 NOT in production m7) vs own_m6 in production m7, '
             'matched in flux x local density (m7 seeds within 1")',
             f'  overall rel {rel:.2f}  (n={tgt.sum()})']
    for lo, hi, n, r in rows:
        lines.append(f'  density [{lo:.0f}, {hi:.0f}) seeds/arcsec-radius: n={n:6d}  rel {r:.2f}')
    lines.append(f'  median density: restored {np.median(dens[tgt]):.0f}, in-prod {np.median(dens[refgrp]):.0f}')
    txt = '\n'.join(lines)
    print(txt, flush=True)
    with open(f'{prefix}_density.txt', 'w') as fh:
        fh.write(txt + '\n')

    # gallery
    _, sep, _ = sc.match_to_catalog_sky(ref)
    matched = sep.to_value(u.mas) < 60
    rng = np.random.default_rng(5)
    picks = rng.choice(np.flatnonzero(tgt), 12, replace=False)
    imgs = [(lab, load(p)) for lab, p in IMGS]
    nrow, ncol = 6, 8
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 2.0 * nrow + 1.0), squeeze=False)
    for a in axes.ravel():
        a.set_axis_off()
    for k, i in enumerate(picks):
        rr, c0 = k // 2, 4 * (k % 2)
        s = sc[i]
        dd, wd = imgs[0][1]
        pxd = HALF / (wcs.utils.proj_plane_pixel_scales(wd)[0] * 3600)
        dcut = np.asarray(Cutout2D(dd, s, (2 * pxd, 2 * pxd), wcs=wd, mode='partial', fill_value=np.nan).data, float)
        n = dcut.shape[0]
        q = max(2, int(round(0.25 / HALF * n / 2)))
        ctr = dcut[n // 2 - q:n // 2 + q + 1, n // 2 - q:n // 2 + q + 1]
        for j, (lab, (d, w)) in enumerate(imgs):
            ax = axes[rr, c0 + j]
            ax.set_axis_on()
            ax.set_xticks([]); ax.set_yticks([])
            px = HALF / (wcs.utils.proj_plane_pixel_scales(w)[0] * 3600)
            try:
                cut = Cutout2D(d, s, (2 * px, 2 * px), wcs=w, mode='partial', fill_value=np.nan)
            except (ValueError, wcs.NoConvergence):
                continue
            img = np.asarray(cut.data, float)
            src = img if j == 3 else dcut
            if j == 3:
                m = img.shape[0]
                qq = max(2, int(round(0.25 / HALF * m / 2)))
                cc = img[m // 2 - qq:m // 2 + qq + 1, m // 2 - qq:m // 2 + qq + 1]
            else:
                cc = ctr
            if np.isfinite(cc).any():
                vlo, vhi = np.nanpercentile(src, 10), np.nanmax(cc)
                ax.imshow(img, origin='lower', cmap='gray_r', vmin=vlo, vmax=max(vhi, vlo + 1e-6))
            x, y = cut.wcs.world_to_pixel(s)
            if j == 0:
                nb = np.flatnonzero(sc.separation(s).arcsec < HALF * 1.5)
                xx, yy = cut.wcs.world_to_pixel(sc[nb])
                ins = (xx >= -0.5) & (xx < img.shape[1] - 0.5) & (yy >= -0.5) & (yy < img.shape[0] - 0.5)
                kk = ins & inprod[nb]
                ax.plot(xx[kk], yy[kk], '.', ms=2.5, color='c')
                ka = ins & (org[nb] == 'own_m6') & ~inprod[nb] & (nb != i)
                ax.plot(xx[ka], yy[ka], 'o', ms=5, mfc='none', mec='C1', mew=0.8)
            ax.plot([x - 6, x - 3], [y, y], color='C1', lw=1)
            ax.plot([x, x], [y + 3, y + 6], color='C1', lw=1)
            if rr == 0:
                ax.set_title(lab, fontsize=8)
        f = float(t['m6_flux'][i])
        snrf = f / float(t['m6_flux_err'][i])
        snrp = f / float(t['m6_flux_err_prop'][i])
        axes[rr, c0].set_ylabel(f'S/N {snrf:.1f}/{snrp:.0f}  nmatch {int(t["m6_nmatch"][i])}\n'
                                f'qfit {float(t["m6_qfit"][i]):.2f} prom {float(t["m6_prominence"][i]):.1f}',
                                fontsize=7.5, color='green' if matched[i] else 'red')
    fig.suptitle(f'Brick F182M: own-band m6 vetted sources the production m7 seed drops (#1015 restores them); '
                 f'{int(matched[tgt].sum())}/{int(tgt.sum())} have an F200W o004 counterpart within 60 mas '
                 f'(chance ~0.07)\ncyan = production m7 vetted, orange circles = other restored seeds, '
                 f'label = m6 S/N frame/prop, nmatch, qfit, prominence; green = F200W counterpart; 2" stamps',
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=0.3, h_pad=0.6)
    fig.savefig(f'{prefix}_gallery.png', dpi=105)
    print(f'{prefix}_gallery.png')
    with open(f'{prefix}_picks.json', 'w') as fh:
        json.dump([int(p) for p in picks], fh)


if __name__ == '__main__':
    main(*sys.argv[1:])
