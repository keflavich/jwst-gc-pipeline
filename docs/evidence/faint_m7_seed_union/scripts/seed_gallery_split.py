"""#1015: stamp gallery of restored own-band seeds, split by the companion cut.

Top half: 6 random restored seeds within 2.5 FWHM of a brighter seed source
(left out by the companion cut).  Bottom half: 6 random restored seeds
>= 2.5 FWHM from every brighter seed source (added when the opt-in union is
on).  Per seed: F182M data, m6 residual, production m7 residual, F200W o004
(independent visit).  Cyan dots = production m7 vetted sources; orange
circles = other restored seeds; magenta x = the nearest brighter seed source.
Label green/red = F200W o004 counterpart within 60 mas.

usage: python seed_gallery_split.py <out_png>
"""
import os
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                      # noqa: E402
from astropy.table import Table                      # noqa: E402
from astropy.coordinates import SkyCoord             # noqa: E402
from astropy.nddata import Cutout2D                  # noqa: E402
from astropy import wcs                              # noqa: E402
import astropy.units as u                            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from realness import match_fraction, in_footprint    # noqa: E402
from seed_realness import REF                        # noqa: E402
from seed_gallery import IMGS, load                  # noqa: E402

warnings.simplefilter('ignore', wcs.FITSFixedWarning)
FWHM = 0.062
CUT = 2.5
HALF = 0.5       # 1" stamps
PEAK_R = 0.07    # stretch top = brightest pixel within this radius (arcsec) of the drawn source


def local(img, w, s):
    """Pixels of img within PEAK_R of s."""
    x, y = w.world_to_pixel(s)
    px = PEAK_R / (wcs.utils.proj_plane_pixel_scales(w)[0] * 3600)
    yy, xx = np.indices(img.shape)
    return img[(xx - x) ** 2 + (yy - y) ** 2 <= px ** 2]


def main(out):
    t = Table.read(f'{HERE}/brick/seed_f182m.fits')
    sc = t['skycoord'] if isinstance(t['skycoord'], SkyCoord) else SkyCoord(t['skycoord'])
    ref = Table.read(REF)['skycoord']
    ref = ref if isinstance(ref, SkyCoord) else SkyCoord(ref)
    fp = in_footprint(sc, ref)
    org = np.asarray(t['seed_origin']).astype(str)
    own = org == 'own_m6'
    inprod = np.asarray(t['sep_prod_m7_mas'], float) < 60
    flux = np.asarray(t['flux'], float)
    fl = np.where(own, np.asarray(t['m6_flux'], float), flux)
    # nearest brighter seed row (same definition as seed_sep_band.py)
    i1, i2, d, _ = sc.search_around_sky(sc, 8 * FWHM * u.arcsec)
    d = d.to_value(u.arcsec) / FWHM
    ok = (i1 != i2) & (fl[i2] > fl[i1])
    sep_b = np.full(len(t), np.inf)
    nb_b = np.full(len(t), -1)
    order = np.lexsort((d[ok], i1[ok]))
    a, dd, bb = i1[ok][order], d[ok][order], i2[ok][order]
    first = np.r_[True, a[1:] != a[:-1]]
    sep_b[a[first]] = dd[first]
    nb_b[a[first]] = bb[first]
    rest = own & ~inprod & fp
    comp = rest & (sep_b < CUT)
    kept = rest & (sep_b >= CUT)
    _, sep, _ = sc.match_to_catalog_sky(ref)
    matched = sep.to_value(u.mas) < 60
    stats = {}
    for name, g in (('comp', comp), ('kept', kept)):
        m, ch = match_fraction(sc[g], ref)
        stats[name] = (int(g.sum()), m, ch)
    rng = np.random.default_rng(11)
    picks = list(rng.choice(np.flatnonzero(comp), 6, replace=False)) + \
        list(rng.choice(np.flatnonzero(kept), 6, replace=False))
    imgs = [(lab, load(p)) for lab, p in IMGS]
    nc, mc, cc_ = stats['comp']
    nk, mk, ck = stats['kept']
    fig = plt.figure(figsize=(15, 14.6), layout='constrained')
    top, bot = fig.subfigures(2, 1)
    top.suptitle(f'Within {CUT} FWHM of a brighter seed source (left out by the companion cut): '
                 f'n={nc}, F200W o004 match {mc:.2f} (chance {cc_:.2f})', fontsize=10, weight='bold')
    bot.suptitle(f'>= {CUT} FWHM from every brighter seed source (added when the union is on): '
                 f'n={nk}, F200W o004 match {mk:.2f} (chance {ck:.2f})', fontsize=10, weight='bold')
    axes = np.vstack([top.subplots(3, 8, squeeze=False), bot.subplots(3, 8, squeeze=False)])
    for a_ in axes.ravel():
        a_.set_axis_off()
    for k, i in enumerate(picks):
        rr, c0 = k // 2, 4 * (k % 2)
        s = sc[i]
        dd_, wd = imgs[0][1]
        pxd = HALF / (wcs.utils.proj_plane_pixel_scales(wd)[0] * 3600)
        dc = Cutout2D(dd_, s, (2 * pxd, 2 * pxd), wcs=wd, mode='partial', fill_value=np.nan)
        dcut = np.asarray(dc.data, float)
        ctr = local(dcut, dc.wcs, s)
        for j, (lab, (dimg, w)) in enumerate(imgs):
            ax = axes[rr, c0 + j]
            ax.set_axis_on()
            ax.set_xticks([])
            ax.set_yticks([])
            px = HALF / (wcs.utils.proj_plane_pixel_scales(w)[0] * 3600)
            try:
                cut = Cutout2D(dimg, s, (2 * px, 2 * px), wcs=w, mode='partial', fill_value=np.nan)
            except (ValueError, wcs.NoConvergence):
                continue
            img = np.asarray(cut.data, float)
            if j == 3:
                cc, src = local(img, cut.wcs, s), img
            else:
                cc, src = ctr, dcut
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
                ka = ins & own[nb] & ~inprod[nb] & (nb != i)
                ax.plot(xx[ka], yy[ka], 'o', ms=7, mfc='none', mec='C1', mew=0.8)
            if nb_b[i] >= 0 and j < 3:
                bx, by = cut.wcs.world_to_pixel(sc[nb_b[i]])
                ax.plot(bx, by, '+', ms=7, color='m', mew=1.0)
            tk = img.shape[0] / 12
            ax.plot([x - 2 * tk, x - tk], [y, y], color='C1', lw=1.2)
            ax.plot([x, x], [y + tk, y + 2 * tk], color='C1', lw=1.2)
            if rr in (0, 3):
                ax.set_title(lab, fontsize=8)
        f = float(t['m6_flux'][i])
        snrp = f / float(t['m6_flux_err_prop'][i])
        axes[rr, c0].set_ylabel(f'{sep_b[i]:.1f} FWHM, ratio {fl[nb_b[i]] / fl[i]:.1f}\n'
                                f'S/N_prop {snrp:.0f} qfit {float(t["m6_qfit"][i]):.2f}',
                                fontsize=7.5, color='green' if matched[i] else 'red')
    fig.suptitle('Brick F182M: own-band m6 vetted sources the production m7 seed drops.  1" stamps, linear stretch '
                 f'from the stamp 10th percentile to the brightest pixel within {PEAK_R}" of the drawn source '
                 '(data stretch also on both residuals).\n'
                 'Orange ticks = drawn source; cyan dots = production m7 vetted; orange circles = other dropped m6 '
                 'sources; magenta + = nearest brighter seed source.\n'
                 'Label: separation and flux ratio to that source, m6 S/N on flux_err_prop, qfit; '
                 'green = F200W o004 counterpart within 60 mas, red = none', fontsize=9)
    fig.savefig(out, dpi=105)
    print(out, stats)


if __name__ == '__main__':
    main(*sys.argv[1:])
