"""Stamp gallery of the sources a branch's vetting adds over the #1015 base.

Rows: random added sources (compare.py, out/<field>_<band>_<v>_added_rowid.npy),
two per row, grouped by distance to the nearest saturated star (0-1", 1-2", >2").
Columns per source:
  data      band data_i2d; cyan dots = base-kept sources (current catalog),
            orange circles = other sources the branch adds (proposed catalog),
            red x = saturated stars, orange ticks = the drawn source
  m6 resid  production m6 residual i2d (every m6 fit subtracted, before
            vetting: what the residual looks like with the source fitted)
  m7 resid  production m7 residual i2d (the current final residual)
  ref       independent-visit image (Brick: 1182/o004 F200W) or other-filter
            image (Sgr B2: F182M, same visit)
Label colour: reference-catalog counterpart within 60 mas (green) or not (red).

usage: python added_gallery.py <field> <variant> <out.png> [per_bin]
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

warnings.simplefilter('ignore', wcs.FITSFixedWarning)
HERE = os.path.dirname(os.path.abspath(__file__))
R = '/orange/adamginsburg/jwst'
CFG = {
    'brick': dict(band='f182m',
                  resid=f'{R}/brick/F182M/pipeline/jw02221-o001_t001_nircam_clear-f182m-merged_resbgsub_m6_daophot_basic_mergedcat_residual_i2d.fits',
                  resid7=f'{R}/brick/F182M/pipeline/jw02221-o001_t001_nircam_clear-f182m-merged_resbgsub_m7_daophot_basic_mergedcat_residual_i2d.fits',
                  refimg=f'{R}/brick/F200W/pipeline/jw01182-o004_t001_nircam_clear-f200w-merged_data_i2d.fits',
                  refcat=f'{R}/brick/catalogs/f200w_merged_o004_indivexp_merged_resbgsub_m7_dao_basic_vetted.fits',
                  reflab='F200W o004 (independent visit)'),
    'sgrb2': dict(band='f187n',
                  resid=f'{R}/sgrb2/F187N/pipeline/jw05365-o001_t001_nircam_clear-f187n-merged_resbgsub_m6_daophot_basic_mergedcat_residual_i2d.fits',
                  resid7=f'{R}/sgrb2/F187N/pipeline/jw05365-o001_t001_nircam_clear-f187n-merged_resbgsub_m7_daophot_basic_mergedcat_residual_i2d.fits',
                  refimg=f'{R}/sgrb2/F182M/pipeline/jw05365-o001_t001_nircam_clear-f182m-merged_data_i2d.fits',
                  refcat=f'{R}/sgrb2/catalogs/f182m_merged_indivexp_merged_resbgsub_m6_dao_basic_vetted.fits',
                  reflab='F182M (same visit)'),
}
PR = {'snr': '#1016 S/N floor on flux_err_prop', 'qsnr': '#1017 qfit noise term',
      'prom': '#1018 prominence keep + guard', 'lsky': '#1019 local sky-clean'}
SAT_BINS = ((0, 1), (1, 2), (2, np.inf))
HALF = 1.0  # arcsec


def load(path):
    # memory-mapped: the mosaics are 3-5 GB and only stamps are read
    h = fits.open(path, memmap=True)
    ext = 'SCI' if 'SCI' in h else 0
    return h[ext].data, wcs.WCS(h[ext].header)


def main(field, v, out, per_bin=4):
    per_bin = int(per_bin)
    c = CFG[field]
    band = c['band']
    base = Table.read(f'{HERE}/out/{field}_{band}_seed.fits')
    prov = json.loads(base.meta['PROVJSON'])
    sc_all = base['skycoord'] if isinstance(base['skycoord'], SkyCoord) else SkyCoord(base['skycoord'])
    kb = np.asarray(base['kept'], bool)
    add = np.load(f'{HERE}/out/{field}_{band}_{v}_added_rowid.npy')
    dsat = np.load(f'{HERE}/out/{field}_{band}_dsat.npy')
    sat = np.asarray(base['is_saturated'], bool)
    flux = np.asarray(base['flux'], float)
    ef = np.asarray(base['flux_err'], float)
    efp = np.asarray(base['flux_err_prop'], float) if 'flux_err_prop' in base.colnames else ef
    qf = np.asarray(base['qfit'], float)
    pr = np.asarray(base['prominence'], float) if 'prominence' in base.colnames else np.full(len(base), np.nan)
    refsc = Table.read(c['refcat'])['skycoord']
    refsc = refsc if isinstance(refsc, SkyCoord) else SkyCoord(refsc)
    _, sep, _ = sc_all[add].match_to_catalog_sky(refsc)
    matched = np.zeros(len(base), bool)
    matched[add] = sep.to_value(u.mas) < 60
    infp = np.zeros(len(base), bool)
    infp[add] = sep.arcsec < 1.0
    addmask = np.zeros(len(base), bool)
    addmask[add] = True

    img = [('data', load(prov['data_i2d'])), ('production m7 residual (current)', load(c['resid7'])),
           ('m6 residual (source fitted)', load(c['resid'])), (c['reflab'], load(c['refimg']))]
    NI = len(img)
    rng = np.random.default_rng(11)
    picks = []
    for lo, hi in SAT_BINS:
        idx = add[(dsat[add] >= lo) & (dsat[add] < hi) & infp[add]]
        sel = rng.choice(idx, min(per_bin, idx.size), replace=False) if idx.size else np.array([], int)
        picks.append(((lo, hi), idx, sel))
    nrow = sum((len(s) + 1) // 2 for _, _, s in picks)
    if nrow == 0:
        print(f'{field} {v}: nothing added')
        return
    fig, axes = plt.subplots(nrow, 2 * NI, figsize=(15, 2.0 * nrow + 1.0), squeeze=False)
    for a in axes.ravel():
        a.set_axis_off()
    r = 0
    for (lo, hi), idx, sel in picks:
        for k, i in enumerate(sel):
            rr, c0 = r + k // 2, NI * (k % 2)
            sc = sc_all[i]
            for j, (lab, (d, w)) in enumerate(img):
                ax = axes[rr, c0 + j]
                ax.set_axis_on()
                ax.set_xticks([]); ax.set_yticks([])
                px = HALF / (wcs.utils.proj_plane_pixel_scales(w)[0] * 3600)
                try:
                    cut = Cutout2D(d, sc, (2 * px, 2 * px), wcs=w, mode='partial', fill_value=np.nan)
                except (ValueError, wcs.NoConvergence):
                    continue
                cut.data = np.asarray(cut.data, float)
                scale_src = cut.data
                if j in (1, 2):
                    dd, wd = img[0][1]
                    pxd = HALF / (wcs.utils.proj_plane_pixel_scales(wd)[0] * 3600)
                    scale_src = np.asarray(Cutout2D(dd, sc, (2 * pxd, 2 * pxd), wcs=wd, mode='partial',
                                                    fill_value=np.nan).data, float)
                n = scale_src.shape[0]
                q = max(2, int(round(0.25 / HALF * n / 2)))
                ctr = scale_src[n // 2 - q:n // 2 + q + 1, n // 2 - q:n // 2 + q + 1]
                if np.isfinite(ctr).any():
                    vlo, vhi = np.nanpercentile(scale_src, 10), np.nanmax(ctr)
                    ax.imshow(cut.data, origin='lower', cmap='gray_r', vmin=vlo, vmax=max(vhi, vlo + 1e-6))
                x, y = cut.wcs.world_to_pixel(sc)
                if j == 0:
                    nb = np.flatnonzero(sc_all.separation(sc).arcsec < HALF * 1.5)
                    xx, yy = cut.wcs.world_to_pixel(sc_all[nb])
                    ins = (xx >= -0.5) & (xx < cut.data.shape[1] - 0.5) & (yy >= -0.5) & (yy < cut.data.shape[0] - 0.5)
                    kk = ins & kb[nb]
                    ax.plot(xx[kk], yy[kk], '.', ms=2.5, color='c')
                    ka = ins & addmask[nb] & (nb != i)
                    ax.plot(xx[ka], yy[ka], 'o', ms=5, mfc='none', mec='C1', mew=0.8)
                    ks = ins & sat[nb]
                    ax.plot(xx[ks], yy[ks], 'x', ms=6, color='r', mew=1)
                ax.plot([x - 6, x - 3], [y, y], color='C1', lw=1)
                ax.plot([x, x], [y + 3, y + 6], color='C1', lw=1)
                if rr == 0:
                    ax.set_title(lab, fontsize=8)
            snr_f = flux[i] / ef[i]
            snr_p = flux[i] / efp[i] if np.isfinite(efp[i]) and efp[i] > 0 else np.nan
            axes[rr, c0].set_ylabel(f'sat {dsat[i]:.1f}"  S/N {snr_f:.1f}/{snr_p:.0f}\nqfit {qf[i]:.2f} prom {pr[i]:.1f}',
                                    fontsize=7.5, color='green' if matched[i] else 'red')
        r += (len(sel) + 1) // 2
    summ = '; '.join(f'{lo}-{hi:g}": {int(matched[idx].sum())}/{idx.size} matched'
                     for (lo, hi), idx, _ in picks)
    fig.suptitle(f'{field} {band.upper()}: sources {PR[v]} adds over the #1015 base, by distance to the nearest '
                 f'saturated star (in reference footprint)\n{summ}; label = sat distance, S/N frame/prop, qfit, '
                 f'prominence; green = reference counterpart within 60 mas; 2" stamps', fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95 if nrow > 3 else 0.9), w_pad=0.3, h_pad=0.6)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(out)


if __name__ == '__main__':
    main(*sys.argv[1:])
