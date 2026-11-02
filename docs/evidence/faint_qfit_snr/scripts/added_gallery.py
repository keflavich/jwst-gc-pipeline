"""Stamp gallery of the sources a branch's vetting adds over (or drops from) the #1015 base.

Rows: random added sources (compare.py, out/<field>_<band>_<v>_added_rowid.npy),
or with kind=lost the base-kept sources the branch drops, two per row, grouped
by distance to the nearest saturated star (0-1", 1-2", >2").  1" stamps; linear
stretch from the stamp 10th percentile to the brightest data pixel within
PEAK_R of the drawn source (the reference image gets its own local peak), so a
faint source sets the stretch rather than its brighter neighbours.  Both
residuals share one stretch with the floor moved down by RESID_FLOOR x the
data range, so the background reads grey and an over-subtracted core reads
white.
Columns per source:
  data      band data_i2d; cyan dots = base-kept sources (current catalog),
            orange circles = other sources the branch adds (proposed catalog),
            red x = saturated stars, orange ticks = the drawn source
  current   residual of the #1015 base catalog: the production m6 residual i2d
  resid     (data - model of the production m6 vetted catalog), corrected by
            propresid.py for any base/production membership difference
  proposed  the same residual with the branch's added sources also subtracted
  resid     (lost mode: the dropped sources added back), each as catalog flux
            x the effective PSF stacked from the production model image
  ref       independent-visit image (Brick: 1182/o004 F200W) or other-filter
            image (Sgr B2: F182M, same visit)
Label colour: reference-catalog counterpart within 60 mas (green) or not (red).

<variant>@<other>: compare with the replay <other> in place of the #1015 base
(the 'current' residual is then <other>'s catalog).

Environment (optional): GALLERY_GROUP=nbr groups by distance to the nearest
BRIGHTER base-kept star (0-5, 5-8, >8 px of the band's pixel grid) in place of
the saturated-star distance; GALLERY_SNR_MIN draws only sources with per-frame
S/N (flux/flux_err) at or above it.

usage: python added_gallery.py <field> <variant>[@<other>] <out.png> [per_bin] [added|lost]
"""
import json
import os
import sys
import textwrap
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import propresid                                     # noqa: E402

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
    'w51': dict(band='f187n',
                resid=f'{R}/w51/F187N/pipeline/jw06151-o001_t001_nircam_clear-f187n-merged_resbgsub_m6_daophot_basic_mergedcat_residual_i2d.fits',
                resid7=f'{R}/w51/F187N/pipeline/jw06151-o001_t001_nircam_clear-f187n-merged_resbgsub_m7_daophot_basic_mergedcat_residual_i2d.fits',
                refimg=f'{R}/w51/F182M/pipeline/jw06151-o001_t001_nircam_clear-f182m-merged_data_i2d.fits',
                refcat=f'{R}/w51/catalogs/f182m_merged_indivexp_merged_resbgsub_m6_dao_basic_vetted.fits',
                reflab='F182M (same visit)'),
}
PR = {'snr': '#1016 S/N floor on flux_err_prop', 'qsnr': '#1017 qfit noise term',
      'prom': '#1018 prominence keep + guard', 'lsky': '#1019 local sky-clean',
      'qsnrp7': '#1017 qfit noise term (prominence guard 7)',
      'promr0p5': '#1018 prominence keep (robust branch off)',
      'promr0p7': '#1018 prominence keep >= 7 (robust branch off)',
      'promr0p10': '#1018 prominence keep >= 10 (robust branch off)',
      'prom2': '#1018 prominence keep >= 7, peak-SB guard 4',
      'prom2p3': '#1018 prominence keep >= 7, peak-SB guard 3',
      'prom2p5': '#1018 prominence keep >= 7, peak-SB guard 5',
      'snrp2': '#1016 S/N floor on flux_err_prop (on #1018)',
      'base1016': 'base (#1015-#1021)',
      'promq0b': 'k 0 (no qfit bound)',
      'promq5b': '#1017 qfit noise bound (k 5)'}
SAT_BINS = ((0, 1), (1, 2), (2, np.inf))
NBR_BINS = ((0, 5), (5, 8), (8, np.inf))   # px to the nearest brighter base-kept star
ABBR = {'prominence_robust': 'rprom', 'core_concentration': 'conc', 'local_structure_snr': 'lstruct'}
HALF = 0.5     # 1" stamps
PEAK_R = 0.07  # stretch top = brightest pixel within this radius (arcsec) of the drawn source
RESID_FLOOR = 0.5  # residual stretch floor = data floor - RESID_FLOOR x (data top - data floor)


def load(path):
    # memory-mapped: the mosaics are 3-5 GB and only stamps are read
    h = fits.open(path, memmap=True)
    ext = 'SCI' if 'SCI' in h else 0
    return h[ext].data, wcs.WCS(h[ext].header)


def local(img, w, s):
    """Pixels of img within PEAK_R of s."""
    x, y = w.world_to_pixel(s)
    px = PEAK_R / (wcs.utils.proj_plane_pixel_scales(w)[0] * 3600)
    yy, xx = np.indices(img.shape)
    return img[(xx - x) ** 2 + (yy - y) ** 2 <= px ** 2]


def nearest_brighter_px(sc_all, kb, flux, rows, band):
    """Distance (px of the band's grid) from each of ``rows`` to the nearest
    base-kept star brighter than it, among the 16 nearest within 20 px (inf if
    none); as lsky_snr_diag.py."""
    from scipy.spatial import cKDTree
    pix = 0.0311 if band in ('f182m', 'f187n', 'f212n', 'f200w', 'f150w') else 0.0629
    off = sc_all.transform_to(sc_all[0].skyoffset_frame())
    xy = np.c_[off.lon.to_value(u.arcsec), off.lat.to_value(u.arcsec)] / pix
    ik = np.flatnonzero(kb)
    dd, jj = cKDTree(xy[ik]).query(xy[rows], k=16, distance_upper_bound=20)
    d_b = np.full(len(sc_all), np.inf)
    for n, i in enumerate(rows):
        for d, j in zip(dd[n], jj[n]):
            if not np.isfinite(d):
                break
            if ik[j] != i and flux[ik[j]] > flux[i]:
                d_b[i] = d
                break
    return d_b


def main(field, v, out, per_bin=4, kind='added'):
    per_bin = int(per_bin)
    assert kind in ('added', 'lost')
    v, _, against = v.partition('@')
    basel = PR[against].split()[0] if against else '#1015 base'
    c = CFG[field]
    band = c['band']
    base = Table.read(f'{HERE}/out/{field}_{band}_seed.fits')
    prov = json.loads(base.meta['PROVJSON'])
    sc_all = base['skycoord'] if isinstance(base['skycoord'], SkyCoord) else SkyCoord(base['skycoord'])
    kb = np.asarray(base['kept'], bool)
    var = Table.read(f'{HERE}/out/{field}_{band}_{v}.fits')
    assert np.array_equal(np.asarray(var['rowid']), np.asarray(base['rowid']))
    kv = np.asarray(var['kept'], bool)
    if against:
        t0 = Table.read(f'{HERE}/out/{field}_{band}_{against}.fits')
        assert np.array_equal(np.asarray(t0['rowid']), np.asarray(base['rowid']))
        kb = np.asarray(t0['kept'], bool)
    if kind == 'added' and against:
        add = np.flatnonzero(kv & ~kb)
    elif kind == 'added':
        add = np.load(f'{HERE}/out/{field}_{band}_{v}_added_rowid.npy')
        assert np.array_equal(add, np.flatnonzero(kv & ~kb))
    else:
        add = np.flatnonzero(kb & ~kv)
    extra = [c for c in ('prominence_robust', 'core_concentration', 'local_structure_snr') if c in var.colnames]
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
    sepmas = np.full(len(base), np.nan)
    sepmas[add] = sep.to_value(u.mas)
    infp = np.zeros(len(base), bool)
    infp[add] = sep.arcsec < 1.0
    addmask = np.zeros(len(base), bool)
    addmask[add] = True

    # production membership: m6-table rows at a production vetted position (within 1 mas)
    pidx, psep, _ = propresid.sky(Table.read(propresid.PROD[field])).match_to_catalog_sky(sc_all)
    prod = np.zeros(len(base), bool)
    prod[pidx[psep.to_value(u.mas) < 1]] = True
    psf = np.load(f'{HERE}/out/{field}_{band}_epsf.npy')
    with open(f'{HERE}/out/{field}_{band}_epsf.json') as fh:
        ecal = json.load(fh)
    # residual of catalog C = production residual - model(C \ prod) + model(prod \ C);
    # current C = #1015 base or <other> (kb), proposed C = branch (kv)
    diffs = {1: (kb & ~prod, prod & ~kb), 2: (kv & ~prod, prod & ~kv)}
    img = [('data', load(prov['data_i2d'])), (f'current residual ({basel})', load(c['resid'])),
           (f'proposed residual ({PR[v].split()[0]})', load(c['resid'])), (c['reflab'], load(c['refimg']))]
    NI = len(img)
    group = os.environ.get('GALLERY_GROUP', 'sat')
    snr_min = float(os.environ.get('GALLERY_SNR_MIN', '0'))
    ok = infp & (flux / ef >= snr_min) if snr_min > 0 else infp
    if group == 'nbr':
        dgrp, bins, unit = nearest_brighter_px(sc_all, kb, flux, add, band), NBR_BINS, ' px'
    else:
        dgrp, bins, unit = dsat, SAT_BINS, '"'
    rng = np.random.default_rng(11)
    picks = []
    for lo, hi in bins:
        idx = add[(dgrp[add] >= lo) & (dgrp[add] < hi) & ok[add]]
        sel = rng.choice(idx, min(per_bin, idx.size), replace=False) if idx.size else np.array([], int)
        picks.append(((lo, hi), idx, sel))
    nrow = sum((len(s) + 1) // 2 for _, _, s in picks)
    if nrow == 0:
        print(f'{field} {v}: nothing added')
        return
    fig, axes = plt.subplots(nrow, 2 * NI, figsize=(15, 2.0 * nrow + 1.7), squeeze=False, layout='constrained')
    for a in axes.ravel():
        a.set_axis_off()
    r = 0
    for (lo, hi), idx, sel in picks:
        for k, i in enumerate(sel):
            rr, c0 = r + k // 2, NI * (k % 2)
            sc = sc_all[i]
            dd, wd = img[0][1]
            pxd = HALF / (wcs.utils.proj_plane_pixel_scales(wd)[0] * 3600)
            dc = Cutout2D(dd, sc, (2 * pxd, 2 * pxd), wcs=wd, mode='partial', fill_value=np.nan)
            dcut = np.asarray(dc.data, float)
            dctr = local(dcut, dc.wcs, sc)
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
                if j in (1, 2):
                    sub, back = diffs[j]
                    cut.data = (cut.data
                                - propresid.stamp_model(cut.wcs, cut.data.shape, sc_all, flux, sub, sc, psf)
                                + propresid.stamp_model(cut.wcs, cut.data.shape, sc_all, flux, back, sc, psf))
                if j == 3:
                    ctr, scale_src = local(cut.data, cut.wcs, sc), cut.data
                else:
                    ctr, scale_src = dctr, dcut
                if np.isfinite(ctr).any():
                    vlo, vhi = np.nanpercentile(scale_src, 10), np.nanmax(ctr)
                    if j in (1, 2):
                        vlo = vlo - RESID_FLOOR * (vhi - vlo)
                    ax.imshow(cut.data, origin='lower', cmap='gray_r', vmin=vlo, vmax=max(vhi, vlo + 1e-6))
                x, y = cut.wcs.world_to_pixel(sc)
                if j == 0:
                    nb = np.flatnonzero(sc_all.separation(sc).arcsec < HALF * 1.5)
                    xx, yy = cut.wcs.world_to_pixel(sc_all[nb])
                    ins = (xx >= -0.5) & (xx < cut.data.shape[1] - 0.5) & (yy >= -0.5) & (yy < cut.data.shape[0] - 0.5)
                    kk = ins & kb[nb]
                    ax.plot(xx[kk], yy[kk], '.', ms=2.5, color='c')
                    ka = ins & addmask[nb] & (nb != i)
                    ax.plot(xx[ka], yy[ka], 'o', ms=7, mfc='none', mec='C1', mew=0.8)
                    ks = ins & sat[nb]
                    ax.plot(xx[ks], yy[ks], 'x', ms=6, color='r', mew=1)
                tk = cut.data.shape[0] / 12
                ax.plot([x - 2 * tk, x - tk], [y, y], color='C1', lw=1.2)
                ax.plot([x, x], [y + tk, y + 2 * tk], color='C1', lw=1.2)
                if rr == 0:
                    ax.set_title(lab, fontsize=8)
            snr_f = flux[i] / ef[i]
            snr_p = flux[i] / efp[i] if np.isfinite(efp[i]) and efp[i] > 0 else np.nan
            ex = ''.join(f' {ABBR[cn]} {float(var[cn][i]):.1f}' for cn in extra)
            nbr = f'nbr {dgrp[i]:.1f} px  ' if group == 'nbr' else ''
            axes[rr, c0].set_ylabel(f'{nbr}sat {dsat[i]:.1f}"  S/N {snr_f:.1f}/{snr_p:.0f}\nqfit {qf[i]:.2f} prom {pr[i]:.1f}{ex}',
                                    fontsize=7.5, color='green' if matched[i] else 'red')
            print(f'row {rr + 1} {"left" if k % 2 == 0 else "right"}: rowid {i} {nbr}sat {dsat[i]:.2f}" '
                  f'nearest reference {sepmas[i]:.0f} mas, S/N {snr_f:.1f}/{snr_p:.0f}, qfit {qf[i]:.2f}, prom {pr[i]:.1f}')
        r += (len(sel) + 1) // 2
    summ = '; '.join(f'{lo}-{hi:g}{unit} {int(matched[idx].sum())}/{idx.size}'
                     for (lo, hi), idx, _ in picks) + ' matched'
    by = ('the nearest brighter base-kept star (px)' if group == 'nbr'
          else 'the nearest saturated star')
    if snr_min > 0:
        by += f', per-frame S/N >= {snr_min:g}'
    what = 'adds over' if kind == 'added' else 'drops from'
    circ = 'other added sources' if kind == 'added' else 'other dropped sources'
    lines = [f'{field} {band.upper()}: sources {PR[v]} {what} the {basel} catalog, by distance to '
             f'{by}.  In the reference footprint: {summ}.',
             f'1" stamps, linear stretch from the stamp 10th percentile to the brightest data pixel within '
             f'{PEAK_R}" of the drawn source; both residuals share that stretch with the floor lowered by '
             f'{RESID_FLOOR:g} x the range (grey = background, white = over-subtracted).',
             ('Proposed residual = current minus the added sources\' catalog flux x effective PSF' if kind == 'added'
              else 'Proposed residual = current plus the dropped sources\' catalog flux x effective PSF (added back)')
             + f' (stacked from the production model; '
             f'held-out isolated stars: scale {ecal["s_iso"]:.2f}, rms mismatch {ecal["frac_rms_iso"]:.0%}; '
             f'random stamps: scale {ecal["s_random"]:.2f}).',
             f'Orange ticks = drawn source; cyan dots = {basel}-kept; orange circles = {circ}; red x = saturated star.',
             'Label: ' + ('brighter-neighbour distance, ' if group == 'nbr' else '') + 'sat distance, S/N per-frame/propagated, qfit, prominence'
             + ''.join(f', {ABBR[cn]}' for cn in extra)
             + '; green = reference counterpart within 60 mas, red = none']
    fig.suptitle('\n'.join(textwrap.fill(t, 150) for t in lines), fontsize=9)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(out)


if __name__ == '__main__':
    main(*sys.argv[1:])
