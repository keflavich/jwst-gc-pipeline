"""Prototype the sigma_ee robust-stack CR rejector for the i2d (issue #161 / #189).

The i2d wants CRs rejected (aesthetics) but the bright-star PSF spikes KEPT.
outlier_detection fails that: it flags with a tolerance built from ERR (photon+
read noise), which is 5-9x too tight where the undersampled PSF disperses the
exposures, so it punches holes along the spikes (~77% of its flags are real
signal; only ~7% are genuine CRs). The fix is to compare each exposure to the
OBSERVED stack dispersion instead:

    flag pixel in exposure k  <=>  (val_k - median) > K * sigma_ee   (and positive)

where median and sigma_ee = 1.4826*MAD are taken across the aligned resampled
exposures at that sky pixel. sigma_ee already inflates exactly where the PSF is
steep (~ERR on flat sky, up to ~9x ERR on the spikes), so K*sigma_ee auto-widens
on the spikes (keeps signal) while a genuine CR -- a many-sigma_ee single-exposure
excursion -- still trips it.

This runs on the 24 saved per-group resampled images (*_outlier_i2d.fits, the
exact stack outlier_detection medians) in a window around the brightest source,
builds the no-rejection and sigma_ee-rejected coadds, and shows the REMOVED
flux: if the rejector is right, the removed flux is point-like CRs, NOT the PSF
spikes. Read-only.
"""
import glob
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy import ndimage

DIST = os.environ.get('OUTLIER_DIST_DIR', '/blue/adamginsburg/adamginsburg/tmp/outlier_dist')
OUT = os.environ.get('SIGEE_OUT', f'{DIST}/sigma_ee_proto')
K = float(os.environ.get('SIGEE_K', '5.0'))     # rejection threshold in sigma_ee
HALF = int(os.environ.get('SIGEE_HALF', '500'))  # window half-size (grid px)
MIN_VALID = 5                                    # need this many exposures for sigma_ee
MASKPT = 0.7


def pick_window():
    """Center the window on the brightest compact source in the median image."""
    medf = sorted(glob.glob(f'{DIST}/*median.fits'))[0]
    med = fits.getdata(medf, 'SCI') if 'SCI' in [h.name for h in fits.open(medf)] \
        else fits.getdata(medf)
    med = np.asarray(med, float)
    sm = ndimage.median_filter(np.nan_to_num(med), size=5)
    # avoid the very edges
    b = HALF + 5
    inner = sm[b:-b, b:-b]
    j = np.unravel_index(np.nanargmax(inner), inner.shape)
    cy, cx = j[0] + b, j[1] + b
    print(f'[sigee] window center (grid) = ({cx},{cy})  peak median = {sm[cy, cx]:.1f}',
          flush=True)
    return med, cy, cx


def load_stack(cy, cx):
    from stcal.outlier_detection.utils import compute_weight_threshold
    fs = sorted(glob.glob(f'{DIST}/*_outlier_i2d.fits'))
    sl = (slice(cy - HALF, cy + HALF), slice(cx - HALF, cx + HALF))
    sci = np.full((len(fs), 2 * HALF, 2 * HALF), np.nan, np.float32)
    for k, f in enumerate(fs):
        with fits.open(f, memmap=True) as h:
            s = np.asarray(h['SCI'].data[sl], float)
            w = np.asarray(h['WHT'].data[sl], float)
            thr = compute_weight_threshold(h['WHT'].data, MASKPT)
        s[(w < thr) | ~np.isfinite(s)] = np.nan
        sci[k] = s
    print(f'[sigee] loaded {len(fs)} groups, window {sci.shape[1]}x{sci.shape[2]}',
          flush=True)
    return sci, sl


def main():
    os.makedirs(OUT, exist_ok=True)
    med_full, cy, cx = pick_window()
    sci, sl = load_stack(cy, cx)

    n_valid = np.isfinite(sci).sum(axis=0)
    median = np.nanmedian(sci, axis=0)
    mad = np.nanmedian(np.abs(sci - median[None]), axis=0)
    sigma_ee = 1.4826 * mad

    # per-exposure positive excursions beyond K*sigma_ee (CRs are positive spikes);
    # only where the stack is deep enough to measure sigma_ee.
    good = (n_valid >= MIN_VALID)
    excursion = sci - median[None]
    with np.errstate(invalid='ignore'):
        flag = (excursion > K * sigma_ee[None]) & good[None] & np.isfinite(sci)
    # symmetric variant just for the count (negatives are PSF gaps -> keep them)
    nflag = int(flag.sum())
    npix = int((good[None] & np.isfinite(sci)).sum())
    print(f'[sigee] K={K}  flagged {nflag} / {npix} valid stack-pixels = '
          f'{100 * nflag / max(1, npix):.3f}%  (per-exposure positive excursions)',
          flush=True)

    # coadds: weighted-ish = simple nan-mean here (WHT already gated); compare
    # no-rejection vs sigma_ee-rejected.
    coadd_norej = np.nanmean(sci, axis=0)
    sci_rej = sci.copy()
    sci_rej[flag] = np.nan
    coadd_rej = np.nanmean(sci_rej, axis=0)
    removed = coadd_norej - coadd_rej   # flux the rejector took out

    # how much of the removed flux sits on the PSF (bright median) vs off it?
    bright = median > np.nanpercentile(median, 99)  # spike/core ridge
    rem_on = np.nansum(np.abs(removed)[bright])
    rem_off = np.nansum(np.abs(removed)[~bright])
    print(f'[sigee] |removed| flux on PSF-ridge (top-1% median): {rem_on:.1f}  '
          f'off-ridge: {rem_off:.1f}  -> off/on = {rem_off / max(rem_on, 1e-9):.2f}')
    print('[sigee] (a GOOD rejector removes CRs OFF the ridge; outlier_detection '
          'removed signal ON it)')

    # ---- figure ----
    fig, ax = plt.subplots(2, 3, figsize=(18, 11))
    vlo, vhi = np.nanpercentile(coadd_norej, [5, 99.5])
    kw = dict(origin='lower', vmin=vlo, vmax=vhi, cmap='gray')
    ax[0, 0].imshow(coadd_norej, **kw); ax[0, 0].set_title('(a) no-rejection coadd (CRs present)')
    ax[0, 1].imshow(coadd_rej, **kw); ax[0, 1].set_title(f'(b) sigma_ee-rejected coadd (K={K})')
    rmax = np.nanpercentile(np.abs(removed), 99.8)
    ax[0, 2].imshow(removed, origin='lower', vmin=-rmax, vmax=rmax, cmap='RdBu_r')
    ax[0, 2].set_title('(c) REMOVED = (a)−(b)\nshould be CR points, NOT spikes')

    # zoom on the central bright star
    z = slice(HALF - 150, HALF + 150)
    ax[1, 0].imshow(coadd_norej[z, z], **kw); ax[1, 0].set_title('(d) star zoom: no-rej')
    ax[1, 1].imshow(coadd_rej[z, z], **kw); ax[1, 1].set_title('(e) star zoom: sigma_ee-rej (spikes kept?)')
    ax[1, 2].imshow(removed[z, z], origin='lower', vmin=-rmax, vmax=rmax, cmap='RdBu_r')
    ax[1, 2].set_title('(f) removed (zoom)')
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])

    fig.suptitle(f'sigma_ee robust-stack rejector prototype — window @grid({cx},{cy}), '
                 f'{sci.shape[0]} exposures, K={K}', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p = f'{OUT}/sigma_ee_rejector_proto.png'
    fig.savefig(p, dpi=110)
    print(f'[sigee] wrote {p}')
    print('[sigee] DONE')


if __name__ == '__main__':
    main()
