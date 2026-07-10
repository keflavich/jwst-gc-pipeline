"""End-to-end regression: a bright blended star that is (transiently) absent from
the per-frame VETTED catalog must survive all 7 crowdsource iterations.

This is the cutout-scale reproduction of the ngc6334 F405N failure (2026-07): two
real, isolated, high-S/N stars (peaks 298 & 112 MJy/sr, 169sigma & 129sigma) at
17:20:55.549 -35:45:33.51 and 17:20:57.905 -35:44:59.71 were DETECTED in the m2
seed (offset <0.03") yet ABSENT from the final m6 catalog (model=0, full flux in
the residual).  Mechanism: a star dropped from the vetted catalog (group-fit flux
collapse, or a prior over-subtraction) leaves its flux as a positive residual
peak; ``_build_source_masked_bg`` masked only the vetted catalog, so the smoother
spread that flux into the reconstructed "background" (the bg reached 164 at the
298-peak star); the next iteration subtracts that bg, drives the fit non-positive,
the ban drops it -- and because it is never re-vetted the bg stays contaminated
and the star erodes out of the data with every iteration.  A self-reinforcing loss.

The fix masks the UNION of the vetted catalog and the i2d detection SEED (which
still lists the coadd-confirmed point source), so the diffuse-bg estimate never
absorbs a detected star, the data is preserved, and the star stays recoverable.

This test drives the real ``_build_source_masked_bg`` through a 7-iteration
subtract-bg -> re-detect loop on a synthetic cutout that mimics the extended-
emission field (bright diffuse floor + a blended pair).  ``star_B`` is held out of
the vetted catalog (the transient drop).  With the fix ``star_B`` survives all 7
iterations and is still detected; with the vetted-only bg it is eroded below
detection -- which is exactly the bug.
"""
import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from jwst_gc_pipeline.photometry.cataloging import _build_source_masked_bg

photutils_detection = pytest.importorskip("photutils.detection")
DAOStarFinder = photutils_detection.DAOStarFinder

FILT = 'F405N'
FWHM_PIX = 2.165          # F405N, fwhm_table.ecsv
DIFFUSE = 7.0             # bright diffuse "extended emission" floor (ngc6334 median)
A_PEAK = 298.0           # star_A: stays vetted/modelled (the survivor)
B_PEAK = 112.0           # star_B: the dropped-but-real blended star (T2 analogue)
NITER = 7


def _wcs():
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crpix = [40.5, 40.5]
    w.wcs.crval = [260.231, -35.759]
    w.wcs.cdelt = [-0.031 / 3600, 0.031 / 3600]
    return w


def _gauss(shape, x0, y0, peak, fwhm=FWHM_PIX):
    sig = fwhm / 2.3548
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return peak * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sig ** 2))


def _write_residual_i2d(tmp_path, arr, w, tag):
    p = str(tmp_path / f'sim_{tag}_clear-f405n-nrca_resbgsub_m6_daophot_basic_mergedcat_residual_i2d.fits')
    fits.HDUList([fits.PrimaryHDU(),
                  fits.ImageHDU(arr.astype('float32'), header=w.to_header(), name='SCI')]).writeto(p, overwrite=True)
    return p


def _cat(path, xy, w):
    if len(xy):
        sc = w.pixel_to_world(np.array([p[0] for p in xy]), np.array([p[1] for p in xy]))
    else:
        sc = SkyCoord([], [], unit='deg')
    Table({'skycoord': sc}).write(path, overwrite=True)


def _detected_near(work, xy, fwhm=FWHM_PIX, thresh_abs=0.25 * B_PEAK):
    """True if DAOStarFinder finds a source within 1.5 px of xy on `work`.

    Threshold is an ABSOLUTE floor above the diffuse median, NOT k*std(work): the
    bright star_A dominates std and would make the threshold order-of-B and hence
    flaky.  The cutout is noiseless, so a fixed floor is deterministic.
    """
    med = float(np.median(work))
    finder = DAOStarFinder(fwhm=fwhm, threshold=thresh_abs)
    tbl = finder(work - med)
    if tbl is None or len(tbl) == 0:
        return False, 0.0
    # photutils >=3 renamed xcentroid->x_centroid; another test may globally set
    # photutils.future_column_names=True, disabling the deprecated alias, so
    # resolve whichever name is present rather than hard-coding 'xcentroid'.
    xc = 'x_centroid' if 'x_centroid' in tbl.colnames else 'xcentroid'
    yc = 'y_centroid' if 'y_centroid' in tbl.colnames else 'ycentroid'
    d = np.hypot(np.asarray(tbl[xc]) - xy[0], np.asarray(tbl[yc]) - xy[1])
    return bool(d.min() < 1.5), float(work[int(round(xy[1])), int(round(xy[0]))])


def _run_iterations(tmp_path, mask_seed):
    """7-iteration subtract-bg -> re-detect loop.  star_A is vetted+modelled every
    round; star_B is held OUT of the vetted catalog (the transient drop) but IS in
    the detection seed.  A and B are placed well apart so A's mask disk does not
    incidentally cover B -- B is protected ONLY if the seed is masked (the fix).
    Returns (detected_B, work_val_B, bg_val_B, orig_val_B)."""
    w = _wcs()
    ny = nx = 80
    A_xy = (25.0, 40.0)
    B_xy = (55.0, 40.0)          # 30 px from A -> outside A's ~8.8 px mask disk
    data = (np.full((ny, nx), DIFFUSE, dtype=float)
            + _gauss((ny, nx), *A_xy, A_PEAK)
            + _gauss((ny, nx), *B_xy, B_PEAK))
    orig_B = data[int(B_xy[1]), int(B_xy[0])]

    vet = str(tmp_path / 'vetted.fits')
    seed = str(tmp_path / 'seed.fits')
    _cat(vet, [A_xy], w)                 # vetted: only star_A (star_B dropped)
    _cat(seed, [A_xy, B_xy], w)          # seed: both (star_B is coadd-confirmed)

    bg = np.zeros_like(data)
    detected_B = work_B = bg_B = None
    bix, biy = int(round(B_xy[0])), int(round(B_xy[1]))
    for it in range(NITER):
        work = data - bg
        detected_B, work_B = _detected_near(work, B_xy)
        # model = star_A only (star_B is NOT in the vetted catalog -> unmodelled)
        model = _gauss((ny, nx), *A_xy, A_PEAK)
        residual = work - model            # star_B's full flux remains here
        rp = _write_residual_i2d(tmp_path, residual, w, f'it{it}')
        extra = [seed] if mask_seed else ()
        bgp = _build_source_masked_bg(rp, vet, FILT, extra_source_catalogs=extra)
        bg = np.nan_to_num(fits.getdata(bgp))
        bg_B = float(bg[biy, bix])
    return detected_B, work_B, bg_B, orig_B


def test_blended_star_survives_all_iterations_with_seed_mask(tmp_path):
    """The fix: masking the seed keeps star_B's flux out of the bg, so across all
    7 iterations the reconstructed bg at star_B stays at the diffuse floor, the
    data is preserved, and the star is still detected."""
    detected, work_B, bg_B, orig_B = _run_iterations(tmp_path, mask_seed=True)
    assert detected, "star_B must remain detectable after 7 iterations with the fix"
    # bg at star_B stayed ~diffuse (did NOT absorb the star)
    assert bg_B < DIFFUSE + 0.2 * B_PEAK, (
        f"bg absorbed star_B ({bg_B:.1f}) despite seed masking (diffuse {DIFFUSE})")
    # data at star_B preserved (not over-subtracted)
    assert work_B > DIFFUSE + 0.5 * B_PEAK, (
        f"star_B flux eroded ({work_B:.1f}); orig {orig_B:.1f}, diffuse {DIFFUSE}")


def test_bg_absorbs_star_without_seed_mask(tmp_path):
    """Guard the discriminator: with the OLD vetted-only bg the same star_B is
    absorbed into the reconstructed background (the ngc6334 164-vs-7 signature).
    This is the contamination that, in the full pipeline, feeds the
    over-subtraction -> non-positive-ban -> permanent-loss cycle.  The fix above
    is meaningless unless this pre-fix path demonstrably contaminates the bg."""
    _detected, _work_B, bg_B, _orig_B = _run_iterations(tmp_path, mask_seed=False)
    # vetted-only bg is strongly elevated at the unmodelled star (bug present).
    # Measured ~45 (diffuse 7); the fix drives it to ~7, so the 0.25*peak (=35)
    # boundary cleanly separates the two regimes.
    assert bg_B > DIFFUSE + 0.25 * B_PEAK, (
        f"expected the vetted-only bg to absorb star_B, but bg={bg_B:.1f} "
        f"(diffuse {DIFFUSE}); the test no longer discriminates the bug")
