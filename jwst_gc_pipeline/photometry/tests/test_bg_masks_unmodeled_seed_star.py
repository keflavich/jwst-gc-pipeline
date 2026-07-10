"""Regression: the source-masked background must not absorb a real star that was
dropped from the per-frame VETTED catalog.

Cause (ngc6334 F405N, 2026-07): a bright star in a blended group is lost from the
vetted catalog -- either a group-fit flux collapse (m3->m4, flux 1327 -> 7) or a
prior bg over-subtraction (healthy through m5, killed at m6).  Its full flux then
remains as a positive peak in the mergedcat residual.  ``_build_source_masked_bg``
masked only the vetted catalog, so the smoother spread that leftover flux into the
"background": the reconstructed bg reached 164 at a 298-peak star (global median
7), 55% of the flux.  The next phase subtracts that bg, drives the star's fit
non-positive, the non-positive-flux ban drops it, and it never re-enters the
vetted catalog -- a self-reinforcing cycle that permanently loses the star.

Fix: mask the UNION of the vetted catalog and the i2d detection seed (which still
lists the coadd-confirmed point source), so the diffuse-bg estimate never absorbs
a detected star.  Seeds are daofind point sources, not emission, so genuine
extended emission is left untouched.
"""
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.wcs import WCS

from jwst_gc_pipeline.photometry.cataloging import _build_source_masked_bg
from jwst_gc_pipeline.photometry.naming import residual_to_smoothed_bg_i2d

FILT = 'F405N'          # present in reduction/fwhm_table.ecsv
DIFFUSE = 10.0          # flat diffuse background level
PEAK = 300.0            # stellar peak above the diffuse background


def _wcs():
    w = WCS(naxis=2)
    w.wcs.crpix = [100, 100]
    w.wcs.crval = [260.23, -35.75]
    w.wcs.cdelt = [-0.03 / 3600, 0.03 / 3600]
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    return w


def _gauss(shape, x0, y0, amp, sig=2.0):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return amp * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sig ** 2))


def _cat(path, xy, w):
    """Write a minimal catalog with a ``skycoord`` column at pixel positions xy."""
    if len(xy):
        sc = w.pixel_to_world(np.array([p[0] for p in xy]),
                              np.array([p[1] for p in xy]))
    else:
        sc = SkyCoord([] * u.deg, [] * u.deg)
    Table({'skycoord': sc}).write(path, overwrite=True)


def _make_residual_i2d(tmp_path):
    w = _wcs()
    ny = nx = 200
    vet_xy = (60.0, 140.0)     # star that IS in the vetted catalog
    missed_xy = (140.0, 60.0)  # real star DROPPED from the vetted catalog
    data = (np.full((ny, nx), DIFFUSE, dtype='float32')
            + _gauss((ny, nx), *vet_xy, PEAK)
            + _gauss((ny, nx), *missed_xy, PEAK))
    mc = str(tmp_path / 'sim_clear-f405n-nrca_resbgsub_m6_daophot_basic_mergedcat_residual_i2d.fits')
    fits.HDUList([fits.PrimaryHDU(),
                  fits.ImageHDU(data=data, header=w.to_header(), name='SCI')]).writeto(mc)
    return mc, w, vet_xy, missed_xy


def _bg_at(path, w, xy):
    d = fits.getdata(path)
    ix, iy = int(round(xy[0])), int(round(xy[1]))
    return float(d[iy, ix])


def test_bg_absorbs_missed_star_without_seed(tmp_path):
    """Pre-fix behaviour: masking only the vetted catalog leaves the missed
    star's flux in the reconstructed bg (this is the bug being guarded)."""
    mc, w, vet_xy, missed_xy = _make_residual_i2d(tmp_path)
    vet = str(tmp_path / 'vetted.fits')
    _cat(vet, [vet_xy], w)               # missed star NOT listed
    out = _build_source_masked_bg(mc, vet, FILT)
    assert out == residual_to_smoothed_bg_i2d(mc)
    # vetted star is masked -> bg ~ diffuse there
    assert _bg_at(out, w, vet_xy) < DIFFUSE + 0.3 * PEAK
    # missed star is NOT masked -> its flux contaminates the bg
    assert _bg_at(out, w, missed_xy) > DIFFUSE + 0.3 * PEAK


def test_seed_masking_keeps_missed_star_out_of_bg(tmp_path):
    """The fix: passing the i2d detection seed (which lists the missed star)
    masks it, so the reconstructed bg stays at the diffuse level there."""
    mc, w, vet_xy, missed_xy = _make_residual_i2d(tmp_path)
    vet = str(tmp_path / 'vetted.fits')
    seed = str(tmp_path / 'seed.fits')
    _cat(vet, [vet_xy], w)                       # vetted: only the surviving star
    _cat(seed, [vet_xy, missed_xy], w)           # seed: both, incl. the missed one
    out = _build_source_masked_bg(mc, vet, FILT, extra_source_catalogs=[seed])
    # both stars now masked -> bg ~ diffuse at BOTH positions
    assert _bg_at(out, w, vet_xy) < DIFFUSE + 0.3 * PEAK
    assert _bg_at(out, w, missed_xy) < DIFFUSE + 0.3 * PEAK, \
        "seed-listed star must be masked out of the reconstructed background"


def test_empty_extra_catalogs_is_noop(tmp_path):
    """An empty/None extra-catalog list reproduces the vetted-only behaviour
    (back-compat: existing callers that pass nothing are unchanged)."""
    mc, w, vet_xy, missed_xy = _make_residual_i2d(tmp_path)
    vet = str(tmp_path / 'vetted.fits')
    _cat(vet, [vet_xy], w)
    out_none = _build_source_masked_bg(mc, vet, FILT, extra_source_catalogs=())
    assert _bg_at(out_none, w, missed_xy) > DIFFUSE + 0.3 * PEAK
