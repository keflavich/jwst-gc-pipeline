"""i2d residual seed: detections with roundness beyond the tight +-0.5 cut are
admitted up to +-0.8 when they rise above their local structure (annulus
prominence), and rejected when they sit in structure.

A faint star distorted by noise or a neighbour's wing fails +-0.5 (Brick F182M
m7 residual peaks at S/N > 7: 55% pass +-0.5, 78% pass +-0.8).  The synthetic
knot below sits inside PSF-scale structure and has low prominence; real
nebular knots often pass the prominence test (W51 F187N), so AUTO keeps the
loose cut off on extended-emission targets.  Diffraction-spike knots pass it
too; the spike guard drops loose detections elongated radially from a bright
star.
"""
import os

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from types import SimpleNamespace

from jwst_gc_pipeline.photometry.cataloging import (_annulus_prominence,
                                                   _auto_seed_round_loose_max,
                                                   _build_i2d_augmented_seed,
                                                   _radially_elongated)
from jwst_gc_pipeline.photometry.manual_defaults import MANUAL_DEFAULTS

N = 140
FWHM = 1.99            # F182M


def _wcs():
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crval = [266.5, -28.8]
    w.wcs.crpix = [N / 2, N / 2]
    w.wcs.cdelt = [-0.063 / 3600, 0.063 / 3600]
    return w


def _blob(x0, y0, amp, q=1.0):
    yy, xx = np.mgrid[0:N, 0:N]
    s = FWHM / 2.3548
    return amp * np.exp(-((xx - x0) ** 2 / (2 * (s * q) ** 2) + (yy - y0) ** 2 / (2 * s ** 2)))


STAR = (40.0, 40.0)        # elongated on flat sky: roundness1 -0.29 (inside +-0.5),
                           # roundness2 -0.62 (outside), the roundness2-only case
KNOT = (100.0, 100.0)      # the same shape inside a patch of emission structure


def _image():
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(4)
    img = rng.normal(0, 1, (N, N))
    img += _blob(*STAR, 20.0, q=1.55)
    img += _blob(*KNOT, 20.0, q=1.55)
    # PSF-scale emission structure (rms ~8) in a 36 px box around the knot
    struct = gaussian_filter(rng.normal(0, 1, (N, N)), 1.2)
    struct *= 8.0 / np.std(struct)
    box = np.zeros((N, N))
    box[82:118, 82:118] = 1
    img += struct * box
    return img


def _write(tmp_path, img):
    hdr = _wcs().to_header()
    det = os.path.join(tmp_path, 'det_i2d.fits')
    fits.HDUList([fits.PrimaryHDU(),
                  fits.ImageHDU(img, header=hdr, name='SCI'),
                  fits.ImageHDU(np.ones_like(img), header=hdr, name='ERR'),
                  fits.ImageHDU(np.ones_like(img), header=hdr, name='WHT')]).writeto(det)
    prev = Table({'skycoord': _wcs().pixel_to_world([5.0], [5.0]), 'flux': [100.0]})
    ppath = os.path.join(tmp_path, 'prev_m5_vetted.fits')
    prev.write(ppath)
    return det, ppath


def _seed_xy(path):
    t = Table.read(path)
    x, y = _wcs().world_to_pixel(t['skycoord'])
    return np.c_[x, y]


def _has(xy, pos, r=1.5):
    return bool(np.any(np.hypot(xy[:, 0] - pos[0], xy[:, 1] - pos[1]) < r))


def test_prominence_flat_vs_structured():
    img = _image()
    p = _annulus_prominence(img, np.array([STAR[0], KNOT[0]]), np.array([STAR[1], KNOT[1]]))
    gate = MANUAL_DEFAULTS["manual_seed_round_loose_prom_min"]
    assert p[0] > 2 * gate and p[1] < gate


@pytest.mark.parametrize('loose, star_in', [(0.0, False), (0.8, True)])
def test_loose_roundness_admits_star_not_knot(tmp_path, loose, star_in):
    det, prev = _write(str(tmp_path), _image())
    out = _build_i2d_augmented_seed(det, prev, 'F182M', local_snr_min=5.0,
                                    roundlo=-0.5, roundhi=0.5,
                                    round_loose_max=loose, round_loose_prom_min=5.0)
    xy = _seed_xy(out)
    assert _has(xy, STAR) == star_in
    assert not _has(xy, KNOT)


def test_pipeline_default():
    assert MANUAL_DEFAULTS['manual_seed_round_max'] == 0.5
    assert MANUAL_DEFAULTS['manual_seed_round_loose_max'] == -1.0      # AUTO
    assert MANUAL_DEFAULTS['manual_seed_round_loose_prom_min'] == 5.0


def test_auto_loose_off_on_extended_emission():
    # elongated nebular knots pass the prominence test on W51-like fields, so
    # AUTO keeps only the tight cut there; an explicit value is used verbatim
    star = SimpleNamespace(target='brick', extended_emission=None)
    nebula = SimpleNamespace(target='w51', extended_emission=None)
    assert _auto_seed_round_loose_max(-1, star) == 0.8
    assert _auto_seed_round_loose_max(-1, nebula) == 0.0
    assert _auto_seed_round_loose_max(0.8, nebula) == 0.8
    assert _auto_seed_round_loose_max(0.0, star) == 0.0
    forced = SimpleNamespace(target='brick', extended_emission=True)
    assert _auto_seed_round_loose_max(-1, forced) == 0.0


def _rot_blob(x0, y0, amp, q, theta_deg):
    """Gaussian of FWHM ``FWHM`` across, ``q`` x FWHM along ``theta_deg``."""
    yy, xx = np.mgrid[0:N, 0:N]
    s = FWHM / 2.3548
    t = np.deg2rad(theta_deg)
    u = (xx - x0) * np.cos(t) + (yy - y0) * np.sin(t)
    v = -(xx - x0) * np.sin(t) + (yy - y0) * np.cos(t)
    return amp * np.exp(-(u ** 2 / (2 * (s * q) ** 2) + v ** 2 / (2 * s ** 2)))


BRIGHT = (70.0, 70.0)


@pytest.mark.parametrize('pos, theta, q, flagged', [
    ((100.0, 70.0), 0.0, 2.0, True),      # along the line to the star (spike knot)
    ((70.0, 100.0), 90.0, 2.0, True),     # same, vertical spike
    ((91.0, 91.0), 45.0, 2.0, True),      # diagonal spike
    ((100.0, 70.0), 90.0, 2.0, False),    # tangential: not spike-like
    ((91.0, 91.0), 135.0, 2.0, False),
    ((100.0, 70.0), 0.0, 1.0, False),     # round: no axis
    ((130.0, 70.0), 0.0, 2.0, False),     # beyond the radius
])
def test_radially_elongated(pos, theta, q, flagged):
    img = np.random.default_rng(1).normal(0, 0.05, (N, N)) + _rot_blob(*pos, 10.0, q, theta)
    f = _radially_elongated(img, [pos[0]], [pos[1]], [BRIGHT[0]], [BRIGHT[1]],
                            radius_pix=48.0, half=3)
    assert bool(f[0]) == flagged


def _spike_case(tmp_path, spike_radius):
    """A bright star (in the previous catalog) with a radial spike knot at 30 px
    and a tangentially elongated star at 30 px on the other side."""
    rng = np.random.default_rng(5)
    img = rng.normal(0, 1, (N, N))
    img += _rot_blob(100.0, 70.0, 20.0, 1.7, 0.0)       # spike knot, radial
    img += _rot_blob(40.0, 70.0, 20.0, 1.7, 90.0)       # star, tangential
    hdr = _wcs().to_header()
    det = os.path.join(tmp_path, 'det_i2d.fits')
    fits.HDUList([fits.PrimaryHDU(),
                  fits.ImageHDU(img, header=hdr, name='SCI'),
                  fits.ImageHDU(np.ones_like(img), header=hdr, name='ERR'),
                  fits.ImageHDU(np.ones_like(img), header=hdr, name='WHT')]).writeto(det)
    # the bright star (subtracted in the residual) is in the previous catalog;
    # 199 faint corner sources put it above the 99.5th flux percentile
    fx = np.r_[70.0, np.full(199, 5.0)]
    fy = np.r_[70.0, np.full(199, 5.0) + np.arange(199) * 1e-3]
    prev = Table({'skycoord': _wcs().pixel_to_world(fx, fy),
                  'flux': np.r_[1e6, np.full(199, 10.0)]})
    ppath = os.path.join(tmp_path, 'prev_m5_vetted.fits')
    prev.write(ppath)
    out = _build_i2d_augmented_seed(det, ppath, 'F182M', local_snr_min=5.0,
                                    roundlo=-0.5, roundhi=0.5, round_loose_max=0.8,
                                    round_loose_prom_min=5.0,
                                    round_loose_spike_radius_as=spike_radius)
    return Table.read(out)


@pytest.mark.parametrize('radius, knot_in', [(0.0, True), (3.0, False)])
def test_spike_guard_in_seed(tmp_path, radius, knot_in):
    t = _spike_case(str(tmp_path), radius)
    x, y = _wcs().world_to_pixel(t['skycoord'])
    xy = np.c_[x, y]
    assert _has(xy, (40.0, 70.0))
    assert _has(xy, (100.0, 70.0)) == knot_in
    loose = np.asarray(t['seed_round_loose'], bool)
    assert loose[np.hypot(xy[:, 0] - 40.0, xy[:, 1] - 70.0) < 1.5].all()
    assert not loose[np.asarray(t['seed_origin']).astype(str) != 'i2d'].any()
