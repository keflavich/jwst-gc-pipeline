"""i2d residual seed: detections with roundness beyond the tight +-0.5 cut are
admitted up to +-0.8 when they rise above their local structure (annulus
prominence), and rejected when they sit in structure.  Off by default; -1 =
AUTO: on for star-dominated NIRCam fields, off on extended-emission targets
and MIRI.

A faint star distorted by noise or a neighbour's wing fails +-0.5 (Brick F182M
m7 residual peaks at S/N > 7: 55% pass +-0.5, 78% pass +-0.8).  The synthetic
knot below sits inside PSF-scale structure and has low prominence; real
nebular knots and diffraction-spike knots often pass the prominence test
(W51 F187N; Brick and Sgr B2 full frame), which is why AUTO leaves the
loose cut off on extended-emission targets.
"""
import os

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from jwst_gc_pipeline.photometry.cataloging import (_annulus_prominence,
                                                   _build_i2d_augmented_seed)
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
    assert MANUAL_DEFAULTS['manual_seed_round_loose_max'] == 0.0      # off; -1 = AUTO
    assert MANUAL_DEFAULTS['manual_seed_round_loose_prom_min'] == 5.0


def test_auto_round_loose_off_on_extended_emission_and_miri():
    from types import SimpleNamespace
    from jwst_gc_pipeline.photometry.cataloging import _auto_seed_round_loose_max
    auto = -1.0
    for target in ('w51', 'sickle', 'wd2', 'ngc6334'):
        assert _auto_seed_round_loose_max(auto, SimpleNamespace(target=target)) == 0.0
    for target in ('brick', 'sgrb2', 'sgra', 'cloudc'):
        assert _auto_seed_round_loose_max(auto, SimpleNamespace(target=target)) == 0.8
        assert _auto_seed_round_loose_max(
            auto, SimpleNamespace(target=target), miri=True) == 0.0
    # --extended-emission / --no-extended-emission override the target list
    assert _auto_seed_round_loose_max(
        auto, SimpleNamespace(target='brick', extended_emission=True)) == 0.0
    assert _auto_seed_round_loose_max(
        auto, SimpleNamespace(target='w51', extended_emission=False)) == 0.8
    # an explicit value is used verbatim, on MIRI too
    assert _auto_seed_round_loose_max(0.0, SimpleNamespace(target='brick')) == 0.0
    assert _auto_seed_round_loose_max(0.7, SimpleNamespace(target='w51')) == 0.7
    assert _auto_seed_round_loose_max(
        0.8, SimpleNamespace(target='brick'), miri=True) == 0.8


def _rot_blob(x0, y0, amp, q, theta_deg):
    """Gaussian of FWHM ``FWHM`` across, ``q`` x FWHM along ``theta_deg``."""
    yy, xx = np.mgrid[0:N, 0:N]
    s = FWHM / 2.3548
    t = np.deg2rad(theta_deg)
    u = (xx - x0) * np.cos(t) + (yy - y0) * np.sin(t)
    v = -(xx - x0) * np.sin(t) + (yy - y0) * np.cos(t)
    return amp * np.exp(-(u ** 2 / (2 * (s * q) ** 2) + v ** 2 / (2 * s ** 2)))


@pytest.mark.parametrize('loose', [0.0, 0.8])
def test_seed_round_loose_column(tmp_path, loose):
    """New seeds admitted only by the loose window carry seed_round_loose;
    round new seeds and rows carried over from the previous catalog do not."""
    rng = np.random.default_rng(5)
    img = rng.normal(0, 1, (N, N))
    img += _rot_blob(40.0, 70.0, 20.0, 1.7, 90.0)       # elongated star
    img += _rot_blob(100.0, 70.0, 20.0, 1.0, 0.0)       # round star
    hdr = _wcs().to_header()
    det = os.path.join(str(tmp_path), 'det_i2d.fits')
    fits.HDUList([fits.PrimaryHDU(),
                  fits.ImageHDU(img, header=hdr, name='SCI'),
                  fits.ImageHDU(np.ones_like(img), header=hdr, name='ERR'),
                  fits.ImageHDU(np.ones_like(img), header=hdr, name='WHT')]).writeto(det)
    prev = Table({'skycoord': _wcs().pixel_to_world([10.0, 130.0], [10.0, 130.0]),
                  'flux': [100.0, 100.0]})
    ppath = os.path.join(str(tmp_path), 'prev_m5_vetted.fits')
    prev.write(ppath)
    t = Table.read(_build_i2d_augmented_seed(det, ppath, 'F182M', local_snr_min=5.0,
                                             roundlo=-0.5, roundhi=0.5, round_loose_max=loose,
                                             round_loose_prom_min=5.0))
    x, y = _wcs().world_to_pixel(t['skycoord'])
    xy = np.c_[x, y]
    loose_col = np.asarray(t['seed_round_loose'], bool)
    near = lambda pos: np.hypot(xy[:, 0] - pos[0], xy[:, 1] - pos[1]) < 1.5  # noqa: E731
    assert _has(xy, (100.0, 70.0)) and not loose_col[near((100.0, 70.0))].any()
    assert _has(xy, (40.0, 70.0)) == (loose > 0)
    assert loose_col[near((40.0, 70.0))].all()
    assert not loose_col[np.asarray(t['seed_origin']).astype(str) != 'i2d'].any()


def _loose_in_structure(t, margin=4):
    """seed_round_loose rows inside the emission-structure box, ``margin`` px
    in from its edge (where the annulus still samples flat sky)."""
    x, y = _wcs().world_to_pixel(t['skycoord'])
    lo, hi = 82 + margin, 118 - margin
    inside = (x >= lo) & (x < hi) & (y >= lo) & (y < hi)
    return int(np.sum(np.asarray(t['seed_round_loose'], bool) & inside))


def test_prominence_gate_rejects_loose_structure_peaks(tmp_path):
    """The prominence gate, and not the S/N cut, removes loose-window peaks in
    emission structure.  With the gate open (prominence >= 0) some of them
    pass S/N and enter the seed; at the default 5 none does, while the
    elongated star on flat sky enters in both runs."""
    n_in = {}
    for pmin in (0.0, 5.0):
        sub = tmp_path / f'p{pmin:g}'
        sub.mkdir()
        det, prev = _write(str(sub), _image())
        t = Table.read(_build_i2d_augmented_seed(det, prev, 'F182M', local_snr_min=5.0,
                                                 roundlo=-0.5, roundhi=0.5, round_loose_max=0.8,
                                                 round_loose_prom_min=pmin))
        x, y = _wcs().world_to_pixel(t['skycoord'])
        assert _has(np.c_[x, y], STAR)
        n_in[pmin] = _loose_in_structure(t)
    assert n_in[0.0] >= 1
    assert n_in[5.0] == 0
