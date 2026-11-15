"""The source-masked bg must not refill a hole from the star's own wing (#694).

The 2 FWHM disk at each catalogued source is NaN-masked and refilled by a
Gaussian convolution of sigma 1 FWHM, whose half-width (1.18 FWHM) is narrower
than the hole.  The fill is drawn mainly from the ring just outside the disk,
where a faint star's residual wing sits, so the bg carries a pedestal at the
star; the next phase subtracts it and the wing grows.  ``fill_sigma_fwhm`` and
``fill_mode='annulus'`` are the opt-in fixes; the defaults are the original
behaviour.
"""
import inspect
import os
import re

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from jwst_gc_pipeline.photometry import cataloging
from argparse import Namespace

from jwst_gc_pipeline.photometry.cataloging import (
    _build_source_masked_bg, _plug_holes_from_annulus, _residual_bg_fill_kwargs)
from jwst_gc_pipeline.photometry.manual_defaults import MANUAL_DEFAULTS

FILT = 'F150W'          # present in reduction/fwhm_table.ecsv
PIX_AS = 0.031
FWHM_PX = 1.6           # approximate; the wing only needs to sit outside 2 FWHM
SKY = 5.0
WING_SIGMA = 2.0        # px; halo is 28% of its amplitude at the 3.2 px disk edge
AMP = 10.0              # wing amplitude at the star centre
STAR_XY = (120.0, 80.0)


def _wcs():
    w = WCS(naxis=2)
    w.wcs.crpix = [100, 100]
    w.wcs.crval = [161.03, -59.75]
    w.wcs.cdelt = [-PIX_AS / 3600, PIX_AS / 3600]
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    return w


def _wing(shape, xy, amp):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    r2 = (xx - xy[0]) ** 2 + (yy - xy[1]) ** 2
    return amp * np.exp(-r2 / (2 * WING_SIGMA ** 2))


def _write(tmp_path, stars=(STAR_XY,), amp=AMP, name='a'):
    """Flat sky plus a residual wing at each star position; every star is in
    the vetted catalogue (as in production, where the disk is masked)."""
    w = _wcs()
    shape = (200, 200)
    data = np.full(shape, SKY)
    for xy in stars:
        data = data + _wing(shape, xy, amp)
    mc = str(tmp_path / f'{name}_clear-f150w-nrca_resbgsub_m6_daophot_basic_mergedcat_residual_i2d.fits')
    fits.HDUList([fits.PrimaryHDU(),
                  fits.ImageHDU(data=data.astype('float32'), header=w.to_header(),
                                name='SCI')]).writeto(mc)
    vet = str(tmp_path / f'{name}_vetted.fits')
    xs, ys = zip(*stars)
    Table({'skycoord': w.pixel_to_world(np.array(xs), np.array(ys))}
          ).write(vet, overwrite=True)
    return mc, vet


def _at(path, xy=STAR_XY):
    return float(fits.getdata(path)[int(round(xy[1])), int(round(xy[0]))])


def _ped(path, xy=STAR_XY):
    return _at(path, xy) - SKY


def test_default_is_bit_identical(tmp_path):
    mc, vet = _write(tmp_path)
    base = fits.getdata(_build_source_masked_bg(mc, vet, FILT)).copy()
    for kw in (dict(fill_sigma_fwhm=1.0),
               dict(fill_mode='gauss'),
               dict(mask_radius_fwhm=2.0, fill_sigma_fwhm=1.0, fill_mode='gauss',
                    fill_annulus_inner_fwhm=4.0)):
        out = fits.getdata(_build_source_masked_bg(mc, vet, FILT, **kw))
        assert np.array_equal(out, base, equal_nan=True)


def test_default_leaves_a_pedestal(tmp_path):
    """Reference for the other tests: the current fill returns a pedestal of a
    sizeable fraction of the wing amplitude at the star."""
    mc, vet = _write(tmp_path)
    assert _ped(_build_source_masked_bg(mc, vet, FILT)) > 0.05 * AMP


def test_wider_kernel_reduces_pedestal(tmp_path):
    mc, vet = _write(tmp_path)
    ped1 = _ped(_build_source_masked_bg(mc, vet, FILT, fill_sigma_fwhm=1.0))
    ped2 = _ped(_build_source_masked_bg(mc, vet, FILT, fill_sigma_fwhm=2.0))
    ped3 = _ped(_build_source_masked_bg(mc, vet, FILT, fill_sigma_fwhm=3.0))
    assert ped2 < 0.7 * ped1
    assert ped3 < ped2


def test_annulus_fill_excludes_own_wing(tmp_path):
    """With the annulus starting where the wing is gone the plug is the sky."""
    mc, vet = _write(tmp_path)
    out = _build_source_masked_bg(mc, vet, FILT, fill_mode='annulus',
                                  fill_annulus_inner_fwhm=4.0)
    assert abs(_ped(out)) < 0.01 * AMP
    # the unmasked image away from the star is untouched by the plug
    assert np.isclose(_at(out, (30, 150)), SKY)


def test_annulus_inner_radius_never_inside_the_disk(tmp_path):
    """An annulus inner radius smaller than the disk is clamped to R + 1 px, so
    the plug cannot read the star's own wing just outside the hole.  Without
    the clamp the annulus starts at 0.5 or 1 FWHM and the plug carries 10-13%
    of the wing amplitude; with it every inner radius below R + 1 gives the
    same output."""
    mc, vet = _write(tmp_path)
    outs = [fits.getdata(_build_source_masked_bg(
        mc, vet, FILT, fill_mode='annulus', fill_annulus_inner_fwhm=f))
        for f in (0.5, 1.0)]
    assert np.all(np.isfinite(outs[0]))
    assert np.array_equal(outs[0], outs[1], equal_nan=True)
    assert abs(float(outs[0][int(STAR_XY[1]), int(STAR_XY[0])]) - SKY) < 0.02 * AMP


def test_annulus_falls_back_when_crowded_or_at_the_edge(tmp_path):
    close = (STAR_XY[0] + 3 * FWHM_PX, STAR_XY[1])
    edge = (1.0, 100.0)
    corner = (198.0, 198.0)
    mc, vet = _write(tmp_path, stars=(STAR_XY, close, edge, corner))
    out = fits.getdata(_build_source_masked_bg(mc, vet, FILT, fill_mode='annulus',
                                               fill_annulus_inner_fwhm=3.0))
    assert np.all(np.isfinite(out))
    assert out.min() > SKY - 1.0
    assert out.max() < SKY + AMP


def test_annulus_leaves_sparse_annulus_to_the_gaussian_fill(tmp_path, capsys):
    """Where the annulus is mostly outside the finite area the hole is left to
    the Gaussian fill: the output equals the plain Gaussian-mode output."""
    mc, vet = _write(tmp_path)
    with fits.open(mc, mode='update') as hl:
        sci = hl['SCI'].data
        keep = np.zeros(sci.shape, bool)
        x, y = int(STAR_XY[0]), int(STAR_XY[1])
        keep[y - 4:y + 5, x - 4:x + 5] = True   # 9 px square: the 3 FWHM
        sci[~keep] = np.nan                     # annulus (4.8-8 px) is >75% NaN
        hl.flush()
    gauss = fits.getdata(_build_source_masked_bg(mc, vet, FILT))
    capsys.readouterr()
    out = fits.getdata(_build_source_masked_bg(mc, vet, FILT, fill_mode='annulus',
                                               fill_annulus_inner_fwhm=3.0))
    assert '0 holes plugged, 1 left to the Gaussian fill' in capsys.readouterr().out
    assert np.isfinite(out[int(STAR_XY[1]), int(STAR_XY[0])])
    assert np.array_equal(out, gauss, equal_nan=True)


def test_annulus_does_not_read_other_plugs(tmp_path):
    """Two neighbours' disks inside each other's annulus stay masked while the
    medians are taken, so one plug cannot feed the other."""
    close = (STAR_XY[0] + 3 * FWHM_PX, STAR_XY[1])
    mc, vet = _write(tmp_path, stars=(STAR_XY, close))
    out = _build_source_masked_bg(mc, vet, FILT, fill_mode='annulus',
                                  fill_annulus_inner_fwhm=4.0)
    for xy in (STAR_XY, close):
        assert abs(_ped(out, xy)) < 0.05 * AMP


def test_unknown_fill_mode_raises(tmp_path):
    mc, vet = _write(tmp_path)
    with pytest.raises(ValueError, match='fill_mode'):
        _build_source_masked_bg(mc, vet, FILT, fill_mode='bogus')


def test_mask_radius_is_reachable(tmp_path):
    """A wider disk removes more of the wing before the fill."""
    mc, vet = _write(tmp_path)
    ped_2 = _ped(_build_source_masked_bg(mc, vet, FILT, mask_radius_fwhm=2.0))
    ped_3 = _ped(_build_source_masked_bg(mc, vet, FILT, mask_radius_fwhm=3.0))
    assert ped_3 < ped_2


def _iterate(tmp_path, n_iter=3, tag='x', **kw):
    """Toy of the compounding loop: the wing of the next phase's residual is
    the base wing plus the pedestal the previous bg removed at the star, which
    the fit cannot absorb.  This coupling (unit gain) is an assumption of the
    test, standing in for the pipeline's subtract-and-refit, not a measurement.
    Returns the wing amplitude entering each iteration."""
    amps, ped = [], 0.0
    for k in range(n_iter):
        amp = AMP + ped
        amps.append(amp)
        mc, vet = _write(tmp_path, amp=amp, name=f'it{tag}{k}')
        ped = max(_ped(_build_source_masked_bg(mc, vet, FILT, **kw)), 0.0)
    return amps


def test_iterated_pedestal_does_not_grow(tmp_path):
    default = _iterate(tmp_path, tag='d')
    # reference behaviour: the wing grows at every iteration
    assert default[0] < default[1] < default[2]
    annulus = _iterate(tmp_path, tag='a', fill_mode='annulus',
                       fill_annulus_inner_fwhm=4.0)
    wide = _iterate(tmp_path, tag='w', fill_sigma_fwhm=3.0)
    for amps in (annulus, wide):
        assert amps[2] - amps[0] < 0.5 * (default[2] - default[0])
    assert annulus[2] < 1.01 * AMP


def test_options_registered_and_wired():
    assert MANUAL_DEFAULTS['manual_residual_bg_mask_fwhm'] == 2.0
    assert MANUAL_DEFAULTS['manual_residual_bg_fill_sigma_fwhm'] == 1.0
    assert MANUAL_DEFAULTS['manual_residual_bg_fill_mode'] == 'gauss'
    assert MANUAL_DEFAULTS['manual_residual_bg_fill_annulus_fwhm'] == 3.0
    src = open(os.path.join(os.path.dirname(__file__), '..',
                            'crowdsource_catalogs_long.py')).read()
    for flag, dest in (('mask-fwhm', 'mask_fwhm'),
                       ('fill-sigma-fwhm', 'fill_sigma_fwhm'),
                       ('fill-mode', 'fill_mode'),
                       ('fill-annulus-fwhm', 'fill_annulus_fwhm')):
        m = re.search(r'add_option\("--residual-bg-%s",(.*?)help=' % flag, src, re.S)
        assert m is not None, flag
        assert f'manual_residual_bg_{dest}' in m.group(1)
    call_src = inspect.getsource(cataloging.run_manual_pipeline)
    call = call_src[call_src.index('bg_for_next[(module, filt)] = _build_source_masked_bg('):]
    call = call[:call.index('_bg_fresh = True')]
    assert '**_residual_bg_fill_kwargs(options)' in call


def test_call_site_kwargs_follow_the_options():
    """The kwargs passed to the bg builder come from the options, and an
    options object without the keys resolves to the builder's own defaults."""
    sig = inspect.signature(_build_source_masked_bg).parameters
    kw = _residual_bg_fill_kwargs(Namespace())
    assert set(kw) <= set(sig)
    for k, v in kw.items():
        assert v == sig[k].default
    kw = _residual_bg_fill_kwargs(Namespace(
        manual_residual_bg_mask_fwhm=3.0,
        manual_residual_bg_fill_sigma_fwhm=2.5,
        manual_residual_bg_fill_mode='annulus',
        manual_residual_bg_fill_annulus_fwhm=4.0))
    assert kw == dict(mask_radius_fwhm=3.0, fill_sigma_fwhm=2.5,
                      fill_mode='annulus', fill_annulus_inner_fwhm=4.0)


def _ramp_image(n=61):
    """work = distance from the centre, so an annulus median is known."""
    yy, xx = np.mgrid[0:n, 0:n]
    c = n // 2
    return np.hypot(xx - c, yy - c).astype(float), c


def test_plug_is_the_median_of_the_bounded_annulus():
    """Annulus 6-10 px (fwhm 2, inner 3, width 2): the median of the radius
    over it is sqrt((36 + 100) / 2) = 8.25.  Dropping the outer bound lets the
    box corners (r up to 14) in and moves it; dropping the inner bound moves it
    down."""
    work, c = _ramp_image()
    fov = np.ones(work.shape, bool)
    work[np.hypot(*np.mgrid[0:61, 0:61] - c) <= 4.0] = np.nan
    n_plug, n_left = _plug_holes_from_annulus(work, fov, [(c, c, 4.0)], 2.0, 3.0)
    assert (n_plug, n_left) == (1, 0)
    assert abs(work[c, c] - 8.25) < 0.2


def test_plug_uses_the_median_not_the_mean():
    work, c = _ramp_image()
    fov = np.ones(work.shape, bool)
    work[c, c + 8] = 1.0e4                       # hot pixel inside the annulus
    work[np.hypot(*np.mgrid[0:61, 0:61] - c) <= 4.0] = np.nan
    _plug_holes_from_annulus(work, fov, [(c, c, 4.0)], 2.0, 3.0)
    assert abs(work[c, c] - 8.25) < 0.3          # the mean would be ~ 58


def test_plug_respects_the_valid_area():
    """A disk straddling the edge of the valid area is plugged only inside it."""
    work, c = _ramp_image()
    yy, xx = np.mgrid[0:61, 0:61]
    fov = xx >= c                                # left half is off the footprint
    work[np.hypot(xx - c, yy - c) <= 4.0] = np.nan
    n_plug, _ = _plug_holes_from_annulus(work, fov, [(c, c, 4.0)], 2.0, 3.0)
    assert n_plug == 1
    assert np.isfinite(work[c, c + 1])
    assert np.isnan(work[c, c - 1])


def test_nondefault_options_leave_header_cards(tmp_path):
    mc, vet = _write(tmp_path)
    hdr = fits.getheader(_build_source_masked_bg(mc, vet, FILT))
    assert 'BGFMODE' not in hdr
    hdr = fits.getheader(_build_source_masked_bg(
        mc, vet, FILT, fill_mode='annulus', fill_annulus_inner_fwhm=4.0))
    assert hdr['BGFMODE'] == 'annulus'
    assert hdr['BGFANN'] == 4.0
    assert hdr['BGFSIG'] == 1.0 and hdr['BGMASKR'] == 2.0
    hdr = fits.getheader(_build_source_masked_bg(mc, vet, FILT, fill_sigma_fwhm=2.0))
    assert hdr['BGFMODE'] == 'gauss' and hdr['BGFSIG'] == 2.0
    assert 'BGFANN' not in hdr
