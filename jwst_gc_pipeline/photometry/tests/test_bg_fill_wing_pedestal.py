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
from jwst_gc_pipeline.photometry.cataloging import _build_source_masked_bg
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
    the plug cannot read the (NaN) hole or its edge."""
    mc, vet = _write(tmp_path)
    out = _build_source_masked_bg(mc, vet, FILT, fill_mode='annulus',
                                  fill_annulus_inner_fwhm=0.5)
    assert np.all(np.isfinite(fits.getdata(out)))


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
    for key in ('mask_fwhm', 'fill_sigma_fwhm', 'fill_mode', 'fill_annulus_fwhm'):
        assert f"mopt(options, 'manual_residual_bg_{key}')" in call
