"""Tests for the two-color HiPS derivation (pure-python core)."""
import os

import numpy as np
from astropy.io import fits

from jwst_gc_pipeline.cmz import hips as H


def _write_mono_tile(root, order, npix, arr):
    p = H.tile_path(root, order, npix, 'fits')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    fits.PrimaryHDU(np.asarray(arr, dtype='float32')).writeto(p, overwrite=True)


def _mono_hips(root, order=0, npix_arrays=None):
    for npix, arr in (npix_arrays or {}).items():
        _write_mono_tile(root, order, npix, arr)
    return root


def test_two_color_tile_channels():
    blue = np.array([[0.0, 1.0]])
    red = np.array([[1.0, 0.0]])
    rgba = H.two_color_tile(blue, red, blue_lims=(0.0, 1.0), red_lims=(0.0, 1.0))
    # asinh(0)->0, asinh(1)->1 at the endpoints
    assert rgba[0, 0, 2] == 0 and rgba[0, 1, 2] == 255       # blue channel
    assert rgba[0, 0, 0] == 255 and rgba[0, 1, 0] == 0       # red channel
    # green = 0.5*(r+b) -> both endpoints ~127
    assert 120 <= rgba[0, 0, 1] <= 135 and 120 <= rgba[0, 1, 1] <= 135
    assert (rgba[..., 3] == 255).all()                       # both finite


def test_two_color_tile_nan_transparent():
    blue = np.array([[np.nan]])
    red = np.array([[np.nan]])
    rgba = H.two_color_tile(blue, red, (0.0, 1.0), (0.0, 1.0))
    assert rgba[0, 0, 3] == 0   # NaN -> alpha 0


def test_global_limits(tmp_path):
    root = str(tmp_path / 'monoblue')
    _mono_hips(root, order=0, npix_arrays={
        0: np.array([[0.0, 10.0], [20.0, 100.0]]),
        1: np.array([[5.0, 50.0], [np.nan, 200.0]]),
    })
    lo, hi = H.global_limits(root, sample_order=0, percentiles=(0, 100))
    assert lo == 0.0 and hi == 200.0


def test_derive_two_color_writes_png_tiles(tmp_path):
    blue = str(tmp_path / 'blue')
    red = str(tmp_path / 'red')
    tile = np.array([[1.0, 2.0], [3.0, 4.0]], dtype='float32')
    _mono_hips(blue, order=0, npix_arrays={0: tile, 1: tile})
    _mono_hips(red, order=0, npix_arrays={0: tile * 2, 1: tile * 2})
    out = str(tmp_path / 'color')
    n = H.derive_two_color_hips(blue, red, out,
                                blue_lims=(0.0, 4.0), red_lims=(0.0, 8.0))
    assert n == 2   # two shared tiles
    assert os.path.exists(H.tile_path(out, 0, 0, 'png'))
    assert os.path.exists(os.path.join(out, 'properties'))
    props = H.read_properties(out)
    assert props['hips_tile_format'] == 'png'
    assert props['hips_frame'] == 'galactic'
    assert 'gc_pipeline_tag' in props


def test_derive_only_shared_tiles(tmp_path):
    # blue has tiles {0,1}, red has {1,2}; only tile 1 is shared
    blue = str(tmp_path / 'b')
    red = str(tmp_path / 'r')
    t = np.ones((2, 2), dtype='float32')
    _mono_hips(blue, order=0, npix_arrays={0: t, 1: t})
    _mono_hips(red, order=0, npix_arrays={1: t, 2: t})
    out = str(tmp_path / 'c')
    n = H.derive_two_color_hips(blue, red, out, blue_lims=(0, 1), red_lims=(0, 1))
    assert n == 1
    assert os.path.exists(H.tile_path(out, 0, 1, 'png'))
    assert not os.path.exists(H.tile_path(out, 0, 0, 'png'))


def test_member_registry(tmp_path):
    reg = H.MemberRegistry(str(tmp_path / 'reg.json'))
    reg.add('/data/brick_f212n_i2d.fits', 'brick', tag='t1').save()
    reg.add('/data/brick_f212n_i2d.fits', 'brick').save()   # dup ignored
    reg.add('/data/sgrc_f212n_i2d.fits', 'sgrc', tag='t2').save()
    reg2 = H.MemberRegistry(str(tmp_path / 'reg.json'))
    assert len(reg2.members) == 2
    assert reg2.i2d_paths() == ['/data/brick_f212n_i2d.fits',
                                '/data/sgrc_f212n_i2d.fits']
