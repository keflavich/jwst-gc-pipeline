"""The two ways a HiPS tile lands on the sky wrong without erroring.

Both produce a picture that looks plausible and is incorrect, which is why
they are pinned rather than left to inspection.
"""
import importlib.util
import os

import numpy as np
import pytest

from astropy.coordinates import Galactic, SkyCoord
from astropy.io import fits

PIL = pytest.importorskip('PIL')
pytest.importorskip('astropy_healpix')
from PIL import Image                                   # noqa: E402
from astropy_healpix import HEALPix                     # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_REL = os.path.normpath(os.path.join(_HERE, '..', '..', '..', 'scripts', 'release'))


@pytest.fixture(scope='module')
def hp_preview():
    spec = importlib.util.spec_from_file_location(
        'hips_preview', os.path.join(_REL, 'hips_preview.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_png_tile_is_flipped_into_fits_orientation(hp_preview, tmp_path):
    """PNG row 0 is the TOP row; a FITS array's row 0 is the BOTTOM, and the
    tile WCS is a FITS WCS.  Reading a tile without flipping it mirrors every
    tile about its own centre -- which at survey scale still looks like a star
    field, and is wrong everywhere."""
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    arr[0, :, :] = 255                      # the PNG's TOP row is bright
    path = tmp_path / 'Npix0.png'
    Image.fromarray(arr).save(path)

    tile = hp_preview.read_tile(str(path))
    assert np.all(tile[-1] == 255), 'the PNG top row must become the FITS top row'
    assert np.all(tile[0] == 0)


def test_a_transparent_pixel_reads_as_no_data(hp_preview, tmp_path):
    """A HiPS tile's uncovered corner is transparent, not black.  Reading it as
    a zero averages real sky towards black wherever tiles overlap."""
    arr = np.zeros((2, 2, 4), dtype=np.uint8)
    arr[..., :3] = 200
    arr[0, 0, 3] = 0                        # transparent
    arr[0, 1, 3] = 255
    arr[1, :, 3] = 255
    path = tmp_path / 'Npix1.png'
    Image.fromarray(arr, mode='RGBA').save(path)

    tile = hp_preview.read_tile(str(path))
    assert np.isnan(tile[-1, 0]).all()
    assert np.all(tile[-1, 1] == 200)


def test_a_border_tile_takes_the_header_that_contains_it(hp_preview):
    """`tile_header_2d` returns TWO candidate headers for a tile the HPX
    projection splits, and picking the wrong one puts that tile somewhere else
    on the sky with no error.  Built from two plain TAN headers rather than by
    hunting a real split tile: what is under test is the choice, and a real
    border tile is rare enough that the search is the slow part of the test."""
    single = fits.Header({'NAXIS1': 512, 'NAXIS2': 512})
    assert hp_preview.choose_header(single, None, 0) is single

    def header_at(lon, lat):
        header = fits.Header()
        header['NAXIS'] = 2
        header['NAXIS1'] = header['NAXIS2'] = 512
        header['CTYPE1'], header['CTYPE2'] = 'GLON-TAN', 'GLAT-TAN'
        header['CRVAL1'], header['CRVAL2'] = lon, lat
        header['CRPIX1'] = header['CRPIX2'] = 256.5
        header['CDELT1'], header['CDELT2'] = -0.001, 0.001
        return header

    hp = HEALPix(nside=2 ** 7, order='nested', frame=Galactic())
    index = int(hp.skycoord_to_healpix(
        SkyCoord(0.3, 0.2, unit='deg', frame='galactic')))
    centre = hp.healpix_to_skycoord(index).galactic

    on_it = header_at(centre.l.deg, centre.b.deg)
    far_away = header_at(centre.l.deg + 40, centre.b.deg + 30)
    assert hp_preview.choose_header((far_away, on_it), hp, index) is on_it
    assert hp_preview.choose_header((on_it, far_away), hp, index) is on_it
