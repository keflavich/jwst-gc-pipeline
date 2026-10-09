"""Plane-to-position pairing of saved STPSF/WebbPSF PSF grids (psf_grid_io).

The fixtures write grids the way ``CreatePSFLibrary.create_grid`` does: the
PSFs are computed over ``itertools.product(loc_list, loc_list)`` (x slowest),
``DET_YX{i}`` records location ``i`` as (y, x), and stpsf / webbpsf >= 1.3.0
save the planes after photutils sorted them by (y, x).  Each plane carries a
distinct total that encodes the position it was computed at.
"""
import itertools

import numpy as np
import pytest
from astropy.io import fits
from astropy.nddata import NDData
from photutils.psf import GriddedPSFModel

from jwst_gc_pipeline.photometry import psf_grid_io as G

OVERSAMP = 2
NPIX = 11 * OVERSAMP + 1
LOC = [0.0, 682.0, 1365.0, 2047.0]


def _total(x, y):
    # asymmetric in x and y, so a transposed assignment changes the total
    return 1.0 + 1e-4 * x + 3e-4 * y


def _plane(x, y):
    yy, xx = np.mgrid[:NPIX, :NPIX] - (NPIX - 1) / 2
    g = np.exp(-(xx ** 2 + yy ** 2) / (2 * 2.0 ** 2))
    return (g / g.sum() * _total(x, y) * OVERSAMP ** 2).astype('float32')


def _write(path, locations, version='2.2.0', sorted_planes=True, nkeys=None):
    """A grid file as create_grid saves it (keys in computation order)."""
    psf_arr = np.array([_plane(x, y) for x, y in locations])
    if sorted_planes:
        data = GriddedPSFModel(NDData(psf_arr, meta={
            'grid_xypos': locations, 'oversampling': OVERSAMP})).data
    else:
        data = psf_arr
    hdr = fits.Header()
    hdr['DETECTOR'] = ('NRCA1', 'Detector')
    hdr['FILTER'] = ('F200W', 'Filter')
    hdr['OVERSAMP'] = (OVERSAMP, 'Oversampling factor')
    for h, (x, y) in enumerate(locations[:nkeys]):
        hdr[f'DET_YX{h}'] = (str((float(y), float(x))),
                             f"The #{h} PSF's (y,x) detector pixel position")
    if version is not None:
        hdr['VERSION'] = (version, 'STPSF software version')
    fits.PrimaryHDU(data, header=hdr).writeto(path)
    return str(path)


def _square():
    return list(itertools.product(LOC, LOC))


def _assert_planes_at_their_positions(grid, locations):
    xy = [tuple(p) for p in np.asarray(grid.grid_xypos)]
    for x, y in locations:
        i = xy.index((x, y))
        assert grid.data[i].sum() / OVERSAMP ** 2 == pytest.approx(_total(x, y), rel=1e-5)


def test_sorted_planes_with_computation_order_keys(tmp_path):
    loc = _square()
    grid = G.load_stpsf_grid(_write(tmp_path / 'g.fits', loc))
    _assert_planes_at_their_positions(grid, loc)


def test_evaluated_model_follows_the_position(tmp_path):
    """Flux of the evaluated model at an off-diagonal node is the total of
    the PSF computed there, not of its transpose."""
    grid = G.load_stpsf_grid(_write(tmp_path / 'g.fits', _square()))
    x, y = 2047.0, 0.0
    half = (NPIX // OVERSAMP) // 2
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    grid.x_0, grid.y_0, grid.flux = x, y, 1.0
    got = grid(xx + x, yy + y).sum()
    assert got == pytest.approx(_total(x, y), rel=2e-3)
    assert abs(got - _total(y, x)) > 0.1


def test_stpsf_loader_transposes_the_same_file(tmp_path):
    """Why this module exists: stpsf.utils.to_griddedpsfmodel pairs plane i
    with DET_YX{i}, so off-diagonal planes land at the transposed node."""
    utils = pytest.importorskip('stpsf.utils')
    fn = _write(tmp_path / 'g.fits', _square())
    grid = utils.to_griddedpsfmodel(fn)
    xy = [tuple(p) for p in np.asarray(grid.grid_xypos)]
    i = xy.index((2047.0, 0.0))
    assert grid.data[i].sum() / OVERSAMP ** 2 == pytest.approx(_total(0.0, 2047.0), rel=1e-5)


def test_rectangular_location_list(tmp_path):
    loc = list(itertools.product([10.0, 1000.0], [5.0, 700.0, 1900.0]))
    _assert_planes_at_their_positions(
        G.load_stpsf_grid(_write(tmp_path / 'g.fits', loc)), loc)


def test_webbpsf_121_planes_in_key_order(tmp_path):
    loc = _square()
    fn = _write(tmp_path / 'g.fits', loc, version='1.2.1', sorted_planes=False)
    _assert_planes_at_their_positions(G.load_stpsf_grid(fn), loc)


@pytest.mark.parametrize('version', ['1.3.0', '1.5.0', '2.0.0', '2.2.0', '2.2.1.dev3+gabc'])
def test_sorted_writer_versions(tmp_path, version):
    loc = _square()
    fn = _write(tmp_path / 'g.fits', loc, version=version)
    _assert_planes_at_their_positions(G.load_stpsf_grid(fn), loc)


@pytest.mark.parametrize('version', [None, '1.2.2.dev7', '2.3.0', '3.0.0', 'unknown'])
def test_unchecked_version_with_unsorted_keys_raises(tmp_path, version):
    fn = _write(tmp_path / 'g.fits', _square(), version=version)
    with pytest.raises(G.PSFGridOrderError):
        G.load_stpsf_grid(fn)


@pytest.mark.parametrize('version', [None, '1.2.1', '2.2.0', '3.0.0'])
def test_keys_in_yx_order_are_unambiguous(tmp_path, version):
    """Keys already sorted by (y, x): both plane orders agree."""
    loc = [(x, y) for y, x in itertools.product(LOC, LOC)]
    fn = _write(tmp_path / 'g.fits', loc, version=version, sorted_planes=False)
    _assert_planes_at_their_positions(G.load_stpsf_grid(fn), loc)


def test_key_count_must_match_planes(tmp_path):
    fn = _write(tmp_path / 'g.fits', _square(), nkeys=15)
    with pytest.raises(G.PSFGridOrderError, match='15 DET_YX keys for 16'):
        G.load_stpsf_grid(fn)


def test_single_psf_2d_data(tmp_path):
    fn = str(tmp_path / 'one.fits')
    hdr = fits.Header({'OVERSAMP': OVERSAMP, 'DET_YX0': '(1024.0, 1024.0)',
                       'VERSION': '2.2.0'})
    fits.PrimaryHDU(_plane(1024.0, 1024.0), header=hdr).writeto(fn)
    grid = G.load_stpsf_grid(fn)
    assert grid.data.shape == (1, NPIX, NPIX)
    assert tuple(grid.grid_xypos[0]) == (1024.0, 1024.0)


def test_hdulist_input_and_meta(tmp_path):
    fn = _write(tmp_path / 'g.fits', _square())
    with fits.open(fn) as hdul:
        grid = G.load_stpsf_grid(hdul)
    assert grid.meta['detector'] == ('NRCA1', 'Detector')
    assert grid.meta['oversamp'][0] == OVERSAMP
    assert int(np.atleast_1d(grid.oversampling)[0]) == OVERSAMP


def test_meta_keys_match_stpsf_loader(tmp_path):
    utils = pytest.importorskip('stpsf.utils')
    fn = _write(tmp_path / 'g.fits', _square())
    assert set(G.load_stpsf_grid(fn).meta) == set(utils.to_griddedpsfmodel(fn).meta)


def test_missing_oversamp_raises(tmp_path):
    fn = str(tmp_path / 'g.fits')
    hdr = fits.Header({'DET_YX0': '(0.0, 0.0)', 'VERSION': '2.2.0'})
    fits.PrimaryHDU(_plane(0.0, 0.0)[np.newaxis], header=hdr).writeto(fn)
    with pytest.raises(KeyError, match='OVERSAMP'):
        G.load_stpsf_grid(fn)
