"""End-to-end characterization test for ``do_photometry_step``.

This is the SAFETY NET for the Phase-6 split of the 1758-line monolith.  It
drives the shortest real path through the function -- unseeded daofind + basic
daophot only (no crowdsource, no iterative, no satstar, no cutout, no bgsub) --
on a synthetic 3-star image, and asserts the basic catalog is written and the
injected sources are recovered at their true positions.

The function is heavily disk/network coupled; four external seams are stubbed so
the test is hermetic and fast, while the REAL seed/finder/fit/dedup/filter/save
logic (blocks A-D, G-J, L', M, P, Q, R of the contract) runs unmodified:
  - get_psf_model        -> returns the synthetic GriddedPSFModel
  - load_or_make_satstar_catalog -> empty (no saturated pixels in the fixture)
  - SvoFps.get_filter_list -> a 1-row filter table (no network)
  - save_residual_datamodel -> no-op (avoids JWST datamodel construction)
  - pl.savefig            -> no-op (avoids diagnostic PNG writes)

Importing catalog_long pulls webbpsf -> slow cold import.
"""
import types

import numpy as np
import pytest

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import NDData
from astropy.table import Table
from astropy.wcs import WCS

# do_photometry_step now lives in the sequestered legacy module; patch the names
# it resolves THERE (the legacy module injects the host's shared-helper namespace).
from jwst_gc_pipeline.photometry.legacy import photometry_step as L

SHAPE = (200, 200)
FWHM_PIX = 2.165  # F405N, from fwhm_table.ecsv
SOURCES = [(50.0, 50.0, 5000.0), (150.0, 80.0, 8000.0), (80.0, 150.0, 6000.0)]


def _wcs():
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crpix = [100.5, 100.5]
    w.wcs.crval = [266.55, -28.80]
    w.wcs.cdelt = [-0.031 / 3600, 0.031 / 3600]
    w.wcs.cunit = ['deg', 'deg']
    return w


def _gridded_psf(fwhm_pix, stamp=25, oversample=1):
    ny = nx = stamp
    yy, xx = np.mgrid[0:ny, 0:nx]
    cy = cx = (stamp - 1) / 2
    sigma = fwhm_pix / 2.3548 * oversample
    g = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)))
    g /= g.sum()
    positions = [(0, 0), (0, SHAPE[0]), (SHAPE[1], 0), (SHAPE[1], SHAPE[0])]
    data = np.array([g, g, g, g])
    nd = NDData(data, meta={'grid_xypos': positions, 'oversampling': oversample})
    from photutils.psf import GriddedPSFModel
    return GriddedPSFModel(nd)


def _render_image():
    # Render with an independent grid so the fixture's grid (returned by the
    # get_psf_model stub) keeps its default params.
    grid = _gridded_psf(FWHM_PIX)
    ny, nx = SHAPE
    yy, xx = np.mgrid[0:ny, 0:nx]
    img = np.zeros(SHAPE, dtype=float)
    for sx, sy, flux in SOURCES:
        grid.x_0 = sx
        grid.y_0 = sy
        grid.flux = flux
        img += np.asarray(grid(xx, yy), dtype=float)
    return img


def _write_i2d(path, ww, img):
    h0 = fits.Header()
    h0['INSTRUME'] = 'NIRCAM'
    h0['TELESCOP'] = 'JWST'
    h0['DATE-OBS'] = '2023-01-01'
    hdul = fits.HDUList([
        fits.PrimaryHDU(header=h0),
        fits.ImageHDU(img.astype('float32'), header=ww.to_header(), name='SCI'),
        fits.ImageHDU(np.ones(SHAPE, dtype='float32'), name='ERR'),
        fits.ImageHDU(np.zeros(SHAPE, dtype=np.int32), name='DQ'),
    ])
    hdul.writeto(path, overwrite=True)


def _options(**over):
    o = types.SimpleNamespace(
        profile_memory=False, desaturated=False, epsf=False, blur=False,
        group=False, cutout_region='', bgsub=False, use_iter3_residual_bg=False,
        each_exposure=True, target='testtarget', max_group_size='unlimited',
        fit_satstar_outside_fov=False, nocrowdsource=True, daophot=True,
        basic_only=True, parallel_workers=1, satstar_artifact_sigK=3.0,
        satstar_artifact_ratio=1.5, proposal_id='9999',
    )
    for k, v in over.items():
        setattr(o, k, v)
    return o


class _FakeSvo:
    @staticmethod
    def get_filter_list(facility=None, instrument=None):
        return Table({'filterID': ['JWST/NIRCam.F405N'],
                      'WavelengthEff': [40520.0]})


@pytest.fixture
def _stubs(monkeypatch):
    grid = _gridded_psf(FWHM_PIX)
    monkeypatch.setattr(L, 'get_psf_model', lambda *a, **k: (grid, None))
    monkeypatch.setattr(L, 'load_or_make_satstar_catalog',
                        lambda *a, **k: Table())
    monkeypatch.setattr(L, 'SvoFps', _FakeSvo)
    monkeypatch.setattr(L, 'save_residual_datamodel', lambda *a, **k: None)
    monkeypatch.setattr(L.pl, 'savefig', lambda *a, **k: None)
    return grid


def _prep(tmp_path):
    """Create the output tree + write the synthetic i2d; return (base, fname)."""
    ww = _wcs()
    base = tmp_path / 'base'
    (base / 'F405N' / 'pipeline').mkdir(parents=True)
    fname = str(base / 'F405N' / 'pipeline' / 'jw09999001_exp_cal.fits')
    _write_i2d(fname, ww, _render_image())
    return base, fname


def _assert_sources_recovered(cat, tol=1.0):
    assert len(cat) >= len(SOURCES)
    xcol = 'x_fit' if 'x_fit' in cat.colnames else 'x_init'
    ycol = 'y_fit' if 'y_fit' in cat.colnames else 'y_init'
    for sx, sy, _flux in SOURCES:
        d = np.hypot(np.asarray(cat[xcol], float) - sx,
                     np.asarray(cat[ycol], float) - sy)
        assert d.min() < tol, f"no fit within {tol}px of ({sx},{sy}); min={d.min():.2f}"


def test_basic_daophot_recovers_injected_sources(tmp_path, _stubs):
    base, fname = _prep(tmp_path)
    L.do_photometry_step(
        _options(), 'F405N', 'nrca', 'nrca', '001', str(base), fname,
        '9999', {}, exposurenumber=1)

    out = base / 'F405N' / 'f405n_nrca_exp00001_daophot_basic.fits'
    assert out.exists(), f"basic catalog not written: {out}"
    _assert_sources_recovered(Table.read(out))


def test_iterative_daophot_recovers_injected_sources(tmp_path, _stubs):
    # basic_only=False runs the iterative path (blocks S-U) after basic.
    base, fname = _prep(tmp_path)
    L.do_photometry_step(
        _options(basic_only=False), 'F405N', 'nrca', 'nrca', '001', str(base),
        fname, '9999', {}, exposurenumber=1)

    out = base / 'F405N' / 'f405n_nrca_exp00001_daophot_iterative.fits'
    assert out.exists(), f"iterative catalog not written: {out}"
    _assert_sources_recovered(Table.read(out))


def test_seeded_basic_recovers_injected_sources(tmp_path, _stubs):
    # seed_catalog != None exercises the seed-assembly path (block L:
    # _resolve_seed_skycoords, dedup, augment, SeededFinder, seeded fit).
    base, fname = _prep(tmp_path)
    ww = _wcs()
    sc = SkyCoord([ww.pixel_to_world(sx, sy) for sx, sy, _ in SOURCES])
    seed = Table({'skycoord': sc, 'flux': [f for *_, f in SOURCES]})

    L.do_photometry_step(
        _options(), 'F405N', 'nrca', 'nrca', '001', str(base), fname,
        '9999', {}, exposurenumber=1, seed_catalog=seed)

    out = base / 'F405N' / 'f405n_nrca_exp00001_daophot_basic.fits'
    assert out.exists(), f"seeded basic catalog not written: {out}"
    _assert_sources_recovered(Table.read(out))
