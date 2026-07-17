"""Tests for the bulk (rigid) astrometry update path."""
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

from jwst_gc_pipeline.photometry import bulk_astrometry_update as B


def _catalog(n=50, seed=0):
    rng = np.random.default_rng(seed)
    ra = 266.40 + rng.uniform(-0.01, 0.01, n)
    dec = -28.90 + rng.uniform(-0.01, 0.01, n)
    t = Table()
    t['x_fit'] = rng.uniform(0, 2048, n)
    t['y_fit'] = rng.uniform(0, 2048, n)
    t['ra'] = ra
    t['dec'] = dec
    t['skycoord'] = SkyCoord(ra * u.deg, dec * u.deg)
    t['flux_fit'] = rng.uniform(10, 1000, n)
    return t


def test_relative_geometry_preserved_and_xfit_untouched():
    t = _catalog()
    before = SkyCoord(t['ra'] * u.deg, t['dec'] * u.deg)
    x0 = np.array(t['x_fit'])
    B.apply_rigid_offset_to_catalog(t, dra_mas=120.0, ddec_mas=-80.0)
    after = SkyCoord(t['ra'] * u.deg, t['dec'] * u.deg)
    # all pairwise separations unchanged (rigid) to sub-microarcsec
    for i in range(0, len(t), 7):
        d_before = before[i].separation(before).arcsec
        d_after = after[i].separation(after).arcsec
        assert np.allclose(d_before, d_after, atol=1e-6)
    # detector positions untouched
    assert np.array_equal(np.array(t['x_fit']), x0)


def test_offset_magnitude_and_direction():
    t = _catalog(n=5)
    before = SkyCoord(t['ra'] * u.deg, t['dec'] * u.deg)
    B.apply_rigid_offset_to_catalog(t, dra_mas=200.0, ddec_mas=150.0)
    after = SkyCoord(t['ra'] * u.deg, t['dec'] * u.deg)
    sep = before.separation(after).to(u.mas).value
    assert np.allclose(sep, np.hypot(200.0, 150.0), rtol=1e-3)
    # dec increased by ~150 mas
    ddec = (after.dec - before.dec).to(u.mas).value
    assert np.allclose(ddec, 150.0, atol=0.5)


def test_data_facet_invariant_only_coords_change():
    # The whole point of the REPROJECT path: every NON-coordinate column
    # (x_fit/y_fit/flux) is byte-identical after the update -- only sky columns
    # move.  This is what keeps the data facet unchanged.
    t = _catalog()
    coord_cols = {'ra', 'dec', 'skycoord'}
    before = {c: np.array(t[c]) for c in t.colnames if c not in coord_cols}
    ra0, dec0 = np.array(t['ra']), np.array(t['dec'])
    B.apply_rigid_offset_to_catalog(t, 250.0, -175.0)
    for c, arr in before.items():
        assert np.array_equal(np.array(t[c]), arr), f"non-coord column {c} changed"
    # and the coordinate columns DID move (tight atol; rtol on RA~266 would
    # otherwise swallow a sub-arcsec shift)
    assert not np.allclose(t['ra'], ra0, rtol=0, atol=1e-9)
    assert not np.allclose(t['dec'], dec0, rtol=0, atol=1e-9)


def test_reversible():
    t = _catalog()
    ra0, dec0 = np.array(t['ra']), np.array(t['dec'])
    B.apply_rigid_offset_to_catalog(t, 137.0, -54.0)
    B.apply_rigid_offset_to_catalog(t, -137.0, 54.0)
    assert np.allclose(t['ra'], ra0, atol=1e-9)
    assert np.allclose(t['dec'], dec0, atol=1e-9)


def test_skycoord_and_radec_stay_consistent():
    t = _catalog()
    B.apply_rigid_offset_to_catalog(t, 90.0, 90.0)
    # the skycoord mixin column and the ra/dec float columns agree after the shift
    sc = t['skycoord']
    assert np.allclose(sc.ra.deg, t['ra'], atol=1e-9)
    assert np.allclose(sc.dec.deg, t['dec'], atol=1e-9)


def test_apply_file_writes_backup_and_meta(tmp_path):
    p = str(tmp_path / 'f200w_nrca_m7.fits')
    _catalog().write(p, overwrite=True)
    touched = B.apply_offset_to_catalog_file(
        p, 100.0, 50.0, utc='2026-07-17T00:00:00Z', method='test', reference='REF',
        restamp=False)
    assert touched
    assert (tmp_path / 'f200w_nrca_m7.fits.pre_bulkastrom').exists()
    out = Table.read(p)
    assert out.meta['ABULKDRA'] == 100.0 and out.meta['ABULKDDE'] == 50.0
    assert out.meta['ABULKREF'] == 'REF'


def test_dry_run_writes_nothing(tmp_path):
    p = str(tmp_path / 'cat.fits')
    _catalog().write(p, overwrite=True)
    ra0 = np.array(Table.read(p)['ra'])
    B.apply_offset_to_catalog_file(p, 500.0, 500.0, dry_run=True, restamp=False)
    assert np.array_equal(np.array(Table.read(p)['ra']), ra0)
    assert not (tmp_path / 'cat.fits.pre_bulkastrom').exists()


# ---- uniformity gate (uses the real measure_offset engine) ----
def _grid_field(n_per_tile=140, seed=1):
    """A dense star field over ~60x60 arcsec, returned as a SkyCoord."""
    rng = np.random.default_rng(seed)
    ra0, dec0 = 266.40, -28.90
    n = n_per_tile * 9
    ra = ra0 + rng.uniform(-0.008, 0.008, n)   # ~±29"
    dec = dec0 + rng.uniform(-0.008, 0.008, n)
    return ra, dec


def test_uniformity_gate_accepts_uniform_shift():
    ra, dec = _grid_field()
    ref = SkyCoord(ra * u.deg, dec * u.deg)
    # catalog = reference shifted uniformly by a known coordinate offset
    a = SkyCoord((ra + 0.00005) * u.deg, (dec + 0.00003) * u.deg)  # ~180/108 mas
    res = B.measure_bulk_offset(a, ref, uniformity_tol_mas=25.0, nx=3, ny=3)
    assert res['ok']
    assert res['worst_tile_dev_mas'] <= 25.0
    # recovers a nonzero coherent offset
    assert np.hypot(res['dra'], res['ddec']) > 50.0


def test_uniformity_gate_rejects_nonuniform():
    ra, dec = _grid_field()
    ref = SkyCoord(ra * u.deg, dec * u.deg)
    # left half shifted differently from right half -> non-uniform residual
    dra = np.where(ra < 266.40, 0.00005, 0.00005)
    ddec = np.where(ra < 266.40, 0.00000, 0.00010)   # right half +360 mas dec
    a = SkyCoord((ra + dra) * u.deg, (dec + ddec) * u.deg)
    with pytest.raises(B.NonUniformResidualError):
        B.measure_bulk_offset(a, ref, uniformity_tol_mas=25.0, nx=3, ny=3)
