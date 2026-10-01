"""m7 seed of one filter = cross-band seed UNION that filter's m6 vetted catalog.

The >=2-filter cross-band seed alone drops every source only ONE band's m6
vetting accepted (a third of the m6 vetted catalog in Brick F182M); those
faint stars leave the m7 model and reappear in the final residual.
"""
import os

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

from jwst_gc_pipeline.photometry.cataloging import _build_m7_band_seed
from jwst_gc_pipeline.photometry.manual_defaults import MANUAL_DEFAULTS

RA0, DEC0 = 266.5, -28.8


def _sc(dx_mas, dy_mas):
    dx = np.asarray(dx_mas, float) / 3.6e6 / np.cos(np.deg2rad(DEC0))
    dy = np.asarray(dy_mas, float) / 3.6e6
    return SkyCoord((RA0 + dx) * u.deg, (DEC0 + dy) * u.deg)


def _write(tmp_path):
    # cross-band seed: three positions, no flux column (as _build_crossband_seed writes it)
    xb = Table({'skycoord': _sc([0, 1000, 2000], [0, 0, 0]),
                'n_filt_confirmed': [2, 2, 3]})
    xpath = os.path.join(tmp_path, 'crossband_seed_manual.fits')
    xb.write(xpath)
    # own-band m6 vetted: a 10 mas neighbour of seed 0, a 25 mas neighbour of
    # seed 1, two single-band sources, and one non-positive flux row
    own = Table({'skycoord': _sc([10, 1025, 5000, 6000, 7000], [0, 0, 0, 0, 0]),
                 'flux': [50.0, 70.0, 3.0, 4.0, -1.0]})
    opath = os.path.join(tmp_path, 'own_m6.fits')
    own.write(opath)
    return xpath, opath


def test_union_and_flux_transfer(tmp_path):
    xpath, opath = _write(str(tmp_path))
    out = Table.read(_build_m7_band_seed(xpath, opath, 'F182M', 'merged',
                                         max_sep_mas=30.0))
    origin = np.asarray(out['seed_origin']).astype(str)
    assert list(origin) == ['crossband'] * 3 + ['own_m6'] * 2
    # cross-band positions take the matched own-band flux, unmatched -> 1.0
    assert list(out['flux'][:3]) == [50.0, 70.0, 1.0]
    # the single-band sources are added with their m6 flux; the negative-flux
    # row and the two matched rows are not duplicated
    assert sorted(out['flux'][3:]) == [3.0, 4.0]
    sc = out['skycoord']
    assert np.allclose(sc[:3].ra.deg, _sc([0, 1000, 2000], [0, 0, 0]).ra.deg)


def test_max_sep_controls_dedup(tmp_path):
    xpath, opath = _write(str(tmp_path))
    out = Table.read(_build_m7_band_seed(xpath, opath, 'F182M', 'merged',
                                         max_sep_mas=20.0))
    # the 25 mas neighbour is now a separate own-band source
    assert len(out) == 6
    assert list(out['flux'][:3]) == [50.0, 1.0, 1.0]


def test_output_path_per_module_and_filter(tmp_path):
    xpath, opath = _write(str(tmp_path))
    pa = _build_m7_band_seed(xpath, opath, 'F182M', 'nrca')
    pb = _build_m7_band_seed(xpath, opath, 'F182M', 'nrcb')
    pc = _build_m7_band_seed(xpath, opath, 'F212N', 'nrca')
    assert len({pa, pb, pc}) == 3
    assert pa.endswith('crossband_seed_manual_nrca_f182m_vetted.fits')
    assert os.path.exists(xpath)            # the shared seed is not rewritten


def test_empty_own_catalog_returns_crossband_seed(tmp_path):
    xpath, _ = _write(str(tmp_path))
    epath = os.path.join(str(tmp_path), 'empty.fits')
    Table({'skycoord': _sc([], []), 'flux': np.zeros(0)}).write(epath)
    assert _build_m7_band_seed(xpath, epath, 'F182M', 'merged') == xpath


def test_pipeline_default_on():
    assert MANUAL_DEFAULTS['manual_m7_seed_own_band'] is True
