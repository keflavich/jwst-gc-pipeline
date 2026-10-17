"""Saturated-star PM offset guard in build_treasury_pm.

The v6 Cloud c and Quintuplet Treasury PM tables carried a PM step at the
saturation magnitude: the source-epoch catalogs' saturated-star-replaced
positions were offset by ~80-120 mas from the unsaturated frame, and the
frame tie (dominated by unsaturated stars) passed that offset into every
saturated star's PM (Cloud c ~-23 mas/yr in Dec). What this pins:

* ``replaced_saturated`` is read (bare or ``_{filt}`` suffixed) and carried
  through the loaders to ``src_saturated``/``ref_saturated`` PM columns;
* saturation_pm_offset reports a large, significant offset and does not
  report a small one (Brick/Sgr B2 show ~2 mas/yr; real cluster-vs-field
  motions are ~5 mas/yr);
* build() end to end on a synthetic field with a 80 mas src-satstar offset
  flags it, records it in meta, and removes those rows from trustworthy.
"""
import json

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u

from jwst_gc_pipeline.astrometry import build_treasury_pm as B


def _pm_table(n_sat, n_unsat, d_dec, rng, scatter=3.0):
    t = Table()
    t['pm_ra'] = rng.normal(0, scatter, n_sat + n_unsat)
    t['pm_dec'] = rng.normal(0, scatter, n_sat + n_unsat)
    t['pm_dec'][:n_sat] += d_dec
    t['trustworthy'] = np.ones(n_sat + n_unsat, bool)
    t['src_saturated'] = np.arange(n_sat + n_unsat) < n_sat
    return t


def test_large_offset_is_reported():
    rng = np.random.default_rng(1)
    d = B.saturation_pm_offset(_pm_table(300, 3000, -22.0, rng), 'src')
    assert d['offset']
    assert d['d_pm_dec'] == pytest.approx(-22.0, abs=1.0)
    assert abs(d['d_pm_ra']) < 1.0


@pytest.mark.parametrize('d_dec', [0.0, 2.0, 5.0])
def test_small_offset_is_not_reported(d_dec):
    rng = np.random.default_rng(2)
    d = B.saturation_pm_offset(_pm_table(300, 3000, d_dec, rng), 'src')
    assert not d['offset']


def test_missing_column_or_too_few_rows_returns_none():
    rng = np.random.default_rng(3)
    t = _pm_table(300, 3000, -22.0, rng)
    assert B.saturation_pm_offset(t, 'ref') is None
    assert B.saturation_pm_offset(_pm_table(5, 3000, -22.0, rng), 'src') is None


def test_untrustworthy_rows_are_ignored():
    rng = np.random.default_rng(4)
    t = _pm_table(300, 3000, -22.0, rng)
    t['trustworthy'][:300] = False
    assert B.saturation_pm_offset(t, 'src') is None


def _write_cat(path, sc, flux, sat, suffix=''):
    t = Table()
    t['skycoord' + suffix] = sc
    t['flux' + suffix] = flux
    t['flux_err' + suffix] = flux / 100.0
    t['std_ra' + suffix] = np.full(len(flux), 5e-7)
    t['std_dec' + suffix] = np.full(len(flux), 5e-7)
    t['replaced_saturated' + suffix] = sat
    t.write(path, overwrite=True)


def test_sat_flag_read_from_bare_and_suffixed_columns(tmp_path):
    sc = SkyCoord([266.5, 266.501] * u.deg, [-28.8, -28.8] * u.deg)
    flux = np.array([1e4, 1e3])
    sat = np.array([True, False])
    _write_cat(tmp_path / 'a.fits', sc, flux, sat)
    _write_cat(tmp_path / 'b.fits', sc, flux, sat, suffix='_f212n')
    for f in ('a.fits', 'b.fits'):
        c = B._load_one(str(tmp_path / f), 'f212n', 2023.0, 5.0, 0)
        assert c['sat'].tolist() == [True, False]


def test_catalog_without_sat_column_loads_as_unsaturated(tmp_path):
    sc = SkyCoord([266.5] * u.deg, [-28.8] * u.deg)
    t = Table()
    t['skycoord'] = sc
    t['flux'] = [1e4]; t['flux_err'] = [1e2]
    t['std_ra'] = [2e-6]; t['std_dec'] = [2e-6]
    t.write(tmp_path / 'c.fits')
    c = B._load_one(str(tmp_path / 'c.fits'), 'f212n', 2023.0, 5.0, 0)
    assert c['sat'].tolist() == [False]


def test_build_flags_and_demotes_offset_src_satstars(tmp_path):
    """Synthetic field: 4000 isolated stars on a grid, brightest 10%
    saturated; src-epoch satstar positions shifted 80 mas in Dec."""
    pytest.importorskip('flystar')
    rng = np.random.default_rng(5)
    n = 4000
    side = int(np.ceil(np.sqrt(n)))
    gx, gy = np.meshgrid(np.arange(side), np.arange(side))
    # 1.2" grid spacing: isolated at the 0.45" isolation beam
    x = (gx.ravel()[:n] - side / 2) * 1.2 + rng.normal(0, 0.1, n)
    y = (gy.ravel()[:n] - side / 2) * 1.2 + rng.normal(0, 0.1, n)
    ra0, dec0 = 266.5, -28.8
    flux = 10 ** rng.uniform(2, 5, n)
    sat = flux > np.percentile(flux, 90)

    def sky(dx, dy):
        return SkyCoord((ra0 + (x + dx) / 3600 / np.cos(np.radians(dec0))) * u.deg,
                        (dec0 + (y + dy) / 3600) * u.deg)
    noise = lambda: rng.normal(0, 0.002, n)
    src_dy = noise() + np.where(sat, 0.080, 0.0)
    _write_cat(tmp_path / 'src.fits', sky(noise(), src_dy), flux, sat)
    _write_cat(tmp_path / 'ref.fits', sky(noise(), noise()), flux, sat,
               suffix='_f212n')
    out = tmp_path / 'pm.fits'
    B.build([str(tmp_path / 'src.fits')], [str(tmp_path / 'ref.fits')], 'f212n',
            2023.0, 2026.0, str(out), verbose=False)
    pm = Table.read(out)
    diag = {d['which']: d for d in json.loads(pm.meta['sat_pm_offset_json'])}
    assert diag['src']['offset']
    # 80 mas over 3 yr, sign: PM = (ref - src)/dt
    assert diag['src']['d_pm_dec'] == pytest.approx(-80 / 3, abs=3)
    assert not np.any(pm['trustworthy'] & pm['src_saturated'])
    assert np.any(pm['trustworthy_presat'] & pm['src_saturated'])
    assert pm['trustworthy'][~pm['src_saturated']].mean() > 0.5
