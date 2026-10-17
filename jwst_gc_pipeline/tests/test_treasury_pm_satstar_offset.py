"""Saturated-star PM offset guard in build_treasury_pm.

The v6 Cloud c and Quintuplet Treasury PM tables carried a PM step at the
saturation magnitude: the source-epoch catalogs' saturated-star-replaced
positions were offset by ~80-120 mas from the unsaturated frame, and the
frame tie (dominated by unsaturated stars) passed that offset into every
saturated star's PM (Cloud c ~-23 mas/yr in Dec; JWST-GC/data-qa#371).
What this pins:

* ``replaced_saturated`` is read (bare or ``_{filt}`` suffixed) and carried
  through the loaders to ``src_saturated``/``ref_saturated`` PM columns;
* saturation_pm_offset reports a large, significant offset and does not
  report a small one (Brick/Sgr B2 show ~2 mas/yr; real cluster-vs-field
  motions are ~5 mas/yr);
* the offset is attributed to an epoch only through rows saturated in that
  epoch ALONE; when the saturated sets coincide, no epoch is blamed;
* build() end to end, in the regime of the real fields (saturated stars a
  small fraction of the tie sample, so the tie follows the unsaturated
  frame): the offset is attributed to src, the shifted rows leave
  'trustworthy', and the surviving trustworthy PMs are ~0.
"""
import json

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u

from jwst_gc_pipeline.astrometry import build_treasury_pm as B


def _pm_table(n_src_only, n_ref_only, n_both, n_neither, d_dec, rng,
              offset_epoch='src', scatter=3.0):
    """Synthetic PM table; rows saturated in ``offset_epoch`` (alone or in
    both epochs) get ``d_dec`` added to pm_dec."""
    n = n_src_only + n_ref_only + n_both + n_neither
    lab = np.repeat(['src', 'ref', 'both', 'neither'],
                    [n_src_only, n_ref_only, n_both, n_neither])
    t = Table()
    t['pm_ra'] = rng.normal(0, scatter, n)
    t['pm_dec'] = rng.normal(0, scatter, n)
    t['trustworthy'] = np.ones(n, bool)
    t['src_saturated'] = (lab == 'src') | (lab == 'both')
    t['ref_saturated'] = (lab == 'ref') | (lab == 'both')
    t['pm_dec'][t[f'{offset_epoch}_saturated']] += d_dec
    t.meta['baseline_yr'] = 3.0
    return t


def _by(diags):
    return {d['which']: d for d in diags}


def test_large_src_offset_is_attributed_to_src():
    rng = np.random.default_rng(1)
    d = _by(B.saturation_pm_offset(_pm_table(300, 200, 2000, 9000, -22.0, rng)))
    assert d['src']['offset'] and d['both']['offset'] and not d['ref']['offset']
    assert d['src']['d_pm_dec'] == pytest.approx(-22.0, abs=1.0)
    assert d['src']['d_pos_dec_mas'] == pytest.approx(-66.0, abs=3.0)
    assert abs(d['src']['d_pm_ra']) < 1.0
    epochs, unattributed = B.saturation_offset_epochs(list(d.values()))
    assert epochs == ['src'] and not unattributed


def test_large_ref_offset_is_attributed_to_ref():
    rng = np.random.default_rng(6)
    diags = B.saturation_pm_offset(
        _pm_table(300, 200, 2000, 9000, +22.0, rng, offset_epoch='ref'))
    assert B.saturation_offset_epochs(diags) == (['ref'], False)


@pytest.mark.parametrize('d_dec', [0.0, 2.0, 5.0])
def test_small_offset_is_not_reported(d_dec):
    rng = np.random.default_rng(2)
    diags = B.saturation_pm_offset(_pm_table(300, 200, 2000, 9000, d_dec, rng))
    assert not any(d['offset'] for d in diags)
    assert B.saturation_offset_epochs(diags) == ([], False)


def test_identical_saturated_sets_are_not_attributed():
    """All saturated stars saturated in both epochs: an offset is visible,
    but neither epoch can be blamed (the reviewer's overlap case)."""
    rng = np.random.default_rng(7)
    d = _by(B.saturation_pm_offset(_pm_table(0, 0, 400, 9000, -26.0, rng)))
    assert set(d) == {'both'} and d['both']['offset']
    assert B.saturation_offset_epochs(list(d.values())) == ([], True)


def test_both_offset_attributed_by_elimination():
    """Quintuplet v6 shape: no src-only saturated stars, a clean ref-only
    set, an offset in the saturated-in-both set -> blame src."""
    rng = np.random.default_rng(10)
    diags = B.saturation_pm_offset(_pm_table(0, 3000, 200, 6000, -39.0, rng))
    assert B.saturation_offset_epochs(diags) == (['src'], False)


def test_missing_columns_and_small_sets():
    rng = np.random.default_rng(3)
    t = _pm_table(300, 0, 0, 3000, -22.0, rng)
    t.remove_column('ref_saturated')
    d = _by(B.saturation_pm_offset(t))
    assert set(d) == {'src'} and d['src']['offset']
    t.remove_column('src_saturated')
    assert B.saturation_pm_offset(t) == []
    small = _by(B.saturation_pm_offset(_pm_table(5, 5, 2000, 3000, -22.0, rng)))
    assert set(small) == {'both'}


def test_untrustworthy_rows_are_ignored():
    rng = np.random.default_rng(4)
    t = _pm_table(300, 0, 0, 3000, -22.0, rng)
    t['trustworthy'][t['src_saturated']] = False
    assert B.saturation_pm_offset(t) == []


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


def _synthetic_field(tmp_path, rng, src_sat_frac, ref_sat_frac, n=4000):
    """Isolated stars on a 1.2" grid; src-epoch positions of the src-
    saturated stars shifted +80 mas in Dec. Saturated = brightest
    ``*_sat_frac`` of each epoch, so the src-only set is the stars between
    the two flux cuts."""
    side = int(np.ceil(np.sqrt(n)))
    gx, gy = np.meshgrid(np.arange(side), np.arange(side))
    x = (gx.ravel()[:n] - side / 2) * 1.2 + rng.normal(0, 0.1, n)
    y = (gy.ravel()[:n] - side / 2) * 1.2 + rng.normal(0, 0.1, n)
    ra0, dec0 = 266.5, -28.8
    flux = 10 ** rng.uniform(2, 5, n)
    src_sat = flux > np.percentile(flux, 100 * (1 - src_sat_frac))
    ref_sat = flux > np.percentile(flux, 100 * (1 - ref_sat_frac))

    def sky(dx, dy):
        return SkyCoord((ra0 + (x + dx) / 3600 / np.cos(np.radians(dec0))) * u.deg,
                        (dec0 + (y + dy) / 3600) * u.deg)
    noise = lambda: rng.normal(0, 0.002, n)
    src_dy = noise() + np.where(src_sat, 0.080, 0.0)
    _write_cat(tmp_path / 'src.fits', sky(noise(), src_dy), flux, src_sat)
    _write_cat(tmp_path / 'ref.fits', sky(noise(), noise()), flux, ref_sat,
               suffix='_f212n')
    out = tmp_path / 'pm.fits'
    B.build([str(tmp_path / 'src.fits')], [str(tmp_path / 'ref.fits')], 'f212n',
            2023.0, 2026.0, str(out), verbose=False)
    return Table.read(out)


def test_build_attributes_and_demotes_offset_src_satstars(tmp_path, capsys):
    """Saturated stars are 4% (src) / 2% (ref) of the field, a small share
    of the brightest-20% tie sample, so the tie follows the unsaturated
    frame as on the real fields."""
    pytest.importorskip('flystar')
    pm = _synthetic_field(tmp_path, np.random.default_rng(5), 0.04, 0.02)
    log = capsys.readouterr().out
    diag = _by(json.loads(pm.meta['sat_pm_offset_json']))
    assert diag['src']['offset']
    # 80 mas over 3 yr, sign: PM = (ref - src)/dt
    assert diag['src']['d_pm_dec'] == pytest.approx(-80 / 3, abs=3)
    assert diag['src']['d_pos_dec_mas'] == pytest.approx(-80, abs=8)
    assert 're-catalog the src epoch' in log
    assert 're-catalog the ref epoch' not in log
    assert not np.any(pm['trustworthy'] & pm['src_saturated'])
    assert np.any(pm['trustworthy_presat'] & pm['src_saturated'])
    kept = pm['trustworthy']
    assert kept.mean() > 0.5
    # the surviving trustworthy rows carry no offset: the tie was not
    # captured by the shifted stars
    assert abs(np.median(pm['pm_dec'][kept])) < 1.0
    assert abs(np.median(pm['pm_ra'][kept])) < 1.0


def test_build_does_not_blame_an_epoch_when_saturated_sets_coincide(tmp_path, capsys):
    pytest.importorskip('flystar')
    pm = _synthetic_field(tmp_path, np.random.default_rng(8), 0.03, 0.03)
    log = capsys.readouterr().out
    diag = _by(json.loads(pm.meta['sat_pm_offset_json']))
    assert set(diag) == {'both'} and diag['both']['offset']
    assert 're-catalog the' not in log
    assert log.count('WARNING') == 1 and 'cannot attribute' in log
    assert not np.any(pm['trustworthy'] & pm['src_saturated'])
    assert abs(np.median(pm['pm_dec'][pm['trustworthy']])) < 1.0


def test_trustworthy_presat_always_written(tmp_path):
    pytest.importorskip('flystar')
    rng = np.random.default_rng(9)
    pm = _synthetic_field(tmp_path, rng, 0.0, 0.0, n=900)
    assert 'trustworthy_presat' in pm.colnames
    assert np.array_equal(pm['trustworthy'], pm['trustworthy_presat'])
