"""Tests for the satstar/daophot luminosity-function continuity check (#925).

Synthetic catalogs draw magnitudes from a power-law LF
(``N(m) ∝ 10**(slope * m)``) with the satstar channel covering stars
brighter than a seam at 12.0.  The defects #925 measured on GC Treasury
o132 are then injected: a hole just faint of the seam, a level step across
it, a pile-up of mis-measured stars just bright of it, and satstar rows pooled
in from other observations' sky.
"""
import os

import numpy as np
import pytest
from astropy.table import Table

from jwst_gc_pipeline.photometry.lf_continuity import (
    assert_lf_continuity, find_seam, lf_continuity, main, occupancy_footprint)

SEAM = 12.0
O132 = ('/orange/adamginsburg/jwst/gc-treasury/catalogs/'
        'basic_merged_indivexp_photometry_tables_merged_resbgsub_m8_o132.fits')
#: Size of the shipped (pre-#925-fix) o132 m8 catalog, 2026-09-16.  The
#: real-data test describes THAT catalog; once o132 is re-merged with the fix
#: the file changes and the test skips instead of reporting the fix as a failure.
O132_SHIPPED_SIZE = 142634880


def _shipped_o132_available():
    return (os.path.exists(O132)
            and os.path.getsize(O132) == O132_SHIPPED_SIZE)


def _draw_lf(rng, n, lo=9.0, hi=16.0, slope=0.3):
    """Inverse-CDF sample of N(m) ∝ 10**(slope m) on [lo, hi)."""
    a = slope * np.log(10)
    u = rng.uniform(size=n)
    return np.log(np.exp(a * lo) + u * (np.exp(a * hi) - np.exp(a * lo))) / a


def _cat(mag, rng, ra=None, dec=None):
    n = mag.size
    if ra is None:
        # a 3' x 3' tile at the Galactic Centre
        ra = 266.4 + rng.uniform(0, 3 / 60, n) / np.cos(np.deg2rad(-28.9))
        dec = -28.9 + rng.uniform(0, 3 / 60, n)
    return Table({'mag_vega_f480m': mag,
                  'replaced_saturated_f480m': mag < SEAM,
                  'forced_filled_f480m': np.zeros(n, bool),
                  'skycoord_ref.ra': ra, 'skycoord_ref.dec': dec})


def _smooth(seed=0, n=60000):
    rng = np.random.default_rng(seed)
    return _draw_lf(rng, n), rng


def test_smooth_lf_passes():
    mag, rng = _smooth()
    r = lf_continuity(_cat(mag, rng), 'f480m')
    assert r['verdict'] == 'pass', r
    assert r['seam'] == pytest.approx(SEAM)
    assert 0.8 < r['ratio_faint'] < 1.25
    assert 0.8 < r['step'] < 1.25


def test_smooth_lf_no_false_alarms_over_seeds():
    """At the default p_fail the smooth LF never fails (100 realisations)."""
    for seed in range(100):
        mag, rng = _smooth(seed, n=20000)
        r = lf_continuity(_cat(mag, rng), 'f480m', footprint=None)
        assert r['verdict'] == 'pass', (seed, r)


def test_hole_faint_of_seam_fails():
    """o132 shape: stars at seam .. seam+0.35 lost from both channels."""
    mag, rng = _smooth()
    mag = mag[~((mag >= SEAM) & (mag < SEAM + 0.35))]
    r = lf_continuity(_cat(mag, rng), 'f480m')
    assert r['verdict'] == 'fail', r
    assert 'ratio_faint' in r['failed']
    assert r['ratio_faint'] < 0.05


def test_partial_hole_fails():
    """Losing 70% of the stars just faint of the seam also fails."""
    mag, rng = _smooth()
    hole = (mag >= SEAM) & (mag < SEAM + 0.3) & (rng.uniform(size=mag.size) < 0.7)
    r = lf_continuity(_cat(mag[~hole], rng), 'f480m')
    assert r['verdict'] == 'fail', r


def test_level_step_fails():
    """Faint channel keeps only 35% of its stars: a step, not a hole."""
    mag, rng = _smooth()
    keep = (mag < SEAM) | (rng.uniform(size=mag.size) < 0.35)
    r = lf_continuity(_cat(mag[keep], rng), 'f480m')
    assert r['verdict'] == 'fail', r
    assert 'step' in r['failed']
    assert r['step'] > 2


def test_bright_pileup_fails():
    """#925 defect 3: stars just faint of the seam measured 0.3 mag bright
    pile up in the satstar window and leave a gap behind."""
    mag, rng = _smooth()
    moved = (mag >= SEAM) & (mag < SEAM + 0.3)
    mag = np.where(moved, mag - 0.3, mag)
    r = lf_continuity(_cat(mag, rng), 'f480m', seam=SEAM)
    assert r['verdict'] == 'fail', r
    assert r['ratio_bright'] > 1.5


def test_thin_bins_are_insufficient_not_fail():
    """A few hundred stars with a real hole: too few to decide, and the
    verdict says so instead of failing."""
    mag, rng = _smooth(n=1000)
    mag = mag[~((mag >= SEAM) & (mag < SEAM + 0.35))]
    r = lf_continuity(_cat(mag, rng), 'f480m', footprint=None, seam=SEAM)
    assert r['verdict'] == 'insufficient', r
    assert r['exp_faint'] < 20


def test_too_few_satstars_to_locate_seam_is_insufficient():
    mag, rng = _smooth(n=300)
    r = lf_continuity(_cat(mag, rng), 'f480m', footprint=None)
    assert r['n_sat'] > 0
    assert r['verdict'] == 'insufficient', r


def test_thin_smooth_never_fails():
    for seed in range(200):
        mag, rng = _smooth(seed, n=3000)
        r = lf_continuity(_cat(mag, rng), 'f480m', footprint=None)
        assert r['verdict'] != 'fail', (seed, r)


def test_no_satstar_population():
    mag, rng = _smooth()
    t = _cat(mag, rng)
    t['replaced_saturated_f480m'] = False
    assert lf_continuity(t, 'f480m')['verdict'] == 'no-sat-population'


def test_forced_fill_rows_excluded():
    """Forced-fill rows do not fill a hole: they are not detections."""
    mag, rng = _smooth()
    hole = (mag >= SEAM) & (mag < SEAM + 0.35)
    t = _cat(mag, rng)
    t['forced_filled_f480m'] = hole
    assert lf_continuity(t, 'f480m')['verdict'] == 'fail'
    assert lf_continuity(t, 'f480m', exclude_forced=False)['verdict'] == 'pass'


def test_pooled_out_of_footprint_satstars_ignored():
    """#925 defect 1: satstar rows from other tiles, far off this tile, land
    in the merged catalog.  Restricted to the footprint the LF is clean;
    with every row it is not."""
    mag, rng = _smooth()
    tile = _cat(mag, rng)
    npool = 20000
    pmag = rng.uniform(10.0, SEAM, npool)
    pool = Table({'mag_vega_f480m': pmag,
                  'replaced_saturated_f480m': np.ones(npool, bool),
                  'forced_filled_f480m': np.zeros(npool, bool),
                  'skycoord_ref.ra': 266.0 + rng.uniform(0, 0.5, npool),
                  'skycoord_ref.dec': -29.5 + rng.uniform(0, 0.3, npool)})
    from astropy.table import vstack
    t = vstack([tile, pool])
    fp = occupancy_footprint(t)
    assert fp[:len(tile)].mean() > 0.9
    assert fp[len(tile):].mean() < 0.01
    assert lf_continuity(t, 'f480m')['verdict'] == 'pass'
    assert lf_continuity(t, 'f480m', footprint=None)['verdict'] == 'fail'


def test_find_seam():
    mag = np.array([10.0] * 10 + [11.93] * 10 + [12.5] * 10)
    rep = mag < 12
    assert find_seam(mag, rep) == pytest.approx(12.0)
    assert np.isnan(find_seam(mag, np.zeros_like(rep)))


def test_assert_lf_continuity_raises_and_names_band():
    mag, rng = _smooth()
    mag = mag[~((mag >= SEAM) & (mag < SEAM + 0.35))]
    with pytest.raises(AssertionError, match='f480m: FAIL'):
        assert_lf_continuity(_cat(mag, rng))
    mag, rng = _smooth()
    res = assert_lf_continuity(_cat(mag, rng))
    assert [r['band'] for r in res] == ['f480m']


def test_cli_exit_status(tmp_path, capsys):
    mag, rng = _smooth()
    good = tmp_path / 'good.fits'
    _cat(mag, rng).write(good)
    bad = tmp_path / 'bad.fits'
    _cat(mag[~((mag >= SEAM) & (mag < SEAM + 0.35))], rng).write(bad)
    assert main([str(good)]) == 0
    assert main([str(bad)]) == 1
    assert 'FAIL' in capsys.readouterr().out


@pytest.mark.skipif(not _shipped_o132_available(),
                    reason='shipped (pre-fix) GC Treasury o132 m8 catalog not available')
def test_real_o132_f480m_hole_fails():
    """Read-only check on the shipped o132 m8 catalog (#925): F480M fails
    on the hole faint of the 12.0 seam; F212N (seam ~15.6) does not."""
    from jwst_gc_pipeline.photometry.lf_continuity import _read_columns
    cols, bands = _read_columns(O132, ['f212n', 'f480m'])
    fp = occupancy_footprint(cols)
    r = lf_continuity(cols, 'f480m', footprint=fp)
    assert r['verdict'] == 'fail'
    assert r['seam'] == pytest.approx(12.0, abs=0.1)
    assert r['ratio_faint'] < 0.1
    assert lf_continuity(cols, 'f212n', footprint=fp)['verdict'] == 'pass'
