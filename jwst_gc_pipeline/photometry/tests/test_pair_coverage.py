"""Tests for the two-band pair-coverage (B2 share) check (#1269).

Synthetic catalogs with known BB / BSAT / B2 / R4 counts.
"""
import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from jwst_gc_pipeline.photometry.pair_coverage import (
    format_result, main, pair_coverage)

COLS = ['independently_detected', 'forced_filled', 'replaced_saturated']


def _rows(n, m2=14.0, i2=True, i4=True, f2=False, f4=False, rs2=False,
          rs4=False, q2=0.1, q4=0.1, m4=13.0):
    return dict(independently_detected_f212n=[i2] * n,
                independently_detected_f480m=[i4] * n,
                forced_filled_f212n=[f2] * n, forced_filled_f480m=[f4] * n,
                replaced_saturated_f212n=[rs2] * n,
                replaced_saturated_f480m=[rs4] * n,
                qfit_f212n=[q2] * n, qfit_f480m=[q4] * n,
                mag_vega_f212n=[m2] * n, mag_vega_f480m=[m4] * n)


def _cat(*blocks):
    out = {}
    for b in blocks:
        for k, v in b.items():
            out.setdefault(k, []).extend(v)
    return Table({k: np.asarray(v) for k, v in out.items()})


def _mixed():
    return _cat(
        _rows(60),                                   # BB
        _rows(5, rs2=True, q2=0.9),                  # BSAT (qfit waived)
        _rows(4, rs4=True, q4=0.5),                  # BB: F480M replaced_saturated, qfit waived
        _rows(3, f4=True),                           # B2: forced-filled F480M
        _rows(2, q4=0.5),                            # B2: poor F480M qfit
        _rows(2, i4=False),                          # B2: no independent F480M
        _rows(7, i2=False),                          # R4
        _rows(9, m2=16.0),                           # outside the magnitude range
    )


def test_channel_counts():
    r = pair_coverage(_mixed(), min_count=1)
    assert (r['n_BB'], r['n_BSAT'], r['n_B2'], r['n_R4']) == (64, 5, 7, 7)
    assert r['b2_share'] == pytest.approx(7 / 76)
    assert r['verdict'] == 'pass'


def test_replaced_saturated_f480m_with_bad_qfit_is_clean():
    r = pair_coverage(_cat(_rows(60, rs4=True, q4=0.5)))
    assert r['n_B2'] == 0 and r['n_BB'] == 60


def test_nan_qfit_counts_as_unclean():
    r = pair_coverage(_cat(_rows(60), _rows(10, q4=np.nan)))
    assert r['n_B2'] == 10


def test_mag_range_is_half_open():
    r = pair_coverage(_cat(_rows(60, m2=13.0), _rows(60, m2=14.5)))
    assert r['n_BB'] == 60


@pytest.mark.parametrize('n_b2, verdict', [(5, 'pass'), (15, 'warn'), (30, 'fail')])
def test_verdict_thresholds(n_b2, verdict):
    r = pair_coverage(_cat(_rows(100 - n_b2), _rows(n_b2, i4=False)))
    assert r['b2_share'] == pytest.approx(n_b2 / 100)
    assert r['verdict'] == verdict


def test_thresholds_are_parameters():
    cat = _cat(_rows(85), _rows(15, i4=False))
    assert pair_coverage(cat, warn_share=0.2)['verdict'] == 'pass'
    assert pair_coverage(cat, fail_share=0.1)['verdict'] == 'fail'


def test_too_few_rows_is_insufficient():
    r = pair_coverage(_cat(_rows(5), _rows(5, i4=False)))
    assert r['verdict'] == 'insufficient'
    assert pair_coverage(_cat(_rows(7, i2=False)))['b2_share'] != r['b2_share']


def test_other_band_pair():
    cat = Table({f'{s}_f187n': v for s, v in (
        ('independently_detected', [True] * 60), ('forced_filled', [False] * 60),
        ('replaced_saturated', [False] * 60), ('qfit', [0.1] * 60),
        ('mag_vega', [14.0] * 60))})
    for s in ('independently_detected', 'forced_filled', 'replaced_saturated',
              'qfit', 'mag_vega'):
        cat[f'{s}_f405n'] = cat[f'{s}_f187n']
    r = pair_coverage(cat, 'f187n', 'f405n')
    assert r['n_BB'] == 60


def test_missing_column_error_names_it():
    cat = _mixed()
    cat.remove_column('qfit_f480m')
    with pytest.raises(KeyError, match='qfit_f480m'):
        pair_coverage(cat)


def test_format_result_mentions_counts():
    line = format_result(pair_coverage(_mixed(), min_count=1))
    assert 'PASS' in line and 'B2 7' in line


def _write(path, cat):
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU(cat)]).writeto(path)


def test_cli_exit_status(tmp_path, capsys):
    good = tmp_path / 'good.fits'
    bad = tmp_path / 'bad.fits'
    _write(good, _cat(_rows(95), _rows(5, i4=False)))
    _write(bad, _cat(_rows(60), _rows(40, i4=False)))
    assert main([str(good)]) == 0
    assert main([str(bad)]) == 1
    assert main([str(good), '--json']) == 0
    out = capsys.readouterr().out
    assert '"verdict": "fail"' not in out.splitlines()[-1]
    assert main([str(bad), '--fail-share', '0.5']) == 0
