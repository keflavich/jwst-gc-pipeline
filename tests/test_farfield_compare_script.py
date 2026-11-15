"""Tests for scripts/analysis/satstar_poisson/farfield_compare.py (issue #996).

The script compares the robust sigma(chi) of two PSF variants in bins of distance to
the nearest field star.  The tests use synthetic frames only, so no pipeline data are
needed.
"""
import functools
import importlib.util
import json
import os
import warnings

import numpy as np
import pytest
from astropy.io import fits

_SCRIPT = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'analysis',
                       'satstar_poisson', 'farfield_compare.py')
_spec = importlib.util.spec_from_file_location('farfield_compare_script', _SCRIPT)
F = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F)

N = 512
SKY = 20.0
SIGMA_PSF = 1.6


@functools.lru_cache(maxsize=None)
def _stars(seed=1, nstar=70):
    rng = np.random.default_rng(seed)
    pad = 20
    # well separated stars, so that the 6 px and 10 px bins do not overlap
    pos = []
    while len(pos) < nstar:
        p = rng.uniform(pad, N - pad, size=2)
        if all(np.hypot(*(p - q)) > 40 for q in pos):
            pos.append(p)
    amp = rng.uniform(300.0, 3000.0, size=nstar)
    return np.array(pos), amp


def _render(pos, amp, width):
    y, x = np.mgrid[:N, :N]
    img = np.zeros((N, N))
    for (px, py), a in zip(pos, amp):
        sl = np.s_[max(int(py) - 15, 0):int(py) + 16, max(int(px) - 15, 0):int(px) + 16]
        img[sl] += a * np.exp(-((x[sl] - px) ** 2 + (y[sl] - py) ** 2) / (2 * width ** 2))
    return img


def _frame(model_width, seed=2):
    """(resid, err, dq, model): truth has SIGMA_PSF, the model has ``model_width``.

    Rendering and the noise draw are cached; every call returns fresh copies.
    """
    return tuple(a.copy() for a in _frame_cached(model_width, seed))


@functools.lru_cache(maxsize=None)
def _frame_cached(model_width, seed):
    pos, amp = _stars()
    truth = _render(pos, amp, SIGMA_PSF)
    model = _render(pos, amp, model_width)
    err = np.sqrt(1.0 + SKY + truth)
    rng = np.random.default_rng(seed)
    data = SKY + truth + rng.normal(size=truth.shape) * err
    resid = data - model
    return resid, err, np.zeros((N, N), dtype=np.int64), model + SKY


def _by_bin(rows):
    return {(r['lo'], r['hi']): r['sigma'] for r in rows}


def test_exact_model_gives_unit_sigma_in_every_bin():
    resid, err, dq, model = _frame(SIGMA_PSF)
    rows = F.farfield_sigma(resid, err, dq, model)
    assert len(rows) == 3
    for row in rows:
        assert row['n'] > 5000
        assert row['sigma'] == pytest.approx(1.0, abs=0.02)


def test_core_width_error_raises_the_near_star_bin_only():
    resid, err, dq, model = _frame(SIGMA_PSF * 1.03)
    sig = _by_bin(F.farfield_sigma(resid, err, dq, model))
    near, far = sig[(0.0, 6.0)], sig[(10.0, np.inf)]
    assert near > far + 0.2
    assert far == pytest.approx(1.0, abs=0.02)


def _total_n(rows):
    return sum(r['n'] for r in rows)


def test_saturated_neighbourhood_is_excluded():
    resid, err, dq, model = _frame(SIGMA_PSF)
    dq[:, :200] |= F.SATURATED_BIT
    with_mask = F.farfield_sigma(resid, err, dq, model, sat_radius=15.0)
    without = F.farfield_sigma(resid, err, np.zeros_like(dq), model)
    # columns 0-214 are excluded (200 saturated + 15 px reach): fewer pixels in every bin
    for a, b in zip(with_mask, without):
        assert a['n'] < b['n']
    assert _total_n(with_mask) < 0.65 * _total_n(without)


def test_sat_radius_excludes_a_disc_around_one_saturated_pixel():
    resid, err, dq, model = _frame(SIGMA_PSF)
    dq[N // 2, N // 2] |= F.SATURATED_BIT
    base = _total_n(F.farfield_sigma(resid, err, np.zeros_like(dq), model))
    # integer lattice points within r of the centre: 1, 81 and 709 for r = 0, 5 and 15
    for radius, npix in ((0.0, 1), (5.0, 81), (15.0, 709)):
        rows = F.farfield_sigma(resid, err, dq, model, sat_radius=radius)
        assert base - _total_n(rows) == npix


@pytest.mark.parametrize('bit, excluded', [(1, True), (2, True), (4, True), (8, False)])
def test_dq_bits_excluded(bit, excluded):
    resid, err, dq, model = _frame(SIGMA_PSF)
    base = _total_n(F.farfield_sigma(resid, err, dq, model))
    dq[100:140, 100:140] |= bit
    rows = F.farfield_sigma(resid, err, dq, model, sat_radius=0.0)
    assert base - _total_n(rows) == (1600 if excluded else 0)


def test_bad_dq_pixels_are_not_counted():
    resid, err, dq, model = _frame(SIGMA_PSF)
    resid_bad = resid.copy()
    resid_bad[100:140, 100:140] = 1e6
    dq[100:140, 100:140] |= 1
    clean = F.farfield_sigma(resid, err, np.zeros_like(dq), model)
    flagged = F.farfield_sigma(resid_bad, err, dq, model)
    assert _total_n(clean) - _total_n(flagged) == 1600
    for a, b in zip(clean, flagged):
        assert b['sigma'] == pytest.approx(a['sigma'], abs=0.02)


def test_bin_edges_and_membership():
    resid, err, dq, model = _frame(SIGMA_PSF)
    rows = F.farfield_sigma(resid, err, dq, model)
    assert [(r['lo'], r['hi']) for r in rows] == [(0.0, 6.0), (6.0, 10.0), (10.0, np.inf)]
    dist = F.distance_to_peaks(F.field_star_peaks(model))
    ok = np.isfinite(F.highpass_chi(resid, err, np.zeros((N, N), bool)))
    # distances of exactly 6 and 10 px occur on the axes: lower edge closed, upper open
    assert (dist == 6.0).any() and (dist == 10.0).any()
    expected = [ok & (dist < 6), ok & (dist >= 6) & (dist < 10), ok & (dist >= 10)]
    assert [r['n'] for r in rows] == [int(m.sum()) for m in expected]


def test_options_change_the_result():
    rng = np.random.default_rng(3)
    white = rng.normal(size=(N, N))
    ones, nobad = np.ones((N, N)), np.zeros((N, N), bool)
    # a small median box leaves less than the unit variance of white noise
    assert F.robust_sigma(F.highpass_chi(white, ones, nobad, bg_size=3)) < 0.9
    assert F.robust_sigma(F.highpass_chi(white, ones, nobad, bg_size=15)) == pytest.approx(1.0, abs=0.03)
    # a smooth component at or above the box scale is removed
    y, x = np.mgrid[:N, :N]
    smooth = 20.0 * np.sin(x / 40.0)
    assert F.robust_sigma(F.highpass_chi(white + smooth, ones, nobad)) == pytest.approx(1.0, abs=0.03)

    resid, err, dq, model = _frame(SIGMA_PSF)
    low = F.farfield_sigma(resid, err, dq, model, peak_threshold=60.0)
    with pytest.warns(UserWarning, match='no field star'):
        high = F.farfield_sigma(resid, err, dq, model, peak_threshold=1e9)
    assert low[0]['n'] > 0 and high[0]['n'] == 0


def test_field_star_peaks_window_and_threshold():
    img = np.zeros((40, 40))
    img[10, 10], img[10, 13] = 100.0, 90.0      # 3 px apart: only the brighter is a maximum
    img[30, 10], img[30, 14] = 100.0, 90.0      # 4 px apart: both are maxima (7 px window)
    img[20, 30] = 50.0                          # below the threshold
    peaks = F.field_star_peaks(img, threshold=60.0)
    assert set(zip(*np.nonzero(peaks))) == {(10, 10), (30, 10), (30, 14)}


def test_no_field_star_warns_and_puts_everything_in_the_far_bin():
    resid, err, dq, model = _frame(SIGMA_PSF)
    with pytest.warns(UserWarning, match='no field star'):
        rows = F.farfield_sigma(resid, err, dq, np.zeros_like(model))
    assert rows[0]['n'] == 0 and np.isnan(rows[0]['sigma'])
    assert rows[1]['n'] == 0
    assert rows[2]['n'] > 0.9 * N * N


def test_field_star_frame_does_not_warn():
    resid, err, dq, model = _frame(SIGMA_PSF)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        F.farfield_sigma(resid, err, dq, model)


def test_robust_sigma_ignores_nonfinite_and_empty():
    assert np.isnan(F.robust_sigma([np.nan, np.inf]))
    assert F.robust_sigma([np.nan, 1.0, -1.0, 1.0, -1.0]) == pytest.approx(1.4826)


def _write_variant(directory, stem, suffix, resid, err, dq, model, date, commit):
    base = os.path.join(directory, stem + suffix)
    hdr = fits.Header()
    hdr['DATE'] = date
    hdr['GCPIPEV'] = commit
    hdr['GCTAG'] = 'tag_' + commit
    fits.HDUList([fits.PrimaryHDU(header=hdr),
                  fits.ImageHDU(resid.astype('f4'), name='SCI'),
                  fits.ImageHDU(err.astype('f4'), name='ERR'),
                  fits.ImageHDU(dq.astype('i4'), name='DQ')]).writeto(base + '_residual.fits')
    # importing jwst_gc_pipeline installs a writeto hook that stamps the running commit
    # into GCPIPEV/GCTAG, so the fixture values are set again on the file afterwards
    fits.setval(base + '_residual.fits', 'GCPIPEV', value=commit)
    fits.setval(base + '_residual.fits', 'GCTAG', value='tag_' + commit)
    fits.HDUList([fits.PrimaryHDU(),
                  fits.ImageHDU(model.astype('f4'), name='SCI')]).writeto(base + '_model.fits')


def test_main_compares_two_variants_from_disk(tmp_path, capsys):
    stem = 'jw10678-o061_t001_nircam_clear-f480m-nrcblong_visit001_vgroup02101_exp00004_resbgsub'
    # the hybrid variant has the exact core, the STPSF variant a 3 % wider one
    resid_s, err, dq, model_s = _frame(SIGMA_PSF * 1.03)
    resid_h, _, _, model_h = _frame(SIGMA_PSF)
    _write_variant(str(tmp_path), stem, F.STPSF_SUFFIX, resid_s, err, dq, model_s,
                   '2026-09-26T08:32:38.839', 'c48ff2a-dirty')
    _write_variant(str(tmp_path), stem, F.HYBRID_SUFFIX, resid_h, err, dq, model_h,
                   '2026-11-07T23:07:12.463', '921d1ef')
    out_json = tmp_path / 'farfield.json'
    assert F.main(['--dir', str(tmp_path), '--json', str(out_json), stem]) == 0
    text = capsys.readouterr().out
    assert 'stpsf: DATE 2026-09-26' in text and 'hybrid: DATE 2026-11-07' in text
    assert 'GCPIPEV c48ff2a-dirty' in text and 'GCPIPEV 921d1ef' in text
    assert 'different pipeline commits' in text
    assert '0-6 px' in text and '10-inf px' in text
    assert out_json.exists()

    results = F.compare_frame(str(tmp_path), stem, {'stpsf': F.STPSF_SUFFIX,
                                                    'hybrid': F.HYBRID_SUFFIX})
    near_s = results['stpsf']['bins'][0]['sigma']
    near_h = results['hybrid']['bins'][0]['sigma']
    assert near_s > near_h + 0.2


def test_missing_variant_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        F.main(['--dir', str(tmp_path), 'no_such_frame_resbgsub'])


def _frame_dirs(tmp_path):
    stem = 'jw10678-o061_t001_nircam_clear-f480m-nrcblong_visit001_vgroup02101_exp00004_resbgsub'
    resid_s, err, dq, model_s = _frame(SIGMA_PSF * 1.03)
    resid_h, _, _, model_h = _frame(SIGMA_PSF)
    # one extra bright star in the hybrid model only: its star set differs from STPSF's
    model_h = model_h + _render([(256.5, 256.5)], [2000.0], SIGMA_PSF)
    _write_variant(str(tmp_path), stem, F.STPSF_SUFFIX, resid_s, err, dq, model_s, 'd1', 'aaa')
    _write_variant(str(tmp_path), stem, F.HYBRID_SUFFIX, resid_h, err, dq, model_h, 'd2', 'bbb')
    return stem


def test_star_model_gives_both_variants_the_same_pixel_sets(tmp_path):
    stem = _frame_dirs(tmp_path)
    suffixes = {'stpsf': F.STPSF_SUFFIX, 'hybrid': F.HYBRID_SUFFIX}
    own = F.compare_frame(str(tmp_path), stem, suffixes)
    assert ([r['n'] for r in own['stpsf']['bins']]
            != [r['n'] for r in own['hybrid']['bins']])
    for ref in ('stpsf', 'hybrid'):
        common = F.compare_frame(str(tmp_path), stem, suffixes, star_model=ref)
        assert ([r['n'] for r in common['stpsf']['bins']]
                == [r['n'] for r in common['hybrid']['bins']])
        assert common['hybrid']['star_model'] == ref
        # the reference variant is unchanged by the choice
        assert common[ref]['bins'] == own[ref]['bins']
    with pytest.raises(ValueError):
        F.compare_frame(str(tmp_path), stem, suffixes, star_model='nope')


def test_main_passes_the_cli_options_through(tmp_path, monkeypatch, capsys):
    stem = _frame_dirs(tmp_path)
    seen = []
    real = F.farfield_sigma

    def spy(resid, err, dq, model, **kwargs):
        seen.append(kwargs)
        return real(resid, err, dq, model, **kwargs)

    monkeypatch.setattr(F, 'farfield_sigma', spy)
    F.main(['--dir', str(tmp_path), '--peak-threshold', '75', '--bg-size', '9',
            '--sat-radius', '3', '--star-model', 'stpsf', stem])
    assert len(seen) == 2
    for kw in seen:
        assert kw['peak_threshold'] == 75.0 and kw['bg_size'] == 9
        assert kw['sat_radius'] == 3.0
    out = capsys.readouterr().out
    # n of both variants is printed, joined by a slash
    assert any('n=' in line and '/' in line.split('n=')[1] for line in out.splitlines())


def _rows(sigma_a, sigma_b):
    bins = lambda s: [{'lo': 0.0, 'hi': 6.0, 'n': 3, 'sigma': s},
                      {'lo': 6.0, 'hi': np.inf, 'n': 0, 'sigma': np.nan}]
    return {'frame': {'stpsf': {'date': '', 'gctag': '', 'gcpipev': 'a', 'bins': bins(sigma_a)},
                      'hybrid': {'date': '', 'gctag': '', 'gcpipev': 'a', 'bins': bins(sigma_b)}}}


def test_format_report_survives_a_zero_sigma_variant():
    text = F.format_report(_rows(0.0, 1.0))
    assert 'nan' in text


def test_json_output_is_strict(tmp_path):
    stem = _frame_dirs(tmp_path)
    out_json = tmp_path / 'ff.json'
    F.main(['--dir', str(tmp_path), '--json', str(out_json), stem])

    def forbid(const):
        raise AssertionError(f'non-standard JSON constant {const}')

    data = json.loads(out_json.read_text(), parse_constant=forbid)
    far = data[stem]['stpsf']['bins'][-1]
    assert far['hi'] is None and far['lo'] == 10.0
