"""Tests for scripts/analysis/satstar_poisson/farfield_compare.py (issue #996).

The script compares the robust sigma(chi) of two PSF variants in bins of distance to
the nearest field star.  The tests use synthetic frames only, so no pipeline data are
needed.
"""
import importlib.util
import os

import numpy as np
import pytest
from astropy.io import fits

_SCRIPT = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'analysis',
                       'satstar_poisson', 'farfield_compare.py')
_spec = importlib.util.spec_from_file_location('farfield_compare_script', _SCRIPT)
F = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F)

N = 1024
SKY = 20.0
SIGMA_PSF = 1.6


def _stars(seed=1, nstar=250):
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
    """(resid, err, dq, model): truth has SIGMA_PSF, the model has ``model_width``."""
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


def test_saturated_neighbourhood_is_excluded():
    resid, err, dq, model = _frame(SIGMA_PSF)
    dq[:, :200] |= F.SATURATED_BIT
    with_mask = F.farfield_sigma(resid, err, dq, model, sat_radius=15.0)
    without = F.farfield_sigma(resid, err, np.zeros_like(dq), model)
    # columns 0-214 are masked, bad-DQ pixels too: fewer pixels in every bin
    for a, b in zip(with_mask, without):
        assert a['n'] < b['n']
    # a saturated pixel at the very edge of the mask excludes nothing beyond sat_radius
    assert sum(r['n'] for r in with_mask) < 0.85 * sum(r['n'] for r in without)


def test_bad_dq_pixels_are_not_counted():
    resid, err, dq, model = _frame(SIGMA_PSF)
    resid_bad = resid.copy()
    resid_bad[100:140, 100:140] = 1e6
    dq[100:140, 100:140] |= 1
    clean = F.farfield_sigma(resid, err, np.zeros_like(dq), model)
    flagged = F.farfield_sigma(resid_bad, err, dq, model)
    for a, b in zip(clean, flagged):
        assert b['sigma'] == pytest.approx(a['sigma'], abs=0.02)


def test_no_field_star_puts_everything_in_the_far_bin():
    resid, err, dq, model = _frame(SIGMA_PSF)
    rows = F.farfield_sigma(resid, err, dq, np.zeros_like(model))
    assert rows[0]['n'] == 0 and np.isnan(rows[0]['sigma'])
    assert rows[1]['n'] == 0
    assert rows[2]['n'] > 0.9 * N * N


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
