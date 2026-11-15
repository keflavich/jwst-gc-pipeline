"""Bright pre-subtraction before the m3..m7 detection (env DAOPHOT_BRIGHT_PRESUB_PEAK_NOISE, #991).

The m3..m7 detection image is the frame itself.  A bright unsubtracted star
inflates ``compute_local_noise_map`` (the squared high-pass residual is
Gaussian-smoothed) over ~12 px, so the local-S/N cut drops the faint neighbours
there.  These tests pin that mechanism, the switch's parsing, and the
bright-seed selection that feeds the pre-fit.
"""
import numpy as np
import pytest
from astropy.table import Table

from jwst_gc_pipeline.photometry import cataloging as C
from jwst_gc_pipeline.photometry import crowdsource_catalogs_long as L

VAR = 'DAOPHOT_BRIGHT_PRESUB_PEAK_NOISE'


def _gauss(shape, x0, y0, peak, sigma=1.0):
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    return peak * np.exp(-0.5 * ((xx - x0) ** 2 + (yy - y0) ** 2) / sigma ** 2)


def _frame(seed=0, ny=96, nx=96):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 5.0, (ny, nx))


@pytest.mark.parametrize('raw, want', [(None, 0.0), ('', 0.0), ('  ', 0.0),
                                       ('0', 0.0), ('30', 30.0), (' 12.5 ', 12.5)])
def test_peak_noise_switch_parses(monkeypatch, raw, want):
    if raw is None:
        monkeypatch.delenv(VAR, raising=False)
    else:
        monkeypatch.setenv(VAR, raw)
    assert C._bright_presub_peak_noise() == want


@pytest.mark.parametrize('raw', ['abc', '-1', 'nan', 'inf', 'on'])
def test_peak_noise_switch_rejects_bad_values(monkeypatch, raw):
    monkeypatch.setenv(VAR, raw)
    with pytest.raises(ValueError, match=VAR):
        C._bright_presub_peak_noise()


def test_bright_star_raises_the_noise_map_at_a_neighbour_until_it_is_subtracted():
    """The neighbour's S/N falls below the 3-sigma cut only while the bright star is in the frame."""
    star = _gauss((96, 96), 48, 48, 5.0e4)
    neighbour = _gauss((96, 96), 58, 48, 100.0)          # 10 px from the bright star
    base = _frame()
    with_star = base + star + neighbour
    subtracted = base + neighbour                        # bright model removed exactly
    snr = {}
    for name, img in (('with', with_star), ('sub', subtracted)):
        noise = L.compute_local_noise_map(img, smooth_sigma_pix=3.0)
        snr[name] = img[48, 58] / noise[48, 58]
    assert snr['with'] < 3.0
    assert snr['sub'] > 3.0


def test_select_marks_only_seeds_above_the_noise_multiple():
    data = _frame() + _gauss((96, 96), 30, 30, 5.0e4) + _gauss((96, 96), 70, 70, 60.0)
    seed = Table({'x_init': [30.0, 70.0, 500.0, 29.6],
                  'y_init': [30.0, 70.0, 10.0, 30.4]})
    bright, noise_ref = C._bright_presub_select(seed, data, 30.0)
    assert noise_ref > 0
    assert bright.tolist() == [True, False, False, True]    # faint star and the off-frame seed stay out


def test_select_threshold_scales_with_the_frame_noise():
    star = _gauss((96, 96), 40, 40, 400.0)
    seed = Table({'x_init': [40.0], 'y_init': [40.0]})
    quiet, nq = C._bright_presub_select(seed, _frame() * 0.2 + star, 30.0)
    loud, nl = C._bright_presub_select(seed, _frame() * 5.0 + star, 30.0)
    assert nl > nq
    assert quiet.tolist() == [True] and loud.tolist() == [False]
