"""Accepted-satstar saturated footprint as NaN in the written residual.

The per-frame residual is the raw crf SCI minus the satstar model.  In the
DQ-SATURATED footprint of an accepted satstar the SCI rate comes from the early
groups and under-reads the model, so ``rate - model`` is not a measurement there
(the residual "donuts"; o102 F480M: peaks 141 / 231 / 638 MJy/sr within 1" by
flux tercile).  Env ``GC_RESID_NAN_SATSTAR_FOOTPRINT`` (default OFF) reports
those pixels as NaN in the WRITTEN residual of the passes
``GC_RESID_NAN_SATSTAR_PASSES`` lists, m7 only by default: the m3-m6 residuals
seed the next pass and have not been run with NaN in them.

These tests pin:

* the mask is DQ SATURATED & model > min_model, and None in gives None out;
* the gate is off by default, m7-only by default, honours the pass list, and a
  malformed value raises (the pass list only when the switch is on);
* ``_save_manual_pass`` writes NaN at exactly the mask in m7 and leaves every
  other pass, and every default-off run, equal to the unmasked residual, while
  the model file and the ``modelsub_bkg`` input stay unmasked.
"""
import os
import types

import numpy as np
import pytest

from jwst_gc_pipeline.photometry import cataloging as C

FOOTPRINT = 'GC_RESID_NAN_SATSTAR_FOOTPRINT'
PASSES = 'GC_RESID_NAN_SATSTAR_PASSES'
MIN_MODEL = 'GC_RESID_NAN_SATSTAR_MIN_MODEL'
SAT = int(C._L.dqflags.pixel['SATURATED'])
DNU = int(C._L.dqflags.pixel['DO_NOT_USE'])
JUMP = int(C._L.dqflags.pixel['JUMP_DET'])
SHAPE = (12, 12)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in [n for n in os.environ if n.startswith('GC_RESID_NAN_')]:
        monkeypatch.delenv(name, raising=False)


def _frame():
    """DQ and satstar model for one 12x12 frame.

    SATURATED block rows/cols 3:8, one of its pixels also DO_NOT_USE.  Model:
    500 over the block's inner 3x3 (4:7), 10 along its top row (below the
    default floor of 20), 50 on an unflagged wing pixel, 500 under a block
    that is flagged but not saturated.  So the default mask is the inner 3x3.
    """
    dq = np.zeros(SHAPE, dtype=np.uint32)
    dq[3:8, 3:8] = SAT
    dq[5, 5] |= DNU
    dq[9:11, 9:11] = DNU | JUMP
    model = np.zeros(SHAPE)
    model[4:7, 4:7] = 500.0
    model[3, 3:8] = 10.0
    model[1, 1] = 50.0
    model[9:11, 9:11] = 500.0
    return dq, model


# -- the mask ---------------------------------------------------------------

def test_mask_is_saturated_bit_and_model_above_floor():
    dq, model = _frame()
    mask = C._satstar_footprint_nan_mask(dq, model)
    want = np.zeros(SHAPE, bool)
    want[4:7, 4:7] = True
    assert mask.dtype == bool and mask.shape == SHAPE
    np.testing.assert_array_equal(mask, want)


def test_mask_tests_the_saturated_bit_not_equality():
    dq, model = _frame()
    assert dq[5, 5] == SAT | DNU
    assert C._satstar_footprint_nan_mask(dq, model)[5, 5]


def test_mask_needs_the_saturated_bit():
    dq, model = _frame()
    dq &= ~np.uint32(SAT)
    assert not C._satstar_footprint_nan_mask(dq, model).any()


@pytest.mark.parametrize('min_model, n', [(0.0, 14), (20.0, 9), (499.9, 9),
                                          (500.0, 0)])
def test_min_model_is_a_strict_threshold(min_model, n):
    dq, model = _frame()
    assert C._satstar_footprint_nan_mask(dq, model, min_model).sum() == n


def test_default_min_model_is_20():
    dq, model = _frame()
    np.testing.assert_array_equal(
        C._satstar_footprint_nan_mask(dq, model),
        C._satstar_footprint_nan_mask(dq, model, 20.0))


def test_unmodelled_saturated_pixels_are_not_masked():
    dq, _ = _frame()
    assert not C._satstar_footprint_nan_mask(dq, np.zeros(SHAPE)).any()


@pytest.mark.parametrize('dq_none, model_none', [(True, False), (False, True),
                                                 (True, True)])
def test_none_input_gives_none(dq_none, model_none):
    dq, model = _frame()
    assert C._satstar_footprint_nan_mask(
        None if dq_none else dq, None if model_none else model) is None


# -- the gate ---------------------------------------------------------------

ALL_PASSES = C._RESID_NAN_PASS_LABELS


def _applies():
    return {p for p in ALL_PASSES if C._resid_nan_satstar_applies(p)}


def test_pass_labels_cover_what_the_writer_is_called_with():
    assert ALL_PASSES == ('m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7')


def test_gate_is_off_by_default_for_every_pass():
    assert _applies() == set()


def test_switch_alone_scopes_to_m7(monkeypatch):
    monkeypatch.setenv(FOOTPRINT, '1')
    assert _applies() == {'m7'}


@pytest.mark.parametrize('raw, want', [
    ('m7', {'m7'}), ('m6,m7', {'m6', 'm7'}), ('m6 m7', {'m6', 'm7'}),
    ('m6, m7', {'m6', 'm7'}), (' M7 ', {'m7'}), ('m3', {'m3'}),
    ('m1,m2', {'m1', 'm2'}), ('', {'m7'}), ('   ', {'m7'})])
def test_pass_list_is_honoured(monkeypatch, raw, want):
    monkeypatch.setenv(FOOTPRINT, '1')
    monkeypatch.setenv(PASSES, raw)
    assert _applies() == want


@pytest.mark.parametrize('bad', ['m8', 'm12', 'm0', '7', 'all', 'm7;m6',
                                 'm6,m77'])
def test_malformed_pass_list_raises(monkeypatch, bad):
    monkeypatch.setenv(FOOTPRINT, '1')
    monkeypatch.setenv(PASSES, bad)
    with pytest.raises(ValueError, match=PASSES):
        C._resid_nan_satstar_applies('m7')


def test_stray_pass_list_cannot_break_a_default_run(monkeypatch):
    monkeypatch.setenv(PASSES, 'bogus')
    assert C._resid_nan_satstar_applies('m7') is False


@pytest.mark.parametrize('raw, want', [
    ('1', True), ('true', True), ('ON', True), ('0', False), ('off', False),
    ('', False), ('  ', False)])
def test_switch_values(monkeypatch, raw, want):
    monkeypatch.setenv(FOOTPRINT, raw)
    assert C._resid_nan_satstar_applies('m7') is want


def test_switch_typo_raises(monkeypatch):
    monkeypatch.setenv(FOOTPRINT, 'maybe')
    with pytest.raises(ValueError, match=FOOTPRINT):
        C._resid_nan_satstar_applies('m7')


def test_min_model_default_and_override(monkeypatch):
    assert C._resid_nan_satstar_min_model() == 20.0
    monkeypatch.setenv(MIN_MODEL, '  ')
    assert C._resid_nan_satstar_min_model() == 20.0
    monkeypatch.setenv(MIN_MODEL, '5.5')
    assert C._resid_nan_satstar_min_model() == 5.5
    monkeypatch.setenv(MIN_MODEL, '0')
    assert C._resid_nan_satstar_min_model() == 0.0


@pytest.mark.parametrize('bad', ['abc', '-1', 'nan', 'inf'])
def test_bad_min_model_raises(monkeypatch, bad):
    monkeypatch.setenv(MIN_MODEL, bad)
    with pytest.raises(ValueError, match=MIN_MODEL):
        C._resid_nan_satstar_min_model()


# -- the writer -------------------------------------------------------------

def _run(monkeypatch, label, model=True):
    """Run the real ``_save_manual_pass`` on the 12x12 frame with the file
    writers stubbed.  Returns what the writers were given, the unmasked
    residual, and the DQ / model the frame was built from."""
    dq, satmodel = _frame()
    data = 100.0 + np.arange(SHAPE[0] * SHAPE[1], dtype=float).reshape(SHAPE)
    modsky = np.full(SHAPE, 2.0)
    seen = types.SimpleNamespace(bkg=None, files={})

    def attach(result, residual, ctx, options, label=''):
        seen.bkg = np.array(residual)
        return result

    def save_residual(filename, path, array):
        seen.files[path.rsplit('_', 1)[-1]] = np.array(array)

    monkeypatch.setattr(C, '_attach_residual_background', attach)
    monkeypatch.setattr(C._L, 'save_photutils_results',
                        lambda *args, **kwargs: 'saved')
    monkeypatch.setattr(C._L, 'save_residual_datamodel', save_residual)
    ctx = types.SimpleNamespace(
        out_basepath='/nonexistent', original_data=data,
        satstar_model_subtracted=satmodel if model else None, dqarr=dq,
        ww=None, filename='x_crf.fits', im1=None, filtername='F480M',
        module='nrca', desat='', bgsub='', exposure_='', visitid_='',
        vgroupid_='', epsf_='', blur_='', group='', background_map=None,
        proposal_id=10678, field='102', inst_token='nircam', pupil='clear')
    saved = C._save_manual_pass(ctx, types.SimpleNamespace(meta={}), modsky,
                                types.SimpleNamespace(blur=False), label,
                                'nrca1')
    assert saved == 'saved'
    plain = data - (satmodel if model else 0.0) - modsky
    seen.modsky = modsky
    return seen, plain, dq, satmodel


def test_m7_residual_is_nan_exactly_on_the_mask(monkeypatch, capsys):
    monkeypatch.setenv(FOOTPRINT, '1')
    seen, plain, dq, satmodel = _run(monkeypatch, 'm7')
    mask = C._satstar_footprint_nan_mask(dq, satmodel)
    out = seen.files['residual.fits']
    assert mask.sum() == 9
    np.testing.assert_array_equal(np.isnan(out), mask)
    np.testing.assert_array_equal(out[~mask], plain[~mask])
    assert '[m7] residual: 9 DQ-SATURATED px' in capsys.readouterr().out


@pytest.mark.parametrize('label', ['m1', 'm2', 'm3', 'm4', 'm5', 'm6'])
def test_other_passes_are_untouched_by_default(monkeypatch, label):
    monkeypatch.setenv(FOOTPRINT, '1')
    seen, plain, _, _ = _run(monkeypatch, label)
    np.testing.assert_array_equal(seen.files['residual.fits'], plain)


def test_default_off_writes_the_plain_residual_at_m7(monkeypatch):
    seen, plain, _, _ = _run(monkeypatch, 'm7')
    np.testing.assert_array_equal(seen.files['residual.fits'], plain)


def test_pass_list_extends_the_nan_to_m6_and_no_further(monkeypatch):
    monkeypatch.setenv(FOOTPRINT, '1')
    monkeypatch.setenv(PASSES, 'm6,m7')
    for label, n_nan in (('m7', 9), ('m6', 9), ('m5', 0)):
        seen, _, _, _ = _run(monkeypatch, label)
        assert np.isnan(seen.files['residual.fits']).sum() == n_nan, label


def test_min_model_env_moves_the_written_set(monkeypatch):
    monkeypatch.setenv(FOOTPRINT, '1')
    monkeypatch.setenv(MIN_MODEL, '0')
    seen, _, _, _ = _run(monkeypatch, 'm7')
    assert np.isnan(seen.files['residual.fits']).sum() == 14


def test_background_input_and_model_file_stay_unmasked(monkeypatch):
    monkeypatch.setenv(FOOTPRINT, '1')
    seen, plain, _, _ = _run(monkeypatch, 'm7')
    assert np.isnan(seen.files['residual.fits']).any()
    np.testing.assert_array_equal(seen.bkg, plain)
    np.testing.assert_array_equal(seen.files['model.fits'], seen.modsky)


def test_no_subtracted_satstar_model_leaves_the_residual_alone(monkeypatch):
    monkeypatch.setenv(FOOTPRINT, '1')
    seen, plain, _, _ = _run(monkeypatch, 'm7', model=False)
    np.testing.assert_array_equal(seen.files['residual.fits'], plain)
