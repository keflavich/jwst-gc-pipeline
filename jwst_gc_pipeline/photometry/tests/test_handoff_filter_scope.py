"""DAOPHOT_HANDOFF_FILTERS limits the opt-in hand-off extensions per bandpass.

One cataloging submission covers every filter of a field (the m7 finalize
does the cross-band merge), and in the gc-treasury o105 m7 A/B the
extensions (``DAOPHOT_HANDOFF_UNACCEPTED_SAT=1`` plus
``DAOPHOT_HANDOFF_RESTORE_ACCEPTED_EXCL_FWHM``) helped F480M and cost F212N
stars.  These tests pin:

* unset or blank keeps the old behaviour (every bandpass);
* a listed bandpass keeps the extensions, an unlisted one runs as if both
  variables were unset, and the default-on gate-reject hand-off is unchanged;
* a malformed list, or a set list with no bandpass to test, raises.
"""
import numpy as np
import pytest

from jwst_gc_pipeline.photometry import cataloging as C
from jwst_gc_pipeline.photometry.tests.test_gate_reject_handoff import (
    FWHM, SAT, SHAPE, STAR, _prepare, _write_rejected)


@pytest.mark.parametrize('raw, filt, expected', [
    (None, 'F212N', True),
    ('', 'F212N', True),
    ('   ', None, True),
    ('F480M', 'F480M', True),
    ('F480M', 'f480m', True),
    ('f480m', 'F480M', True),
    ('F480M', 'F212N', False),
    ('F480M,F405N', 'F405N', True),
    ('F480M, F405N F466N', 'F466N', True),
    ('F150W2', 'F150W2', True),
    ('F1130W', 'F1130W', True),
])
def test_filter_scope_parse(monkeypatch, raw, filt, expected):
    if raw is None:
        monkeypatch.delenv('DAOPHOT_HANDOFF_FILTERS', raising=False)
    else:
        monkeypatch.setenv('DAOPHOT_HANDOFF_FILTERS', raw)
    assert C._daophot_handoff_in_filter_scope(filt) is expected


@pytest.mark.parametrize('raw', ['F48OM', 'F480', 'yes', 'F480M;F212N',
                                 'nrcalong'])
def test_filter_scope_rejects_non_bandpass(monkeypatch, raw):
    monkeypatch.setenv('DAOPHOT_HANDOFF_FILTERS', raw)
    with pytest.raises(ValueError, match='DAOPHOT_HANDOFF_FILTERS'):
        C._daophot_handoff_in_filter_scope('F480M')


def test_filter_scope_needs_a_bandpass_when_set(monkeypatch):
    monkeypatch.setenv('DAOPHOT_HANDOFF_FILTERS', 'F480M')
    with pytest.raises(ValueError, match='no bandpass'):
        C._daophot_handoff_in_filter_scope(None)


def test_restore_excl_scoped(monkeypatch):
    monkeypatch.setenv('DAOPHOT_HANDOFF_RESTORE_ACCEPTED_EXCL_FWHM', '1.5')
    monkeypatch.delenv('DAOPHOT_HANDOFF_FILTERS', raising=False)
    assert C._daophot_handoff_restore_accepted_excl_fwhm() == 1.5
    assert C._daophot_handoff_restore_accepted_excl_fwhm('F212N') == 1.5
    monkeypatch.setenv('DAOPHOT_HANDOFF_FILTERS', 'F480M')
    assert C._daophot_handoff_restore_accepted_excl_fwhm('F480M') == 1.5
    assert C._daophot_handoff_restore_accepted_excl_fwhm('F212N') is None
    # the value is still parsed outside the scope
    monkeypatch.setenv('DAOPHOT_HANDOFF_RESTORE_ACCEPTED_EXCL_FWHM', 'x')
    with pytest.raises(ValueError,
                       match='DAOPHOT_HANDOFF_RESTORE_ACCEPTED_EXCL_FWHM'):
        C._daophot_handoff_restore_accepted_excl_fwhm('F212N')


def test_daophot_handoff_xy_component_scope(tmp_path, monkeypatch, capsys):
    """Outside the scope only the default-on gate-reject position is left."""
    path = _write_rejected(tmp_path / 'r.fits',
                           [(STAR[0], STAR[1], 'implied_peak_gate')])
    dq = np.zeros(SHAPE, dtype=np.uint32)
    sci = np.full(SHAPE, 10.0)
    dq[80:83, 80:83] |= SAT                       # no satstar row at all
    sci[81, 81] = 2500.0
    monkeypatch.setenv('DAOPHOT_HANDOFF_UNACCEPTED_SAT', '1')
    monkeypatch.setenv('DAOPHOT_HANDOFF_FILTERS', 'F480M')
    xy, _ = C._daophot_handoff_xy(dq, sci, None, path, FWHM, data_floor=0.0,
                                  label='t', filtername='F480M')
    assert xy.shape == (2, 2)
    xy, _ = C._daophot_handoff_xy(dq, sci, None, path, FWHM, data_floor=0.0,
                                  label='t', filtername='F212N')
    assert xy.shape == (1, 2)
    np.testing.assert_allclose(xy[0], STAR, atol=0.01)
    assert 'does not list F212N' in capsys.readouterr().out
    # the switch off: no scope message, same single position
    monkeypatch.delenv('DAOPHOT_HANDOFF_UNACCEPTED_SAT')
    xy, _ = C._daophot_handoff_xy(dq, sci, None, path, FWHM, data_floor=0.0,
                                  label='t', filtername='F212N')
    assert xy.shape == (1, 2)
    assert 'does not list' not in capsys.readouterr().out


@pytest.mark.parametrize('scope, handed', [
    (None, True),
    ('F480M', True),
    ('F212N F480M', True),
    ('F212N', False),
])
def test_prepare_component_handoff_follows_scope(tmp_path, monkeypatch,
                                                 scope, handed):
    """The frame preparation (bandpass F480M) hands the component off only
    when F480M is in scope; out of scope it matches the switch being off."""
    monkeypatch.setenv('DAOPHOT_HANDOFF_UNACCEPTED_SAT', '1')
    if scope is None:
        monkeypatch.delenv('DAOPHOT_HANDOFF_FILTERS', raising=False)
    else:
        monkeypatch.setenv('DAOPHOT_HANDOFF_FILTERS', scope)
    ctx, _ = _prepare(tmp_path, monkeypatch, reject_reason='fit_quality_gate')
    assert (ctx.handoff_xy is not None) is handed
