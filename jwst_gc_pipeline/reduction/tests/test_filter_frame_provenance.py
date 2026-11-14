"""The FFRAMCOR provenance scan (issue #1133): which F162M/F164N frames are stale.

Synthetic headers and tiny synthetic FITS files only; no production data.
"""
import importlib.util
import os

import numpy as np
import pytest
from astropy.io import fits

from jwst_gc_pipeline.reduction import filter_frame_correction as ffc

_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                       os.pardir, os.pardir, 'scripts', 'analysis',
                       'scan_filter_frame_provenance.py')


def _load_script():
    spec = importlib.util.spec_from_file_location('_scan_ffp', _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _hdrs(filt='F162M', pupil=None, det='NRCA1', corrected=False, pending=False,
          anchor='F212N', instrume='NIRCAM'):
    hdr0 = fits.Header()
    hdr0['INSTRUME'] = instrume
    hdr0['DETECTOR'] = det
    hdr0['FILTER'] = filt
    if pupil:
        hdr0['PUPIL'] = pupil
    hdr = fits.Header()
    if corrected:
        hdr[ffc.MARKER] = True
        hdr['FFRAMANC'] = anchor
        hdr[ffc.PENDING] = False
    elif pending:
        hdr[ffc.PENDING] = True
    return hdr0, hdr


@pytest.mark.parametrize('kwargs, status', [
    (dict(corrected=True), ffc.STATUS_CORRECTED),
    (dict(), ffc.STATUS_STALE),
    (dict(pending=True), ffc.STATUS_PENDING),
    (dict(corrected=True, anchor='F210M'), ffc.STATUS_WRONG_ANCHOR),
    (dict(filt='F212N'), ffc.STATUS_NOT_SELECTED),
    (dict(det='NRCALONG'), ffc.STATUS_NOT_SELECTED),
    (dict(instrume='MIRI'), ffc.STATUS_NOT_SELECTED),
])
def test_status_matrix(kwargs, status):
    hdr0, hdr = _hdrs(**kwargs)
    assert ffc.frame_provenance(hdr0, hdr, mode='default')['status'] == status


def test_pupil_wheel_filter_is_resolved_from_pupil():
    hdr0, hdr = _hdrs(filt='F150W2', pupil='F162M')
    rec = ffc.frame_provenance(hdr0, hdr, mode='default')
    assert rec['filter'] == 'F162M'
    assert rec['status'] == ffc.STATUS_STALE


def test_completed_apply_with_false_pending_flag_is_corrected():
    # apply_filter_frame_correction leaves FFRAMPND=False beside FFRAMCOR=True
    hdr0, hdr = _hdrs(corrected=True)
    assert hdr[ffc.PENDING] is False
    assert ffc.frame_provenance(hdr0, hdr, mode='default')['status'] == 'corrected'


def test_selection_follows_mode():
    hdr0, hdr = _hdrs(filt='F182M')
    assert ffc.frame_provenance(hdr0, hdr, mode='default')['status'] == 'not-selected'
    assert ffc.frame_provenance(hdr0, hdr, mode='all')['status'] == 'stale'
    hdr0, hdr = _hdrs()
    assert ffc.frame_provenance(hdr0, hdr, mode='off')['status'] == 'not-selected'


def test_mode_defaults_to_environment(monkeypatch):
    hdr0, hdr = _hdrs()
    monkeypatch.setenv('FILTER_FRAME_CORRECTION', '0')
    assert ffc.frame_provenance(hdr0, hdr)['status'] == 'not-selected'
    monkeypatch.delenv('FILTER_FRAME_CORRECTION')
    assert ffc.frame_provenance(hdr0, hdr)['status'] == 'stale'


@pytest.mark.parametrize('filt', ['F212N', 'F210M', 'F115W'])
def test_mode_all_leaves_filters_outside_the_table_alone(filt):
    # apply_configured_filter_frame_correction returns None for these, so a
    # frame without FFRAMCOR is not stale
    hdr0, hdr = _hdrs(filt=filt)
    assert ffc.frame_provenance(hdr0, hdr, mode='all')['status'] == 'not-selected'


def test_mixed_groups_needs_both_corrected_and_uncorrected():
    def rec(status):
        return {'status': status}
    records = [
        (('a', '1', 'F162M'), rec('corrected')), (('a', '1', 'F162M'), rec('stale')),
        (('a', '2', 'F162M'), rec('stale')), (('a', '2', 'F162M'), rec('stale')),
        (('a', '3', 'F162M'), rec('corrected')),
        (('a', '4', 'F212N'), rec('not-selected')), (('a', '4', 'F212N'), rec('stale')),
    ]
    assert ffc.mixed_groups(records) == [('a', '1', 'F162M')]


def _write_frame(path, **kwargs):
    hdr0, hdr = _hdrs(**kwargs)
    hdr0['PROGRAM'] = '02092'
    hdr0['OBSERVTN'] = '002'
    sci = fits.ImageHDU(np.zeros((2, 2), dtype='f4'), header=hdr, name='SCI')
    fits.HDUList([fits.PrimaryHDU(header=hdr0), sci]).writeto(path)


def test_scan_flags_a_mixed_band(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv('FILTER_FRAME_CORRECTION', raising=False)
    pipe = tmp_path / 'F162M' / 'pipeline'
    pipe.mkdir(parents=True)
    _write_frame(pipe / 'jw02092002001_02101_00001_nrca1_destreak_o002_crf.fits',
                 corrected=True)
    _write_frame(pipe / 'jw02092002001_02101_00001_nrca2_destreak_o002_crf.fits')
    scan = _load_script()
    assert scan.main([str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert 'MIXED' in out
    assert scan.main([str(tmp_path), '--check', '--list']) == 1
    assert 'nrca2_destreak_o002_crf.fits' in capsys.readouterr().out


def test_scan_keeps_align_and_destreak_crf_lineages_apart(tmp_path, capsys, monkeypatch):
    # cloudef o002: a regenerated destreak_crf set beside older align_crf copies
    monkeypatch.delenv('FILTER_FRAME_CORRECTION', raising=False)
    pipe = tmp_path / 'F162M' / 'pipeline'
    pipe.mkdir(parents=True)
    _write_frame(pipe / 'jw02092002001_02101_00001_nrca1_destreak_o002_crf.fits',
                 corrected=True)
    _write_frame(pipe / 'jw02092002001_02101_00001_nrca1_align_o002_crf.fits')
    scan = _load_script()
    assert scan.main([str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert 'MIXED' not in out
    assert 'destreak_crf' in out and 'align_crf' in out


def test_scan_of_corrected_field_passes_check(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv('FILTER_FRAME_CORRECTION', raising=False)
    pipe = tmp_path / 'F164N' / 'pipeline'
    pipe.mkdir(parents=True)
    for det in ('nrca1', 'nrcb4'):
        _write_frame(pipe / f'jw02092002001_02101_00001_{det}_align_o002_crf.fits',
                     filt='F164N', det=det.upper(), corrected=True)
    scan = _load_script()
    assert scan.main([str(tmp_path), '--check']) == 0
    assert 'ok' in capsys.readouterr().out


def test_scan_with_no_frames_exits_2(tmp_path):
    assert _load_script().main([str(tmp_path)]) == 2
