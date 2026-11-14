"""The satstar data and severity floors are keyed by the science bandpass (#950).

NIRCam pupil-wheel bands carry the wide blocker in FILTER (F405N and F466N read
``FILTER=F444W``, F323N reads ``F322W2``, F164N and F162M read ``F150W2``), so a
FILTER lookup gave them the blocker's floor row or none.
"""
import os
import re

import numpy as np
import pytest
from astropy.io import fits

from jwst_gc_pipeline.reduction import saturated_star_finding as SSF

READOUT = {'INSTRUME': 'NIRCAM', 'NGROUPS': 7, 'NFRAMES': 1, 'GROUPGAP': 0,
           'TFRAME': 58.0}


def _floors(header):
    band = SSF._satstar_floor_band(fits.Header(header))
    return (SSF._resolve_satstar_data_floor(band),
            SSF._resolve_satstar_severity_floor(band))


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in [n for n in os.environ if 'SATSTAR' in n]:
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize('filt, pupil, want', [
    ('F444W', 'F405N', (1000., 5000.)),
    ('F444W', 'F466N', (0., 5000.)),
    ('F322W2', 'F323N', (1000., 5000.)),
    ('F150W2', 'F164N', (1000., 8000.)),
    ('F150W2', 'F162M', (1000., 5000.)),
    ('F444W', 'F470N', (1000., 5000.)),
    # filter-wheel bands and wide bands keep their rows
    ('F212N', 'CLEAR', (0., 4000.)),
    ('F405N', 'CLEAR', (1000., 5000.)),
    ('F444W', 'CLEAR', (0., 2500.)),
    ('F150W', 'CLEAR', (0., 0.)),
    ('F410M', 'CLEAR', (800., 2500.)),
])
def test_floors_follow_the_bandpass(filt, pupil, want):
    assert _floors({'FILTER': filt, 'PUPIL': pupil}) == want


def test_floors_without_pupil_keyword_and_miri():
    assert _floors({'FILTER': 'F410M'}) == (800., 2500.)
    assert _floors({'FILTER': 'F770W'}) == (0., 0.)
    # a header naming no bandpass gives floor 0 rather than raising
    assert _floors({'FILTER': 'CLEAR', 'PUPIL': 'CLEAR'}) == (0., 0.)
    assert _floors({}) == (0., 0.)


def _crf(tmp_path, primary):
    fn = tmp_path / 'jw03523005001_16101_00001_nrcblong_align_o005_crf.fits'
    sci = fits.ImageHDU(np.zeros((8, 8), 'float32'), name='SCI')
    sci.header['PHOTMJSR'] = 0.86
    fits.HDUList([fits.PrimaryHDU(header=fits.Header(primary)), sci]
                 ).writeto(fn, overwrite=True)
    return str(fn)


def test_data_floor_signature_uses_the_pupil_band(tmp_path):
    fn = _crf(tmp_path, dict(READOUT, FILTER='F444W', PUPIL='F405N'))
    assert SSF._satstar_data_floor_signature(fn) == 'dfr0.5'
    fn = _crf(tmp_path, dict(READOUT, FILTER='F444W', PUPIL='CLEAR'))
    assert SSF._satstar_data_floor_signature(fn) == ''


def test_fit_switch_signature_marks_only_changed_lookups(tmp_path, monkeypatch):
    monkeypatch.setenv('SATSTAR_SEED_CORE_DQ', '0')
    base = SSF.satstar_fit_switch_signature(
        _crf(tmp_path, dict(READOUT, FILTER='F444W', PUPIL='CLEAR')))
    assert 'sf' not in base
    f405n = SSF.satstar_fit_switch_signature(
        _crf(tmp_path, dict(READOUT, FILTER='F444W', PUPIL='F405N')))
    assert 'sf0-2500to1000-5000' in f405n
    assert f405n != base
    f466n = SSF.satstar_fit_switch_signature(
        _crf(tmp_path, dict(READOUT, FILTER='F444W', PUPIL='F466N')))
    assert 'sf0-2500to0-5000' in f466n
    # a filter-wheel band and an unchanged band keep their keys
    f212n = SSF.satstar_fit_switch_signature(
        _crf(tmp_path, dict(READOUT, FILTER='F212N', PUPIL='CLEAR')))
    assert 'sf' not in f212n
    f212n_nopupil = SSF.satstar_fit_switch_signature(
        _crf(tmp_path, dict(READOUT, FILTER='F212N')))
    assert f212n_nopupil == f212n


def test_warning_once_per_unlisted_narrow_band(capsys, monkeypatch):
    monkeypatch.setattr(SSF, '_SATSTAR_FLOOR_UNLISTED_WARNED', set())
    for band in ('f999n', 'f999n', 'f998m'):
        SSF._warn_unlisted_floor_band(band)
    out = capsys.readouterr().out
    assert out.count('F999N has no entry') == 1
    assert out.count('F998M has no entry') == 1
    # listed, wide, MIRI and empty bands stay quiet
    for band in ('f405n', 'f466n', 'f212n', 'f444w', 'f150w2', 'f770w',
                 'f1065c', ''):
        SSF._warn_unlisted_floor_band(band)
    assert capsys.readouterr().out == ''


#: Wide NIRCam bands without a data floor by design (0 = gate off).
_DATA_FLOOR_OFF = {'f115w', 'f150w', 'f150w2', 'f200w', 'f356w', 'f444w',
                   'f212n', 'f466n'}
_SEVERITY_OFF = {'f150w', 'f150w2'}


def test_every_nircam_band_in_fields_yaml_is_listed_or_allowed():
    path = os.path.join(os.path.dirname(SSF.__file__), '..', 'fields.yaml')
    text = open(path).read()
    bands = {'f' + b.lower() for b in re.findall(r'\bF(\d{3}[WMN]2?)\b', text)}
    nircam = {b for b in bands if int(b[1:4]) <= 500}
    missing_data = sorted(b for b in nircam
                          if b not in SSF._SATSTAR_DATA_FLOOR
                          and b not in _DATA_FLOOR_OFF)
    missing_sev = sorted(b for b in nircam
                         if b not in SSF.SAT_SEVERITY_FLOOR
                         and b not in _SEVERITY_OFF)
    assert not missing_data, f'NIRCam bands with no data floor row: {missing_data}'
    assert not missing_sev, f'NIRCam bands with no severity floor row: {missing_sev}'
