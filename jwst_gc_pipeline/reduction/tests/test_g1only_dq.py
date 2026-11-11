"""Single-group (G1-only) NIRCam pixels get SATURATED|DO_NOT_USE from the ramp.

With every group after the first saturated, ramp_fit returns G1/tgroup, ~0.75x
the true rate for NFRAMES=2, and the crf DQ stays clean.  See
reduction/g1only_dq.py and brick 2221 F182M nrcb4 (427, 1794).
"""
import os
import warnings

import numpy as np
import pytest
from astropy.io import fits

from jwst_gc_pipeline.reduction import g1only_dq

SAT, DNU, JUMP = 2, 1, 4
STEM = 'jw02221001001_07101_00001_nrcb4'
NY = NX = 32


@pytest.fixture(autouse=True)
def _fresh_state(monkeypatch):
    monkeypatch.delenv(g1only_dq.ENV, raising=False)
    g1only_dq.COUNTS.clear()
    g1only_dq.MISSING.clear()
    g1only_dq._MASK_CACHE.clear()
    yield


def _bright2_groupdq():
    """NGROUPS=2 GROUPDQ shaped like the brick star: a 2x2 core saturated in
    G2, a one-pixel growth ring also flagged in G2, and one pixel saturated in
    both groups (zero usable groups: already handled by ramp_fit)."""
    gdq = np.zeros((1, 2, NY, NX), dtype=np.uint8)
    gdq[0, 1, 14:18, 14:18] = SAT          # 4x4 = core 2x2 + growth ring
    gdq[0, :, 3, 3] = SAT                  # all groups saturated
    gdq[0, 1, 25, 25] = JUMP               # a jump is not saturation
    return gdq


def _write_pair(tmp_path, gdq, *, instrume='NIRCAM', stem=STEM, suffix='destreak_o001_crf',
                dq=None, write_ramp=True):
    sci = np.full((NY, NX), 100.0, dtype=np.float32)
    err = np.ones((NY, NX), dtype=np.float32)
    if dq is None:
        dq = np.zeros((NY, NX), dtype=np.uint32)
    hdul = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(sci, name='SCI'),
                         fits.ImageHDU(err, name='ERR'), fits.ImageHDU(dq, name='DQ'),
                         fits.ImageHDU(err.copy(), name='VAR_POISSON')])
    hdul[0].header['INSTRUME'] = instrume
    hdul[0].header['TELESCOP'] = 'JWST'
    hdul[0].header['DATE-OBS'] = '2022-08-28'
    crf = os.path.join(tmp_path, f'{stem}_{suffix}.fits')
    hdul.writeto(crf)
    if write_ramp:
        ramp = fits.HDUList([fits.PrimaryHDU(),
                             fits.ImageHDU(np.zeros(gdq.shape, np.float32), name='SCI'),
                             fits.ImageHDU(gdq, name='GROUPDQ')])
        ramp.writeto(os.path.join(tmp_path, f'{stem}_ramp.fits'))
    return crf


def test_mask_is_exactly_the_g1_only_pixels():
    m = g1only_dq.single_group_mask(_bright2_groupdq())
    assert m.sum() == 16
    assert m[14:18, 14:18].all()
    assert not m[3, 3]           # zero usable groups: not ours
    assert not m[25, 25]         # JUMP only: two usable groups


def test_ngroups_ge3_only_flags_single_usable_group():
    gdq = np.zeros((1, 4, NY, NX), dtype=np.uint8)
    gdq[0, 1:, 5, 5] = SAT       # only G1 usable -> flagged
    gdq[0, 2:, 6, 6] = SAT       # G1+G2 usable -> fit has a difference
    gdq[0, 3:, 7, 7] = SAT
    gdq[0, 0, 8, 8] = DNU        # G1 DO_NOT_USE, G2 usable, G3-4 saturated
    gdq[0, 2:, 8, 8] = SAT
    m = g1only_dq.single_group_mask(gdq)
    assert m[5, 5] and m[8, 8]
    assert not m[6, 6] and not m[7, 7]
    assert m.sum() == 2


def test_multi_integration_uses_best_integration():
    gdq = np.zeros((2, 3, NY, NX), dtype=np.uint8)
    gdq[0, 1:, 5, 5] = SAT                  # single-group in int 1 only
    gdq[:, 1:, 6, 6] = SAT                  # single-group in both
    m = g1only_dq.single_group_mask(gdq)
    assert not m[5, 5] and m[6, 6]


def test_ngroups1_flags_nothing():
    assert not g1only_dq.single_group_mask(np.zeros((1, 1, NY, NX), np.uint8)).any()


def test_apply_flags_nans_and_keeps_everything_else(tmp_path):
    dq0 = np.zeros((NY, NX), dtype=np.uint32)
    dq0[14, 14] = JUMP            # pre-existing bit must survive
    dq0[0, 0] = 1024
    crf = _write_pair(str(tmp_path), _bright2_groupdq(), dq=dq0)
    with fits.open(crf) as h:
        n = g1only_dq.apply_to_hdul(h, crf)
        dq, sci, err, vp = (h[e].data for e in ('DQ', 'SCI', 'ERR', 'VAR_POISSON'))
        assert h[0].header[g1only_dq.HEADER_KEY] == 16
    assert n == 16
    m = np.zeros((NY, NX), bool)
    m[14:18, 14:18] = True
    assert ((dq[m] & (SAT | DNU)) == (SAT | DNU)).all()
    assert dq[14, 14] & JUMP
    assert np.array_equal(dq[~m], dq0[~m])
    for a in (sci, err, vp):
        assert np.isnan(a[m]).all() and np.isfinite(a[~m]).all()
    assert np.all(sci[~m] == 100.0)
    assert g1only_dq.COUNTS['applied'] == 1 and g1only_dq.COUNTS['flagged_px'] == 16


def test_apply_to_file_is_idempotent(tmp_path):
    crf = _write_pair(str(tmp_path), _bright2_groupdq())
    assert g1only_dq.apply_to_file(crf) == 16
    with fits.open(crf) as h:
        dq1, sci1 = h['DQ'].data.copy(), h['SCI'].data.copy()
    assert g1only_dq.apply_to_file(crf) == 16
    with fits.open(crf) as h:
        assert np.array_equal(h['DQ'].data, dq1)
        assert np.array_equal(np.isnan(h['SCI'].data), np.isnan(sci1))


def test_missing_ramp_is_loud_and_counted(tmp_path, capsys):
    crf = _write_pair(str(tmp_path), _bright2_groupdq(), write_ramp=False)
    with fits.open(crf) as h:
        with pytest.warns(g1only_dq.G1OnlyRampMissingWarning):
            assert g1only_dq.apply_to_hdul(h, crf) is None
        assert np.isfinite(h['SCI'].data).all()
    assert g1only_dq.COUNTS['missing_ramp'] == 1
    assert g1only_dq.MISSING and 'no sibling _ramp.fits' in g1only_dq.MISSING[0][1]
    assert 'WARNING' in capsys.readouterr().out


def test_shape_mismatch_is_loud(tmp_path):
    crf = _write_pair(str(tmp_path), np.zeros((1, 2, 8, 8), np.uint8))
    with fits.open(crf) as h:
        with pytest.warns(g1only_dq.G1OnlyRampMissingWarning):
            assert g1only_dq.apply_to_hdul(h, crf) is None
    assert g1only_dq.COUNTS['shape_mismatch'] == 1


def test_env_toggle_disables(tmp_path, monkeypatch):
    monkeypatch.setenv(g1only_dq.ENV, '0')
    crf = _write_pair(str(tmp_path), _bright2_groupdq())
    with fits.open(crf) as h:
        assert g1only_dq.apply_to_hdul(h, crf) is None
        assert (h['DQ'].data == 0).all()
    assert g1only_dq.signature(crf) == ''


@pytest.mark.parametrize('instrume,stem', [('MIRI', 'jw02221002001_02101_00001_mirimage'),
                                           ('NIRCAM', 'jw02221-o001_t001_nircam_clear-f182m')])
def test_out_of_scope_products_are_silent(tmp_path, instrume, stem):
    crf = _write_pair(str(tmp_path), _bright2_groupdq(), instrume=instrume, stem=stem,
                      suffix='i2d', write_ramp=False)
    with fits.open(crf) as h:
        with warnings.catch_warnings():
            warnings.simplefilter('error', g1only_dq.G1OnlyRampMissingWarning)
            assert g1only_dq.apply_to_hdul(h, crf) is None
    assert g1only_dq.COUNTS['missing_ramp'] == 0


def test_signature_marks_only_frames_with_flags(tmp_path):
    a = tmp_path / 'a'
    b = tmp_path / 'b'
    a.mkdir()
    b.mkdir()
    crf = _write_pair(str(a), _bright2_groupdq())
    assert g1only_dq.signature(crf) == 'g1'
    clean = _write_pair(str(b), np.zeros((1, 2, NY, NX), np.uint8))
    assert g1only_dq.signature(clean) == ''


def test_cataloging_load_data_applies_it(tmp_path):
    from jwst_gc_pipeline.photometry.crowdsource_catalogs_long import load_data
    crf = _write_pair(str(tmp_path), _bright2_groupdq())
    fh, im1, data, wht, err, *_ = load_data(crf)
    try:
        assert np.isnan(data[14:18, 14:18]).all()
        assert ((im1['DQ'].data[14:18, 14:18] & SAT) > 0).all()
        assert np.isnan(err[14:18, 14:18]).all()
    finally:
        fh.close()
    with fits.open(crf) as h:            # in memory only: file untouched
        assert (h['DQ'].data == 0).all()


def test_reduction_flags_crf_after_every_crf_write():
    """Source pin: each write_perexposure_crf call in main() is followed by the
    G1-only flagging of the same pass (the module is not imported: jwst+CRDS)."""
    import ast
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1]
           / 'PipelineRerunNIRCAM-LONG.py').read_text()
    main = next(n for n in ast.walk(ast.parse(src))
                if isinstance(n, ast.FunctionDef) and n.name == 'main')
    calls = [n.func.id for n in ast.walk(main)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id in ('write_perexposure_crf', 'flag_single_group_pixels_in_crf')]
    assert calls.count('write_perexposure_crf') == 2
    assert calls.count('flag_single_group_pixels_in_crf') == 2
    lines = src.splitlines()
    for i, ln in enumerate(lines):
        if ln.strip().startswith('write_perexposure_crf(asn_data'):
            follow = '\n'.join(lines[i:i + 5])
            assert 'flag_single_group_pixels_in_crf(' in follow
