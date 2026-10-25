"""The header-rate ZEROFRAME rim (SATSTAR_ZF_R_HEADER) minus the cal - crf level.

R_header x group 0 is a cal-scale value.  A legacy destreak (no DESTRKMD)
subtracted each row's 10th percentile per amplifier from the cal and restored
nothing, so the crf sits below the cal by the local sky level: on w51 LW the
median cal - crf is 7-16 MJy/sr.  Pinned here:

1. ``cal_offset_map`` recovers a per-row, per-amplifier level, ignores
   SATURATED / DO_NOT_USE / non-finite pixels, interpolates sparse rows and
   reads 0 in a chunk with no valid pixel;
2. under the header rate the rim and the buffer threshold are
   R_header x group 0 - offset, and the log says so;
3. a measured curve, an explicit R or an offset of the wrong shape leave the
   rim as without the offset;
4. the R-curve satcheck fallback to R_header subtracts the offset with
   SATSTAR_ZF_R_HEADER off, and ``remove_saturated_stars`` reads the level
   whenever a first read is handed over;
5. ``zeroframe_cal_offset`` reads the sibling cal and returns None when there
   is none, when the PHOTMJSR differs, or when cal - crf is 0.
"""
import builtins

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

import jwst_gc_pipeline.reduction.saturated_star_finding as SSF
from jwst_gc_pipeline.reduction.saturated_star_finding import (
    _RCURVE_SATCHECK_MIN_PX, _find_cal_for, cal_offset_map,
    zeroframe_cal_offset, zeroframe_fit_anchor, zeroframe_recover_saturated)
from jwst_gc_pipeline.reduction.tests.test_zeroframe_first_frame import (
    _write_pair)
from jwst_gc_pipeline.reduction.tests.test_zeroframe_r_header import (
    R_TRUE, WING_INFL, _scene)
from jwst_gc_pipeline.reduction.tests.test_zeroframe_rcurve_satcheck import (
    R_TRUE as SC_R_TRUE, _scene as _satcheck_scene)

SAT = 2
DNU = 1
_ENV = ('SATSTAR_ZF_R_HEADER', 'NIRCAM_SATSTAR_RECOVERED_CAP',
        'SATSTAR_ZF_KEEP_FINITE', 'SATSTAR_ZF_G0_GROUPDQ',
        'SATSTAR_ZF_FIRST_FRAME', 'SATSTAR_ZF_RCURVE_GUARD',
        'SATSTAR_ZF_RCURVE_MAXSTEP', 'SATSTAR_ZF_RCURVE_SATCHECK',
        'SATSTAR_ZF_RIM_BADPIX')


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in _ENV:
        monkeypatch.delenv(k, raising=False)


def _destreaked(ny=64, nx=128, noutputs=4, seed=3):
    """A cal frame and its legacy-destreaked crf: one level per row and
    amplifier chunk removed."""
    rng = np.random.default_rng(seed)
    cal = rng.normal(20.0, 2.0, (ny, nx))
    level = rng.uniform(5.0, 15.0, (ny, noutputs))
    off = np.repeat(level, nx // noutputs, axis=1)
    return cal, cal - off, off


def test_map_recovers_the_row_amplifier_level():
    cal, crf, off = _destreaked()
    assert np.allclose(cal_offset_map(cal, crf, None), off)


def test_map_ignores_flagged_and_nonfinite_pixels():
    cal, crf, off = _destreaked()
    dq = np.zeros(cal.shape, dtype=np.uint32)
    dq[::3, ::5] = SAT
    dq[1::4, 2::7] = DNU
    crf = crf.copy()
    crf[dq != 0] = 1e5                   # clipped / zeroed values
    crf[5, :10] = np.nan
    assert np.allclose(cal_offset_map(cal, crf, dq), off)


def test_map_interpolates_sparse_rows_and_zeroes_empty_chunks():
    cal, crf, off = _destreaked()
    crf = crf.copy()
    crf[10, :32] = np.nan                # row 10 of chunk 0: no valid pixel
    crf[:, 96:] = np.nan                 # chunk 3: none at all
    got = cal_offset_map(cal, crf, None)
    assert np.allclose(got[10, :32], 0.5 * (off[9, 0] + off[11, 0]))
    assert np.allclose(got[:, 96:], 0.0)
    assert np.allclose(got[:10, :96], off[:10, :96])


def test_header_rim_subtracts_the_offset(monkeypatch, capsys):
    monkeypatch.setenv('SATSTAR_ZF_R_HEADER', '1')
    data, dq, g0, core, rim, wing = _scene()
    off = np.full(data.shape, 3.0)
    rec, rim_mask, deep, R = zeroframe_recover_saturated(
        data, dq, g0, R_header=R_TRUE, cal_offset=off)
    out = capsys.readouterr().out
    assert R == R_TRUE
    assert np.allclose(rec[rim], R_TRUE * 25000.0 - 3.0)
    assert deep[core].all()
    assert 'minus the cal - crf level, median 3 MJy/sr' in out


def test_fit_anchor_passes_the_offset(monkeypatch):
    monkeypatch.setenv('SATSTAR_ZF_R_HEADER', '1')
    data, dq, g0, core, rim, wing = _scene()
    off = np.full(data.shape, 3.0)
    rec, _, _, _ = zeroframe_fit_anchor(data, dq, g0, R_header=R_TRUE,
                                        cal_offset=off)
    assert np.allclose(rec[rim], R_TRUE * 25000.0 - 3.0)


def test_buffer_threshold_follows_the_offset(monkeypatch):
    """Buffer wings read WING_INFL x R_TRUE x group 0, below the 1.10
    threshold of R_header x group 0; once the threshold is taken from
    R_header x group 0 - offset they are flagged as inflated."""
    monkeypatch.setenv('SATSTAR_ZF_R_HEADER', '1')
    data, dq, g0, core, rim, wing = _scene()
    buf = wing & ~((dq & SAT) != 0)
    _, rim0, _, _ = zeroframe_recover_saturated(data, dq, g0,
                                                R_header=R_TRUE)
    off = np.full(data.shape, 10.0)      # > R g0 (1 - 1.08 / 1.10) at 5000 DN
    rec, rim1, _, _ = zeroframe_recover_saturated(
        data, dq, g0, R_header=R_TRUE, cal_offset=off)
    assert WING_INFL < 1.10
    assert not (rim0 & buf).any()
    newly = rim1 & buf
    assert newly.any()
    assert np.allclose(rec[newly], R_TRUE * g0[newly] - 10.0)


def test_curve_explicit_r_and_bad_shape_ignore_the_offset(monkeypatch, capsys):
    data, dq, g0, core, rim, wing = _scene()
    off = np.full(data.shape, 3.0)
    rec_c, _, _, R_c = zeroframe_recover_saturated(data, dq, g0,
                                                   R_header=R_TRUE)
    rec_co, _, _, _ = zeroframe_recover_saturated(
        data, dq, g0, R_header=R_TRUE, cal_offset=off)
    assert np.allclose(rec_co, rec_c)            # switch off: measured curve
    monkeypatch.setenv('SATSTAR_ZF_R_HEADER', '1')
    rec_r, _, _, _ = zeroframe_recover_saturated(
        data, dq, g0, R=0.05, R_header=R_TRUE, cal_offset=off)
    assert np.allclose(rec_r[rim], 0.05 * 25000.0)
    capsys.readouterr()
    rec_s, _, _, _ = zeroframe_recover_saturated(
        data, dq, g0, R_header=R_TRUE, cal_offset=np.zeros((3, 3)))
    assert np.allclose(rec_s[rim], R_TRUE * 25000.0)
    assert 'not applied' in capsys.readouterr().out


def test_satcheck_header_fallback_subtracts_the_offset(monkeypatch, capsys):
    """Too few measured SATURATED pixels to rebuild a junk curve: the rate
    falls back to R_header with SATSTAR_ZF_R_HEADER off, and the rim takes
    the offset as under the switch."""
    monkeypatch.setenv('SATSTAR_ZF_R_HEADER', '0')
    data, dq, g0, core, meas = _satcheck_scene(
        n_sat_meas=_RCURVE_SATCHECK_MIN_PX - 1)
    off = np.full(data.shape, 0.5)
    rec, rim, deep, R = zeroframe_recover_saturated(
        data, dq, g0, R_header=SC_R_TRUE, cal_offset=off)
    out = capsys.readouterr().out
    assert R == SC_R_TRUE
    assert 'the header value is used' in out
    assert 'minus the cal - crf level, median 0.5 MJy/sr' in out
    assert rim[meas].all()
    assert np.allclose(rec[meas], SC_R_TRUE * g0[meas] - 0.5)


@pytest.mark.parametrize('env, zf_on', [
    ({}, True),
    ({'SATSTAR_ZF_R_HEADER': '1'}, True),
    ({'NIRCAM_SATSTAR_RECOVERED_CAP': '1'}, True),
    ({'SATSTAR_ZEROFRAME_FIT': '0'}, False),
])
def test_remove_saturated_stars_hands_the_offset_with_a_first_read(
        tmp_path, monkeypatch, env, zf_on):
    """The level is read whenever a first read is handed over, with the
    header switch on or off, since the satcheck fallback can take the
    header rate either way."""
    fn, _ = _write_pair(tmp_path, shape=(8, 8))
    for attr in ('satstar_wingcal_measurements', 'satstar_rejected',
                 'satstar_model', 'satstar_resid', 'satstar_flagimg'):
        monkeypatch.delattr(builtins, attr, raising=False)
    monkeypatch.delenv('SATSTAR_ZEROFRAME_FIT', raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(SSF, '_find_zeroframe_for',
                        lambda filename: np.ones((8, 8)))
    monkeypatch.setattr(SSF, '_find_first_frame_for', lambda filename: None)
    monkeypatch.setattr(SSF, '_find_group0_saturation_for',
                        lambda filename, do_not_use=False:
                        np.zeros((8, 8), bool))
    level = np.full((8, 8), 2.0)
    calls = []

    def _offset(filename, data, dq, photmjsr=None):
        calls.append(filename)
        return level
    monkeypatch.setattr(SSF, 'zeroframe_cal_offset', _offset)
    seen = {}

    def _fit(fh, **kw):
        seen.update(kw)
        return Table({'flux_fit': [1.0]})
    monkeypatch.setattr(SSF, 'get_saturated_stars', _fit)
    SSF.remove_saturated_stars(fn, recovery_signature='off')
    assert ('zeroframe' in seen) is zf_on
    assert bool(calls) is zf_on
    if zf_on:
        assert seen['zeroframe_cal_offset'] is level
    else:
        assert 'zeroframe_cal_offset' not in seen


def _write(path, sci, photmjsr=1.0, dq=None):
    ph = fits.PrimaryHDU()
    ph.header['NOUTPUTS'] = 4
    sh = fits.ImageHDU(sci.astype(np.float32), name='SCI')
    sh.header['PHOTMJSR'] = photmjsr
    hl = [ph, sh]
    if dq is not None:
        hl.append(fits.ImageHDU(dq, name='DQ'))
    fits.HDUList(hl).writeto(path)


def test_zeroframe_cal_offset_reads_the_sibling_cal(tmp_path, capsys):
    cal, crf, off = _destreaked()
    stem = 'jw01234001001_02101_00001_nrcalong'
    crf_fn = tmp_path / f'{stem}_destreak_o001_crf.fits'
    cal_fn = tmp_path / f'{stem}_cal.fits'
    _write(cal_fn, cal, photmjsr=0.85)
    assert _find_cal_for(str(crf_fn)) == str(cal_fn)
    assert _find_cal_for(str(cal_fn)) is None
    got = zeroframe_cal_offset(str(crf_fn), crf.astype(np.float32), None,
                               photmjsr=0.85)
    assert np.allclose(got, off, atol=1e-4)
    assert 'cal - crf level from' in capsys.readouterr().out
    assert zeroframe_cal_offset(str(crf_fn), crf, None, photmjsr=0.9) is None
    assert 'PHOTMJSR' in capsys.readouterr().out
    assert zeroframe_cal_offset(
        str(crf_fn), cal.astype(np.float32), None, photmjsr=0.85) is None
    assert zeroframe_cal_offset(
        str(tmp_path / 'jw09999001001_02101_00001_nrcb1_crf.fits'), crf,
        None) is None
