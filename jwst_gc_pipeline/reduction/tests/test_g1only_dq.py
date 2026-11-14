"""Single-group (G1-only) NIRCam pixels get SATURATED|DO_NOT_USE from the ramp.

With every group after the first saturated, a ramp fit that divides G1 by
TGROUP returns 0.75x the true rate for BRIGHT2, and the crf DQ stays clean.
The default G1ONLY_SAT_DQ=auto gate measures m = SCI / (G1/TGROUP) per frame
and flags only frames with m ~ 1 (biased), not m ~ r = 1.333 (unbiased).  See
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
    g1only_dq._RAMP_CACHE.clear()
    g1only_dq._DECISION_CACHE.clear()
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


# ---------------------------------------------------------------------------
# The auto gate.  Synthetic BRIGHT2 frames with a real ramp: sky ref pixels
# with G2 - G1 well above REF_MIN_DN, and a star whose core has only G1 usable.
# The crf SCI = C * rate + B, with the rate at the single-group pixels either
# G1/TGROUP (biased, m = 1) or G1/((NF+1) TF/2) (unbiased, m = r = 1.333).
# ---------------------------------------------------------------------------
TFRAME, NFRAMES = 10.73677, 2
TGROUP = 2 * TFRAME
T1 = (NFRAMES + 1) * TFRAME / 2
R_BRIGHT2 = TGROUP / T1
CONV, OFFSET = 0.31, 0.7
N2 = 64


def _readout(hdr, nframes=NFRAMES, tframe=TFRAME, tgroup=TGROUP, readpatt='BRIGHT2'):
    hdr['READPATT'] = readpatt
    hdr['NFRAMES'] = nframes
    hdr['TFRAME'] = tframe
    hdr['TGROUP'] = tgroup
    hdr['GROUPGAP'] = int(round(tgroup / tframe)) - nframes


def _write_gate_pair(tmp_path, *, biased, core=6, nframes=NFRAMES, tframe=TFRAME,
                     tgroup=TGROUP, readpatt='BRIGHT2', bias_factor=None, offset=OFFSET,
                     ramp_sci=True, stem=STEM):
    """crf + ramp (NGROUPS=2, NINTS=1) with a ``core``x``core`` G1-only star."""
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[:N2, :N2]
    rate = 20.0 + 5.0 * xx / N2 + rng.normal(0, 0.2, (N2, N2))          # DN/s sky
    stars = rng.random((N2, N2)) < 0.15                       # unsaturated sources
    rate[stars] += rng.uniform(30, 1200, int(stars.sum()))
    t1 = (nframes + 1) * tframe / 2
    g1 = rate * t1
    g2 = g1 + rate * tgroup
    gdq = np.zeros((1, 2, N2, N2), np.uint8)
    c0 = 20
    star = (slice(c0, c0 + core), slice(c0, c0 + core))
    star_rate = 900.0 + rng.normal(0, 30, (core, core))
    g1[star] = star_rate * t1
    g2[star] = 60000.0
    gdq[0, 1][star] = SAT
    fit = rate.copy()                                 # (G2 - G1)/TGROUP elsewhere
    if bias_factor is None:
        bias_factor = 1.0 if biased else tgroup / t1
    fit[star] = g1[star] / tgroup * bias_factor
    sci = (CONV * fit + offset).astype(np.float32)
    sci *= (1 + rng.normal(0, 0.01, sci.shape)).astype(np.float32)   # flat-field scatter
    hdul = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(sci, name='SCI'),
                         fits.ImageHDU(np.ones_like(sci), name='ERR'),
                         fits.ImageHDU(np.zeros((N2, N2), np.uint32), name='DQ'),
                         fits.ImageHDU(np.ones_like(sci), name='VAR_POISSON')])
    for k, v in (('INSTRUME', 'NIRCAM'), ('TELESCOP', 'JWST'), ('DATE-OBS', '2022-08-28')):
        hdul[0].header[k] = v
    _readout(hdul[0].header, nframes, tframe, tgroup, readpatt)
    crf = os.path.join(tmp_path, f'{stem}_destreak_o001_crf.fits')
    hdul.writeto(crf)
    rsci = np.stack([g1, g2])[None].astype(np.float32)
    ramp = fits.HDUList([fits.PrimaryHDU()]
                        + ([fits.ImageHDU(rsci, name='SCI')] if ramp_sci else [])
                        + [fits.ImageHDU(gdq, name='GROUPDQ')])
    _readout(ramp[0].header, nframes, tframe, tgroup, readpatt)
    ramp.writeto(os.path.join(tmp_path, f'{stem}_ramp.fits'))
    return crf, core * core


@pytest.mark.parametrize('value,expected', [
    (None, 'auto'), ('auto', 'auto'), (' AUTO ', 'auto'),
    ('1', 'force'), ('true', 'force'), ('on', 'force'), ('force', 'force'),
    ('0', 'off'), ('false', 'off'), ('off', 'off'), ('', 'off')])
def test_mode_parsing(value, expected):
    env = {} if value is None else {g1only_dq.ENV: value}
    assert g1only_dq.mode(env) == expected
    assert g1only_dq.enabled(env) == (expected != 'off')


@pytest.mark.parametrize('value', ['2', 'maybe', 'autoo', 'flag'])
def test_unknown_env_value_raises(tmp_path, monkeypatch, value):
    with pytest.raises(ValueError, match=g1only_dq.ENV):
        g1only_dq.mode({g1only_dq.ENV: value})
    monkeypatch.setenv(g1only_dq.ENV, value)
    crf, _ = _write_gate_pair(str(tmp_path), biased=True)
    with fits.open(crf) as h:
        with pytest.raises(ValueError, match=g1only_dq.ENV):
            g1only_dq.apply_to_hdul(h, crf)
    with pytest.raises(ValueError, match=g1only_dq.ENV):
        g1only_dq.signature(crf)


def test_expected_ratio_from_readout_keywords():
    hdr = fits.Header()
    _readout(hdr)
    r, tg = g1only_dq.expected_ratio(hdr)
    assert r == pytest.approx(4 / 3) and tg == pytest.approx(TGROUP)
    shallow2 = fits.Header()
    _readout(shallow2, nframes=2, tframe=TFRAME, tgroup=5 * TFRAME, readpatt='SHALLOW2')
    assert g1only_dq.expected_ratio(shallow2)[0] == pytest.approx(10 / 3)
    del shallow2['TGROUP']                       # falls back to (NF + GAP) * TFRAME
    assert g1only_dq.expected_ratio(shallow2)[0] == pytest.approx(10 / 3)
    assert g1only_dq.expected_ratio(fits.Header(), None) == (None, None)


def test_auto_flags_biased_frame_and_records_m(tmp_path, capsys):
    crf, npx = _write_gate_pair(str(tmp_path), biased=True)
    with fits.open(crf) as h:
        n = g1only_dq.apply_to_hdul(h, crf, label='t')
        hdr = h[0].header
        assert n == npx
        assert hdr[g1only_dq.HEADER_KEY] == npx
        assert hdr[g1only_dq.DECISION_KEY] == 'flag'
        assert hdr[g1only_dq.M_KEY] == pytest.approx(1.0, abs=0.03)
        assert hdr[g1only_dq.R_KEY] == pytest.approx(R_BRIGHT2, abs=1e-3)
        assert hdr[g1only_dq.N_KEY] == npx
        assert hdr[g1only_dq.REASON_KEY].startswith('measured: m<1.167')
        assert np.isnan(h['SCI'].data[20:26, 20:26]).all()
        assert ((h['DQ'].data[20:26, 20:26] & (SAT | DNU)) == (SAT | DNU)).all()
    out = capsys.readouterr().out
    assert 'G1-only DQ (t)' in out and 'flagged 36' in out and 'm=0.99' in out and 'r=1.333' in out
    assert g1only_dq.signature(crf) == 'g1'


def test_auto_skips_unbiased_frame(tmp_path, capsys):
    crf, npx = _write_gate_pair(str(tmp_path), biased=False)
    with fits.open(crf) as h:
        sci0 = h['SCI'].data.copy()
        n = g1only_dq.apply_to_hdul(h, crf, label='t')
        hdr = h[0].header
        assert n == 0
        assert hdr[g1only_dq.HEADER_KEY] == 0
        assert hdr[g1only_dq.DECISION_KEY] == 'skip'
        assert hdr[g1only_dq.M_KEY] == pytest.approx(R_BRIGHT2, abs=0.04)
        assert hdr[g1only_dq.N_KEY] == npx
        assert (h['DQ'].data == 0).all()
        assert np.array_equal(h['SCI'].data, sci0)
    assert 'skip 36 single-group px' in capsys.readouterr().out
    assert g1only_dq.signature(crf) == ''
    assert g1only_dq.COUNTS['skipped_unbiased'] == 1 and g1only_dq.COUNTS['applied'] == 0


@pytest.mark.parametrize('offset', [0.0, -15.0, 40.0])
def test_auto_is_insensitive_to_additive_offsets(tmp_path, offset):
    """destreak / bgsub shift the crf by a constant; B absorbs it."""
    for biased, want in ((True, 1.0), (False, R_BRIGHT2)):
        d = tmp_path / f'{biased}'
        d.mkdir()
        crf, _ = _write_gate_pair(str(d), biased=biased, offset=offset)
        with fits.open(crf) as h:
            mask, dec = g1only_dq.frame_plan(crf, h[0].header, h['SCI'].data, h['DQ'].data)
        assert dec.flag is biased and dec.source == 'measured'
        assert dec.m == pytest.approx(want, abs=0.04)


def test_faint_g1_pixels_do_not_enter_m(tmp_path):
    """Single-group pixels with G1 < G1_MIN_DN (e.g. DO_NOT_USE in group 2 on
    sky) give m dominated by any residual the fit's B does not absorb; they
    still get flagged but are left out of the median."""
    crf, npx = _write_gate_pair(str(tmp_path), biased=True)
    ramp = crf.replace('_destreak_o001_crf.fits', '_ramp.fits')
    faint = (slice(45, 51), slice(5, 15))                    # 60 px, away from the star
    g1_faint = 40.0
    with fits.open(ramp, mode='update') as r:
        r['SCI'].data[0, 0][faint] = g1_faint
        r['GROUPDQ'].data[0, 1][faint] = DNU
    with fits.open(crf, mode='update') as h:                  # biased rate + 5 DN/s residual
        h['SCI'].data[faint] = CONV * (g1_faint / TGROUP + 5.0) + OFFSET
    with fits.open(crf) as h:
        mask, dec = g1only_dq.frame_plan(crf, h[0].header, h['SCI'].data, h['DQ'].data)
    assert mask.sum() == npx + 60
    assert dec.flag and dec.source == 'measured'
    assert dec.n == npx
    assert dec.m == pytest.approx(1.0, abs=0.03)


def test_auto_too_few_pixels_falls_back_to_flag(tmp_path, capsys):
    crf, npx = _write_gate_pair(str(tmp_path), biased=False, core=4)   # 16 < N_MIN_PX
    assert npx < g1only_dq.N_MIN_PX
    with fits.open(crf) as h:
        assert g1only_dq.apply_to_hdul(h, crf) == npx
        assert h[0].header[g1only_dq.DECISION_KEY] == 'flag'
        assert h[0].header[g1only_dq.REASON_KEY].startswith('fallback: 16 measurable px')
    assert g1only_dq.COUNTS['fallback_flag'] == 1


def test_auto_missing_ramp_sci_falls_back_to_flag(tmp_path):
    crf, npx = _write_gate_pair(str(tmp_path), biased=False, ramp_sci=False)
    with fits.open(crf) as h:
        mask, dec = g1only_dq.frame_plan(crf, h[0].header, h['SCI'].data, h['DQ'].data)
    assert mask.sum() == npx
    assert dec.flag and dec.source == 'fallback' and 'ramp SCI unreadable' in dec.reason


def test_auto_missing_ramp_is_still_loud(tmp_path):
    crf, _ = _write_gate_pair(str(tmp_path), biased=True)
    os.remove(os.path.join(str(tmp_path), f'{STEM}_ramp.fits'))
    with fits.open(crf) as h:
        with pytest.warns(g1only_dq.G1OnlyRampMissingWarning):
            assert g1only_dq.apply_to_hdul(h, crf) is None
    assert g1only_dq.COUNTS['missing_ramp'] == 1


def test_rapid_readout_is_skipped_without_measuring(tmp_path, monkeypatch):
    """NFRAMES=1: G1/TGROUP is the right rate whichever timing ramp_fit used."""
    crf, _ = _write_gate_pair(str(tmp_path), biased=True, nframes=1, tgroup=TFRAME,
                              readpatt='RAPID')
    called = []
    monkeypatch.setattr(g1only_dq, 'measure_ratio',
                        lambda *a, **k: called.append(1) or (1.0, 99, None, 999))
    with fits.open(crf) as h:
        assert g1only_dq.apply_to_hdul(h, crf) == 0
        assert h[0].header[g1only_dq.DECISION_KEY] == 'skip'
        assert h[0].header[g1only_dq.R_KEY] == pytest.approx(1.0)
    assert not called


def test_force_flags_an_unbiased_frame(tmp_path, monkeypatch):
    monkeypatch.setenv(g1only_dq.ENV, '1')
    crf, npx = _write_gate_pair(str(tmp_path), biased=False)
    with fits.open(crf) as h:
        assert g1only_dq.apply_to_hdul(h, crf) == npx
        assert h[0].header[g1only_dq.DECISION_KEY] == 'flag'
        assert h[0].header[g1only_dq.REASON_KEY].startswith('forced')
    assert g1only_dq.signature(crf) == 'g1'


def test_off_ignores_a_biased_frame(tmp_path, monkeypatch):
    monkeypatch.setenv(g1only_dq.ENV, 'off')
    crf, _ = _write_gate_pair(str(tmp_path), biased=True)
    with fits.open(crf) as h:
        assert g1only_dq.apply_to_hdul(h, crf) is None
        assert g1only_dq.DECISION_KEY not in h[0].header
    assert g1only_dq.signature(crf) == ''


@pytest.mark.parametrize('biased', [True, False])
def test_persisted_decision_is_reused(tmp_path, monkeypatch, biased):
    """A crf flagged on disk has NaN where m is measured: the header decides."""
    crf, npx = _write_gate_pair(str(tmp_path), biased=biased)
    first = g1only_dq.apply_to_file(crf, label='reduction')
    assert first == (npx if biased else 0)
    g1only_dq._DECISION_CACHE.clear()
    monkeypatch.setattr(g1only_dq, 'measure_ratio',
                        lambda *a, **k: pytest.fail('re-measured despite header'))
    with fits.open(crf) as h:
        assert h[0].header[g1only_dq.DECISION_KEY] == ('flag' if biased else 'skip')
        assert g1only_dq.apply_to_hdul(h, crf, label='load_data') == first
        assert h[0].header[g1only_dq.REASON_KEY].startswith('measured')   # kept
    assert g1only_dq.signature(crf) == ('g1' if biased else '')


def test_pre_gate_flagged_crf_stays_flagged(tmp_path, monkeypatch):
    monkeypatch.setenv(g1only_dq.ENV, '1')
    crf, npx = _write_gate_pair(str(tmp_path), biased=False)
    assert g1only_dq.apply_to_file(crf) == npx
    with fits.open(crf, mode='update') as h:          # as the pre-gate code left it
        for k in (g1only_dq.DECISION_KEY, g1only_dq.REASON_KEY, g1only_dq.R_KEY):
            h[0].header.remove(k, ignore_missing=True)
    monkeypatch.delenv(g1only_dq.ENV)
    g1only_dq._DECISION_CACHE.clear()
    with fits.open(crf) as h:
        mask, dec = g1only_dq.frame_plan(crf, h[0].header, h['SCI'].data, h['DQ'].data)
    assert dec.flag and dec.source == 'header' and 'pre-gate' in dec.reason


def test_decision_is_measured_once_per_frame(tmp_path, monkeypatch):
    crf, _ = _write_gate_pair(str(tmp_path), biased=False)
    calls = []
    real = g1only_dq.measure_ratio
    monkeypatch.setattr(g1only_dq, 'measure_ratio',
                        lambda *a, **k: calls.append(1) or real(*a, **k))
    assert g1only_dq.signature(crf) == ''
    for label in ('satstar', 'load_data'):
        with fits.open(crf) as h:
            assert g1only_dq.apply_to_hdul(h, crf, label=label) == 0
    assert g1only_dq.signature(crf) == ''
    assert len(calls) == 1


def test_ambiguous_m_warns(tmp_path):
    crf, _ = _write_gate_pair(str(tmp_path), biased=True, bias_factor=1.17)
    with fits.open(crf) as h:
        with pytest.warns(g1only_dq.G1OnlyAmbiguousWarning):
            mask, dec = g1only_dq.frame_plan(crf, h[0].header, h['SCI'].data, h['DQ'].data)
    assert dec.source == 'measured' and 1.1 < dec.m < 1.25


def test_dry_run_reports_decision(tmp_path, capsys):
    crf, _ = _write_gate_pair(str(tmp_path), biased=False)
    assert g1only_dq.main(['--dry-run', crf]) == 0
    out = capsys.readouterr().out
    assert '36 single-group px skip: m=1.3' in out
    with fits.open(crf) as h:
        assert g1only_dq.DECISION_KEY not in h[0].header


@pytest.mark.parametrize('biased', [True, False])
def test_call_sites_follow_the_gate(tmp_path, biased):
    """load_data, the satstar fit-switch key and the satstar cache key all
    route through the gate: flagged + keyed for biased frames only."""
    from jwst_gc_pipeline.photometry.crowdsource_catalogs_long import load_data
    from jwst_gc_pipeline.reduction.saturated_star_finding import satstar_fit_switch_signature
    from jwst_gc_pipeline.photometry import satstar_cache
    crf, _ = _write_gate_pair(str(tmp_path), biased=biased)
    fh, im1, data, *_ = load_data(crf)
    try:
        assert bool(np.isnan(data[20:26, 20:26]).all()) is biased
        assert np.isfinite(data[~np.isnan(data)]).all()
    finally:
        fh.close()
    assert ('g1' in satstar_fit_switch_signature(crf).split('_')) is biased
    src = satstar_cache.satstar_content_key.__code__.co_names
    assert 'g1only_dq' in src and 'signature' in src


def test_multi_integration_measures_only_single_group_integrations(tmp_path):
    """G1 enters m only from integrations where group 1 is the lone usable
    group; an integration saturated in every group carries no rate there."""
    crf, npx = _write_gate_pair(str(tmp_path), biased=True)
    rpath = os.path.join(str(tmp_path), f'{STEM}_ramp.fits')
    with fits.open(rpath) as r:
        sci1, gdq1, hdr = r['SCI'].data.copy(), r['GROUPDQ'].data.copy(), r[0].header.copy()
    sci2, gdq2 = sci1.copy(), gdq1.copy()
    sci2[0, :, 20:26, 20:26] = 65000.0                 # fully saturated integration
    gdq2[0, :, 20:26, 20:26] = SAT
    fits.HDUList([fits.PrimaryHDU(header=hdr),
                  fits.ImageHDU(np.concatenate([sci1, sci2]), name='SCI'),
                  fits.ImageHDU(np.concatenate([gdq1, gdq2]), name='GROUPDQ')]
                 ).writeto(rpath, overwrite=True)
    with fits.open(crf) as h:
        mask, dec = g1only_dq.frame_plan(crf, h[0].header, h['SCI'].data, h['DQ'].data)
    assert mask.sum() == npx
    assert dec.flag and dec.m == pytest.approx(1.0, abs=0.03) and dec.n == npx
