"""Satstar severity-gate drops are persisted and handed off to daophot.

``find_saturated_stars`` drops a DQ-SATURATED component with no NaN-variance
pixel whose data peak is below ``SAT_SEVERITY_FLOOR``: the star still has a
valid rate, so it belongs to the daophot channel.  The drop used to write
nothing, so no hand-off reached it, and daophot masked its any-group SATURATED
core and deleted fits next to it -- the star was modelled by neither channel
(10678 F480M: ~200-300 such stars per NRCBLONG frame, median core ~1200
MJy/sr).
"""
import builtins

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

import jwst_gc_pipeline.photometry.cataloging as C
import jwst_gc_pipeline.photometry.crowdsource_catalogs_long as L
import jwst_gc_pipeline.reduction.saturated_star_finding as ssf
from jwst_gc_pipeline.photometry.tests.test_gate_reject_handoff import (
    BIG, BIG_AMP, SHAPE, _ctx_pass, _gaussian_grid_psf_fwhm, _has_fit_near,
    _prep_options, _write_crf)

SAT = L.dqflags.pixel['SATURATED']
FWHM = 2.574


# ---------------------------------------------------------------------------
# satstar side: recording the drops
# ---------------------------------------------------------------------------

def _star(img, x, y, amp, fwhm=FWHM):
    yy, xx = np.mgrid[:img.shape[0], :img.shape[1]]
    sig = fwhm / 2.3548
    img += amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sig ** 2))


def _hdul(stars, shape=(120, 120)):
    """SCI/DQ/VAR_POISSON frame; each star's pixels above 40% of its peak are
    DQ-SATURATED with a finite rate (late-group saturation)."""
    img = np.full(shape, 10.0)
    dq = np.zeros(shape, dtype=np.uint32)
    for x, y, amp in stars:
        one = np.zeros(shape)
        _star(one, x, y, amp)
        img += one
        dq[one > 0.4 * amp] |= SAT
    return fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(img.astype('float32'), name='SCI'),
        fits.ImageHDU(dq, name='DQ'),
        fits.ImageHDU(np.ones(shape, dtype='float32'), name='VAR_POISSON'),
    ])


WEAK = (40.0, 60.0, 1200.0)      # below 0.35 x 5000: dropped, never re-seeded
RESEEDED = (80.0, 60.0, 3000.0)  # below 5000 but above 1750: sub-floor re-seed
DEEP = (60.0, 25.0, 2.0e4)       # truly lost core (NaN variance): kept


def _scene(stars):
    h = _hdul(stars)
    for x, y, amp in stars:
        if amp >= 1e4:
            h['VAR_POISSON'].data[int(y), int(x)] = np.nan
    return h


def test_severity_gate_records_every_drop():
    drops = []
    ssf.find_saturated_stars(_hdul([WEAK, RESEEDED]), severity_floor=5000.0,
                             severity_drops=drops)
    assert len(drops) == 2
    by_x = sorted(drops, key=lambda r: r['xcentroid'])
    for rec, (x, y, amp) in zip(by_x, (WEAK, RESEEDED)):
        assert (rec['xcentroid'], rec['ycentroid']) == (x, y)
        assert rec['core_peak'] == pytest.approx(amp + 10.0, rel=1e-3)
        assert rec['npix'] > 0


def test_drop_table_keeps_only_drops_nothing_reseeded():
    drops = []
    saturated, *_ = ssf.find_saturated_stars(
        _scene([WEAK, RESEEDED, DEEP]), severity_floor=5000.0,
        severity_drops=drops)
    assert saturated[int(RESEEDED[1]), int(RESEEDED[0])], 'sub-floor re-seed'
    tab = ssf.severity_drop_table(drops, saturated, 5000.0)
    assert list(tab['xcentroid']) == [WEAK[0]]
    assert list(tab['ycentroid']) == [WEAK[1]]
    assert list(tab['sat_severity_floor']) == [5000.0]
    assert tab.meta['NDROPPED'] == 2
    assert tuple(tab.colnames) == ssf.SEVERITY_DROP_COLUMNS


def test_drop_table_keeps_every_drop_when_the_gate_empties_the_frame():
    """When the gate drops every component, find_saturated_stars returns
    before any re-seeding, so no drop was taken back."""
    drops = []
    saturated, *_ = ssf.find_saturated_stars(
        _hdul([WEAK, RESEEDED]), severity_floor=5000.0, severity_drops=drops)
    assert not saturated.any()
    tab = ssf.severity_drop_table(drops, saturated, 5000.0)
    assert sorted(tab['xcentroid']) == [WEAK[0], RESEEDED[0]]


def test_no_gate_no_records():
    drops = []
    ssf.find_saturated_stars(_hdul([WEAK]), severity_floor=0.0,
                             severity_drops=drops)
    assert drops == []


def test_nan_variance_component_is_not_dropped():
    """A truly-lost core (NaN variance) is a real saturated star: kept."""
    h = _hdul([WEAK])
    h['VAR_POISSON'].data[60, 40] = np.nan
    drops = []
    ssf.find_saturated_stars(h, severity_floor=5000.0, severity_drops=drops)
    assert drops == []


def test_empty_drop_table_round_trips(tmp_path):
    """An empty table is written too, so a rerun overwrites a stale file."""
    tab = ssf.severity_drop_table([], np.zeros((4, 4), bool), 5000.0)
    fn = tmp_path / 'x_satstar_severity_dropped.fits'
    tab.write(fn)
    back = Table.read(fn)
    assert len(back) == 0 and tuple(back.colnames) == ssf.SEVERITY_DROP_COLUMNS


def test_remove_saturated_stars_writes_drops_when_nothing_is_accepted(
        tmp_path, monkeypatch):
    fn = str(tmp_path / 'jw01234001001_02101_00001_nrcblong_crf.fits')
    _hdul([WEAK]).writeto(fn)
    table = ssf.severity_drop_table(
        [{'xcentroid': 40.0, 'ycentroid': 60.0, 'core_peak': 1210.0,
          'npix': 9}], np.zeros((120, 120), bool), 5000.0)

    def fake_get(fh, **kw):
        builtins.satstar_severity_dropped = table
        return None
    monkeypatch.setattr(ssf, 'get_saturated_stars', fake_get)
    monkeypatch.setattr(ssf, '_zeroframe_fit_enabled', lambda: False)
    ssf.remove_saturated_stars(fn, file_suffix='_m7')
    out = Table.read(fn.replace('.fits', '_m7_satstar_severity_dropped.fits'))
    assert list(out['xcentroid']) == [40.0]


def test_remove_saturated_stars_clears_a_previous_frames_drops(
        tmp_path, monkeypatch):
    """The builtins side channel is reset per call: a run that never reaches
    the gate must not write the previous frame's drops."""
    fn = str(tmp_path / 'jw01234001001_02101_00002_nrcblong_crf.fits')
    _hdul([WEAK]).writeto(fn)
    builtins.satstar_severity_dropped = ssf.severity_drop_table(
        [{'xcentroid': 1.0, 'ycentroid': 1.0, 'core_peak': 1.0, 'npix': 1}],
        np.zeros((4, 4), bool), 5000.0)
    monkeypatch.setattr(ssf, 'get_saturated_stars', lambda fh, **kw: None)
    monkeypatch.setattr(ssf, '_zeroframe_fit_enabled', lambda: False)
    ssf.remove_saturated_stars(fn, file_suffix='_m7')
    assert not (tmp_path / ('jw01234001001_02101_00002_nrcblong_crf_m7'
                            '_satstar_severity_dropped.fits')).exists()


def test_cache_reuse_copies_the_drop_file():
    from jwst_gc_pipeline.photometry.satstar_cache import (
        SATSTAR_PRODUCT_SUFFIXES)
    assert '_satstar_severity_dropped.fits' in SATSTAR_PRODUCT_SUFFIXES


# ---------------------------------------------------------------------------
# daophot side: the hand-off
# ---------------------------------------------------------------------------

def test_drop_path_sits_beside_the_rejected_file():
    assert (C.severity_dropped_path('/p/a_crf_m7_satstar_rejected.fits')
            == '/p/a_crf_m7_satstar_severity_dropped.fits')
    assert C.severity_dropped_path(None) is None
    assert C.severity_dropped_path('/p/other.fits') is None


def _write_drops(path, xy):
    ssf.severity_drop_table(
        [{'xcentroid': x, 'ycentroid': y, 'core_peak': 1200.0, 'npix': 20}
         for x, y in xy], np.zeros((4, 4), bool), 5000.0).write(path)


def test_reader_hands_off_drops_away_from_accepted_satstars(tmp_path):
    fn = str(tmp_path / 'f_m7_satstar_severity_dropped.fits')
    _write_drops(fn, [(10.0, 10.0), (50.0, 50.0)])
    acc = Table({'xcentroid': [51.0], 'ycentroid': [50.5]})
    xy, r = C._severity_drop_handoff_xy(fn, acc, FWHM)
    np.testing.assert_allclose(xy, [[10.0, 10.0]])
    assert r == pytest.approx(max(1.0, 0.5 * FWHM))


@pytest.mark.parametrize('rows', [None, []])
def test_reader_without_drops_hands_off_nothing(tmp_path, rows):
    fn = str(tmp_path / 'f_m7_satstar_severity_dropped.fits')
    if rows is not None:
        _write_drops(fn, rows)
    assert C._severity_drop_handoff_xy(fn, None, FWHM) == (None, 0.0)


def _prepare_with_drop(tmp_path, monkeypatch, *, write_drop=True):
    for k, v in (('NIRCAM_SATSTAR_TIGHT_BOUND', '0'),
                 ('SATSTAR_COMPONENT_OVERLAP_FRAC', '0'),
                 ('NIRCAM_SATSTAR_LOCK_POS', '0'),
                 ('NIRCAM_SATSTAR_RECOVERED_CAP', '0')):
        monkeypatch.setenv(k, v)
    d = tmp_path / 'F480M' / 'pipeline'
    d.mkdir(parents=True)
    fn = str(d / 'jw10678132001_02101_00004_nrcalong_destreak_o132_crf.fits')
    img = _write_crf(fn)
    fits.PrimaryHDU(np.zeros(SHAPE)).writeto(
        fn.replace('.fits', '_m7_satstar_model.fits'))
    # the satstar run rejected nothing: the star never reached its fit
    if write_drop:
        _write_drops(fn.replace('.fits', '_m7_satstar_severity_dropped.fits'),
                     [(round(BIG[0]), round(BIG[1]))])
    monkeypatch.setattr(L, 'get_psf_model',
                        lambda *a, **k: (_gaussian_grid_psf_fwhm(FWHM), None))
    monkeypatch.setattr(L, 'load_or_make_satstar_catalog',
                        lambda *a, **k: None)
    ctx = C._prepare_frame_for_photometry(
        _prep_options(), 'F480M', 'nrcalong', '132', str(tmp_path), fn,
        '10678', exposurenumber=4, visit_id=1, vgroup_id='02101',
        bg_boxsizes=None, use_webbpsf=True, pupil='clear',
        resbg_path=None, satstar_label='m7')
    return ctx, img


def _core():
    core = np.zeros(SHAPE, bool)
    core[47:54, 37:44] = True
    return core


def test_prepare_fits_a_severity_dropped_star(tmp_path, monkeypatch):
    """End to end: the dropped star's late-group core goes back to the fit and
    daophot measures its flux."""
    for k in ('DAOPHOT_HANDOFF_SEVERITY_DROPS', 'DAOPHOT_HANDOFF_UNACCEPTED_SAT'):
        monkeypatch.delenv(k, raising=False)
    ctx, img = _prepare_with_drop(tmp_path, monkeypatch)
    assert ctx.handoff_xy is not None
    assert not ctx.mask[_core()].any()
    np.testing.assert_allclose(ctx.nan_replaced_data[_core()], img[_core()],
                               rtol=1e-5)
    res = _ctx_pass(ctx, handoff=True)
    assert _has_fit_near(res, BIG)
    d = np.hypot(np.asarray(res['x_fit']) - BIG[0],
                 np.asarray(res['y_fit']) - BIG[1])
    flux = float(np.asarray(res['flux_fit'])[np.argmin(d)])
    true = BIG_AMP * 2 * np.pi * (FWHM / 2.3548) ** 2
    assert flux == pytest.approx(true, rel=0.05)


def test_prepare_switch_off_leaves_the_star_unmodelled(tmp_path, monkeypatch):
    """DAOPHOT_HANDOFF_SEVERITY_DROPS=0 restores the old behaviour: core masked
    and model-filled, no daophot row."""
    monkeypatch.setenv('DAOPHOT_HANDOFF_SEVERITY_DROPS', '0')
    monkeypatch.delenv('DAOPHOT_HANDOFF_UNACCEPTED_SAT', raising=False)
    ctx, _ = _prepare_with_drop(tmp_path, monkeypatch)
    assert ctx.handoff_xy is None
    assert ctx.mask[_core()].all()
    assert not _has_fit_near(_ctx_pass(ctx, handoff=False), BIG)


def test_prepare_without_drop_file_is_unchanged(tmp_path, monkeypatch):
    """A satstar catalog cached before the drop file existed hands off nothing."""
    for k in ('DAOPHOT_HANDOFF_SEVERITY_DROPS', 'DAOPHOT_HANDOFF_UNACCEPTED_SAT'):
        monkeypatch.delenv(k, raising=False)
    ctx, _ = _prepare_with_drop(tmp_path, monkeypatch, write_drop=False)
    assert ctx.handoff_xy is None
    assert ctx.mask[_core()].all()


def test_component_handoff_supersedes_the_drop_file(tmp_path, monkeypatch):
    """With DAOPHOT_HANDOFF_UNACCEPTED_SAT on, every unaccepted component is
    handed off already; the drop file is not read on top of it."""
    monkeypatch.setenv('DAOPHOT_HANDOFF_UNACCEPTED_SAT', '1')
    calls = []
    real = C._severity_drop_handoff_xy
    monkeypatch.setattr(C, '_severity_drop_handoff_xy',
                        lambda *a, **k: calls.append(a) or real(*a, **k))
    ctx, _ = _prepare_with_drop(tmp_path, monkeypatch)
    assert calls == []
    assert ctx.handoff_xy is not None


def test_drop_switch_rejects_typos(tmp_path, monkeypatch):
    monkeypatch.setenv('DAOPHOT_HANDOFF_SEVERITY_DROPS', 'ture')
    monkeypatch.delenv('DAOPHOT_HANDOFF_UNACCEPTED_SAT', raising=False)
    with pytest.raises(ValueError, match='DAOPHOT_HANDOFF_SEVERITY_DROPS'):
        C._daophot_handoff_xy(np.zeros(SHAPE, np.uint32), np.zeros(SHAPE),
                              None, str(tmp_path / 'x_satstar_rejected.fits'),
                              FWHM)
