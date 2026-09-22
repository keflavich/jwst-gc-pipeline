"""Satstar implied-peak-gate rejects must reach the daophot channel (#925).

The post-fit severity gate in ``get_saturated_stars`` rejects a fitted
"satstar" whose model cannot reach the filter's saturation level, and its
comment promised the star "falls through to the daophot channel".  It did
not: the star's core still carries any-group SATURATED DQ (late-group
saturation, valid rate), so ``_filter_near_saturation`` deleted the daophot fit
and the star vanished from both channels (gc-treasury o132 F480M: 403 stars at
12.0-12.9 with a satstar fit and no catalog row in m2-m7).

These tests pin the hand-off:

* a gate-rejected star with a late-group-SATURATED core keeps its daophot fit,
  including when its SATURATED core is larger than the 5x5 fit box (the
  gc-treasury F480M case: 12-65 px cores): the frame preparation gives the
  valid-rate core pixels back to the fit and keeps them out of the satstar fill;
* a spike / wing detection on SATURATED pixels a few px from that star, and one
  on an ACCEPTED satstar's spike, are still vetoed;
* fit-quality rejects and rejects on an accepted satstar's core are never
  handed off;
* with no implied-peak-gate rejects the pass is identical to the old one.
"""
import types

import numpy as np
import pytest
from astropy.table import Table
from astropy.nddata import NDData
from photutils.psf import GriddedPSFModel
from photutils.psf import SourceGrouper

from jwst_gc_pipeline.photometry import cataloging as C
from jwst_gc_pipeline.photometry import crowdsource_catalogs_long as L

SAT = L.dqflags.pixel['SATURATED']
FWHM = 2.5
SHAPE = (100, 100)
STAR = (30.3, 40.2)          # gate-rejected, lightly saturated star (x, y)
SPIKE = (35.0, 40.0)         # spurious detection on SATURATED pixels 4.7 px away
BRIGHT = (70.0, 70.0)        # accepted satstar
BRIGHT_SPIKE = (77.0, 70.0)  # spurious detection on the accepted satstar's spike


def _scene():
    """Frame with one lightly saturated star plus an accepted satstar.

    The light star's 2x2 core is SATURATED (late group: finite data, no
    DO_NOT_USE).  A 1-px-wide SATURATED streak runs from x=34 to x=36 at its
    row (a charge-bleed / spike segment carrying a spurious detection).  The
    accepted satstar has its own saturated core and a spike at x=76..78.
    """
    yy, xx = np.mgrid[:SHAPE[0], :SHAPE[1]]
    sig = FWHM / 2.3548
    img = np.full(SHAPE, 10.0)
    # 150-amplitude bumps on the streak / spike make the spurious detections
    # stay put in the fit instead of sliding onto the neighbouring star.
    for (x0, y0), amp in ((STAR, 3000.0), (BRIGHT, 50000.0),
                          (SPIKE, 150.0), (BRIGHT_SPIKE, 1500.0)):
        img += amp * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sig ** 2))
    rng = np.random.default_rng(3)
    img += rng.normal(0, 1.0, SHAPE)
    dq = np.zeros(SHAPE, dtype=np.uint32)
    dq[40:42, 30:32] |= SAT              # light star core (late-group)
    dq[40, 34:37] |= SAT                 # streak under SPIKE
    dq[68:73, 68:73] |= SAT              # accepted satstar core
    dq[70, 76:79] |= SAT                 # accepted satstar spike
    return img, dq


def _gaussian_grid_psf():
    """Single-point GriddedPSFModel of a Gaussian (the pipeline renders model
    stamps through ``GriddedPSFModel.evaluate``)."""
    ov = 4
    n = 25 * ov + 1
    c = (n - 1) / 2
    yy, xx = np.mgrid[:n, :n]
    sig = FWHM / 2.3548 * ov
    p = np.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2 * sig ** 2))
    p /= p.sum() / ov ** 2
    return GriddedPSFModel(NDData(p[None], meta={'grid_xypos': [(0, 0)],
                                                 'oversampling': ov}))


def _options():
    return types.SimpleNamespace(group=False, satstar_artifact_sigK=3.0,
                                 satstar_artifact_ratio=1.0)


def _run_pass(img, dq, seed_xy, *, handoff_xy=None, handoff_radius_pix=0.0):
    err = np.ones(SHAPE)
    mask = (dq & SAT) != 0             # the fit masks any-group SATURATED
    seed = Table()
    seed['x_init'] = np.array([p[0] for p in seed_xy], float)
    seed['y_init'] = np.array([p[1] for p in seed_xy], float)
    seed['flux_init'] = np.full(len(seed_xy), 2.0e4)
    psf = _gaussian_grid_psf()
    res, _, _ = C._manual_phot_pass(
        data=img, mask=mask, err=err, bad=np.zeros(SHAPE, bool),
        dao_psf_model=psf, init_params=seed, aperture_radius_pix=2 * FWHM,
        localbkg_inner=6, localbkg_outer=10, grouper=SourceGrouper(2 * FWHM),
        options=_options(), dq=dq, satstar_model_subtracted=np.zeros(SHAPE),
        label='t', near_sat_dist_pix=1.0,
        handoff_xy=handoff_xy, handoff_radius_pix=handoff_radius_pix)
    return res


def _has_fit_near(res, xy, r=1.0):
    if res is None or len(res) == 0:
        return False
    d = np.hypot(np.asarray(res['x_fit'], float) - xy[0],
                 np.asarray(res['y_fit'], float) - xy[1])
    return bool(np.any(d <= r))


def _write_rejected(path, rows):
    t = Table()
    t['xcentroid'] = np.array([r[0] for r in rows], float)
    t['ycentroid'] = np.array([r[1] for r in rows], float)
    t['flux_fit'] = np.full(len(rows), 1.0e4)
    t['reject_reason'] = np.array([r[2] for r in rows])
    t.write(path, overwrite=True)
    return str(path)


def _accepted_table():
    t = Table()
    t['xcentroid'] = np.array([BRIGHT[0]])
    t['ycentroid'] = np.array([BRIGHT[1]])
    return t


# ---------------------------------------------------------------------------
# The loader: which rejects are handed off
# ---------------------------------------------------------------------------

def test_loader_hands_off_only_implied_peak_gate_rejects(tmp_path):
    path = _write_rejected(tmp_path / 'r.fits', [
        (STAR[0] + 0.1, STAR[1] - 0.2, 'implied_peak_gate'),
        (50.0, 20.0, 'fit_quality_gate'),            # garbage fit: not handed off
        (BRIGHT[0] + 0.5, BRIGHT[1], 'implied_peak_gate'),  # on an accepted core
        (np.nan, 10.0, 'implied_peak_gate'),         # no position
    ])
    xy, r = C._gate_reject_handoff_xy(path, _accepted_table(), FWHM, label='t')
    assert xy is not None and xy.shape == (1, 2)
    np.testing.assert_allclose(xy[0], [STAR[0] + 0.1, STAR[1] - 0.2])
    assert r == pytest.approx(max(1.0, 0.5 * FWHM))


@pytest.mark.parametrize('rows', [None, [], [(10.0, 10.0, 'fit_quality_gate')]])
def test_loader_returns_none_without_implied_peak_rejects(tmp_path, rows):
    path = (str(tmp_path / 'missing.fits') if rows is None
            else _write_rejected(tmp_path / 'r.fits', rows) if rows
            else str(tmp_path / 'absent.fits'))
    assert C._gate_reject_handoff_xy(path, None, FWHM) == (None, 0.0)


# ---------------------------------------------------------------------------
# The pass: the star survives, spike/wing detections do not
# ---------------------------------------------------------------------------

def test_gate_reject_is_vetoed_without_handoff():
    """The #925 failure: without the hand-off the lightly saturated star's
    daophot fit is deleted by the near-saturation filter."""
    img, dq = _scene()
    res = _run_pass(img, dq, [STAR, SPIKE, BRIGHT_SPIKE])
    assert not _has_fit_near(res, STAR)


def test_gate_reject_survives_with_handoff(tmp_path):
    img, dq = _scene()
    path = _write_rejected(tmp_path / 'r.fits',
                           [(STAR[0] + 0.1, STAR[1] - 0.2, 'implied_peak_gate')])
    hxy, hr = C._gate_reject_handoff_xy(path, _accepted_table(), FWHM)
    res = _run_pass(img, dq, [STAR, SPIKE, BRIGHT_SPIKE],
                    handoff_xy=hxy, handoff_radius_pix=hr)
    assert _has_fit_near(res, STAR), 'the gate-rejected star must keep its fit'
    # the fit is of the star, not of a wing: flux within 10% of the truth
    d = np.hypot(np.asarray(res['x_fit']) - STAR[0],
                 np.asarray(res['y_fit']) - STAR[1])
    flux = float(np.asarray(res['flux_fit'])[np.argmin(d)])
    true = 3000.0 * 2 * np.pi * (FWHM / 2.3548) ** 2
    assert flux == pytest.approx(true, rel=0.10)
    # spike / wing detections on SATURATED pixels are still vetoed
    assert not _has_fit_near(res, SPIKE, r=1.0)
    assert not _has_fit_near(res, BRIGHT_SPIKE, r=1.0)


def test_fit_quality_reject_is_not_handed_off(tmp_path):
    img, dq = _scene()
    path = _write_rejected(tmp_path / 'r.fits',
                           [(STAR[0], STAR[1], 'fit_quality_gate')])
    hxy, hr = C._gate_reject_handoff_xy(path, None, FWHM)
    res = _run_pass(img, dq, [STAR], handoff_xy=hxy, handoff_radius_pix=hr)
    assert not _has_fit_near(res, STAR)


def test_no_handoff_is_identical_to_previous_pass():
    """Frames without implied-peak-gate rejects (MIRI, most fields) take the
    old path exactly: same rows, same values."""
    img, dq = _scene()
    seed = [STAR, SPIKE, BRIGHT_SPIKE, (15.0, 80.0)]
    a = _run_pass(img, dq, seed)
    b = _run_pass(img, dq, seed, handoff_xy=None, handoff_radius_pix=0.0)
    assert a.colnames == b.colnames and len(a) == len(b)
    for c in ('x_fit', 'y_fit', 'flux_fit'):
        np.testing.assert_array_equal(np.asarray(a[c]), np.asarray(b[c]))


# ---------------------------------------------------------------------------
# The filter itself
# ---------------------------------------------------------------------------

class _FakePhot:
    def __init__(self, results):
        self.results = results


def test_filter_exempts_only_within_radius():
    dq = np.zeros(SHAPE, dtype=np.uint32)
    dq[50, 50] |= SAT
    dq[50, 53] |= SAT
    res = Table()
    res['id'] = [1, 2]
    res['x_fit'] = [50.2, 53.0]      # at the reject; 2.8 px away on a SAT px
    res['y_fit'] = [50.1, 50.0]
    res['flux_fit'] = [1e4, 1e3]
    phot = _FakePhot(res)
    n = L._filter_near_saturation(phot, dq, max_sat_dist_pix=1.0, label='t',
                                  handoff_xy=np.array([[50.0, 50.0]]),
                                  handoff_radius_pix=1.25)
    assert n == 1
    assert list(phot.results['id']) == [1]


# ---------------------------------------------------------------------------
# Wiring: do_photometry_step_manual hands the rejects to every pass
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('target', ['gc-treasury', 'w51'])
def test_manual_step_passes_handoff_for_every_target(tmp_path, monkeypatch, target):
    path = _write_rejected(tmp_path / 'r.fits',
                           [(STAR[0], STAR[1], 'implied_peak_gate')])
    ctx = types.SimpleNamespace(
        im1=None, fwhm_pix=FWHM, satstar_rejected_path=path,
        handoff_xy=C._gate_reject_handoff_xy(path, None, FWHM)[0],
        handoff_radius=C._gate_reject_handoff_xy(path, None, FWHM)[1],
        satstar_table=None, nan_replaced_data=np.zeros(SHAPE),
        mask=np.zeros(SHAPE, bool), err=np.ones(SHAPE), bad=np.zeros(SHAPE, bool),
        dao_psf_model=None, aperture_radius_pix=5.0, localbkg_inner=6,
        localbkg_outer=10, grouper=None, dqarr=None,
        satstar_model_subtracted=None, data=np.zeros(SHAPE), ww=None)
    seen = {}
    monkeypatch.setattr(C, '_prepare_frame_for_photometry',
                        lambda *a, **k: ctx)
    monkeypatch.setattr(C, '_build_manual_seed', lambda **k: Table())

    def _fake_pass(**kw):
        seen.update(kw)
        return Table(), np.zeros(SHAPE), None
    monkeypatch.setattr(C, '_manual_phot_pass', _fake_pass)
    monkeypatch.setattr(C, '_save_manual_pass', lambda **k: None)
    opts = types.SimpleNamespace(target=target)
    C.do_photometry_step_manual(opts, 'F480M', 'nrcalong', 'nrcalong', '132',
                                str(tmp_path), 'f.fits', '10678',
                                manual_phase='m7')
    assert seen['handoff_xy'] is not None
    np.testing.assert_allclose(seen['handoff_xy'], [[STAR[0], STAR[1]]])
    assert seen['handoff_radius_pix'] == pytest.approx(max(1.0, 0.5 * FWHM))


# ---------------------------------------------------------------------------
# Frame preparation: a core larger than the fit box (the o132 F480M regime)
# ---------------------------------------------------------------------------

BIG = (40.3, 50.2)            # gate reject with a 7x7 late-group SATURATED core
BIG_AMP = 3000.0


def _write_crf(path, with_truly_lost=False):
    from astropy.io import fits
    from astropy.wcs import WCS
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crpix = [50.5, 50.5]
    w.wcs.crval = [266.55, -28.80]
    w.wcs.cdelt = [-0.063 / 3600, 0.063 / 3600]
    yy, xx = np.mgrid[:SHAPE[0], :SHAPE[1]]
    sig = 2.574 / 2.3548
    img = 10.0 + BIG_AMP * np.exp(-((xx - BIG[0]) ** 2 + (yy - BIG[1]) ** 2)
                                  / (2 * sig ** 2))
    img += np.random.default_rng(5).normal(0, 1.0, SHAPE)
    dq = np.zeros(SHAPE, dtype=np.uint32)
    dq[47:54, 37:44] |= SAT                       # 49 px > 25 px fit box
    if with_truly_lost:
        dq[50, 40] |= L.dqflags.pixel['DO_NOT_USE']
        img[50, 40] = np.nan
    h0 = fits.Header()
    h0['INSTRUME'] = 'NIRCAM'
    h0['TELESCOP'] = 'JWST'
    h0['DATE-OBS'] = '2025-01-01'
    fits.HDUList([
        fits.PrimaryHDU(header=h0),
        fits.ImageHDU(img.astype('float32'), header=w.to_header(), name='SCI'),
        fits.ImageHDU(np.ones(SHAPE, dtype='float32'), name='ERR'),
        fits.ImageHDU(dq, name='DQ'),
    ]).writeto(path, overwrite=True)
    return img


def _prep_options():
    return types.SimpleNamespace(
        desaturated=False, epsf=False, blur=False, group=False, cutout_region='',
        bgsub=False, use_iter3_residual_bg=False, each_exposure=True,
        target='gc-treasury', max_group_size='unlimited',
        fit_satstar_outside_fov=False, satstar_partner_seed=False,
        satstar_zeroframe_recover=False, satstar_ramp_recover=False,
        deblend_satstars=False, satstar_artifact_sigK=3.0,
        satstar_artifact_ratio=1.0)


def _prepare(tmp_path, monkeypatch, *, reject_reason='implied_peak_gate',
             with_truly_lost=False):
    from astropy.io import fits
    for k, v in (('NIRCAM_SATSTAR_TIGHT_BOUND', '0'),
                 ('SATSTAR_COMPONENT_OVERLAP_FRAC', '0'),
                 ('NIRCAM_SATSTAR_LOCK_POS', '0'),
                 ('NIRCAM_SATSTAR_RECOVERED_CAP', '0')):
        monkeypatch.setenv(k, v)
    d = tmp_path / 'F480M' / 'pipeline'
    d.mkdir(parents=True)
    fn = str(d / 'jw10678132001_02101_00004_nrcalong_destreak_o132_crf.fits')
    img = _write_crf(fn, with_truly_lost=with_truly_lost)
    # an accepted satstar elsewhere: its (zero here) model file triggers the fill
    fits.PrimaryHDU(np.zeros(SHAPE)).writeto(
        fn.replace('.fits', '_m7_satstar_model.fits'))
    _write_rejected(fn.replace('.fits', '_m7_satstar_rejected.fits'),
                    [(BIG[0], BIG[1], reject_reason)])
    monkeypatch.setattr(L, 'get_psf_model',
                        lambda *a, **k: (_gaussian_grid_psf_fwhm(2.574), None))
    monkeypatch.setattr(L, 'load_or_make_satstar_catalog',
                        lambda *a, **k: None)
    ctx = C._prepare_frame_for_photometry(
        _prep_options(), 'F480M', 'nrcalong', '132', str(tmp_path), fn,
        '10678', exposurenumber=4, visit_id=1, vgroup_id='02101',
        bg_boxsizes=None, use_webbpsf=True, pupil='clear', resbg_path=None,
        satstar_label='m7')
    return ctx, img


def _gaussian_grid_psf_fwhm(fwhm):
    ov = 4
    n = 25 * ov + 1
    c = (n - 1) / 2
    yy, xx = np.mgrid[:n, :n]
    sig = fwhm / 2.3548 * ov
    p = np.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2 * sig ** 2))
    p /= p.sum() / ov ** 2
    return GriddedPSFModel(NDData(p[None], meta={'grid_xypos': [(0, 0)],
                                                 'oversampling': ov}))


def _ctx_pass(ctx, *, handoff):
    seed = Table()
    seed['x_init'] = [BIG[0]]
    seed['y_init'] = [BIG[1]]
    seed['flux_init'] = [2.0e4]
    res, _, _ = C._manual_phot_pass(
        data=ctx.nan_replaced_data, mask=ctx.mask, err=ctx.err, bad=ctx.bad,
        dao_psf_model=ctx.dao_psf_model, init_params=seed,
        aperture_radius_pix=ctx.aperture_radius_pix,
        localbkg_inner=ctx.localbkg_inner, localbkg_outer=ctx.localbkg_outer,
        grouper=None, options=_prep_options(), dq=ctx.dqarr,
        satstar_model_subtracted=np.zeros(SHAPE), label='t',
        near_sat_dist_pix=1.0,
        handoff_xy=ctx.handoff_xy if handoff else None,
        handoff_radius_pix=ctx.handoff_radius if handoff else 0.0)
    return res


def test_prepare_returns_valid_rate_core_to_the_fit(tmp_path, monkeypatch):
    ctx, img = _prepare(tmp_path, monkeypatch)
    core = np.zeros(SHAPE, bool)
    core[47:54, 37:44] = True
    assert ctx.handoff_xy is not None
    assert not ctx.mask[core].any(), 'late-group core must be fittable'
    # the satstar fill must not overwrite the measured core with the model
    np.testing.assert_allclose(ctx.nan_replaced_data[core], img[core],
                               rtol=1e-5)
    res = _ctx_pass(ctx, handoff=True)
    assert _has_fit_near(res, BIG), 'the handed-off star must be catalogued'
    d = np.hypot(np.asarray(res['x_fit']) - BIG[0],
                 np.asarray(res['y_fit']) - BIG[1])
    flux = float(np.asarray(res['flux_fit'])[np.argmin(d)])
    true = BIG_AMP * 2 * np.pi * (2.574 / 2.3548) ** 2
    assert flux == pytest.approx(true, rel=0.05)


def test_prepare_without_implied_peak_reject_keeps_core_masked(tmp_path,
                                                               monkeypatch):
    """A fit-quality reject is not handed off: the frame is prepared exactly as
    before (core masked and model-filled), and the star has no daophot row."""
    ctx, _ = _prepare(tmp_path, monkeypatch, reject_reason='fit_quality_gate')
    assert ctx.handoff_xy is None
    core = np.zeros(SHAPE, bool)
    core[47:54, 37:44] = True
    assert ctx.mask[core].all()
    assert np.all(ctx.nan_replaced_data[core] == 0.0)
    assert not _has_fit_near(_ctx_pass(ctx, handoff=False), BIG)


def test_prepare_never_restores_truly_lost_pixels(tmp_path, monkeypatch):
    ctx, _ = _prepare(tmp_path, monkeypatch, with_truly_lost=True)
    assert ctx.mask[50, 40], 'SATURATED & DO_NOT_USE has no rate; keep it masked'
    assert not ctx.mask[49, 39]


def test_restore_skips_component_holding_an_accepted_satstar():
    dq = np.zeros(SHAPE, dtype=np.uint32)
    dq[20:30, 20:30] |= SAT
    data = np.ones(SHAPE)
    bad = np.zeros(SHAPE, bool)
    acc = Table({'xcentroid': [25.0], 'ycentroid': [25.0]})
    r = C._handoff_restore_pixels(dq, data, bad, np.array([[21.0, 21.0]]),
                                  acc, FWHM)
    assert not r.any()
    r = C._handoff_restore_pixels(dq, data, bad, np.array([[21.0, 21.0]]),
                                  None, FWHM)
    assert r.any()
    # capped at 3 FWHM from the hand-off position
    yy, xx = np.nonzero(r)
    assert np.hypot(xx - 21.0, yy - 21.0).max() <= 3 * FWHM
