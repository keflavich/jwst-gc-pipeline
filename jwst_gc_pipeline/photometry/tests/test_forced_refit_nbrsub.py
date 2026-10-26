"""Neighbour-subtracted overshoot refit (#932), default on.

``_manual_phot_pass`` re-fits each overshooting source as forced photometry
at its seed.  The solve used to run on the raw frame as a single source in a
5x5 box, so light from neighbours a few px away entered the flux: a ring or
wing seed next to a bright star took part of the star's flux and the
residual showed a hole at the seed.  The solve now runs on the frame with
every non-overshooting source's model removed, and overshooting sources are
solved brightest first, each with the brighter overshooting sources' refit
models removed (``_overshoot_refit_tiers``).  Env ``DAOPHOT_REFIT_NBRSUB=0``
restores the raw-frame solve.
"""
import types

import numpy as np
import pytest
from astropy.nddata import NDData
from astropy.table import Table
from photutils.psf import GriddedPSFModel, SourceGrouper

from jwst_gc_pipeline.photometry import cataloging as C

FWHM = 2.5


def _gaussian_grid_psf():
    ov = 4
    n = 25 * ov + 1
    c = (n - 1) / 2
    yy, xx = np.mgrid[:n, :n]
    sig = FWHM / 2.3548 * ov
    p = np.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2 * sig ** 2))
    p /= p.sum() / ov ** 2
    return GriddedPSFModel(NDData(p[None], meta={'grid_xypos': [(0, 0)],
                                                 'oversampling': ov}))


def _gauss(xx, yy, flux, x0, y0, fwhm=FWHM):
    sig = fwhm / 2.3548
    return (flux / (2 * np.pi * sig ** 2)
            * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sig ** 2)))


def _set_nbrsub(monkeypatch, nbrsub):
    if nbrsub:
        # the default: leave the variable unset
        monkeypatch.delenv('DAOPHOT_REFIT_NBRSUB', raising=False)
    else:
        monkeypatch.setenv('DAOPHOT_REFIT_NBRSUB', '0')


# ---------------------------------------------------------------------------
# env switch
# ---------------------------------------------------------------------------
def test_refit_nbrsub_env(monkeypatch):
    monkeypatch.delenv('DAOPHOT_REFIT_NBRSUB', raising=False)
    assert C._daophot_refit_nbrsub() is True
    monkeypatch.setenv('DAOPHOT_REFIT_NBRSUB', '0')
    assert C._daophot_refit_nbrsub() is False
    monkeypatch.setenv('DAOPHOT_REFIT_NBRSUB', ' off ')
    assert C._daophot_refit_nbrsub() is False
    monkeypatch.setenv('DAOPHOT_REFIT_NBRSUB', '1')
    assert C._daophot_refit_nbrsub() is True


@pytest.mark.parametrize('raw', ['abc', '2', 'nan'])
def test_refit_nbrsub_env_rejects_malformed_values(monkeypatch, raw):
    monkeypatch.setenv('DAOPHOT_REFIT_NBRSUB', raw)
    with pytest.raises(ValueError, match='DAOPHOT_REFIT_NBRSUB'):
        C._daophot_refit_nbrsub()


# ---------------------------------------------------------------------------
# _subset_model_image
# ---------------------------------------------------------------------------
class _FakePhot:
    """Minimal stand-in exposing ``results`` and ``make_model_image``."""

    def __init__(self, results):
        self.results = results

    def make_model_image(self, shape, psf_shape=None, **kw):
        yy, xx = np.mgrid[:shape[0], :shape[1]].astype(float)
        out = np.zeros(shape)
        for r in self.results:
            out += _gauss(xx, yy, r['flux_fit'], r['x_fit'], r['y_fit'])
        return out


def test_subset_model_image_renders_only_kept_rows_and_restores():
    res = Table({'x_fit': [20.0, 40.0], 'y_fit': [20.0, 20.0],
                 'flux_fit': [100.0, 300.0]})
    phot = _FakePhot(res)
    m = C._subset_model_image(phot, np.array([False, True]), (41, 61))
    assert phot.results is res and len(phot.results) == 2
    assert m.sum() == pytest.approx(300.0, rel=1e-3)
    assert m[20, 20] < 1e-6 * m[20, 40]
    none = C._subset_model_image(phot, np.array([False, False]), (41, 61))
    assert not none.any()


def test_subset_model_image_skips_nonfinite_rows():
    # a forced solve can return a NaN flux; that row adds nothing
    res = Table({'x_fit': [20.0, 40.0, np.nan], 'y_fit': [20.0, 20.0, 20.0],
                 'flux_fit': [100.0, np.nan, 50.0]})
    m = C._subset_model_image(_FakePhot(res), np.ones(3, bool), (41, 61))
    assert np.isfinite(m).all()
    assert m.sum() == pytest.approx(100.0, rel=1e-3)


def test_subset_model_image_nan_flux_with_photutils():
    from jwst_gc_pipeline.photometry.psf_fitting import _make_psfphotometry
    shape = (41, 61)
    yy, xx = np.mgrid[:shape[0], :shape[1]].astype(float)
    img = _gauss(xx, yy, 100.0, 20.0, 20.0) + _gauss(xx, yy, 300.0, 40.0, 20.0)
    phot = _make_psfphotometry(localbkg_estimator=None,
                               psf_model=_gaussian_grid_psf(),
                               fit_shape=(5, 5), aperture_radius=4)
    phot(img, init_params=Table({'x': [20.0, 40.0], 'y': [20.0, 20.0],
                                 'flux': [100.0, 300.0]}))
    phot.results['flux_fit'][1] = np.nan
    m = C._subset_model_image(phot, np.ones(2, bool), shape)
    assert np.isfinite(m).all()
    assert m.sum() == pytest.approx(100.0, rel=1e-3)


# ---------------------------------------------------------------------------
# _overshoot_refit_tiers
# ---------------------------------------------------------------------------
def _peaks_frame(xy, peaks, shape=(60, 120)):
    img = np.zeros(shape)
    for (x, y), p in zip(xy, peaks):
        img[int(round(y)), int(round(x))] = p
    return img


def test_tiers_isolated_rows_are_all_tier_zero():
    xy = [(10.0, 10.0), (40.0, 10.0), (70.0, 10.0)]
    img = _peaks_frame(xy, [5.0, 50.0, 500.0])
    x, y = np.array(xy).T
    assert C._overshoot_refit_tiers(img, x, y).tolist() == [0, 0, 0]


def test_tiers_fainter_row_waits_for_brighter_neighbour():
    # a chain: 0 is brightest, 1 is 3 px from 0, 2 is 3 px from 1 and 6 px
    # from 0; 3 is beyond the radius from all of them
    xy = [(20.0, 20.0), (23.0, 20.0), (26.0, 20.0), (60.0, 20.0)]
    img = _peaks_frame(xy, [1000.0, 300.0, 100.0, 10.0])
    x, y = np.array(xy).T
    tier = C._overshoot_refit_tiers(img, x, y)
    assert tier.tolist() == [0, 1, 2, 0]
    # the order goes by the data peak, not by row order
    tier_r = C._overshoot_refit_tiers(img, x[::-1], y[::-1])
    assert tier_r.tolist() == [0, 2, 1, 0]


def test_tiers_radius_and_ties_and_nonfinite():
    xy = [(20.0, 20.0), (20.0 + C._REFIT_TIER_RADIUS_PIX + 1, 20.0),
          (20.0, 25.0), (np.nan, 20.0)]
    img = _peaks_frame(xy[:3], [100.0, 1000.0, 100.0])
    x, y = np.array(xy).T
    tier = C._overshoot_refit_tiers(img, x, y)
    # 1 is out of range of 0 and 2; 0 and 2 tie on the peak, so the lower
    # index goes first; a non-finite seed stays at tier 0
    assert tier.tolist() == [0, 0, 1, 0]
    assert C._overshoot_refit_tiers(img, x[:1], y[:1]).tolist() == [0]


# ---------------------------------------------------------------------------
# _manual_phot_pass: faint star next to a bright one
# ---------------------------------------------------------------------------
SHAPE = (100, 100)
BRIGHT = (50.0, 50.0)
FAINT = (54.0, 50.0)
F_BRIGHT = 5.0e4
F_FAINT = 3.0e3


def _scene(seed=3):
    yy, xx = np.mgrid[:SHAPE[0], :SHAPE[1]].astype(float)
    img = _gauss(xx, yy, F_BRIGHT, *BRIGHT) + _gauss(xx, yy, F_FAINT, *FAINT)
    return img + np.random.default_rng(seed).normal(0, 1.0, SHAPE)


def _run_pass(monkeypatch, nbrsub):
    """Run one pass with the faint star forced into the overshoot refit."""
    _set_nbrsub(monkeypatch, nbrsub)
    real = C._filter_or_flag_model_overshoot

    def flag_faint(phot_obj, modsky, data, **kw):
        over = real(phot_obj, modsky, data, **kw)
        if kw.get('action') == 'flag' and kw.get('label') == 't':
            res = phot_obj.results
            d = np.hypot(np.asarray(res['x_init'], float) - FAINT[0],
                         np.asarray(res['y_init'], float) - FAINT[1])
            over = np.asarray(over, bool).copy()
            over[np.argmin(d)] = True
            res['model_overshoot'] = over
        return over

    monkeypatch.setattr(C, '_filter_or_flag_model_overshoot', flag_faint)
    img = _scene()
    seed = Table({'x_init': [BRIGHT[0], FAINT[0]],
                  'y_init': [BRIGHT[1], FAINT[1]],
                  'flux_init': [F_BRIGHT, F_FAINT]})
    zeros = np.zeros(SHAPE)
    opts = types.SimpleNamespace(group=False, satstar_artifact_sigK=3.0,
                                 satstar_artifact_ratio=1.0)
    res, _, _ = C._manual_phot_pass(
        data=img, mask=np.zeros(SHAPE, bool), err=np.ones(SHAPE),
        bad=np.zeros(SHAPE, bool), dao_psf_model=_gaussian_grid_psf(),
        init_params=seed, aperture_radius_pix=2 * FWHM, localbkg_inner=6,
        localbkg_outer=10, grouper=None, options=opts,
        dq=np.zeros(SHAPE, np.uint32), satstar_model_subtracted=zeros,
        label='t', near_sat_dist_pix=1.0)
    d = np.hypot(np.asarray(res['x_fit'], float) - FAINT[0],
                 np.asarray(res['y_fit'], float) - FAINT[1])
    k = int(np.argmin(d))
    if d[k] > 0.5:
        return None, res
    assert bool(res['forced_refit'][k])
    return float(res['flux_fit'][k]), res


def test_refit_on_raw_frame_over_brightens_faint_neighbour(monkeypatch):
    flux, _ = _run_pass(monkeypatch, nbrsub=False)
    # the raw-frame solve takes in the bright star's wing (or is so inflated
    # that the final phantom drop removes the row)
    assert flux is None or flux > 1.1 * F_FAINT


def test_nbrsub_refit_recovers_faint_neighbour_flux(monkeypatch):
    flux, res = _run_pass(monkeypatch, nbrsub=True)
    assert flux == pytest.approx(F_FAINT, rel=0.05)
    # the bright star is untouched by the refit
    d = np.hypot(np.asarray(res['x_fit'], float) - BRIGHT[0],
                 np.asarray(res['y_fit'], float) - BRIGHT[1])
    k = int(np.argmin(d))
    assert not bool(res['forced_refit'][k])
    assert float(res['flux_fit'][k]) == pytest.approx(F_BRIGHT, rel=0.02)


# ---------------------------------------------------------------------------
# _manual_phot_pass: the ring-seed scene of
# test_dedup_seed_handover.test_grouped_fit_keeps_the_star_on_its_own_seed
# ---------------------------------------------------------------------------
RS_SHAPE = (80, 80)
STAR = (40.3, 40.2)
STAR_SEED = (40.5, 40.1)
# 2.5 px from the star, on its wing, with ~18% of its flux: the F212N
# first-ring detection.
RING_SEED = (STAR[0] - 2.5 / np.sqrt(2), STAR[1] - 2.5 / np.sqrt(2))
RS_PEDESTAL = 10.0


def _ring_seed_pass(monkeypatch, nbrsub):
    """A star broader than the PSF (FWHM 3.0 vs 2.5) with its own seed and a
    ring seed, fit as one group.  Both grouped rows read a model/data peak
    ratio of ~1.06, so with ``overshoot_ratio=1.05`` both go to the refit.

    Returns the rows, the true flux, the star's data peak above the pedestal
    and the residual frame (data - pedestal - model of every row).
    """
    _set_nbrsub(monkeypatch, nbrsub)
    yy, xx = np.mgrid[:RS_SHAPE[0], :RS_SHAPE[1]].astype(float)
    flux_true = 3000.0 * 2 * np.pi * (3.0 / 2.3548) ** 2
    star = _gauss(xx, yy, flux_true, *STAR, fwhm=3.0)
    img = (RS_PEDESTAL + star
           + np.random.default_rng(1).normal(0, 1.0, RS_SHAPE))
    seed = Table()
    seed['x_init'] = [STAR_SEED[0], RING_SEED[0]]
    seed['y_init'] = [STAR_SEED[1], RING_SEED[1]]
    seed['flux_init'] = [0.8 * flux_true, 0.18 * flux_true]
    # group=True is the legacy switch; manual_group_fit is the #1246 one
    opts = types.SimpleNamespace(group=True, manual_group_fit=True,
                                 satstar_artifact_sigK=3.0,
                                 satstar_artifact_ratio=1.0)
    psf = _gaussian_grid_psf()
    res, _, _ = C._manual_phot_pass(
        data=img, mask=np.zeros(RS_SHAPE, bool), err=np.ones(RS_SHAPE),
        bad=np.zeros(RS_SHAPE, bool), dao_psf_model=psf,
        init_params=seed, aperture_radius_pix=2 * FWHM, localbkg_inner=6,
        localbkg_outer=10, grouper=SourceGrouper(2 * FWHM), options=opts,
        dq=np.zeros(RS_SHAPE, np.uint32),
        satstar_model_subtracted=np.zeros(RS_SHAPE), label='t',
        near_sat_dist_pix=1.0, overshoot_ratio=1.05)
    model = np.zeros(RS_SHAPE)
    for r in res:
        model += np.asarray(psf.evaluate(xx, yy, float(r['flux_fit']),
                                         float(r['x_fit']),
                                         float(r['y_fit'])), float)
    resid = img - RS_PEDESTAL - model
    return res, flux_true, float(star.max()), resid


def _at(resid, xy):
    ix, iy = int(round(xy[0])), int(round(xy[1]))
    return float(resid[iy - 1:iy + 2, ix - 1:ix + 2].mean())


def _ring_row(res):
    d = np.hypot(np.asarray(res['x_init'], float) - RING_SEED[0],
                 np.asarray(res['y_init'], float) - RING_SEED[1])
    k = int(np.argmin(d))
    assert d[k] < 1e-6, 'the ring seed keeps its own row in this scene'
    return res[k]


def test_ring_seed_scene_reaches_the_refit_with_both_rows(monkeypatch):
    res, _, _, _ = _ring_seed_pass(monkeypatch, nbrsub=True)
    assert len(res) == 2
    assert np.all(np.asarray(res['forced_refit'], bool))


def test_ring_seed_raw_frame_refit_digs_a_hole(monkeypatch):
    # the failure: solved on the raw frame, the ring seed takes ~0.26x the
    # star's flux from the star's wing, the rows sum to more than the star,
    # and the residual at the ring seed is ~15% of the star's peak deep
    res, flux_true, peak, resid = _ring_seed_pass(monkeypatch, nbrsub=False)
    assert float(_ring_row(res)['flux_fit']) > 0.2 * flux_true
    assert float(np.sum(res['flux_fit'])) > flux_true
    assert _at(resid, RING_SEED) < -0.10 * peak


def test_ring_seed_nbrsub_refit_leaves_no_hole(monkeypatch):
    res, flux_true, peak, resid = _ring_seed_pass(monkeypatch, nbrsub=True)
    # the star's row is solved first and keeps its flux
    d = np.hypot(np.asarray(res['x_init'], float) - STAR_SEED[0],
                 np.asarray(res['y_init'], float) - STAR_SEED[1])
    star_row = res[int(np.argmin(d))]
    assert float(star_row['flux_fit']) > 0.7 * flux_true
    # the ring seed is solved on the frame with the star's refit model
    # removed, so it only takes the light the star's row left behind (the
    # star is broader than the PSF), and the rows do not sum past the star
    assert float(_ring_row(res)['flux_fit']) < 0.15 * flux_true
    assert float(np.sum(res['flux_fit'])) <= flux_true
    assert _at(resid, RING_SEED) > -0.03 * peak


# ---------------------------------------------------------------------------
# _manual_phot_pass: a bright -> mid -> faint chain, all overshooting
# ---------------------------------------------------------------------------
CH_SHAPE = (100, 100)
# each ~4 px from the next, so each is inside the next one's refit stamp
CHAIN = {'bright': (50.0, 50.0, 5.0e4), 'mid': (54.0, 50.0, 1.0e4),
         'faint': (52.0, 53.5, 2.0e3)}


def _chain_pass(monkeypatch, capsys):
    """All three rows forced into the refit, fit as one group.  Returns the
    rows, the residual frame (data - model of every row) and the pass's
    stdout."""
    _set_nbrsub(monkeypatch, True)
    real = C._filter_or_flag_model_overshoot

    def flag_all(phot_obj, modsky, data, **kw):
        over = real(phot_obj, modsky, data, **kw)
        if kw.get('action') == 'flag' and kw.get('label') == 't':
            over = np.ones(len(phot_obj.results), bool)
            phot_obj.results['model_overshoot'] = over
        return over

    monkeypatch.setattr(C, '_filter_or_flag_model_overshoot', flag_all)
    yy, xx = np.mgrid[:CH_SHAPE[0], :CH_SHAPE[1]].astype(float)
    img = sum(_gauss(xx, yy, f, x, y) for x, y, f in CHAIN.values())
    img = img + np.random.default_rng(5).normal(0, 1.0, CH_SHAPE)
    xs, ys, fs = np.array(list(CHAIN.values())).T
    seed = Table({'x_init': xs, 'y_init': ys, 'flux_init': fs})
    opts = types.SimpleNamespace(group=True, manual_group_fit=True,
                                 satstar_artifact_sigK=3.0,
                                 satstar_artifact_ratio=1.0)
    psf = _gaussian_grid_psf()
    capsys.readouterr()
    res, _, _ = C._manual_phot_pass(
        data=img, mask=np.zeros(CH_SHAPE, bool), err=np.ones(CH_SHAPE),
        bad=np.zeros(CH_SHAPE, bool), dao_psf_model=psf,
        init_params=seed, aperture_radius_pix=2 * FWHM, localbkg_inner=6,
        localbkg_outer=10, grouper=SourceGrouper(8.0), options=opts,
        dq=np.zeros(CH_SHAPE, np.uint32),
        satstar_model_subtracted=np.zeros(CH_SHAPE), label='t',
        near_sat_dist_pix=1.0)
    out = capsys.readouterr().out
    model = np.zeros(CH_SHAPE)
    for r in res:
        model += np.asarray(psf.evaluate(xx, yy, float(r['flux_fit']),
                                         float(r['x_fit']),
                                         float(r['y_fit'])), float)
    return res, img - model, out


def test_chain_refit_runs_in_three_tiers(monkeypatch, capsys):
    res, _, out = _chain_pass(monkeypatch, capsys)
    assert len(res) == 3 and np.all(np.asarray(res['forced_refit'], bool))
    xs, ys, _ = np.array(list(CHAIN.values())).T
    yy, xx = np.mgrid[:CH_SHAPE[0], :CH_SHAPE[1]].astype(float)
    img = sum(_gauss(xx, yy, f, x, y) for x, y, f in CHAIN.values())
    assert C._overshoot_refit_tiers(img, xs, ys).tolist() == [0, 1, 2]
    assert 'refit 3 overshooting sources' in out and '3 tier(s)' in out


def test_chain_faint_row_solved_on_cumulative_subtraction(monkeypatch, capsys):
    # the faint row is solved last, on the frame with both the bright and
    # the mid row's refit models removed; with only the mid row removed it
    # would take in the bright star's wing 4 px away
    res, resid, _ = _chain_pass(monkeypatch, capsys)
    for name, (x, y, f) in CHAIN.items():
        d = np.hypot(np.asarray(res['x_fit'], float) - x,
                     np.asarray(res['y_fit'], float) - y)
        k = int(np.argmin(d))
        assert d[k] < 1e-6, name
        assert float(res['flux_fit'][k]) == pytest.approx(f, rel=0.02), name
    fx, fy, ff = CHAIN['faint']
    peak = float(_gauss(fx, fy, ff, fx, fy))
    assert abs(_at(resid, (fx, fy))) < 0.03 * peak


# ---------------------------------------------------------------------------
# _manual_phot_pass, ungrouped (--no-group-fit): a 1.3 FWHM pair
# ---------------------------------------------------------------------------
UP_SHAPE = (100, 100)
# the close pair of test_group_fit_default.py
UP_PAIR = {'bright': (30.0, 40.0, 2.0e4), 'faint': (30.0 + 1.3 * FWHM, 40.4, 1.0e4)}


def _ungrouped_pair_pass(monkeypatch, nbrsub):
    """Rows of the bright and faint member of a pair fit without grouping.
    The overshoot test itself picks the faint member for the refit."""
    _set_nbrsub(monkeypatch, nbrsub)
    yy, xx = np.mgrid[:UP_SHAPE[0], :UP_SHAPE[1]].astype(float)
    img = 10.0 + sum(_gauss(xx, yy, f, x, y) for x, y, f in UP_PAIR.values())
    img = img + np.random.default_rng(7).normal(0, 1.0, UP_SHAPE)
    xs, ys, fs = np.array(list(UP_PAIR.values())).T
    # seeds 0.3 px off the truth, as a detection would be
    seed = Table({'x_init': xs + 0.3, 'y_init': ys - 0.2, 'flux_init': fs})
    opts = types.SimpleNamespace(group=False, manual_group_fit=False,
                                 satstar_artifact_sigK=3.0,
                                 satstar_artifact_ratio=1.0)
    res, _, _ = C._manual_phot_pass(
        data=img, mask=np.zeros(UP_SHAPE, bool), err=np.ones(UP_SHAPE),
        bad=np.zeros(UP_SHAPE, bool), dao_psf_model=_gaussian_grid_psf(),
        init_params=seed, aperture_radius_pix=2 * FWHM, localbkg_inner=6,
        localbkg_outer=10, grouper=None, options=opts,
        dq=np.zeros(UP_SHAPE, np.uint32),
        satstar_model_subtracted=np.zeros(UP_SHAPE), label='t',
        near_sat_dist_pix=1.0)
    rows = {}
    for name, (x, y, _) in UP_PAIR.items():
        d = np.hypot(np.asarray(res['x_fit'], float) - x,
                     np.asarray(res['y_fit'], float) - y)
        k = int(np.argmin(d))
        assert d[k] < 1.0, name
        rows[name] = res[k]
    return rows


@pytest.mark.parametrize('nbrsub, faint_ratio', [(False, 1.063), (True, 0.954)])
def test_ungrouped_pair_faint_member_refit_flux(monkeypatch, nbrsub,
                                                faint_ratio):
    # Pins a known side effect of the neighbour subtraction on ungrouped
    # runs.  Each ungrouped member absorbs part of the other's wing: the
    # bright member fits at 1.045x and is not refit, and the faint member
    # fits at 1.126x, overshoots (model/data peak 1.21) and is refit.  The
    # raw-frame refit gives 1.063x.  With the neighbour subtraction the
    # refit solves on a frame with the bright member's 1.045x model removed,
    # which also takes out part of the faint member's light, so it gives
    # 0.954x (pair sum 1.015x).  Grouped fitting fits this pair at 1.000x
    # each with no refit (test_group_fit_default.py).
    rows = _ungrouped_pair_pass(monkeypatch, nbrsub)
    bright, faint = rows['bright'], rows['faint']
    assert not bool(bright['forced_refit'])
    assert bool(faint['forced_refit'])
    assert float(bright['flux_fit']) / UP_PAIR['bright'][2] == \
        pytest.approx(1.045, abs=0.01)
    assert float(faint['flux_fit']) / UP_PAIR['faint'][2] == \
        pytest.approx(faint_ratio, abs=0.01)
