"""The post-fit dedup survivor must carry the seed nearest its fit (#1239).

A seed a few px from a bright star (an Airy-ring or wing detection) can
converge onto the star in the free fit.  The post-fit dedup in
``_manual_phot_pass`` keeps one of the two rows by flux or qfit, and it could
keep the neighbour's row, which still carried the neighbour's ``x_init/y_init``.
The overshoot refit then pinned that row at the neighbour's seed, 2-4 px off
the star, and the star lost its row in the frame (gc-treasury o105 F212N: 241
stars with a prodD m7 row lost from the rf re-catalog this way).
"""
import types

import numpy as np
import pytest
from astropy.nddata import NDData
from astropy.table import Table
from photutils.psf import GriddedPSFModel, SourceGrouper

from jwst_gc_pipeline.photometry import cataloging as C
from jwst_gc_pipeline.photometry.psf_fitting import _dedup_close_sources


# ---------------------------------------------------------------------------
# _dedup_close_sources(return_owner=True)
# ---------------------------------------------------------------------------

def test_default_return_is_unchanged():
    xy = np.array([[0.0, 0.0], [0.5, 0.0], [10.0, 10.0]])
    out = _dedup_close_sources(xy, np.array([100.0, 50.0, 10.0]), 1.0)
    assert len(out) == 2
    keep, n_disagree = out
    assert keep.tolist() == [True, False, True]
    assert n_disagree == 1


def test_owner_points_at_the_survivor():
    xy = np.array([[0.0, 0.0], [0.5, 0.0], [10.0, 10.0], [10.3, 10.0]])
    flux = np.array([100.0, 50.0, 10.0, 10.2])
    keep, _, owner = _dedup_close_sources(xy, flux, 1.0, return_owner=True)
    assert keep.tolist() == [True, False, False, True]
    assert owner.tolist() == [0, 0, 3, 3]


def test_owner_follows_a_winner_absorbed_later():
    # Cluster A: seed 0 (brightest) absorbs 1, but qfit makes 1 the winner, so
    # 0 is dropped and owned by 1.  Then 2 (fainter, processed later) is within
    # 1 px of 1 and wins ITS cluster on qfit, absorbing 1.  Row 0 must end up
    # owned by 2, the entry that survived.
    xy = np.array([[0.0, 0.0], [0.9, 0.0], [1.8, 0.0]])
    flux = np.array([100.0, 50.0, 20.0])
    qfit = np.array([0.5, 0.2, 0.1])
    keep, _, owner = _dedup_close_sources(xy, flux, 1.0, quality=qfit,
                                          return_owner=True)
    assert keep.tolist() == [False, False, True]
    assert owner.tolist() == [2, 2, 2]
    assert keep[owner].all()


@pytest.mark.parametrize('xy, flux', [
    (np.zeros((1, 2)), np.array([1.0])),                       # n < 2
    (np.array([[0.0, 0.0], [np.nan, 0.0]]), np.ones(2)),       # < 2 finite xy
    (np.array([[0.0, 0.0], [0.5, 0.0]]), np.full(2, np.nan)),  # none eligible
])
def test_early_returns_carry_an_identity_owner(xy, flux):
    keep, n_disagree, owner = _dedup_close_sources(xy, flux, 1.0,
                                                   return_owner=True)
    assert keep.all()
    assert n_disagree == 0
    assert owner.tolist() == list(range(len(flux)))


# ---------------------------------------------------------------------------
# _seed_nearest_fit
# ---------------------------------------------------------------------------

def test_survivor_takes_the_absorbed_star_seed():
    # Row 1 (neighbour seed 3 px out) won the dedup on qfit; its fit sits on
    # the star, where row 0's seed is.
    xfit = np.array([10.0, 10.1])
    yfit = np.array([10.0, 10.0])
    xinit = np.array([10.2, 7.9])
    yinit = np.array([10.0, 7.9])
    src = C._seed_nearest_fit(xfit, yfit, xinit, yinit, owner=[1, 1])
    assert src[1] == 0


def test_survivor_keeps_its_own_seed_when_nearest():
    xfit = np.array([10.0, 10.1])
    yfit = np.array([10.0, 10.0])
    xinit = np.array([7.9, 10.2])
    yinit = np.array([7.9, 10.0])
    src = C._seed_nearest_fit(xfit, yfit, xinit, yinit, owner=[1, 1])
    assert src.tolist() == [0, 1]


def test_tie_and_non_finite_seed_keep_the_survivor():
    # Equal distances -> the survivor's own seed; a NaN seed never wins, and a
    # survivor whose candidates are all non-finite keeps itself.
    xfit = np.array([0.0, 0.0, 0.0, 5.0, 5.0])
    yfit = np.zeros(5)
    xinit = np.array([1.0, -1.0, np.nan, np.nan, np.nan])
    yinit = np.zeros(5)
    src = C._seed_nearest_fit(xfit, yfit, xinit, yinit,
                              owner=[0, 0, 0, 3, 3])
    assert src[0] == 0
    assert src[3] == 3


# ---------------------------------------------------------------------------
# _manual_phot_pass: the o105 F212N failure in a synthetic frame
# ---------------------------------------------------------------------------

FWHM = 2.5
SHAPE = (80, 80)
STAR = (40.3, 40.2)
STAR_SEED = (40.5, 40.1)
# 2.5 px from the star, on its wing, with ~18% of its flux: the F212N
# first-ring detection.
NB_SEED = (STAR[0] - 2.5 / np.sqrt(2), STAR[1] - 2.5 / np.sqrt(2))


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


def _run(group_fit=False):
    """A star broader than the PSF (FWHM 3.0 vs 2.5), so its free fit reads a
    model/data peak ratio of ~1.06, plus the star's seed and a neighbour seed.
    With ``group_fit=False`` the fits are independent (the prodD state), so
    the neighbour slides onto the star; the qfit tiebreak keeps the
    neighbour's row; an overshoot ratio of 1.05 sends it to the seed-pinned
    refit.  With ``group_fit=True`` (the default since #1246) the two seeds
    are fit jointly."""
    yy, xx = np.mgrid[:SHAPE[0], :SHAPE[1]]
    sig = 3.0 / 2.3548
    amp = 3000.0
    img = 10.0 + amp * np.exp(-((xx - STAR[0]) ** 2 + (yy - STAR[1]) ** 2)
                              / (2 * sig ** 2))
    img += np.random.default_rng(1).normal(0, 1.0, SHAPE)
    flux_true = amp * 2 * np.pi * sig ** 2
    seed = Table()
    seed['x_init'] = [STAR_SEED[0], NB_SEED[0]]
    seed['y_init'] = [STAR_SEED[1], NB_SEED[1]]
    seed['flux_init'] = [0.8 * flux_true, 0.18 * flux_true]
    opts = types.SimpleNamespace(group=False, manual_group_fit=group_fit,
                                 satstar_artifact_sigK=3.0,
                                 satstar_artifact_ratio=1.0)
    res, _, _ = C._manual_phot_pass(
        data=img, mask=np.zeros(SHAPE, bool), err=np.ones(SHAPE),
        bad=np.zeros(SHAPE, bool), dao_psf_model=_gaussian_grid_psf(),
        init_params=seed, aperture_radius_pix=2 * FWHM, localbkg_inner=6,
        localbkg_outer=10, grouper=SourceGrouper(2 * FWHM), options=opts,
        dq=np.zeros(SHAPE, np.uint32),
        satstar_model_subtracted=np.zeros(SHAPE), label='t',
        near_sat_dist_pix=1.0, overshoot_ratio=1.05)
    return res, flux_true


def test_refit_stays_on_the_star():
    res, flux_true = _run()
    assert len(res) == 1, 'the dedup merges the two rows on the star'
    row = res[0]
    assert bool(row['forced_refit']), 'the scene must reach the refit'
    # The refit pins at the star's seed, not at the neighbour's 2.5 px away.
    assert (row['x_init'], row['y_init']) == pytest.approx(STAR_SEED)
    assert np.hypot(row['x_fit'] - STAR[0], row['y_fit'] - STAR[1]) < 0.5
    # Before the fix the refit at the neighbour's seed read 0.26x the star's
    # flux; at the star's seed it reads ~0.8x (the PSF is narrower than this
    # star).
    assert float(row['flux_fit']) > 0.7 * flux_true


def test_grouped_fit_keeps_the_star_on_its_own_seed():
    # Fit jointly, the neighbour seed no longer slides onto the star, so no
    # hand-over is needed: the star's row keeps its own seed and its flux.
    # The neighbour seed keeps a separate row in this scene; this test does
    # not pin that row.
    res, flux_true = _run(group_fit=True)
    on_star = [r for r in res
               if (r['x_init'], r['y_init']) == pytest.approx(STAR_SEED)]
    assert len(on_star) == 1
    row = on_star[0]
    assert np.hypot(row['x_fit'] - STAR[0], row['y_fit'] - STAR[1]) < 0.5
    assert float(row['flux_fit']) > 0.7 * flux_true
