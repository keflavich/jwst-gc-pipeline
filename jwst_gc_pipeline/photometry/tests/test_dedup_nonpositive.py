"""A non-positive-flux fit must not win the post-fit dedup against a star.

The grouped free fit of a bright star and a seed ~2 px away can converge to a
degenerate pair: the star keeps a flux inflated by the partner's magnitude and
the partner lands on the star with a negative flux and a negative ``qfit``
(``qfit`` is the summed absolute residual divided by the flux).  The dedup's
``argmin(qfit)`` then keeps the negative row, the non-positive-flux ban removes
it, and the star has no row in the frame (gc-treasury 10678 F212N o130, m7
exposure 4 detector nrca2: a 4.4e4 star beside a -1.0e4 row 0.2 px away; 11
positive-flux rows lost this way in that frame).  Env ``DAOPHOT_DEDUP_NONPOSITIVE``
(default off) keeps the star, re-fits it at its seed, and gives a separate
neighbour seed its own row.
"""
import types

import numpy as np
import pytest
from astropy.nddata import NDData
from astropy.table import Table
from photutils.psf import GriddedPSFModel, PSFPhotometry, SourceGrouper

from jwst_gc_pipeline.photometry import cataloging as C
from jwst_gc_pipeline.photometry.psf_fitting import (
    _dedup_close_sources, forced_psf_photometry)


# ---------------------------------------------------------------------------
# _dedup_close_sources(nonpositive_loses=...)
# ---------------------------------------------------------------------------

# star + the degenerate partner 0.2 px away (the o130 exposure-4 values)
_XY = np.array([[10.0, 10.0], [10.2, 10.0]])
_FLUX = np.array([5.46e4, -1.04e4])
_QFIT = np.array([0.007, -0.025])


def test_default_keeps_the_negative_row():
    # Pins the unchanged default: argmin(qfit) picks the negative-flux row.
    keep, n_disagree, owner = _dedup_close_sources(
        _XY, _FLUX, 1.0, quality=_QFIT, return_owner=True)
    assert keep.tolist() == [False, True]
    assert owner.tolist() == [1, 1]
    assert n_disagree == 1


def test_nonpositive_loses_keeps_the_star():
    keep, n_disagree, owner = _dedup_close_sources(
        _XY, _FLUX, 1.0, quality=_QFIT, return_owner=True,
        nonpositive_loses=True)
    assert keep.tolist() == [True, False]
    assert owner.tolist() == [0, 0]
    assert n_disagree == 1


def test_zero_flux_row_also_loses():
    keep, _ = _dedup_close_sources(_XY, np.array([5.0e4, 0.0]), 1.0,
                                   quality=np.array([0.007, -0.01]),
                                   nonpositive_loses=True)
    assert keep.tolist() == [True, False]


def test_nan_quality_star_still_beats_a_negative_row():
    # a star without a finite qfit and a negative row with one
    keep, _ = _dedup_close_sources(_XY, _FLUX, 1.0,
                                   quality=np.array([np.nan, -0.025]),
                                   nonpositive_loses=True)
    assert keep.tolist() == [True, False]


def test_all_nonpositive_cluster_is_unchanged():
    # fmax <= 0 never reaches the quality rule; the switch changes nothing
    xy = np.array([[0.0, 0.0], [0.5, 0.0]])
    flux = np.array([-1.0e3, -2.0e3])
    q = np.array([-0.5, -0.1])
    base = _dedup_close_sources(xy, flux, 1.0, quality=q)
    new = _dedup_close_sources(xy, flux, 1.0, quality=q,
                               nonpositive_loses=True)
    assert base[0].tolist() == new[0].tolist() == [True, False]


def test_positive_clusters_are_unchanged():
    rng = np.random.default_rng(7)
    for _ in range(50):
        n = 12
        xy = rng.uniform(0, 4, size=(n, 2))
        flux = rng.uniform(1.0, 100.0, size=n)
        q = rng.uniform(0.0, 1.0, size=n)
        a = _dedup_close_sources(xy, flux, 1.0, quality=q, return_owner=True)
        b = _dedup_close_sources(xy, flux, 1.0, quality=q, return_owner=True,
                                 nonpositive_loses=True)
        assert a[0].tolist() == b[0].tolist()
        assert a[1] == b[1]
        assert a[2].tolist() == b[2].tolist()


# ---------------------------------------------------------------------------
# _dedup_nonpositive_cluster_rows
# ---------------------------------------------------------------------------

def _rows(xfit, yfit, flux, xinit, yinit, keep, owner):
    return C._dedup_nonpositive_cluster_rows(
        np.array(xfit, float), np.array(yfit, float), np.array(flux, float),
        np.array(xinit, float), np.array(yinit, float),
        np.array(keep, bool), np.array(owner), min_sep_pix=1.0)


def test_a_separate_neighbour_seed_is_rescued():
    # the star keeps row 0; row 1 converged onto it with a negative flux, but
    # its seed is 2.2 px from the star's: a source of its own
    refit, rescue = _rows(xfit=[10.0, 10.2], yfit=[10.0, 10.0],
                          flux=[5.46e4, -1.04e4],
                          xinit=[10.1, 12.3], yinit=[10.0, 10.0],
                          keep=[True, False], owner=[0, 0])
    assert refit.tolist() == [True, True]
    assert rescue.tolist() == [False, True]


def test_a_duplicate_seed_is_not_rescued():
    # the absorbed row's seed is within 1 px of the survivor's: a duplicate
    refit, rescue = _rows(xfit=[10.0, 10.2], yfit=[10.0, 10.0],
                          flux=[5.46e4, -1.04e4],
                          xinit=[10.1, 10.6], yinit=[10.0, 10.0],
                          keep=[True, False], owner=[0, 0])
    assert refit.tolist() == [True, False]
    assert rescue.tolist() == [False, False]


def test_a_seed_on_another_kept_row_is_not_rescued():
    # row 2 is a kept star at (12.4, 10); the absorbed seed (12.3, 10) is its
    refit, rescue = _rows(xfit=[10.0, 10.2, 12.4], yfit=[10.0, 10.0, 10.0],
                          flux=[5.46e4, -1.04e4, 8.0e3],
                          xinit=[10.1, 12.3, 12.4], yinit=[10.0, 10.0, 10.0],
                          keep=[True, False, True], owner=[0, 0, 2])
    assert refit.tolist() == [True, False, False]
    assert rescue.tolist() == [False, False, False]


def test_two_absorbed_rows_on_one_seed_give_one_row():
    # rows 1 and 2 both converged onto the star; their seeds are 0.4 px apart,
    # 2.2 px from the star's: one source, one rescued row
    refit, rescue = _rows(xfit=[10.0, 10.2, 10.1], yfit=[10.0, 10.0, 10.0],
                          flux=[5.46e4, -1.04e4, -3.0e3],
                          xinit=[10.1, 12.3, 12.7], yinit=[10.0, 10.0, 10.0],
                          keep=[True, False, False], owner=[0, 0, 0])
    assert refit.tolist() == [True, True, False]
    assert rescue.tolist() == [False, True, False]


def test_a_positive_duplicate_touches_nothing():
    refit, rescue = _rows(xfit=[10.0, 10.2], yfit=[10.0, 10.0],
                          flux=[5.46e4, 5.0e4],
                          xinit=[10.1, 12.3], yinit=[10.0, 10.0],
                          keep=[True, False], owner=[0, 0])
    assert not refit.any()
    assert not rescue.any()


def test_a_non_positive_survivor_is_not_refit():
    # an all-non-positive cluster: nothing for the refit to rescue
    refit, rescue = _rows(xfit=[10.0, 10.2], yfit=[10.0, 10.0],
                          flux=[-1.0e3, -2.0e3],
                          xinit=[10.1, 12.3], yinit=[10.0, 10.0],
                          keep=[True, False], owner=[0, 0])
    assert not refit.any()
    assert not rescue.any()


def test_non_finite_values_are_ignored():
    refit, rescue = _rows(xfit=[10.0, 10.2, 30.0], yfit=[10.0, 10.0, 30.0],
                          flux=[5.46e4, np.nan, -3.0],
                          xinit=[10.1, 12.3, np.nan], yinit=[10.0, 10.0, 30.0],
                          keep=[True, False, False], owner=[0, 0, 0])
    # row 1 has no finite flux; row 2 has a non-finite seed and sits far off
    assert not rescue.any()
    assert refit.tolist() == [True, False, False]


def test_the_helper_needs_no_dedup_when_nothing_was_absorbed():
    refit, rescue = _rows(xfit=[1.0, 20.0], yfit=[1.0, 20.0], flux=[5.0, -2.0],
                          xinit=[1.0, 20.0], yinit=[1.0, 20.0],
                          keep=[True, True], owner=[0, 1])
    assert not refit.any()
    assert not rescue.any()


# ---------------------------------------------------------------------------
# the switch
# ---------------------------------------------------------------------------

def test_the_switch_defaults_off(monkeypatch):
    monkeypatch.delenv('DAOPHOT_DEDUP_NONPOSITIVE', raising=False)
    assert C._dedup_nonpositive_refit() is False


@pytest.mark.parametrize('raw, want', [('1', True), ('on', True),
                                       (' True ', True), ('0', False),
                                       ('off', False), ('', False)])
def test_the_switch_parses_the_usual_values(monkeypatch, raw, want):
    monkeypatch.setenv('DAOPHOT_DEDUP_NONPOSITIVE', raw)
    assert C._dedup_nonpositive_refit() is want


def test_a_malformed_switch_names_the_variable(monkeypatch):
    monkeypatch.setenv('DAOPHOT_DEDUP_NONPOSITIVE', 'maybe')
    with pytest.raises(ValueError, match='DAOPHOT_DEDUP_NONPOSITIVE'):
        C._dedup_nonpositive_refit()


# ---------------------------------------------------------------------------
# _manual_phot_pass: a star and a neighbour seed, collapsed in the free fit
# ---------------------------------------------------------------------------

FWHM = 2.5
SHAPE = (80, 80)
STAR = (40.3, 40.2)
STAR_SEED = (40.5, 40.1)
# a real neighbour 2.2 px from the star, 20 % of its flux
COMP = (STAR[0] - 2.2, STAR[1])
COMP_SEED = (COMP[0] - 0.1, COMP[1] + 0.1)
F_STAR = 2.0e4
F_COMP = 4.0e3


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


def _frame():
    yy, xx = np.mgrid[:SHAPE[0], :SHAPE[1]]
    sig = FWHM / 2.3548
    img = np.full(SHAPE, 10.0)
    for (x0, y0), f in ((STAR, F_STAR), (COMP, F_COMP)):
        img += f * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2)
                          / (2 * sig ** 2)) / (2 * np.pi * sig ** 2)
    img += np.random.default_rng(3).normal(0, 1.0, SHAPE)
    return img


def _collapse(monkeypatch):
    """Make the free fit return the degenerate pair of the o130 frame: the
    star with a flux inflated by the partner's magnitude and positive qfit,
    the partner 0.2 px from it with a negative flux, negative qfit and flag 4.
    """
    real_call = PSFPhotometry.__call__

    def collapsed(self, data, **kwargs):
        out = real_call(self, data, **kwargs)
        res = self.results
        res['x_fit'][0], res['y_fit'][0] = STAR
        res['flux_fit'][0] = 1.24 * F_STAR
        res['qfit'][0] = 0.007
        res['x_fit'][1], res['y_fit'][1] = STAR[0] + 0.2, STAR[1]
        res['flux_fit'][1] = -0.25 * F_STAR
        res['qfit'][1] = -0.025
        res['flags'][1] = int(res['flags'][1]) | 4
        return out

    monkeypatch.setattr(PSFPhotometry, '__call__', collapsed)


def _run_pass():
    seed = Table()
    seed['x_init'] = [STAR_SEED[0], COMP_SEED[0]]
    seed['y_init'] = [STAR_SEED[1], COMP_SEED[1]]
    seed['flux_init'] = [F_STAR, F_COMP]
    opts = types.SimpleNamespace(group=False, manual_group_fit=False,
                                 satstar_artifact_sigK=3.0,
                                 satstar_artifact_ratio=1.0)
    res, _, _ = C._manual_phot_pass(
        data=_frame(), mask=np.zeros(SHAPE, bool), err=np.ones(SHAPE),
        bad=np.zeros(SHAPE, bool), dao_psf_model=_gaussian_grid_psf(),
        init_params=seed, aperture_radius_pix=2 * FWHM, localbkg_inner=6,
        localbkg_outer=10, grouper=SourceGrouper(2 * FWHM), options=opts,
        dq=np.zeros(SHAPE, np.uint32),
        satstar_model_subtracted=np.zeros(SHAPE), label='t',
        near_sat_dist_pix=1.0, overshoot_ratio=1.05)
    return res


def _row_at(res, seed):
    hit = [r for r in res
           if (r['x_init'], r['y_init']) == pytest.approx(seed)]
    assert len(hit) == 1, f'expected one row seeded at {seed}, got {len(hit)}'
    return hit[0]


def test_default_loses_the_star_in_the_collapse(monkeypatch):
    # The unchanged default: the negative row wins the dedup and the ban
    # removes it, so the frame has no row for the star.
    monkeypatch.delenv('DAOPHOT_DEDUP_NONPOSITIVE', raising=False)
    _collapse(monkeypatch)
    res = _run_pass()
    assert len(res) == 0


def test_switch_keeps_the_star_and_its_neighbour(monkeypatch):
    monkeypatch.setenv('DAOPHOT_DEDUP_NONPOSITIVE', '1')
    _collapse(monkeypatch)
    res = _run_pass()
    assert len(res) == 2
    assert 'dedup_refit' not in res.colnames
    star = _row_at(res, STAR_SEED)
    comp = _row_at(res, COMP_SEED)
    for row in (star, comp):
        assert bool(row['forced_refit'])
        assert float(row['flux_fit']) > 0
        assert not int(row['flags']) & 4
    # the rescued row's free fit had a negative flux, so a negative qfit and
    # a sign-flipped cfit; the refit gives it its own
    assert float(comp['qfit']) > 0
    assert np.isfinite(float(comp['cfit']))
    # Both are pinned at their seeds, and the star's flux is no longer the
    # inflated 1.24 x truth of the free fit.  The seeds sit 0.2 px (star) and
    # 0.14 px (neighbour) off the true positions, which at 2.2 px separation
    # moves light between the two: the refit reads +5 % (star) and +28 %
    # (neighbour) on this frame.
    assert (star['x_fit'], star['y_fit']) == pytest.approx(STAR_SEED)
    assert (comp['x_fit'], comp['y_fit']) == pytest.approx(COMP_SEED)
    assert float(star['flux_fit']) == pytest.approx(F_STAR, rel=0.15)
    assert float(comp['flux_fit']) == pytest.approx(F_COMP, rel=0.35)


def test_switch_is_a_no_op_without_a_non_positive_row(monkeypatch):
    # the same frame fit normally gives the same rows with the switch on or off
    monkeypatch.delenv('DAOPHOT_DEDUP_NONPOSITIVE', raising=False)
    off = _run_pass()
    monkeypatch.setenv('DAOPHOT_DEDUP_NONPOSITIVE', '1')
    on = _run_pass()
    assert len(on) == len(off)
    for col in ('x_fit', 'y_fit', 'flux_fit'):
        assert np.asarray(on[col]) == pytest.approx(np.asarray(off[col]))


# ---------------------------------------------------------------------------
# forced_psf_photometry(return_fit_quality=True)
# ---------------------------------------------------------------------------

def test_forced_fit_quality_is_the_photutils_definition():
    psf = _gaussian_grid_psf()
    yy, xx = np.mgrid[:SHAPE[0], :SHAPE[1]]
    x0, y0, flux = 40.0, 40.0, 1000.0
    img = np.asarray(psf.evaluate(xx, yy, flux, x0, y0), dtype=float)
    seed = Table({'x_init': [x0, 20.0], 'y_init': [y0, 20.0]})
    out = forced_psf_photometry(img, psf, seed, fit_shape=(5, 5),
                                return_fit_quality=True)
    assert float(out['flux_fit'][0]) == pytest.approx(flux, rel=1e-3)
    assert abs(float(out['qfit'][0])) < 1e-3
    assert abs(float(out['cfit'][0])) < 1e-3
    # an empty stamp solves flux 0: no qfit
    assert float(out['flux_fit'][1]) == pytest.approx(0.0, abs=1e-6)
    assert np.isnan(float(out['qfit'][1])) or float(out['flux_fit'][1]) > 0

    # the same star solved 0.5 px off the seed: the residual sums to a qfit
    # of the order of the PSF mismatch, and equals sum|resid| / flux
    off = Table({'x_init': [x0 + 0.5], 'y_init': [y0]})
    o2 = forced_psf_photometry(img, psf, off, fit_shape=(5, 5),
                               return_fit_quality=True)
    f2 = float(o2['flux_fit'][0])
    stamp = np.asarray(psf.evaluate(xx[38:43, 38:43], yy[38:43, 38:43], f2,
                                    x0 + 0.5, y0), dtype=float)
    want = np.abs(img[38:43, 38:43] - stamp).sum() / f2
    assert float(o2['qfit'][0]) == pytest.approx(want, rel=1e-6)
    assert float(o2['qfit'][0]) > 0.01


def test_forced_photometry_columns_are_unchanged_by_default():
    psf = _gaussian_grid_psf()
    yy, xx = np.mgrid[:SHAPE[0], :SHAPE[1]]
    img = np.asarray(psf.evaluate(xx, yy, 1000.0, 40.0, 40.0), dtype=float)
    out = forced_psf_photometry(img, psf,
                                Table({'x_init': [40.0], 'y_init': [40.0]}))
    assert 'qfit' not in out.colnames
    assert 'cfit' not in out.colnames
