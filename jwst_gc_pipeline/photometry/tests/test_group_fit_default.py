"""Grouped fitting is the manual daophot default (PR3, crater track).

Ungrouped fits of a close pair (0.75-2 FWHM) each absorb part of the
neighbour's wing, so the summed pair model over-subtracts and the residual
shows a crater at each member (o105 30" cutout: pair model/data relative to
isolated stars 1.08 F212N / 1.10 F480M ungrouped).  These tests pin:

* grouping is ON by default, ``--no-group-fit`` turns it off, the legacy
  ``--group`` flag still turns it on;
* the CLI defaults (cap 10, LevMar steps 25) and the opt-out flags;
* product names stay canonical: the ``_group`` token follows the legacy
  ``--group`` flag only, so a default (grouped) run writes the names every
  consumer reads (merge globs, the residual HiPS builder);
* a grouped pass fits a close pair jointly (group_size 2), recovers the pair
  flux an ungrouped pass over-estimates, and leaves an isolated star's fit
  unchanged;
* the header records GRPFIT / GRPSEP / GRPMAX / FITMAXFE;
* the LevMar evaluation budget scales with the largest group so grouped fits
  converge (photutils flags bit 8 clear, finite flux_err).
"""
import optparse
import types

import numpy as np
import pytest
from astropy.nddata import NDData
from astropy.table import Table
from photutils.psf import GriddedPSFModel

from jwst_gc_pipeline.photometry import cataloging as C
from jwst_gc_pipeline.photometry import crowdsource_catalogs_long as L
from jwst_gc_pipeline.photometry.manual_defaults import (
    MANUAL_DEFAULTS, manual_group_fit_enabled)

FWHM = 2.5
SHAPE = (100, 100)
SIG = FWHM / 2.3548
PAIR = ((30.0, 40.0), (30.0 + 1.3 * FWHM, 40.4))   # 1.3 FWHM apart
PAIR_FLUX = (2.0e4, 1.0e4)
ISO = (70.2, 69.7)
ISO_FLUX = 2.0e4


# ---------------------------------------------------------------------------
# option plumbing
# ---------------------------------------------------------------------------

def test_grouped_fit_is_the_default():
    assert MANUAL_DEFAULTS['manual_group_fit'] is True
    assert manual_group_fit_enabled(types.SimpleNamespace())
    assert manual_group_fit_enabled(types.SimpleNamespace(group=False))


def test_no_group_fit_opts_out_and_legacy_group_wins():
    assert not manual_group_fit_enabled(
        types.SimpleNamespace(group=False, manual_group_fit=False))
    assert manual_group_fit_enabled(
        types.SimpleNamespace(group=True, manual_group_fit=False))


class _Parsed(Exception):
    pass


_REAL_PARSE_ARGS = optparse.OptionParser.parse_args


def _parse(monkeypatch, argv):
    """Build main()'s OptionParser and parse ``argv`` without running main."""
    captured = {}

    def fake(self, args=None, values=None):
        captured['opts'] = _REAL_PARSE_ARGS(self, argv)[0]
        raise _Parsed

    monkeypatch.setattr(optparse.OptionParser, 'parse_args', fake)
    with pytest.raises(_Parsed):
        L.main()
    return captured['opts']


def test_cli_defaults(monkeypatch):
    o = _parse(monkeypatch, [])
    assert o.group is False
    assert o.manual_group_fit is True
    assert o.max_group_size == '10'
    assert L.resolve_max_group_size(o.max_group_size) == 10
    assert o.manual_group_min_sep_fwhm == 2.0
    assert o.manual_group_fit_lm_steps == 25
    assert manual_group_fit_enabled(o)


def test_cli_opt_out(monkeypatch):
    o = _parse(monkeypatch, ['--no-group-fit'])
    assert o.manual_group_fit is False
    assert not manual_group_fit_enabled(o)
    o = _parse(monkeypatch, ['--no-group-fit', '--group-fit'])
    assert o.manual_group_fit is True
    o = _parse(monkeypatch, ['--manual-group-fit-lm-steps=0',
                             '--max-group-size=unlimited'])
    assert o.manual_group_fit_lm_steps == 0
    assert L.resolve_max_group_size(o.max_group_size) is None


def test_default_run_keeps_canonical_product_names():
    """The ``_group`` token follows the legacy --group flag only."""
    import inspect
    for mod in (C, L):
        src = inspect.getsource(mod)
        # every product-name token is built from options.group ...
        assert "'_group' if options.group else ''" in src or \
            '"_group" if options.group else ""' in src
        # ... and never from the new default switch
        assert "'_group' if manual_group_fit_enabled" not in src
        assert "_group' if mopt(options, 'manual_group_fit')" not in src
    default = types.SimpleNamespace(group=False)
    assert manual_group_fit_enabled(default)
    assert ('_group' if default.group else '') == ''


# ---------------------------------------------------------------------------
# synthetic fits
# ---------------------------------------------------------------------------

def _gaussian_grid_psf():
    ov = 4
    n = 25 * ov + 1
    c = (n - 1) / 2
    yy, xx = np.mgrid[:n, :n]
    sig = SIG * ov
    p = np.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2 * sig ** 2))
    p /= p.sum() / ov ** 2
    psf = GriddedPSFModel(NDData(p[None], meta={'grid_xypos': [(0, 0)],
                                                'oversampling': ov}))
    psf.flux.min = 0
    return psf


def _scene(seed=7):
    yy, xx = np.mgrid[:SHAPE[0], :SHAPE[1]]
    img = np.full(SHAPE, 10.0)
    for (x0, y0), f in list(zip(PAIR, PAIR_FLUX)) + [(ISO, ISO_FLUX)]:
        img += f / (2 * np.pi * SIG ** 2) * np.exp(
            -((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * SIG ** 2))
    img += np.random.default_rng(seed).normal(0, 1.0, SHAPE)
    return img


def _seed():
    t = Table()
    xy = list(PAIR) + [ISO]
    # seeds 0.3 px off the truth, as a detection would be
    t['x_init'] = np.array([p[0] + 0.3 for p in xy], float)
    t['y_init'] = np.array([p[1] - 0.2 for p in xy], float)
    t['flux_init'] = np.array(list(PAIR_FLUX) + [ISO_FLUX], float)
    return t


def _options(**kw):
    o = dict(group=False, satstar_artifact_sigK=3.0, satstar_artifact_ratio=1.0)
    o.update(kw)
    return types.SimpleNamespace(**o)


def _run(options, img=None, grouper=None):
    img = _scene() if img is None else img
    if grouper is None:
        grouper = L.CappedSourceGrouper(2.0 * FWHM, max_size=10)
    res, _, _ = C._manual_phot_pass(
        data=img, mask=np.zeros(SHAPE, bool), err=np.ones(SHAPE),
        bad=np.zeros(SHAPE, bool), dao_psf_model=_gaussian_grid_psf(),
        init_params=_seed(), aperture_radius_pix=2 * FWHM,
        localbkg_inner=6, localbkg_outer=10, grouper=grouper,
        options=options, dq=np.zeros(SHAPE, np.uint32),
        satstar_model_subtracted=np.zeros(SHAPE), label='t',
        near_sat_dist_pix=1.0)
    return res


def _nearest(res, xy):
    d = np.hypot(np.asarray(res['x_fit'], float) - xy[0],
                 np.asarray(res['y_fit'], float) - xy[1])
    i = int(np.argmin(d))
    assert d[i] < 1.0
    return res[i]


def test_default_pass_fits_the_pair_jointly_and_records_it():
    res = _run(_options())
    a, b = (_nearest(res, p) for p in PAIR)
    assert a['group_size'] == 2 and b['group_size'] == 2
    assert a['group_id'] == b['group_id']
    assert _nearest(res, ISO)['group_size'] == 1
    assert res.meta['GRPFIT'][0] is True
    assert res.meta['GRPSEP'][0] == pytest.approx(2.0 * FWHM)
    assert res.meta['GRPMAX'][0] == 10
    # largest group 2 -> 25 * (3*2 + 1)
    assert res.meta['FITMAXFE'][0] == 25 * 7


def test_opt_out_pass_is_ungrouped_and_records_it():
    res = _run(_options(manual_group_fit=False))
    assert set(np.asarray(res['group_size']).tolist()) == {1}
    assert res.meta['GRPFIT'][0] is False
    assert 'GRPSEP' not in res.meta
    assert res.meta['FITMAXFE'][0] == 100


def test_grouped_pair_flux_is_closer_to_truth_than_ungrouped():
    """Each ungrouped member absorbs the neighbour's wing (sum too high)."""
    img = _scene()
    g = _run(_options(), img=img)
    u = _run(_options(manual_group_fit=False), img=img)
    true = sum(PAIR_FLUX)
    sg = sum(float(_nearest(g, p)['flux_fit']) for p in PAIR)
    su = sum(float(_nearest(u, p)['flux_fit']) for p in PAIR)
    assert su > 1.03 * true, su / true
    assert abs(sg / true - 1) < 0.01, sg / true
    assert abs(sg / true - 1) < abs(su / true - 1)


def test_isolated_star_fit_unchanged_by_grouping():
    img = _scene()
    g = _nearest(_run(_options(), img=img), ISO)
    u = _nearest(_run(_options(manual_group_fit=False), img=img), ISO)
    for col in ('flux_fit', 'x_fit', 'y_fit', 'flux_err'):
        assert float(g[col]) == pytest.approx(float(u[col]), rel=1e-6, abs=1e-6)


def test_maxiters_scales_with_largest_group():
    grouper = L.CappedSourceGrouper(2.0 * FWHM, max_size=10)
    seed = _seed()
    assert C._grouped_fit_maxiters(grouper, seed, _options()) == 25 * 7
    assert C._grouped_fit_maxiters(None, seed, _options()) == 100
    assert C._grouped_fit_maxiters(
        grouper, seed, _options(manual_group_fit_lm_steps=0)) == 100
    # a 12-source chain capped at 10 splits into two groups of 6
    chain = Table()
    chain['x_init'] = 10.0 + 2.0 * np.arange(12)
    chain['y_init'] = np.full(12, 50.0)
    assert C._grouped_fit_maxiters(grouper, chain, _options()) == 25 * (3 * 6 + 1)
    # isolated sources only: the photutils default
    iso = Table()
    iso['x_init'] = np.array([10.0, 60.0])
    iso['y_init'] = np.array([10.0, 60.0])
    assert C._grouped_fit_maxiters(grouper, iso, _options()) == 100


def _chain_scene(n=6, sep=1.1 * FWHM, seed=11):
    yy, xx = np.mgrid[:SHAPE[0], :SHAPE[1]]
    img = np.full(SHAPE, 10.0)
    xs = 25.0 + sep * np.arange(n)
    ys = 50.0 + 0.4 * np.sin(np.arange(n))
    fl = 1.0e4 * (1.0 + 0.5 * np.cos(np.arange(n)))
    for x0, y0, f in zip(xs, ys, fl):
        img += f / (2 * np.pi * SIG ** 2) * np.exp(
            -((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * SIG ** 2))
    img += np.random.default_rng(seed).normal(0, 1.0, SHAPE)
    t = Table()
    # seeds 0.8 px off and at 30% of the true flux (a crowded m2 seed)
    t['x_init'] = xs + 0.8
    t['y_init'] = ys - 0.8
    t['flux_init'] = fl * 0.3
    return img, t, fl


@pytest.mark.parametrize('steps, converged', [(25, True), (0, False)])
def test_group_budget_lets_a_six_source_group_converge(steps, converged):
    """At the flat photutils budget (100 evaluations = 5 LM steps for 18
    parameters) a 6-source blend stops early and loses its covariance; the
    scaled budget converges it."""
    img, seed, fl = _chain_scene()
    res, _, _ = C._manual_phot_pass(
        data=img, mask=np.zeros(SHAPE, bool), err=np.ones(SHAPE),
        bad=np.zeros(SHAPE, bool), dao_psf_model=_gaussian_grid_psf(),
        init_params=seed, aperture_radius_pix=2 * FWHM,
        localbkg_inner=6, localbkg_outer=10,
        grouper=L.CappedSourceGrouper(2.0 * FWHM, max_size=10),
        options=_options(manual_group_fit_lm_steps=steps),
        dq=np.zeros(SHAPE, np.uint32),
        satstar_model_subtracted=np.zeros(SHAPE), label='t',
        near_sat_dist_pix=1.0)
    assert len(res) == len(seed)
    assert set(np.asarray(res['group_size']).tolist()) == {len(seed)}
    maxiter_hit = (np.asarray(res['flags']).astype(int) & 8) > 0
    finite_err = np.isfinite(np.asarray(res['flux_err'], float))
    if converged:
        assert not maxiter_hit.any()
        assert finite_err.all()
        assert float(np.sum(res['flux_fit'])) == pytest.approx(float(fl.sum()), rel=0.01)
    else:
        # flags 8 (maxiters) + 16 (no covariance): flux_err NaN, which fails
        # the m7 vetting S/N terms (the o105 F480M vetting loss)
        assert maxiter_hit.all()
        assert not finite_err.any()


# ---------------------------------------------------------------------------
# KD-tree partition == photutils SourceGrouper partition
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('n, side, sep, seed', [
    (2000, 400.0, 5.0, 1),     # sparse: mostly singletons
    (3000, 200.0, 5.0, 2),     # <k> ~ 1.8: chains and large groups
    (1500, 120.0, 6.0, 3),     # near percolation
    (500, 50.0, 2.0, 4),
])
def test_kdtree_groups_equal_sourcegrouper(n, side, sep, seed):
    from photutils.psf import SourceGrouper
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, side, n)
    y = rng.uniform(0, side, n)
    # exact-separation pairs: SourceGrouper (fcluster t, '<=') joins them
    x[:10] = 10.0 + np.arange(10) * 30.0
    y[:10] = 5.0
    x[10:20] = x[:10] + sep
    y[10:20] = 5.0
    ref = np.asarray(SourceGrouper(sep)(x, y))
    got = L.kdtree_group_ids(x, y, sep)
    np.testing.assert_array_equal(got, ref)
    np.testing.assert_array_equal(L.CappedSourceGrouper(sep, max_size=0)(x, y), ref)


def test_kdtree_groups_edge_cases():
    assert L.kdtree_group_ids([3.0], [4.0], 5.0).tolist() == [1]
    assert L.kdtree_group_ids([], [], 5.0).size == 0
    with pytest.raises(ValueError):
        L.kdtree_group_ids([1.0, np.nan], [1.0, 2.0], 5.0)


def test_capped_grouper_partition_refines_the_uncapped_one():
    rng = np.random.default_rng(5)
    x = rng.uniform(0, 150, 2500)
    y = rng.uniform(0, 150, 2500)
    full = L.CappedSourceGrouper(5.0, max_size=0)(x, y)
    cap = L.CappedSourceGrouper(5.0, max_size=10)(x, y)
    assert np.unique(full, return_counts=True)[1].max() > 10
    assert np.unique(cap, return_counts=True)[1].max() <= 10
    # every capped group lies inside one uncapped group
    for g in np.unique(cap):
        assert np.unique(full[cap == g]).size == 1
