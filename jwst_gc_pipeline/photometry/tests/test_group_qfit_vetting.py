"""Group-aware qfit for the vetting of close-pair members (#1245).

In a grouped fit the brighter member takes most of the blended flux, so the
fainter member's qfit (sum|resid in own box| / own flux) roughly doubles and
the merged-catalogue vetting drops it.  ``qfit_group`` replaces the member's
flux by the light in its fit box, partners included.  These tests pin:

* ``compute_group_qfit``: singleton equality, the monotonic reduction for the
  faint member, the flux-ratio, flag-8, non-finite-error and overshoot-refit
  guards;
* ``_manual_phot_pass`` writes the column only when
  ``manual_ext_group_qfit`` is on, and the column never exceeds qfit;
* the merge averages the column like qfit;
* ``_filter_extended_emission`` reads it only with ``group_qfit=True``, keeps
  ``low_fit_quality`` on the member's own qfit, and leaves the overshoot and
  previous-phase gates in force;
* the documented limits: the ``qfit_confident`` bypass reads the group value
  and so skips the local S/N floor, and the discount factors that admit a
  spurious faint member at ``min_ratio`` 0.1;
* the log lines for a missing or partial column;
* the option plumbing and the defaults.
"""
import optparse
import types

import numpy as np
import pytest
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.nddata import NDData
from astropy.table import Table
from astropy.wcs import WCS
from photutils.psf import GriddedPSFModel

from jwst_gc_pipeline.photometry import cataloging as C
from jwst_gc_pipeline.photometry import crowdsource_catalogs_long as L
from jwst_gc_pipeline.photometry import merge_catalogs as MC
from jwst_gc_pipeline.photometry.catalog_dtypes import FLOAT32_COLUMNS
from jwst_gc_pipeline.photometry.group_qfit import compute_group_qfit
from jwst_gc_pipeline.photometry.manual_defaults import MANUAL_DEFAULTS
from jwst_gc_pipeline.photometry.tests.test_combine_singleframe_split_rows import (
    _frame, _near_origin)

FWHM = 2.5
SIG = FWHM / 2.3548
SHAPE = (100, 100)


def _gaussian_grid_psf(sig=SIG):
    ov = 4
    n = 25 * ov + 1
    c = (n - 1) / 2
    yy, xx = np.mgrid[:n, :n]
    s = sig * ov
    p = np.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2 * s ** 2))
    p /= p.sum() / ov ** 2
    psf = GriddedPSFModel(NDData(p[None], meta={'grid_xypos': [(0, 0)],
                                                'oversampling': ov}))
    psf.flux.min = 0
    return psf


def _results(xs, fluxes, qfits, gids, **cols):
    n = len(xs)
    t = Table({'x_fit': np.asarray(xs, float), 'y_fit': np.full(n, 40.0),
               'flux_fit': np.asarray(fluxes, float),
               'qfit': np.asarray(qfits, float),
               'group_id': np.asarray(gids, int),
               'flux_err': np.full(n, 30.0),
               'flags': np.zeros(n, int)})
    for k, v in cols.items():
        t[k] = v
    return t


def _qg(t, **kw):
    return compute_group_qfit(t, _gaussian_grid_psf(), SHAPE, **kw)


# ---------------------------------------------------------------------------
# compute_group_qfit
# ---------------------------------------------------------------------------

def test_singleton_equals_qfit():
    t = _results([30.0, 70.0], [1e4, 5e3], [0.31, 0.57], [1, 2])
    np.testing.assert_array_equal(_qg(t), t['qfit'])


def test_faint_member_is_reduced_and_bright_member_nearly_unchanged():
    # 1.5 FWHM apart, flux ratio 0.55
    t = _results([30.0, 30.0 + 1.5 * FWHM], [1.0e4, 5.5e3], [0.30, 0.57], [1, 1])
    out = _qg(t)
    bright, faint = out[0] / 0.30, out[1] / 0.57
    assert faint < 1.0 and bright < 1.0
    # the partner's light in the bright member's box is smaller than the
    # bright member's light in the faint member's box
    assert faint < bright
    assert bright > 0.85
    assert np.all(out <= np.asarray(t['qfit']))


def test_reduction_grows_as_the_pair_closes():
    prev = np.inf
    for sep in (3.0, 2.0, 1.5, 1.0):
        t = _results([30.0, 30.0 + sep * FWHM], [1.0e4, 5.5e3], [0.30, 0.57], [1, 1])
        faint = _qg(t)[1]
        assert faint < prev
        prev = faint


def test_matches_the_estimator_on_a_known_geometry():
    """qfit_group = qfit * flux / (flux + in-box partner light)."""
    t = _results([30.0, 33.0], [1.0e4, 4.0e3], [0.30, 0.60], [1, 1])
    psf = _gaussian_grid_psf()
    dy, dx = np.mgrid[-2:3, -2:3].astype(float)
    xx, yy = 33 + dx, 40 + dy
    partner = float(psf.evaluate(xx, yy, 1.0e4, 30.0, 40.0).sum())
    assert _qg(t)[1] == pytest.approx(0.60 * 4.0e3 / (4.0e3 + partner), rel=1e-6)


def test_flux_ratio_guard_keeps_qfit():
    t = _results([30.0, 33.0], [1.0e4, 4.0e2], [0.30, 0.60], [1, 1])
    # flux 400 / 1e4 = 0.04 is below 0.1: the guard keeps qfit
    assert _qg(t, min_ratio=0.1)[1] == 0.60
    assert _qg(t, min_ratio=0.0)[1] < 0.60


def test_flag_8_nan_error_and_refit_rows_keep_qfit():
    base = dict(xs=[30.0, 33.0], fluxes=[1.0e4, 4.0e3], qfits=[0.30, 0.60],
                gids=[1, 1])
    assert _qg(_results(**base))[1] < 0.60
    t = _results(**base)
    t['flags'][1] = 8 | 16
    assert _qg(t)[1] == 0.60
    t = _results(**base)
    t['flux_err'][1] = np.nan
    assert _qg(t)[1] == 0.60
    t = _results(**base, forced_refit=[False, True])
    assert _qg(t)[1] == 0.60
    # the refit row is still a partner: the bright member sees its light
    assert _qg(t)[0] < 0.30


def test_partner_outside_the_group_or_the_radius_adds_no_light():
    t = _results([30.0, 33.0], [1.0e4, 4.0e3], [0.30, 0.60], [1, 2])
    np.testing.assert_array_equal(_qg(t), t['qfit'])
    t = _results([30.0, 60.0], [1.0e4, 4.0e3], [0.30, 0.60], [1, 1])
    np.testing.assert_array_equal(_qg(t), t['qfit'])


def test_excluded_pixels_carry_no_partner_light():
    t = _results([30.0, 33.0], [1.0e4, 4.0e3], [0.30, 0.60], [1, 1])
    assert _qg(t, excluded=np.ones(SHAPE, bool))[1] == 0.60


def test_masked_group_id_names_no_group():
    t = _results([30.0, 33.0], [1.0e4, 4.0e3], [0.30, 0.60], [1, 1])
    t['group_id'] = np.ma.masked_array([1, 1], mask=[True, True])
    np.testing.assert_array_equal(_qg(t), t['qfit'])


def test_missing_columns_and_nonfinite_values_are_safe():
    t = _results([30.0, 33.0], [1.0e4, 4.0e3], [0.30, np.nan], [1, 1])
    out = _qg(t)
    assert np.isnan(out[1]) and out[0] < 0.30
    del t['group_id']
    np.testing.assert_array_equal(_qg(t), t['qfit'])


def test_returns_a_plain_float_array():
    out = _qg(_results([30.0, 33.0], [1.0e4, 4.0e3], [0.30, 0.60], [1, 1]))
    assert type(out) is np.ndarray and out.dtype == np.float64


def test_flux_ratio_boundary_is_reduced():
    # flux exactly min_ratio x the brightest: only strictly below is guarded
    t = _results([30.0, 33.0], [8.0e3, 4.0e3], [0.30, 0.60], [1, 1])
    assert _qg(t, min_ratio=0.5)[1] < 0.60
    assert _qg(t, min_ratio=0.51)[1] == 0.60


def test_only_flag_bit_8_keeps_qfit():
    base = dict(xs=[30.0, 33.0], fluxes=[1.0e4, 4.0e3], qfits=[0.30, 0.60],
                gids=[1, 1])
    for flag, kept in ((8, True), (16, False), (1 | 2 | 4 | 32 | 64, False),
                       (8 | 1, True)):
        t = _results(**base)
        t['flags'][1] = flag
        assert bool(_qg(t)[1] == 0.60) is kept, flag


def _box_partner_light(psf, xc, yc, xp, yp, fp):
    dy, dx = np.mgrid[-2:3, -2:3].astype(float)
    return float(np.clip(psf.evaluate(round(xc) + dx, round(yc) + dy, fp, xp, yp),
                         0.0, None).sum())


def test_box_is_centred_on_the_seed_position_not_the_fit():
    """photutils fits in the 5x5 box around the rounded seed (x_init, y_init)."""
    psf = _gaussian_grid_psf()
    t = _results([30.0, 33.0], [1.0e4, 4.0e3], [0.30, 0.60], [1, 1],
                 x_init=[30.0, 34.2], y_init=[40.0, 41.2])
    seeded = _box_partner_light(psf, 34.2, 41.2, 30.0, 40.0, 1.0e4)
    fitted = _box_partner_light(psf, 33.0, 40.0, 30.0, 40.0, 1.0e4)
    assert seeded != pytest.approx(fitted, rel=1e-2)
    assert _qg(t)[1] == pytest.approx(0.60 * 4.0e3 / (4.0e3 + seeded), rel=1e-6)


def test_negative_psf_lobes_are_clipped_to_zero():
    class _Lobed:
        def evaluate(self, x, y, flux, x0, y0):
            r2 = (x - x0) ** 2 + (y - y0) ** 2
            return flux * (np.exp(-r2 / (2 * SIG ** 2)) / 10.0 - 0.01)

    t = _results([30.0, 33.0], [1.0e4, 4.0e3], [0.30, 0.60], [1, 1])
    dy, dx = np.mgrid[-2:3, -2:3].astype(float)
    raw = _Lobed().evaluate(33 + dx, 40 + dy, 1.0e4, 30.0, 40.0)
    assert raw.min() < 0
    clipped = float(raw[raw > 0].sum())
    out = compute_group_qfit(t, _Lobed(), SHAPE)
    assert out[1] == pytest.approx(0.60 * 4.0e3 / (4.0e3 + clipped), rel=1e-6)


# ---------------------------------------------------------------------------
# the per-frame pass
# ---------------------------------------------------------------------------

PAIR = ((30.0, 40.0), (30.0 + 1.3 * FWHM, 40.4))
PAIR_FLUX = (2.0e4, 1.0e4)
ISO = (70.2, 69.7)
ISO_FLUX = 2.0e4


def _scene(seed=7):
    """Stars 1.1x wider than the fit PSF, so the fit leaves a residual."""
    sig = 1.1 * SIG
    yy, xx = np.mgrid[:SHAPE[0], :SHAPE[1]]
    img = np.full(SHAPE, 10.0)
    for (x0, y0), f in list(zip(PAIR, PAIR_FLUX)) + [(ISO, ISO_FLUX)]:
        img += f / (2 * np.pi * sig ** 2) * np.exp(
            -((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sig ** 2))
    img += np.random.default_rng(seed).normal(0, 1.0, SHAPE)
    return img


def _seed():
    t = Table()
    xy = list(PAIR) + [ISO]
    t['x_init'] = np.array([p[0] + 0.3 for p in xy], float)
    t['y_init'] = np.array([p[1] - 0.2 for p in xy], float)
    t['flux_init'] = np.array(list(PAIR_FLUX) + [ISO_FLUX], float)
    return t


def _options(**kw):
    o = dict(group=False, satstar_artifact_sigK=3.0, satstar_artifact_ratio=1.0)
    o.update(kw)
    return types.SimpleNamespace(**o)


def _run(options, mask=None, bad=None):
    res, _, _ = C._manual_phot_pass(
        data=_scene(),
        mask=np.zeros(SHAPE, bool) if mask is None else mask,
        err=np.ones(SHAPE),
        bad=np.zeros(SHAPE, bool) if bad is None else bad,
        dao_psf_model=_gaussian_grid_psf(),
        init_params=_seed(), aperture_radius_pix=2 * FWHM,
        localbkg_inner=6, localbkg_outer=10,
        grouper=L.CappedSourceGrouper(2.0 * FWHM, max_size=10),
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


def test_pass_writes_no_column_by_default():
    res = _run(_options())
    assert 'qfit_group' not in res.colnames


def test_pass_writes_the_column_when_enabled():
    res = _run(_options(manual_ext_group_qfit=True))
    qg = np.asarray(res['qfit_group'], float)
    qf = np.asarray(res['qfit'], float)
    assert np.all(qg <= qf + 1e-12)
    iso = _nearest(res, ISO)
    assert iso['group_size'] == 1
    assert float(iso['qfit_group']) == float(iso['qfit'])
    faint = _nearest(res, PAIR[1])
    bright = _nearest(res, PAIR[0])
    assert faint['group_size'] == 2 and bright['group_size'] == 2
    # how much each member gains depends on where its box falls relative to
    # the partner; both are judged on a lower value than their own qfit
    assert float(faint['qfit_group']) < 0.95 * float(faint['qfit'])
    assert float(bright['qfit_group']) < float(bright['qfit'])


def test_pass_log_counts_only_material_reductions(capsys):
    _run(_options(manual_ext_group_qfit=True))
    out = capsys.readouterr().out
    # the two pair members, not the isolated star
    assert "group-aware qfit: 2 of 3 rows more than 0.1% below" in out


def test_pass_log_ignores_wing_sized_reductions(monkeypatch, capsys):
    """A partner outside the 5x5 box lowers qfit by ~1e-8: not counted."""
    def tiny(results, *a, **kw):
        return np.asarray(results['qfit'], dtype=float) * (1.0 - 1e-6)

    monkeypatch.setattr(C, 'compute_group_qfit', tiny)
    _run(_options(manual_ext_group_qfit=True))
    assert "group-aware qfit: 0 of 3 rows more than 0.1% below" in \
        capsys.readouterr().out


def test_estimator_default_ratio_matches_the_pipeline_default():
    import inspect
    default = inspect.signature(compute_group_qfit).parameters['min_ratio'].default
    assert default == MANUAL_DEFAULTS['manual_ext_group_qfit_min_ratio']


def test_pass_hands_the_final_exclusion_mask_to_the_estimator(monkeypatch):
    """Pixels the fit did not use (mask | bad) must carry no partner light."""
    seen = {}
    real = C.compute_group_qfit

    def spy(*a, **kw):
        seen['excluded'] = kw.get('excluded')
        return real(*a, **kw)

    monkeypatch.setattr(C, 'compute_group_qfit', spy)
    mask = np.zeros(SHAPE, bool)
    bad = np.zeros(SHAPE, bool)
    mask[5, 5] = True
    bad[90, 10] = True
    _run(_options(manual_ext_group_qfit=True), mask=mask, bad=bad)
    ex = seen['excluded']
    assert ex is not None and ex.shape == SHAPE
    assert ex[5, 5] and ex[90, 10] and int(ex.sum()) == 2


def test_option_off_never_calls_the_estimator(monkeypatch):
    def boom(*a, **kw):
        raise AssertionError('compute_group_qfit called with the option off')

    monkeypatch.setattr(C, 'compute_group_qfit', boom)
    _run(_options())


def test_run_manual_pipeline_forwards_the_switch_to_the_vetting():
    """Every ``_filter_extended_emission`` call of the phase loop passes the
    option (a source check: a run needs SLURM-sized inputs)."""
    import inspect
    src = inspect.getsource(C.run_manual_pipeline)
    assert ("group_qfit=bool(mopt(opts_phase, 'manual_ext_group_qfit'))"
            in src)


def test_empty_seed_carries_the_column_when_enabled():
    def run(opts):
        res, _, _ = C._manual_phot_pass(
            data=_scene(), mask=np.zeros(SHAPE, bool), err=np.ones(SHAPE),
            bad=np.zeros(SHAPE, bool), dao_psf_model=_gaussian_grid_psf(),
            init_params=_seed()[:0], aperture_radius_pix=2 * FWHM,
            localbkg_inner=6, localbkg_outer=10,
            grouper=L.CappedSourceGrouper(2.0 * FWHM, max_size=10),
            options=opts, dq=np.zeros(SHAPE, np.uint32),
            satstar_model_subtracted=np.zeros(SHAPE), label='t')
        return res
    assert 'qfit_group' in run(_options(manual_ext_group_qfit=True)).colnames
    assert 'qfit_group' not in run(_options()).colnames


# ---------------------------------------------------------------------------
# merge and dtype
# ---------------------------------------------------------------------------

def test_merge_averages_qfit_group_like_qfit(tmp_path):
    tbls = [_frame(tmp_path, f, [(0.0, 0.0)], [1000.0], dao=True)
            for f in range(3)]
    for tbl, q in zip(tbls, (0.30, 0.40, 0.50)):
        tbl['qfit_group'] = np.full(len(tbl), 0.5 * q, dtype='float32')
        tbl['qfit'][:] = q
    out = MC.combine_singleframe(tbls, nanaverage=MC.nanaverage_numpy,
                                 verbose=False)
    row = _near_origin(out)[0]
    assert row['qfit_group_avg'] == pytest.approx(0.5 * row['qfit_avg'], rel=1e-5)
    assert row['qfit_group_avg'] == pytest.approx(0.2, rel=1e-5)
    assert 'std_qfit_group_avg' in out.colnames


def test_merge_without_the_column_is_unchanged(tmp_path):
    tbls = [_frame(tmp_path, f, [(0.0, 0.0)], [1000.0], dao=True)
            for f in range(3)]
    out = MC.combine_singleframe(tbls, nanaverage=MC.nanaverage_numpy,
                                 verbose=False)
    assert 'qfit_group_avg' not in out.colnames


def test_merge_warns_when_the_first_frame_lacks_the_column(tmp_path, capsys):
    tbls = [_frame(tmp_path, f, [(0.0, 0.0)], [1000.0], dao=True)
            for f in range(3)]
    for tbl in tbls[1:]:
        tbl['qfit_group'] = np.full(len(tbl), 0.2, dtype='float32')
    out = MC.combine_singleframe(tbls, nanaverage=MC.nanaverage_numpy,
                                 verbose=False)
    assert 'qfit_group_avg' not in out.colnames
    log = capsys.readouterr().out
    assert 'WARNING: qfit_group is dropped because tbls[0] lacks it' in log
    assert '2 of 3 frames carry it' in log


def test_merge_warns_when_only_some_frames_carry_the_column(tmp_path, capsys):
    tbls = [_frame(tmp_path, f, [(0.0, 0.0)], [1000.0], dao=True)
            for f in range(3)]
    tbls[0]['qfit_group'] = np.full(len(tbls[0]), 0.2, dtype='float32')
    out = MC.combine_singleframe(tbls, nanaverage=MC.nanaverage_numpy,
                                 verbose=False)
    assert _near_origin(out)[0]['qfit_group_avg'] == pytest.approx(0.2, rel=1e-5)
    assert ('WARNING: qfit_group_avg averages only the 1 of 3 frames'
            in capsys.readouterr().out)


def test_merge_is_quiet_when_every_frame_carries_the_column(tmp_path, capsys):
    tbls = [_frame(tmp_path, f, [(0.0, 0.0)], [1000.0], dao=True)
            for f in range(3)]
    for tbl in tbls:
        tbl['qfit_group'] = np.full(len(tbl), 0.2, dtype='float32')
    MC.combine_singleframe(tbls, nanaverage=MC.nanaverage_numpy, verbose=False)
    assert 'WARNING: qfit_group' not in capsys.readouterr().out


def test_qfit_group_is_downcast_like_qfit():
    assert 'qfit_group' in FLOAT32_COLUMNS


# ---------------------------------------------------------------------------
# the vetting
# ---------------------------------------------------------------------------

NY = NX = 200
STAR = (50, 50)


def _wcs():
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crpix = [NX / 2, NY / 2]
    w.wcs.crval = [266.5, -28.7]
    w.wcs.cdelt = [-0.063 / 3600, 0.063 / 3600]
    return w


def _image(seed=3):
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0, 1.0, (NY, NX))
    yy, xx = np.mgrid[0:NY, 0:NX]
    data += 30.0 * np.exp(-((xx - STAR[0]) ** 2 + (yy - STAR[1]) ** 2) / 2.0)
    return data


def _vet(qfit, qfit_group, *, snr=12.0, group_qfit=True, flux_err=None,
         extra=None, **kw):
    """One grouped faint member: S/N 12, qfit 0.60, qfit_group 0.31 in the
    issue.  local_bkg < 0 keeps it out of the peak_SB branch, so only the
    noise-bound prominence keep (bound 0.46 at S/N 12) can admit it."""
    w = _wcs()
    ferr = 10.0 if flux_err is None else flux_err
    cat = Table({'skycoord': w.pixel_to_world([STAR[0]], [STAR[1]]),
                 'qfit': [qfit], 'flags': [0], 'local_bkg': [-0.05],
                 'flux': [ferr * snr], 'flux_err': [ferr],
                 'group_size': [2], 'id': [0]})
    if qfit_group is not None:
        cat['qfit_group'] = [qfit_group]
    for k, v in (extra or {}).items():
        cat[k] = [v]
    with np.errstate(divide='ignore', invalid='ignore'):
        out = _filter_extended_emission(
            cat, data_i2d_image=_image(), ww_i2d=w, star_prom_min=7.0,
            star_prom_peak_min=4.0, qfit_snr_k=5.0, local_snr_min=5.0,
            group_qfit=group_qfit, sky_clean_keep=False, label='test', **kw)
    return out


_filter_extended_emission = C._filter_extended_emission


def test_noise_bound_keep_admits_the_member_only_with_the_option():
    assert len(_vet(0.60, 0.31, group_qfit=True)) == 1
    assert len(_vet(0.60, 0.31, group_qfit=False)) == 0


def test_unreduced_group_value_stays_dropped():
    # the per-frame guards (flux ratio, flag 8) leave qfit_group == qfit
    assert len(_vet(0.60, 0.60, group_qfit=True)) == 0


def test_nan_flux_err_keeps_the_own_qfit():
    assert len(_vet(0.60, 0.31, flux_err=np.nan)) == 0
    # a NaN qfit_group falls back to qfit
    assert len(_vet(0.60, np.nan)) == 0


def test_option_without_the_column_matches_the_old_path():
    assert len(_vet(0.60, None, group_qfit=True)) == 0
    assert len(_vet(0.30, None, group_qfit=True)) == 1


def test_qfit_group_never_raises_qfit():
    # an averaged qfit_group above qfit (frames without the column) must not
    # demote a row the own qfit keeps: qfit 0.15 passes the flat 0.2 gate, and
    # without the min() the 0.90 would drop it
    assert len(_vet(0.15, 0.90, group_qfit=False)) == 1
    assert len(_vet(0.15, 0.90, group_qfit=True)) == 1
    # and it does not rescue a row the own qfit drops
    assert len(_vet(0.60, 0.90, group_qfit=True)) == 0


def test_nan_flux_err_cannot_reach_the_qfit_confident_bypass():
    # the qfit-confident bypass keeps a row whatever its S/N, so a row whose
    # discounted value would clear qfit_max must still be refused when its
    # flux_err is not finite (no covariance behind the fit)
    assert len(_vet(0.60, 0.15, flux_err=np.nan, group_qfit=True)) == 0
    assert len(_vet(0.60, 0.15, group_qfit=True)) == 1


def test_qfit_confident_bypass_reads_the_group_value_and_skips_the_snr_floor():
    """The documented limit: with the option on, a grouped member whose
    qfit_group <= qfit_max is kept REGARDLESS of S/N (here 1.5 against
    local_snr_min 5), where its own qfit 1.2 would drop it."""
    assert len(_vet(1.2, 0.18, snr=1.5, group_qfit=True)) == 1
    assert len(_vet(1.2, 0.18, snr=1.5, group_qfit=False)) == 0
    # a member whose discounted value stays above qfit_max is still dropped
    assert len(_vet(1.2, 0.50, snr=1.5, group_qfit=True)) == 0


def test_discount_factor_at_min_ratio_admits_a_spurious_faint_member():
    """Numbers quoted in the docs: within 1 FWHM the faint member's qfit is
    scaled by ~0.13 / 0.31 at flux ratio 0.1 / 0.3."""
    f = {}
    for ratio in (0.1, 0.3):
        t = _results([30.0, 30.0 + FWHM], [5.0e4, 5.0e4 * ratio], [0.30, 1.0],
                     [1, 1])
        f[ratio] = _qg(t, min_ratio=0.0)[1]
    assert f[0.1] == pytest.approx(0.129, abs=0.005)
    assert f[0.3] == pytest.approx(0.307, abs=0.005)
    # own qfit 1.2 -> 0.155 (< 0.2, passes the flat gate at the default ratio
    # 0.1) but 0.37 at ratio 0.3
    assert 1.2 * f[0.1] < 0.2 < 1.2 * f[0.3]


def test_missing_column_is_logged_when_the_option_is_on(capsys):
    _vet(0.30, None, group_qfit=True)
    assert 'has no qfit_group column' in capsys.readouterr().out
    _vet(0.30, None, group_qfit=False)
    assert 'has no qfit_group column' not in capsys.readouterr().out
    _vet(0.30, 0.2, group_qfit=True)
    out = capsys.readouterr().out
    assert 'has no qfit_group column' not in out
    assert 'rows judged on qfit_group' in out


def _vet_prominence_exempt(qfit, qfit_group, group_qfit):
    """One source of prominence 2-3 on a nebular-style annulus, merged S/N 75
    (the scene of test_ext_prom_floor_exempt), so only the exemption of the
    prominence floor can keep it."""
    ny = nx = 200
    w = _wcs()
    rng = np.random.default_rng(2)
    data = rng.normal(0.0, 0.2, (ny, nx))
    yy, xx = np.mgrid[0:ny, 0:nx]
    x, y = 80, 40
    data += 8.0 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 2.0)
    r = np.hypot(xx - x, yy - y)
    ring = (r >= 4) & (r <= 10)
    data[ring] += 3.0 * np.sin(3 * np.arctan2(yy - y, xx - x))[ring]
    t = Table({'id': [0], 'skycoord': w.pixel_to_world([x], [y]),
               'qfit': [qfit], 'qfit_group': [qfit_group],
               'flux': [300.0], 'flux_err': [30.0], 'flux_err_prop': [4.0],
               'flags': [0], 'local_bkg': [0.0], 'group_size': [2]})
    out = _filter_extended_emission(
        t, data_i2d_image=data, ww_i2d=w, ext_prom_min=3.0,
        ext_prom_exempt_qfit=0.2, ext_prom_exempt_snr=40.0,
        ext_prom_exempt_prom_min=2.0, snr_floor_propagated=True,
        sky_clean_keep=False, group_qfit=group_qfit, label='exempt')
    return len(out) == 1, float(t['prominence'][0])


def test_prominence_exempt_stays_on_the_own_qfit():
    kept, prom = _vet_prominence_exempt(0.15, 0.15, True)
    assert 2.0 <= prom < 3.0
    assert kept
    # own qfit 0.3 is not a tight fit: the exemption must not read the
    # group value 0.15, so the prominence floor still drops it
    assert not _vet_prominence_exempt(0.30, 0.15, True)[0]
    assert not _vet_prominence_exempt(0.30, 0.15, False)[0]


def test_overshoot_gate_still_applies():
    assert len(_vet(0.60, 0.31, extra={'model_overshoot': True})) == 0
    assert len(_vet(0.60, 0.31, extra={'model_overshoot': False})) == 1


def test_low_fit_quality_stays_on_the_own_qfit():
    out = _vet(0.60, 0.31, low_fit_quality_qfit=0.5)
    assert len(out) == 1
    assert bool(out['low_fit_quality'][0]) is True


def test_singletons_are_identical_with_the_option_on():
    t = Table({'qfit': [0.30, 0.60], 'flags': [0, 0],
               'flux': [6381.0, 360.0], 'flux_err': [31.7, 30.0],
               'group_size': [1.0, 1.0]})
    t['qfit_group'] = t['qfit']
    kw = dict(min_prominence=0.0, qfit_max=0.2, local_snr_min=5.0,
              snr_high_keep=20.0, qfit_high_keep_max=0.4, label='nircam-test',
              sky_clean_keep=False)
    on = C._filter_extended_emission(t.copy(), group_qfit=True, **kw)
    off = C._filter_extended_emission(t.copy(), group_qfit=False, **kw)
    assert on.colnames == off.colnames
    for col in on.colnames:
        np.testing.assert_array_equal(np.asarray(on[col]), np.asarray(off[col]))


def test_hysteresis_keep_reads_the_group_value():
    t = Table({
        'id': [0],
        'skycoord': SkyCoord([266.5] * u.deg, [-28.8] * u.deg),
        'qfit': [0.65], 'qfit_group': [0.45],
        'flux': [300.0], 'flux_err': [10.0], 'flags': [0],
        'group_size': [2], 'local_bkg': [0.0]})
    prev = SkyCoord([266.5] * u.deg, [-28.8] * u.deg).spherical_offsets_by(
        0.01 * u.arcsec, 0 * u.arcsec)

    def kept(group_qfit, **extra):
        tt = t.copy()
        for k, v in extra.items():
            tt[k] = [v]
        out = C._filter_extended_emission(
            tt, sky_clean_keep=False, prev_vetted_skycoord=prev,
            hysteresis_qfit_max=0.6, hysteresis_snr_min=10.0,
            hysteresis_match_arcsec=0.036, group_qfit=group_qfit)
        return len(out) == 1

    assert kept(True)
    assert not kept(False)
    assert not kept(True, model_overshoot=True)


# ---------------------------------------------------------------------------
# options
# ---------------------------------------------------------------------------

class _Parsed(Exception):
    pass


_REAL_PARSE_ARGS = optparse.OptionParser.parse_args


def _parse(monkeypatch, argv):
    captured = {}

    def fake(self, args=None, values=None):
        captured['opts'] = _REAL_PARSE_ARGS(self, argv)[0]
        raise _Parsed

    monkeypatch.setattr(optparse.OptionParser, 'parse_args', fake)
    with pytest.raises(_Parsed):
        L.main()
    return captured['opts']


def test_defaults_are_off_and_cli_follows_them(monkeypatch):
    assert MANUAL_DEFAULTS['manual_ext_group_qfit'] is False
    assert MANUAL_DEFAULTS['manual_ext_group_qfit_min_ratio'] == 0.1
    o = _parse(monkeypatch, [])
    assert o.manual_ext_group_qfit is False
    assert o.manual_ext_group_qfit_min_ratio == 0.1
    o = _parse(monkeypatch, ['--manual-ext-group-qfit',
                             '--manual-ext-group-qfit-min-ratio=0.3'])
    assert o.manual_ext_group_qfit is True
    assert o.manual_ext_group_qfit_min_ratio == 0.3
    o = _parse(monkeypatch, ['--manual-ext-group-qfit',
                             '--no-manual-ext-group-qfit'])
    assert o.manual_ext_group_qfit is False
