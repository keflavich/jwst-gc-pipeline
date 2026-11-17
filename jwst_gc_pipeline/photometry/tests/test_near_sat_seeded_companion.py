"""Seeded-companion exemption from the near-saturation veto
(env ``SATPROX_SEEDED_COMPANION``, default off).

gc-treasury F480M residual-pox stars 2, 3 and 6: i2d-seeded companions 4-5 px
from a handed-off ~2e4 primary are fitted, then dropped by
``_filter_near_saturation`` because the primary's late-group SATURATED
footprint -- pixels ``_handoff_restore_pixels`` gave back to the fit -- reaches
within 1 px of them.  The exemption keeps a fit only when it

* sits within 0.5 px of a seed position,
* has no SATURATED pixel within ``max_sat_dist_pix`` that is NOT restored, and
* has a finite ``flux_fit > 0`` and a finite ``qfit``.

A fresh (unseeded) fit on the restored footprint stays vetoed: that is the
spike/wing case ``test_gate_reject_handoff`` pins.
"""
import types

import numpy as np
import pytest
from astropy.table import Table

from jwst_gc_pipeline.photometry import cataloging as C
from jwst_gc_pipeline.photometry import crowdsource_catalogs_long as L

SAT = L.dqflags.pixel['SATURATED']
HANDOFF = np.array([[20.0, 20.0]])
SEED = np.array([[25.5, 20.0]])


class _Snap:
    def __init__(self, results):
        self.results = results
        self.init_params = None


class _FakePhot:
    def __init__(self, results, with_snapshot=False):
        self.results = results
        if with_snapshot:
            self.fit_results = [_Snap(Table(results, copy=True))]


def _scene(flux=5e3, qfit=0.05, with_qfit=True):
    """Primary at (20, 20) with a SATURATED core and an arm reaching 1 px from
    a companion at (25.6, 20.1), plus an unseeded wing fit on the core
    (21, 22) that is always dropped -- so the per-iteration snapshot loop,
    which only runs when something is dropped, is exercised.  Returns
    ``(dq, results, restored)`` with every SATURATED pixel restored."""
    dq = np.zeros((40, 40), dtype=np.uint32)
    dq[18:23, 18:23] |= SAT
    dq[20, 23:26] |= SAT
    res = Table()
    res['id'] = [1, 2, 3]
    res['x_fit'] = [20.0, 25.6, 21.0]
    res['y_fit'] = [20.0, 20.1, 22.0]
    res['flux_fit'] = [2e4, flux, 800.0]
    if with_qfit:
        res['qfit'] = [0.01, qfit, 0.2]
    restored = (dq & SAT).astype(bool)
    return dq, res, restored


def _run(res, dq, *, with_snapshot=False, **kw):
    phot = _FakePhot(res, with_snapshot=with_snapshot)
    L._filter_near_saturation(phot, dq, max_sat_dist_pix=1.0, label='t',
                              handoff_xy=HANDOFF, handoff_radius_pix=1.3, **kw)
    return phot


def _companion_kw(seed=SEED, restored=None):
    return dict(companion_seed_xy=seed, companion_seed_radius_pix=0.5,
                companion_restored_mask=restored)


def test_companion_vetoed_without_exemption():
    dq, res, _ = _scene()
    assert list(_run(res, dq).results['id']) == [1]


def test_seeded_companion_on_restored_footprint_kept_in_results_and_snapshot():
    dq, res, restored = _scene()
    phot = _run(res, dq, with_snapshot=True, **_companion_kw(restored=restored))
    assert list(phot.results['id']) == [1, 2]
    # the model image is rendered from the snapshots: the star must stay there too
    assert list(phot.fit_results[0].results['id']) == [1, 2]


def test_unseeded_fit_on_restored_footprint_still_vetoed():
    dq, res, restored = _scene()
    phot = _run(res, dq, **_companion_kw(seed=np.array([[30.0, 30.0]]),
                                         restored=restored))
    assert list(phot.results['id']) == [1]


def test_seeded_fit_next_to_unrestored_pixel_vetoed():
    dq, res, restored = _scene()
    restored[20, 25] = False           # arm tip still lost, 0.6 px from the fit
    phot = _run(res, dq, with_snapshot=True, **_companion_kw(restored=restored))
    assert list(phot.results['id']) == [1]
    assert list(phot.fit_results[0].results['id']) == [1]


def test_no_restored_mask_exempts_nothing():
    dq, res, _ = _scene()
    phot = _run(res, dq, **_companion_kw(restored=None))
    assert list(phot.results['id']) == [1]


@pytest.mark.parametrize('flux, qfit, with_qfit', [
    (0.0, 0.05, True),
    (-10.0, 0.05, True),
    (np.nan, 0.05, True),
    (5e3, np.nan, True),
    (5e3, 0.05, False),
])
def test_junk_fit_not_exempt(flux, qfit, with_qfit):
    """Flux <= 0 / non-finite flux / non-finite or missing qfit: the
    full-frame runs kept ~3 such fits per frame (flux 0, qfit nan) before
    this guard."""
    dq, res, restored = _scene(flux=flux, qfit=qfit, with_qfit=with_qfit)
    phot = _run(res, dq, with_snapshot=True, **_companion_kw(restored=restored))
    assert list(phot.results['id']) == [1]
    assert list(phot.fit_results[0].results['id']) == [1]


def test_all_saturated_restored_corner_fit_kept():
    """Every SATURATED pixel restored -> nothing is lost, so a seeded fit next
    to a restored pixel at the array corner is exempt.  (A distance transform
    of the all-True "not lost" array would put it 1.0 px from a lost pixel.)"""
    dq = np.zeros((40, 40), dtype=np.uint32)
    dq[0:3, 0:3] |= SAT
    res = Table()
    res['id'] = [1]
    res['x_fit'] = [0.2]
    res['y_fit'] = [0.1]
    res['flux_fit'] = [5e3]
    res['qfit'] = [0.05]
    restored = (dq & SAT).astype(bool)
    phot = _FakePhot(res)
    L._filter_near_saturation(phot, dq, max_sat_dist_pix=1.0, label='t',
                              **_companion_kw(seed=np.array([[0.0, 0.0]]),
                                              restored=restored))
    assert list(phot.results['id']) == [1]


def test_restored_mask_shape_mismatch_raises():
    dq, res, _ = _scene()
    with pytest.raises(ValueError, match='companion_restored_mask shape'):
        _run(res, dq, **_companion_kw(restored=np.zeros((10, 10), bool)))


def test_default_call_identical_to_no_companion_kwargs():
    """The defaults of the new kwargs reproduce the pre-change call exactly:
    same rows, same bytes, same snapshot rows."""
    dq, res, _ = _scene()
    a = _run(Table(res, copy=True), dq, with_snapshot=True)
    b = _run(Table(res, copy=True), dq, with_snapshot=True,
             companion_seed_xy=None, companion_seed_radius_pix=0.0,
             companion_restored_mask=None)
    assert a.results.as_array().tobytes() == b.results.as_array().tobytes()
    assert (a.fit_results[0].results.as_array().tobytes()
            == b.fit_results[0].results.as_array().tobytes())


# --- cataloging wiring: _seeded_companion_kwargs ---------------------------

class _IdentityWCS:
    def world_to_pixel(self, sc):
        return np.asarray(sc.ra.deg), np.asarray(sc.dec.deg)


def _ctx(restored):
    return types.SimpleNamespace(ww=_IdentityWCS(), handoff_restore=restored)


def _seed_file(tmp_path):
    from astropy.coordinates import SkyCoord
    t = Table()
    t['skycoord'] = SkyCoord([25.5, 3.0], [20.0, 4.0], unit='deg')
    p = tmp_path / 'seed.ecsv'
    t.write(p)
    return str(p)


def test_kwargs_empty_when_env_unset(tmp_path, monkeypatch):
    monkeypatch.delenv('SATPROX_SEEDED_COMPANION', raising=False)
    restored = np.ones((40, 40), bool)
    assert C._seeded_companion_kwargs(_ctx(restored), _seed_file(tmp_path),
                                      False) == {}


def test_kwargs_when_on(tmp_path, monkeypatch):
    monkeypatch.setenv('SATPROX_SEEDED_COMPANION', '1')
    restored = np.zeros((40, 40), bool)
    restored[20, 23:26] = True
    kw = C._seeded_companion_kwargs(_ctx(restored), _seed_file(tmp_path), False)
    assert kw['companion_seed_radius_pix'] == 0.5
    assert np.allclose(kw['companion_seed_xy'], [[25.5, 20.0], [3.0, 4.0]])
    assert kw['companion_restored_mask'].sum() == 3


@pytest.mark.parametrize('is_miri, seedcat, restored', [
    (True, True, np.ones((4, 4), bool)),     # MIRI never hands off
    (False, False, np.ones((4, 4), bool)),   # m1/m2: no seed catalog
    (False, True, None),                     # nothing restored on this frame
])
def test_kwargs_empty_when_not_applicable(tmp_path, monkeypatch, is_miri,
                                          seedcat, restored):
    monkeypatch.setenv('SATPROX_SEEDED_COMPANION', '1')
    path = _seed_file(tmp_path) if seedcat else None
    assert C._seeded_companion_kwargs(_ctx(restored), path, is_miri) == {}


def test_kwargs_bad_value_raises(tmp_path, monkeypatch):
    monkeypatch.setenv('SATPROX_SEEDED_COMPANION', 'maybe')
    with pytest.raises(ValueError, match='SATPROX_SEEDED_COMPANION'):
        C._seeded_companion_kwargs(_ctx(np.ones((4, 4), bool)),
                                   _seed_file(tmp_path), False)


class _NanWCS:
    """Identity WCS whose second seed falls off the frame (NaN pixel)."""
    def world_to_pixel(self, sc):
        px = np.asarray(sc.ra.deg, dtype=float).copy()
        py = np.asarray(sc.dec.deg, dtype=float).copy()
        px[1] = np.nan
        return px, py


def test_kwargs_drop_nonfinite_seed_and_filter_runs(tmp_path, monkeypatch):
    """A seed whose pixel position is NaN is left out of the kwargs; the
    filter then runs (a NaN in the seed tree would raise)."""
    monkeypatch.setenv('SATPROX_SEEDED_COMPANION', '1')
    dq, res, restored = _scene()
    ctx = types.SimpleNamespace(ww=_NanWCS(), handoff_restore=restored)
    kw = C._seeded_companion_kwargs(ctx, _seed_file(tmp_path), False)
    np.testing.assert_allclose(kw['companion_seed_xy'], [[25.5, 20.0]])
    phot = _run(res, dq, **kw)
    assert list(phot.results['id']) == [1, 2]


# --- wiring: ctx -> _pass -> _manual_phot_pass -> _filter_near_saturation ---

def test_prepare_frame_returns_handoff_restore(tmp_path, monkeypatch):
    """_prepare_frame_for_photometry hands the restored pixels to the ctx the
    manual step builds the companion kwargs from."""
    from jwst_gc_pipeline.photometry.tests import test_gate_reject_handoff as G
    ctx, _ = G._prepare(tmp_path, monkeypatch)
    core = np.zeros(G.SHAPE, bool)
    core[47:54, 37:44] = True
    assert ctx.handoff_restore is not None
    assert ctx.handoff_restore[core].all()
    assert not ctx.handoff_restore[~core].any()


def test_manual_phot_pass_forwards_companion_kwargs(monkeypatch):
    from jwst_gc_pipeline.photometry.tests import test_gate_reject_handoff as G
    seen = {}

    def _capture(phot, dq, **kw):
        seen.update(kw)
        return 0
    monkeypatch.setattr(C, '_filter_near_saturation', _capture)
    shape = (40, 40)
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    sig = 2.574 / 2.3548
    img = 1e3 * np.exp(-((xx - 20.0) ** 2 + (yy - 20.0) ** 2) / (2 * sig ** 2))
    seed = Table()
    seed['x_init'] = [20.0]
    seed['y_init'] = [20.0]
    seed['flux_init'] = [1e4]
    restored = np.zeros(shape, bool)
    restored[20, 20] = True
    C._manual_phot_pass(
        data=img, mask=np.zeros(shape, bool), err=np.ones(shape),
        bad=np.zeros(shape, bool), dao_psf_model=G._gaussian_grid_psf_fwhm(2.574),
        init_params=seed, aperture_radius_pix=3.0, localbkg_inner=6,
        localbkg_outer=10, grouper=None, options=G._prep_options(),
        dq=np.zeros(shape, np.uint32), satstar_model_subtracted=np.zeros(shape),
        label='t', **_companion_kw(restored=restored))
    np.testing.assert_allclose(seen['companion_seed_xy'], SEED)
    assert seen['companion_seed_radius_pix'] == 0.5
    assert seen['companion_restored_mask'] is restored


def test_manual_step_passes_companion_kwargs_to_every_pass(tmp_path,
                                                           monkeypatch):
    """do_photometry_step_manual builds the kwargs from ctx.handoff_restore and
    the pass's seed catalog and hands them to _manual_phot_pass."""
    monkeypatch.setenv('SATPROX_SEEDED_COMPANION', '1')
    shape = (40, 40)
    restored = np.zeros(shape, bool)
    restored[20, 23:26] = True
    ctx = types.SimpleNamespace(
        im1=None, fwhm_pix=2.574, satstar_rejected_path=None,
        handoff_xy=HANDOFF, handoff_radius=1.3, handoff_restore=restored,
        satstar_table=None, nan_replaced_data=np.zeros(shape),
        mask=np.zeros(shape, bool), err=np.ones(shape),
        bad=np.zeros(shape, bool), dao_psf_model=None, aperture_radius_pix=5.0,
        localbkg_inner=6, localbkg_outer=10, grouper=None, dqarr=None,
        satstar_model_subtracted=None, data=np.zeros(shape), ww=_IdentityWCS())
    calls = []
    monkeypatch.setattr(C, '_prepare_frame_for_photometry',
                        lambda *a, **k: ctx)
    monkeypatch.setattr(C, '_build_manual_seed', lambda **k: Table())

    def _fake_pass(**kw):
        calls.append(kw)
        return Table(), np.zeros(shape), None
    monkeypatch.setattr(C, '_manual_phot_pass', _fake_pass)
    monkeypatch.setattr(C, '_save_manual_pass', lambda **k: None)
    C.do_photometry_step_manual(
        types.SimpleNamespace(target='gc-treasury'), 'F480M', 'nrcblong',
        'nrcblong', '105', str(tmp_path), 'f.fits', '10678',
        manual_phase='m7', prev_seed_catalog=_seed_file(tmp_path))
    assert calls
    for kw in calls:
        np.testing.assert_allclose(kw['companion_seed_xy'],
                                   [[25.5, 20.0], [3.0, 4.0]])
        assert kw['companion_seed_radius_pix'] == 0.5
        assert kw['companion_restored_mask'].sum() == 3
