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
