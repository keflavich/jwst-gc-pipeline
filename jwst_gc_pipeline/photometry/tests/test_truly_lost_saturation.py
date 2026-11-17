"""``SATURATED & DO_NOT_USE`` -- the saturation the ramp fit could NOT recover.

The cal/crf ``SATURATED`` flag marks a pixel saturated in ANY ramp group.  A
pixel that saturates in a LATE group still has a good group 0, so the ramp fit
produces a valid rate and the pixel carries no ``DO_NOT_USE``; only pixels with
no usable group do.  Keying ``is_saturated`` off the raw ``SATURATED`` bit
therefore masks recovered flux out of the daophot fit and lets
``_filter_near_saturation`` veto real point sources on bright emission (#418).

How much of the mask this removes is a property of the REDUCTION, not of the
instrument.  Census over all 140 delivered bands, 8 frames each (2026-08-26,
``scripts/analysis/truly_lost_saturation_census.py``), grouped by ``CAL_VER``:

    CAL_VER                     bands   min%   med%   max%   n at 100%
    1.14.1.dev43+g4641c6a09         5  100.0  100.0  100.0           5
    1.15.1                          1  100.0  100.0  100.0           1
    1.18.0                         13  100.0  100.0  100.0          13
    1.20.2                         20  100.0  100.0  100.0          20
    1.21.0.dev314+g61bd2fe47      101    0.5   15.7  100.0           9

Every band reduced with ``jwst <= 1.20.2`` reads 100%: there ``SATURATED`` and
``DO_NOT_USE`` are coextensive and the restriction is a bit-for-bit no-op.  On
current products the median band keeps 15.7% of its saturation mask, on BOTH
instruments -- NIRCam 4.1% (sgrc F470N) to 100%, MIRI 0.5% (sickle F1500W) to
100%.  An earlier version of this docstring claimed the restriction was a
"no-op on NIRCam"; that rested on two bands (brick F212N, sgrb2 F480M) which
read 100% because of their reduction generation, and it is wrong -- 80 of 125
NIRCam bands are below 100%.  Instrument is not the axis that separates them,
so an instrument gate would not scope this change either.

Unlike ``correct_dq_first_group_saturation`` this reads only the frame's own DQ
-- no sibling ``_ramp.fits``, no environment gate, no instrument branch.
"""
import types

import numpy as np
import pytest
from jwst.datamodels import dqflags

from jwst_gc_pipeline.reduction.saturated_star_finding import (
    truly_lost_saturated_mask)

SAT = dqflags.pixel['SATURATED']
DNU = dqflags.pixel['DO_NOT_USE']


def _miri_shaped_dq(ny=40, nx=40):
    """A big any-group-saturated region with a small truly-lost core inside it.

    The shape of the cloudc F770W blob: emission floods SATURATED over a wide
    area, and only the star cores inside it are unrecoverable.
    """
    dq = np.zeros((ny, nx), dtype=np.uint32)
    dq[5:35, 5:35] |= SAT                     # 900 px, any-group
    dq[18:22, 18:22] |= SAT | DNU             # 16 px, truly lost
    return dq


def test_restricts_to_the_truly_lost_core():
    dq = _miri_shaped_dq()
    assert int(((dq & SAT) > 0).sum()) == 900
    mask = truly_lost_saturated_mask(dq)
    assert int(mask.sum()) == 16
    assert mask[20, 20] and not mask[6, 6]


def test_is_a_noop_where_every_saturated_pixel_is_lost():
    """The ``jwst <= 1.20.2`` case (39 of 140 bands, both instruments): where
    SATURATED and DO_NOT_USE are coextensive the restriction must not remove a
    single pixel.  This is a property of those reductions, not of NIRCam."""
    dq = np.zeros((20, 20), dtype=np.uint32)
    dq[8:12, 8:12] |= SAT | DNU
    assert np.array_equal(truly_lost_saturated_mask(dq), (dq & SAT) > 0)


def test_falls_back_to_the_full_mask_when_no_pixel_carries_do_not_use():
    """Without the fallback, a frame with SATURATED but no DO_NOT_USE anywhere
    would get an EMPTY saturation mask and silently turn every saturated core
    into ordinary data.

    This is a SYNTHETIC-data guard and nothing more: of the 140 delivered bands
    in the census, ZERO have no saturated pixel carrying DO_NOT_USE, so the
    fallback never fires on real products.  The lowest band is sickle F1500W at
    6,837 / 1,330,121 = 0.51%.  It is therefore not a safety net for the bands
    where the restriction cuts the most -- arches F323N has 14,413 such pixels,
    so the fallback does not trigger and the mask shrinks by 92%.
    """
    dq = np.zeros((20, 20), dtype=np.uint32)
    dq[8:12, 8:12] |= SAT
    mask = truly_lost_saturated_mask(dq)
    assert int(mask.sum()) == 16
    assert np.array_equal(mask, (dq & SAT) > 0)


def test_unsaturated_do_not_use_is_not_saturation():
    """DO_NOT_USE alone marks dead/hot pixels and CR hits; only the conjunction
    is a lost saturated core."""
    dq = np.zeros((20, 20), dtype=np.uint32)
    dq[3, 3] |= DNU
    dq[8:12, 8:12] |= SAT | DNU
    mask = truly_lost_saturated_mask(dq)
    assert not mask[3, 3]
    assert int(mask.sum()) == 16


def test_empty_dq_gives_an_empty_mask():
    dq = np.zeros((10, 10), dtype=np.uint32)
    assert not truly_lost_saturated_mask(dq).any()


def test_env_gate_restores_the_any_group_mask(monkeypatch):
    """``SATSTAR_REQUIRE_DO_NOT_USE=0`` is the escape hatch back to the
    pre-#418 behaviour, for comparing a re-catalog against an old one."""
    dq = _miri_shaped_dq()
    monkeypatch.setenv('SATSTAR_REQUIRE_DO_NOT_USE', '0')
    assert int(truly_lost_saturated_mask(dq).sum()) == 900
    monkeypatch.setenv('SATSTAR_REQUIRE_DO_NOT_USE', '1')
    assert int(truly_lost_saturated_mask(dq).sum()) == 16


def test_the_daophot_path_uses_it():
    """`_prepare_frame_for_photometry` is where is_saturated is built.
    Reverting that call site to the raw SATURATED bit is the regression."""
    import ast
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1]
           / 'cataloging.py').read_text()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef)
              and n.name == '_prepare_frame_for_photometry')
    body = ast.get_source_segment(src, fn)
    assert 'truly_lost_saturated_mask' in body, (
        '_prepare_frame_for_photometry no longer restricts is_saturated to the '
        'truly-lost core; the any-group SATURATED bit over-masks MIRI by 87%')


# ---------------------------------------------------------------------------
# What the restriction does NOT reach, as currently wired.
#
# `_prepare_frame_for_photometry` builds the fit mask as
#
#     mask |= is_saturated
#     mask |= (dqarr & _bad_dq_bitmask(instrument)) != 0
#
# and `_bad_dq_bitmask` includes 'SATURATED' for BOTH instruments
# (_BAD_DQ_FLAGS_NIRCAM = ('DO_NOT_USE', 'SATURATED')), so the second line puts
# every any-group saturated pixel straight back.  Separately,
# `_filter_near_saturation` is handed `ctx.dqarr` -- the unmodified DQ array --
# and recomputes `(dq & SATURATED)` itself, so the near-saturation veto never
# sees the restriction either.
#
# The two tests below pin that.  They are not an endorsement: they exist so the
# #418 decision is taken against what the branch does rather than against what
# its rationale claims.  Making the restriction actually reach the fit mask and
# the veto means changing `_bad_dq_bitmask` and the `_filter_near_saturation`
# call, which is a much larger blast radius and a separate change.
# ---------------------------------------------------------------------------

def test_the_fit_mask_is_unchanged_because_bad_dq_readds_saturated():
    """Verified against real frames (arches F323N NIRCam, cloudc F770W MIRI):
    the mask `_prepare_frame_for_photometry` ends up with is bit-identical with
    and without the restriction, because `_bad_dq_bitmask` re-ORs SATURATED."""
    from jwst_gc_pipeline.photometry.crowdsource_catalogs_long import (
        _bad_dq_bitmask, _BAD_DQ_FLAGS_NIRCAM, _BAD_DQ_FLAGS_MIRI)

    assert 'SATURATED' in _BAD_DQ_FLAGS_NIRCAM
    assert 'SATURATED' in _BAD_DQ_FLAGS_MIRI

    dq = _miri_shaped_dq()
    data = np.zeros(dq.shape, dtype=float)
    bad = np.zeros(dq.shape, dtype=bool)

    def build_mask(is_saturated, instrument):
        mask = np.isnan(data) | bad
        mask |= is_saturated
        mask |= (dq & _bad_dq_bitmask(instrument)) != 0
        return mask

    for instrument in ('NIRCAM', 'MIRI'):
        restricted = build_mask(truly_lost_saturated_mask(dq), instrument)
        any_group = build_mask((dq & SAT) != 0, instrument)
        assert np.array_equal(restricted, any_group), instrument
        # and both mask the full 900-px any-group region, not the 16-px core
        assert int(restricted.sum()) == 900, instrument


def test_the_near_saturation_veto_still_uses_the_any_group_mask():
    """`_filter_near_saturation` recomputes `(dq & SATURATED)` from the DQ array
    it is passed, which the restriction does not modify.  A fit sitting on a
    RECOVERED (saturated, not DO_NOT_USE) pixel is therefore still dropped."""
    from astropy.table import Table
    from jwst_gc_pipeline.photometry.crowdsource_catalogs_long import (
        _filter_near_saturation)

    class _FakePhot:
        def __init__(self, results):
            self.results = results

    dq = _miri_shaped_dq()                 # 900 any-group, 16 truly-lost core
    assert not truly_lost_saturated_mask(dq)[7, 7]   # recovered, would be kept
    assert (dq[7, 7] & SAT) != 0

    results = Table()
    results['id'] = np.array([1, 2])
    results['x_fit'] = np.array([7.0, 1.0])   # on a recovered px; far control
    results['y_fit'] = np.array([7.0, 1.0])
    results['flux_fit'] = np.array([1000.0, 500.0])
    phot = _FakePhot(results)

    n_drop = _filter_near_saturation(phot, dq, max_sat_dist_pix=1.0,
                                     label='truly_lost_wiring')
    assert n_drop == 1, (
        'the veto no longer drops fits on recovered-saturated pixels -- if this '
        'now passes, _filter_near_saturation was rewired and the #418 '
        'measurement in this module needs redoing')
    assert set(np.asarray(phot.results['id'])) == {2}


# ---------------------------------------------------------------------------
# 2026-08-27: `cataloging._prepare_frame_for_photometry` called
# `_L.truly_lost_saturated_mask(dqarr)`, but crowdsource_catalogs_long
# re-exported only `remove_saturated_stars` and
# `correct_dq_first_group_saturation` from saturated_star_finding.  Every
# per-frame fit raised AttributeError -- 16 of 16 sgra m12 fan-out tasks in
# under a minute each, and the finalize path with them.
#
# The tests above exercise the FUNCTION.  Nothing checked that the name
# cataloging.py actually uses resolves, which is the whole failure.
# ---------------------------------------------------------------------------


def test_every_L_attribute_cataloging_uses_actually_resolves():
    """Guard the class, not the instance.

    cataloging.py imports crowdsource_catalogs_long as `_L` and reaches through
    it for helpers that live in other modules.  Any `_L.<name>` that is not
    re-exported is an AttributeError at fit time -- never at import time, so it
    survives every smoke test and dies once per frame in production.
    """
    import ast
    import pathlib
    from jwst_gc_pipeline.photometry import crowdsource_catalogs_long as _L

    src = pathlib.Path(
        _L.__file__).parent.joinpath('cataloging.py').read_text()
    tree = ast.parse(src)

    used = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == '_L'):
            used.add(node.attr)

    assert used, 'cataloging.py no longer reaches through _L; drop this test'
    missing = sorted(n for n in used if not hasattr(_L, n))
    assert not missing, (
        f'cataloging.py calls _L.{{{", ".join(missing)}}}, which '
        f'crowdsource_catalogs_long does not export. This raises AttributeError '
        f'per frame at fit time, not at import, so nothing else catches it.')


def test_the_two_dq_helpers_are_exported_together():
    """They are called on consecutive lines; exporting one without the other is
    exactly the break above."""
    from jwst_gc_pipeline.photometry import crowdsource_catalogs_long as _L
    assert hasattr(_L, 'correct_dq_first_group_saturation')
    assert hasattr(_L, 'truly_lost_saturated_mask')


# ---------------------------------------------------------------------------
# #567: opt-in `--truly-lost-saturation-in-fit` plumbing.  Flag off keeps the
# two pins above; flag on lets the restriction reach the fit mask and the veto.
# ---------------------------------------------------------------------------

def test_bad_dq_bitmask_keep_recovered_saturation():
    from jwst_gc_pipeline.photometry.crowdsource_catalogs_long import (
        _bad_dq_bitmask)

    nonsci = dqflags.pixel['NON_SCIENCE']
    persist = dqflags.pixel['PERSISTENCE']
    for instrument in ('NIRCAM', 'MIRI'):
        default = _bad_dq_bitmask(instrument)
        assert default == _bad_dq_bitmask(instrument,
                                          keep_recovered_saturation=False)
        assert default & SAT
        reduced = _bad_dq_bitmask(instrument, keep_recovered_saturation=True)
        assert not reduced & SAT
        assert reduced & DNU
        assert reduced == default & ~SAT
    miri = _bad_dq_bitmask('MIRI', keep_recovered_saturation=True)
    assert miri & nonsci and miri & persist


@pytest.mark.parametrize('instrument', ['NIRCAM', 'MIRI'])
def test_fit_mask_with_flag_keeps_only_the_truly_lost_core(instrument):
    from jwst_gc_pipeline.photometry.cataloging import _truly_lost_fit_products
    from jwst_gc_pipeline.photometry.crowdsource_catalogs_long import (
        _bad_dq_bitmask)

    dq = _miri_shaped_dq()
    truly_lost = truly_lost_saturated_mask(dq)
    dq_before = dq.copy()

    assert int(((dq & _bad_dq_bitmask(instrument)) != 0).sum()) == 900
    fit_add, _ = _truly_lost_fit_products(dq, truly_lost, instrument)
    assert int(fit_add.sum()) == 16
    assert not fit_add[7, 7]
    assert np.array_equal(dq, dq_before)


def test_veto_with_restricted_dq_keeps_recovered_and_drops_the_core():
    from astropy.table import Table
    from jwst_gc_pipeline.photometry.crowdsource_catalogs_long import (
        _filter_near_saturation)

    class _FakePhot:
        def __init__(self, results):
            self.results = results

    dq = _miri_shaped_dq()
    truly_lost = truly_lost_saturated_mask(dq)
    from jwst_gc_pipeline.photometry.cataloging import _truly_lost_fit_products
    _, restricted = _truly_lost_fit_products(dq, truly_lost, 'MIRI')
    assert restricted.dtype == dq.dtype
    assert int(((restricted & SAT) != 0).sum()) == 16

    results = Table()
    results['id'] = np.array([1, 2, 3, 4])
    # recovered px; inside the core; 1 px from the core; far control
    results['x_fit'] = np.array([7.0, 19.0, 17.0, 1.0])
    results['y_fit'] = np.array([7.0, 19.0, 19.0, 1.0])
    results['flux_fit'] = np.array([1000.0, 1000.0, 1000.0, 500.0])
    phot = _FakePhot(results)
    n_drop = _filter_near_saturation(phot, restricted, max_sat_dist_pix=1.0,
                                     label='tlsatfit')
    assert n_drop == 2
    assert set(np.asarray(phot.results['id'])) == {1, 4}


def test_flag_is_wired_through_the_parser():
    import pathlib

    here = pathlib.Path(__file__).resolve().parents[1]
    src = (here / 'crowdsource_catalogs_long.py').read_text()
    assert '"--truly-lost-saturation-in-fit"' in src
    assert 'dest="truly_lost_saturation_in_fit"' in src


# The tests below run the real _prepare_frame_for_photometry (the #925 hand-off
# fixture: a 7x7 any-group SATURATED block around a star, a zero satstar model
# file present so the satstar fill runs) with the flag off and on.  The hand-off
# is switched off (fit-quality reject, a target outside its default list) so
# the only thing that can un-mask the recovered pixels is the flag.

_BLOCK = np.zeros((100, 100), bool)
_BLOCK[47:54, 37:44] = True
_DNU_ROW = np.zeros((100, 100), bool)
_DNU_ROW[47, 37:44] = True


def _prep(tmp_path, monkeypatch, flag, **kw):
    from jwst_gc_pipeline.photometry.tests import test_gate_reject_handoff as G
    monkeypatch.delenv('DAOPHOT_HANDOFF_UNACCEPTED_SAT', raising=False)
    return G._prepare(tmp_path, monkeypatch, reject_reason='fit_quality_gate',
                      target=G.NO_DEFAULT_TARGET,
                      truly_lost_saturation_in_fit=flag, **kw)


def test_flag_off_is_the_any_group_mask(tmp_path, monkeypatch):
    ctx, img = _prep(tmp_path, monkeypatch, False, dnu_row=True)
    assert ctx.mask[_BLOCK].all()
    assert ctx.dq_for_satveto is ctx.dqarr
    assert ctx.truly_lost_in_fit is False
    # satstar fill still overwrites every any-group SATURATED pixel
    assert np.all(ctx.nan_replaced_data[_BLOCK] == 0.0)


def test_an_options_object_without_the_attribute_means_flag_off(tmp_path,
                                                                monkeypatch):
    """Callers that build their own options (older drivers, tests) never set
    the attribute; that must read as the default, off."""
    from jwst_gc_pipeline.photometry.tests import test_gate_reject_handoff as G
    monkeypatch.delenv('DAOPHOT_HANDOFF_UNACCEPTED_SAT', raising=False)
    ctx, _ = G._prepare(tmp_path, monkeypatch, reject_reason='fit_quality_gate',
                        target=G.NO_DEFAULT_TARGET, dnu_row=True)
    assert ctx.truly_lost_in_fit is False
    assert ctx.mask[_BLOCK].all()


def test_flag_on_unmasks_recovered_pixels_and_keeps_their_data(tmp_path,
                                                               monkeypatch):
    ctx, img = _prep(tmp_path, monkeypatch, True, dnu_row=True)
    assert ctx.truly_lost_in_fit is True
    assert np.array_equal(ctx.mask[_BLOCK], _DNU_ROW[_BLOCK])
    recovered = _BLOCK & ~_DNU_ROW
    assert not ctx.mask[recovered].any()
    # The satstar fill must not replace the un-masked pixels by the model (0
    # away from an accepted satstar): the fit has to read their measured rate.
    np.testing.assert_allclose(ctx.nan_replaced_data[recovered], img[recovered],
                               rtol=1e-5)
    # the veto sees only the truly-lost row; ctx.dqarr stays the any-group DQ
    assert int(((ctx.dq_for_satveto & SAT) != 0).sum()) == 7
    assert np.array_equal((ctx.dq_for_satveto & SAT) != 0, _DNU_ROW)
    assert int(((ctx.dqarr & SAT) != 0).sum()) == 49
    assert ctx.dq_for_satveto.dtype == ctx.dqarr.dtype


def test_flag_on_without_do_not_use_keeps_the_whole_block(tmp_path,
                                                          monkeypatch):
    """No DO_NOT_USE anywhere: truly_lost_saturated_mask falls back to every
    SATURATED pixel, so nothing is released and the veto DQ is unchanged."""
    ctx, _ = _prep(tmp_path, monkeypatch, True)
    assert ctx.mask[_BLOCK].all()
    assert np.array_equal((ctx.dq_for_satveto & SAT) != 0, _BLOCK)
    assert np.all(ctx.nan_replaced_data[_BLOCK] == 0.0)


def test_flag_on_takes_truly_lost_before_the_data_floor(tmp_path, monkeypatch):
    """A floor above every pixel empties the floor-restricted set; the fit mask
    and the veto DQ still carry the truly-lost row (flag on), and with the flag
    off the floor releases nothing from the any-group fit mask."""
    ctx_on, _ = _prep(tmp_path / 'on', monkeypatch, True, dnu_row=True,
                      saturation_data_floor=1.0e9)
    assert np.array_equal(ctx_on.mask[_BLOCK], _DNU_ROW[_BLOCK])
    assert np.array_equal((ctx_on.dq_for_satveto & SAT) != 0, _DNU_ROW)
    ctx_off, _ = _prep(tmp_path / 'off', monkeypatch, False, dnu_row=True,
                       saturation_data_floor=1.0e9)
    assert ctx_off.mask[_BLOCK].all()


def test_flag_on_recovers_the_flux_that_the_veto_drops_with_it_off(
        tmp_path, monkeypatch):
    from jwst_gc_pipeline.photometry.tests import test_gate_reject_handoff as G
    ctx_off, _ = _prep(tmp_path / 'off', monkeypatch, False, dnu_row=True)
    assert not G._has_fit_near(
        G._ctx_pass(ctx_off, handoff=False, dq=ctx_off.dq_for_satveto), G.BIG)
    ctx_on, _ = _prep(tmp_path / 'on', monkeypatch, True, dnu_row=True)
    res = G._ctx_pass(ctx_on, handoff=False, dq=ctx_on.dq_for_satveto)
    assert G._has_fit_near(res, G.BIG)
    d = np.hypot(np.asarray(res['x_fit']) - G.BIG[0],
                 np.asarray(res['y_fit']) - G.BIG[1])
    flux = float(np.asarray(res['flux_fit'])[np.argmin(d)])
    true = G.BIG_AMP * 2 * np.pi * (2.574 / 2.3548) ** 2
    assert flux == pytest.approx(true, rel=0.02)


def test_the_manual_pass_gets_the_veto_dq(monkeypatch):
    """do_photometry_step_manual hands _manual_phot_pass ctx.dq_for_satveto,
    not ctx.dqarr."""
    import types
    from astropy.table import Table
    from jwst_gc_pipeline.photometry import cataloging as C
    from jwst_gc_pipeline.photometry.tests import test_gate_reject_handoff as G

    dq_any = np.zeros(G.SHAPE, np.uint32)
    dq_veto = dq_any + 1
    ctx = types.SimpleNamespace(
        im1=None, fwhm_pix=G.FWHM, satstar_rejected_path=None, handoff_xy=None,
        handoff_radius=0.0, satstar_table=None,
        nan_replaced_data=np.zeros(G.SHAPE), mask=np.zeros(G.SHAPE, bool),
        err=np.ones(G.SHAPE), bad=np.zeros(G.SHAPE, bool), dao_psf_model=None,
        aperture_radius_pix=5.0, localbkg_inner=6, localbkg_outer=10,
        grouper=None, dqarr=dq_any, dq_for_satveto=dq_veto,
        satstar_model_subtracted=None, data=np.zeros(G.SHAPE), ww=None)
    seen = {}
    monkeypatch.setattr(C, '_prepare_frame_for_photometry', lambda *a, **k: ctx)
    monkeypatch.setattr(C, '_build_manual_seed', lambda **k: Table())

    def _fake_pass(**kw):
        seen.update(kw)
        return Table(), np.zeros(G.SHAPE), None
    monkeypatch.setattr(C, '_manual_phot_pass', _fake_pass)
    monkeypatch.setattr(C, '_save_manual_pass', lambda **k: None)
    C.do_photometry_step_manual(types.SimpleNamespace(target='brick'), 'F212N',
                                'nrca1', 'nrca1', '001', '/nonexistent',
                                'f.fits', '2221', manual_phase='m7')
    assert seen['dq'] is dq_veto


@pytest.mark.parametrize('flag', [False, True])
def test_the_arm_is_written_to_the_per_frame_header(tmp_path, monkeypatch,
                                                    flag):
    """TLSATFIT in the per-frame catalog header tells the A and B arms apart:
    they write the same product names."""
    from astropy.table import Table
    from jwst_gc_pipeline.photometry import cataloging as C
    from jwst_gc_pipeline.photometry import crowdsource_catalogs_long as L
    ctx, _ = _prep(tmp_path, monkeypatch, flag, dnu_row=True)
    captured = {}

    def _fake_save(result, *a, **k):
        captured['meta'] = dict(result.meta)
        return 'saved'
    monkeypatch.setattr(L, 'save_photutils_results', _fake_save)
    monkeypatch.setattr(L, 'save_residual_datamodel', lambda *a, **k: None)
    monkeypatch.setattr(C, '_attach_residual_background', lambda *a, **k: None)
    ctx.satstar_model_subtracted = None
    result = Table()
    C._save_manual_pass(ctx, result, np.zeros(ctx.data.shape),
                        types.SimpleNamespace(blur=False), 'm7', 'nrcalong')
    card = captured['meta']['TLSATFIT']
    assert (card[0] if isinstance(card, tuple) else card) is flag
