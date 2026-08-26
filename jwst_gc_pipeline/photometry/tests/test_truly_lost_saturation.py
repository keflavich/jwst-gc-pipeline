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
    """Synthetic frames and older products whose per-group DQ was not
    propagated: restricting would return an EMPTY saturation mask and silently
    turn every saturated core into ordinary data."""
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
