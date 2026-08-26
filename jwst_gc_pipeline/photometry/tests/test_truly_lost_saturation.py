"""``SATURATED & DO_NOT_USE`` -- the saturation the ramp fit could NOT recover.

The cal/crf ``SATURATED`` flag marks a pixel saturated in ANY ramp group.  A
pixel that saturates in a LATE group still has a good group 0, so the ramp fit
produces a valid rate and the pixel carries no ``DO_NOT_USE``; only pixels with
no usable group do.  Keying ``is_saturated`` off the raw ``SATURATED`` bit
therefore masks recovered flux out of the daophot fit and lets
``_filter_near_saturation`` veto real point sources on bright emission (#418).

Measured on delivered products (2026-08-25), summing over the DQ extension:

    cloudc F770W  4 frames   301,901 SATURATED    39,148 also DO_NOT_USE  (13%)
    brick  F212N  6 frames     3,188 SATURATED     3,188 also DO_NOT_USE (100%)
    sgrb2  F480M  6 frames    15,981 SATURATED    15,981 also DO_NOT_USE (100%)

So the restriction removes 87% of the MIRI mask and is a **no-op on NIRCam**,
where every saturated pixel is already truly lost.  Both halves are pinned
below, because the second is what makes an on-by-default, instrument-agnostic
change safe for every NIRCam field.

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
    """The NIRCam case, measured at 100% on brick F212N and sgrb2 F480M: the
    restriction must not remove a single pixel there."""
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
    """`_prepare_frame_for_photometry` is where is_saturated is built, and it
    feeds the fit mask AND `_filter_near_saturation` (which reads ctx.dqarr).
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
