"""Seed provenance survives the ZEROFRAME deblender (issue #212).

With ``--deblend-satstars`` on, ``get_saturated_stars`` replaced the
one-record-per-component loop (which carried ``_seed_kinds[ii]``) with
``build_deblended_source_records`` followed by a bare
``_r.setdefault('seed_kind', 'dqsat')``, so every seed -- peak, subfloor,
partner -- was reported as ``dqsat`` exactly when the deblender was the thing
being debugged.

``build_deblended_source_records`` stamps ``'label'`` (= component index + 1) on
every record it emits, and ``find_saturated_stars`` returns ``seed_kinds`` in
``np.arange(nsource) + 1`` label order, so ``seed_kinds[label - 1]`` recovers the
parent component's kind without a signature change.  These tests drive the real
``find_saturated_stars`` -> ``build_deblended_source_records`` ->
``stamp_seed_kinds`` chain that the ``zeroframe_deblend`` branch runs (the PSF
fitting that follows it in ``get_saturated_stars`` is not exercised).

``seed_kind`` is diagnostic only; no photometry reads it.
"""
import inspect

import numpy as np
import pytest
from astropy.io import fits
from scipy.ndimage import sum_labels

from jwst_gc_pipeline.reduction import saturated_star_finding as ssf
from jwst_gc_pipeline.reduction.saturated_star_finding import (
    find_saturated_stars, stamp_seed_kinds)
from jwst_gc_pipeline.reduction.satstar_deblend import (
    build_deblended_source_records)

SATBIT = 2
FLOOR = 4000.0


def _fitsdata(data, satmask=None):
    ny, nx = data.shape
    dq = np.zeros((ny, nx), dtype=np.uint32)
    if satmask is not None:
        dq[satmask] = SATBIT
    return fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(data=data.astype('float32'), name='SCI'),
        fits.ImageHDU(data=dq, name='DQ'),
        fits.ImageHDU(data=np.ones((ny, nx), dtype='float32'),
                      name='VAR_POISSON'),
    ])


def _disk(shape, x, y, r=3):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return (xx - x) ** 2 + (yy - y) ** 2 <= r ** 2


def _gauss(shape, x, y, amp, sig=1.5):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sig ** 2))


def _two_kind_scene():
    """One merged 'peak' component (two touching cores) + one 'partner' seed.

    Returns (records, kinds) after the deblend branch's own call chain.
    """
    shape = (160, 160)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    data = np.full(shape, 5.0)
    # two saturated cores bridged into ONE connected component
    pair = (_disk(shape, 60, 60, r=4) | _disk(shape, 70, 60, r=4)
            | ((np.abs(yy - 60) <= 2) & (np.abs(xx - 65) <= 5)))
    data[pair] = 1.5 * FLOOR              # above the severity floor, no DQ -> 'peak'
    data[_disk(shape, 120, 120, r=3)] = 0.30 * FLOOR   # -> 'partner'

    sat, sources, coms, kinds = find_saturated_stars(
        _fitsdata(data), severity_floor=FLOOR, partner_xy=[(120, 120)])

    # ZEROFRAME: saturates ~Ngroup higher, so both cores of the merged pair are
    # resolved point sources in it.
    zf = np.full(shape, 100.0)
    zf += (_gauss(shape, 60, 60, 5e4) + _gauss(shape, 70, 60, 4e4)
           + _gauss(shape, 120, 120, 2e4))
    zf += np.random.default_rng(0).normal(0.0, 1.0, shape)

    sizes = sum_labels(sat, sources, np.arange(len(coms)) + 1)
    records = build_deblended_source_records(
        sat, sources, coms, sizes, zf, data, 2.0,
        confirm_xy=[(60, 60), (70, 60), (120, 120)])
    return stamp_seed_kinds(records, kinds), kinds


def test_scene_has_two_kinds_and_a_split_component():
    """Guard the fixture: without a split component and two distinct kinds the
    seed-kind assertions below would pass trivially."""
    records, kinds = _two_kind_scene()
    assert sorted(kinds) == ['partner', 'peak']
    labels = [r['label'] for r in records]
    assert labels.count(1) == 2, "merged component should deblend into 2 stars"
    assert labels.count(2) == 1


def test_deblended_records_carry_parent_seed_kind():
    """The bug: every record read 'dqsat'.  Each now inherits its component."""
    records, kinds = _two_kind_scene()
    by_label = {r['label']: r['seed_kind'] for r in records}
    assert by_label[1] == kinds[0] == 'peak'
    assert by_label[2] == kinds[1] == 'partner'
    assert 'dqsat' not in {r['seed_kind'] for r in records}


def test_split_stars_share_the_component_kind():
    """Deliberate semantics: several stars deblended out of one component all
    inherit that component's seed kind (the deblender does not re-classify)."""
    records, _ = _two_kind_scene()
    split = [r['seed_kind'] for r in records if r['label'] == 1]
    assert split == ['peak', 'peak']


# ---- stamp_seed_kinds unit behaviour ---------------------------------------
def test_stamp_uses_label_minus_one():
    recs = [{'label': 1}, {'label': 3}, {'label': 2}]
    stamp_seed_kinds(recs, ['dqsat', 'peak', 'subfloor'])
    assert [r['seed_kind'] for r in recs] == ['dqsat', 'subfloor', 'peak']


def test_stamp_falls_back_to_dqsat_out_of_range():
    recs = [{'label': 9}, {'label': 0}, {}]
    stamp_seed_kinds(recs, ['peak'])
    assert [r['seed_kind'] for r in recs] == ['dqsat', 'dqsat', 'dqsat']


def test_stamp_does_not_override_explicit_kind():
    recs = [{'label': 1, 'seed_kind': 'forced'}]
    stamp_seed_kinds(recs, ['peak'])
    assert recs[0]['seed_kind'] == 'forced'


def test_stamp_empty_kinds_is_safe():
    recs = [{'label': 1}, {'label': 2}]
    stamp_seed_kinds(recs, [])
    assert all(r['seed_kind'] == 'dqsat' for r in recs)


# ---- call-site guard --------------------------------------------------------
def test_deblend_branch_routes_through_stamp_seed_kinds():
    """The unit tests above cannot see the wiring: reverting the call site to the
    bare ``setdefault('seed_kind', 'dqsat')`` would leave them green."""
    src = inspect.getsource(ssf.get_saturated_stars)
    assert 'stamp_seed_kinds(source_records, _seed_kinds)' in src
    assert "setdefault('seed_kind', 'dqsat')" not in src


if __name__ == '__main__':
    pytest.main([__file__])
