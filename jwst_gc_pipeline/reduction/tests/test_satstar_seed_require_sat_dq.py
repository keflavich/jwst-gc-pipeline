"""Opt-in SAT-DQ requirement for peak-based and sub-floor satstar seeds (#991).

F212N has a severity floor of 4000 MJy/sr but a measured no-NaN SATURATED
onset of 38017: on gc-treasury o043, 91.2% of the accepted F212N satstars
(915 per frame) have no SATURATED DQ pixel within 2 px, because the peak and
sub-floor seeds pick up ordinary unsaturated stars between the floor and the
onset.  ``find_saturated_stars(seed_require_sat_dq=True)`` drops those seeds;
``SATSTAR_SEED_REQUIRE_SAT_DQ_FILTERS`` turns it on per bandpass.  Default
off, so every existing catalog and cache key is unchanged.

The test reads the DQ array, not the seed kind: a flagged component that the
severity gate dropped is re-seeded as ``subfloor`` and still carries the flag.
"""
import numpy as np
import pytest
from astropy.io import fits

from jwst_gc_pipeline.reduction.saturated_star_finding import (
    find_saturated_stars, satstar_fit_switch_signature,
    satstar_seed_require_sat_dq_filters)

SATBIT = 2
FLOOR = 4000.0
ENV = 'SATSTAR_SEED_REQUIRE_SAT_DQ_FILTERS'


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


def _blob(shape, x, y, r=2):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return (xx - x) ** 2 + (yy - y) ** 2 <= r ** 2


def _find(data, satmask=None, require=False, **kw):
    _, _, coms, kinds = find_saturated_stars(
        _fitsdata(data, satmask), severity_floor=FLOOR,
        seed_require_sat_dq=require, **kw)
    return coms, kinds


def _sky():
    return np.full((100, 100), 5.0)


# --------------------------------------------------------------------------
# seeds with no SATURATED DQ
# --------------------------------------------------------------------------

def test_peak_seed_without_dq_dropped_only_when_required():
    data = _sky()
    data[_blob(data.shape, 50, 50)] = 1.5 * FLOOR
    coms, kinds = _find(data, require=False)
    assert kinds == ['peak'] and len(coms) == 1      # the default is unchanged
    coms, kinds = _find(data, require=True)
    assert kinds == [] and len(coms) == 0


def test_subfloor_seed_without_dq_dropped_only_when_required():
    data = _sky()
    data[_blob(data.shape, 50, 50)] = 0.6 * FLOOR
    coms, kinds = _find(data, require=False)
    assert kinds == ['subfloor'] and len(coms) == 1
    coms, kinds = _find(data, require=True)
    assert kinds == [] and len(coms) == 0


def test_only_the_unflagged_seed_is_dropped():
    data = _sky()
    data[_blob(data.shape, 25, 25)] = 1.5 * FLOOR           # no DQ
    flagged = _blob(data.shape, 75, 75)
    data[flagged] = 2.0 * FLOOR
    coms, kinds = _find(data, satmask=flagged, require=True)
    assert kinds == ['dqsat'] and len(coms) == 1
    assert abs(coms[0][0] - 75) < 1 and abs(coms[0][1] - 75) < 1


# --------------------------------------------------------------------------
# seeds that do carry the flag
# --------------------------------------------------------------------------

def test_peak_seed_merged_with_a_flagged_component_is_kept():
    data = _sky()
    flagged = _blob(data.shape, 50, 50)
    data[flagged] = 2.0 * FLOOR
    data[48:53, 53:56] = 1.2 * FLOOR                         # unflagged shoulder
    sat0, _, coms0, kinds0 = find_saturated_stars(
        _fitsdata(data, flagged), severity_floor=FLOOR)
    sat1, _, coms1, kinds1 = find_saturated_stars(
        _fitsdata(data, flagged), severity_floor=FLOOR,
        seed_require_sat_dq=True)
    assert kinds0 == kinds1 == ['dqsat']
    assert (sat0 == sat1).all()


def test_flagged_component_the_severity_gate_dropped_is_kept():
    """Weak flagged star: the severity gate drops it (no NaN core, peak under
    the floor) and sub-floor seeding takes it back as ``subfloor``.  Its seed
    kind says no DQ; its DQ array says SATURATED.  Requiring SAT DQ must keep
    it, so the test cannot be ``seed_kind == 'dqsat'``.  A second, strongly
    flagged star keeps the gate from emptying the frame (it returns early
    when it drops every component, so nothing is re-seeded)."""
    data = _sky()
    strong = _blob(data.shape, 20, 20)
    data[strong] = 2.0 * FLOOR
    weak = _blob(data.shape, 75, 75)
    data[weak] = 0.6 * FLOOR
    satmask = strong | weak

    def _kind_at(coms, kinds, x, y):
        near = [k for (cy, cx), k in zip(coms, kinds)
                if abs(cx - x) < 2 and abs(cy - y) < 2]
        assert len(near) == 1
        return near[0]

    coms, kinds = _find(data, satmask=satmask, require=False)
    assert _kind_at(coms, kinds, 75, 75) == 'subfloor'
    coms, kinds = _find(data, satmask=satmask, require=True)
    assert len(coms) == 2
    assert _kind_at(coms, kinds, 75, 75) == 'subfloor'
    assert _kind_at(coms, kinds, 20, 20) == 'dqsat'


# --------------------------------------------------------------------------
# partner / sibling seeds are not touched
# --------------------------------------------------------------------------

def test_partner_seed_without_dq_is_kept():
    data = _sky()
    data[_blob(data.shape, 50, 50)] = 0.3 * FLOOR     # under the sub-floor seed
    kw = dict(partner_xy=[(50.0, 50.0)], partner_frac=0.25)
    coms, kinds = _find(data, require=False, **kw)
    assert kinds == ['partner']
    coms, kinds = _find(data, require=True, **kw)
    assert kinds == ['partner'] and len(coms) == 1


def test_partner_seed_claims_a_dropped_peak_seed():
    """The drop runs before the partner block, so a star with partner
    evidence keeps its component, as a ``partner`` seed."""
    data = _sky()
    data[_blob(data.shape, 50, 50)] = 1.5 * FLOOR
    kw = dict(partner_xy=[(50.0, 50.0)], partner_frac=0.25)
    assert _find(data, require=False, **kw)[1] == ['peak']
    coms, kinds = _find(data, require=True, **kw)
    assert kinds == ['partner'] and len(coms) == 1


def test_sibling_seed_without_dq_is_kept():
    data = _sky()
    data[_blob(data.shape, 50, 50)] = 100.0
    coms, kinds = _find(data, require=True, sibling_xy=[(50.0, 50.0)])
    assert kinds == ['sibling'] and len(coms) == 1


def test_no_severity_floor_no_change():
    data = _sky()
    flagged = _blob(data.shape, 50, 50)
    data[flagged] = 2.0 * FLOOR
    out0 = find_saturated_stars(_fitsdata(data, flagged), severity_floor=0.0)
    out1 = find_saturated_stars(_fitsdata(data, flagged), severity_floor=0.0,
                                seed_require_sat_dq=True)
    assert (out0[0] == out1[0]).all() and out0[3] == out1[3] == ['dqsat']


# --------------------------------------------------------------------------
# per-bandpass switch and cache key
# --------------------------------------------------------------------------

def test_filter_list_parsing():
    assert satstar_seed_require_sat_dq_filters({}) == frozenset()
    assert satstar_seed_require_sat_dq_filters({ENV: '  '}) == frozenset()
    got = satstar_seed_require_sat_dq_filters({ENV: 'f212n, F405N  '})
    assert got == {'F212N', 'F405N'}


def test_filter_list_typo_raises():
    with pytest.raises(ValueError, match='not bandpass names'):
        satstar_seed_require_sat_dq_filters({ENV: 'F212N,all'})


def _write(path, filt, pupil):
    fits.PrimaryHDU(header=fits.Header({'FILTER': filt, 'PUPIL': pupil})
                    ).writeto(path, overwrite=True)
    return str(path)


def test_signature_only_for_listed_bandpass(tmp_path):
    listed = _write(tmp_path / 'a_cal.fits', 'F212N', 'CLEAR')
    other = _write(tmp_path / 'b_cal.fits', 'F444W', 'F405N')
    env = {ENV: 'f212n'}
    base = satstar_fit_switch_signature(listed, env={})
    sig = satstar_fit_switch_signature(listed, env=env)
    assert sig == '_'.join(p for p in (base, 'sd') if p) and sig != base
    assert satstar_fit_switch_signature(other, env=env) == \
        satstar_fit_switch_signature(other, env={})
