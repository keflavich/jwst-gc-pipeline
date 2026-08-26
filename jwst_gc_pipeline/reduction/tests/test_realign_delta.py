"""A frame whose baked correction disagrees with its offsets table is
RE-CORRECTED by the difference, not reported (issue #274).

The policy lives in ``unified_alignment.alignment_apply_plan`` so it has a
verdict without a FITS file, a GWCS or a reduction environment; the driver reads
that verdict.  The last test pins the wiring, because a correct policy nothing
calls is what the previous two rounds of this issue produced.
"""

import ast
import os

import numpy as np
import pytest
from astropy.io import fits

from jwst_gc_pipeline.reduction import unified_alignment as ua


def _hdr(ra=None, dec=None, **extra):
    """A minimal SCI header, optionally carrying a baked correction."""
    h = fits.Header()
    if ra is not None:
        h[ua.TOTAL_RA_KEY] = ra
    if dec is not None:
        h[ua.TOTAL_DEC_KEY] = dec
    for k, v in extra.items():
        h[k] = v
    return h


def _shift(ra=0.0, dec=0.0, **kw):
    kw.setdefault('source', 'TABLE_LOCKED')
    kw.setdefault('reference_frame', 'VIRAC2')
    return ua.AlignmentShift(bulk_ra=ra, bulk_dec=dec, **kw)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv('REALIGN_DELTA_ON_DISAGREE', raising=False)
    monkeypatch.delenv('FORCE_REALIGN_ON_DISAGREE', raising=False)
    monkeypatch.delenv('RAOFFSET_DISAGREE_TOL_ARCSEC', raising=False)


# ---------------------------------------------------------------------------
# the delta itself
# ---------------------------------------------------------------------------

def test_the_delta_is_the_table_minus_what_the_frame_carries():
    # brick-1182 v001: the frame holds +1.9", the corrected table says -17.5".
    d = ua.realign_delta(_hdr(1.9, 0.5), _shift(-17.5, 0.5), 'v001.fits')
    assert d.ok
    assert d.dra == pytest.approx(-19.4)
    assert d.ddec == pytest.approx(0.0)
    assert (d.baked_ra, d.baked_dec) == (1.9, 0.5)


def test_applying_the_delta_lands_on_the_table_value():
    baked, target = 0.0719, 0.1234
    d = ua.realign_delta(_hdr(baked, 0.0), _shift(target, 0.0))
    assert baked + d.dra == pytest.approx(target)


def test_an_unconfigured_field_is_refused_rather_than_zeroed():
    """``resolve_shift`` returns (0,0) for a field with no alignment_config
    entry.  Treating that as the target would UNDO a real correction."""
    d = ua.realign_delta(_hdr(-17.5, 0.5),
                         ua.AlignmentShift(configured=False, table_present=False),
                         'unconfigured.fits')
    assert not d.ok
    assert 'no configured alignment' in d.refusal
    assert d.dra == 0.0 and d.ddec == 0.0


def test_a_missing_table_is_refused_rather_than_zeroed():
    d = ua.realign_delta(_hdr(-17.5, 0.5), _shift(0.0, 0.0, table_present=False))
    assert not d.ok
    assert 'does not exist' in d.refusal


@pytest.mark.parametrize('header', [
    _hdr(1.9),                                  # RAOFFSET without DEOFFSET
    _hdr(1.9, 0.5, **{ua.TOTAL_RA_KEY: 'UNKNOWN'}),
])
def test_an_unreadable_baked_offset_is_refused(header):
    """A delta needs a minuend: a frame that cannot say what it already carries
    cannot have anything subtracted from the new value.  This must be a REFUSAL
    and not a ``ValueError`` from the middle of a reduction."""
    d = ua.realign_delta(header, _shift(-17.5, 0.5), 'broken.fits')
    assert not d.ok
    assert 'unreadable' in d.refusal


def test_a_frame_with_no_baked_offset_is_the_first_apply_path():
    d = ua.realign_delta(_hdr(), _shift(-17.5, 0.5))
    assert not d.ok
    assert ua.TOTAL_RA_KEY in d.refusal


# ---------------------------------------------------------------------------
# the policy
# ---------------------------------------------------------------------------

def test_a_virgin_frame_gets_the_whole_shift():
    verdict, delta, _ = ua.alignment_apply_plan(_hdr(), _shift(-17.5, 0.5))
    assert verdict == ua.APPLY_FULL
    assert delta is None


def test_a_frame_that_agrees_with_its_table_is_left_alone():
    verdict, delta, _ = ua.alignment_apply_plan(_hdr(-17.5, 0.5), _shift(-17.5, 0.5))
    assert verdict == ua.SKIP_CURRENT
    assert delta is None


def test_a_stale_frame_is_re_corrected_not_reported():
    """This is the whole issue: the historical behaviour was SKIP + warn."""
    verdict, delta, why = ua.alignment_apply_plan(
        _hdr(1.9, 0.5), _shift(-17.5, 0.5), 'v001.fits')
    assert verdict == ua.APPLY_DELTA
    assert delta.dra == pytest.approx(-19.4)
    assert 'STALE ASTROMETRY' in why


def test_the_sickle_f187n_double_count_direction_is_handled_too():
    """The other direction: the table absorbed a frame change the frames had
    not made.  103 mas, all 24 rows of the filter."""
    verdict, delta, _ = ua.alignment_apply_plan(
        _hdr(0.0719, -0.1801), _shift(-0.0177, -0.1033))
    assert verdict == ua.APPLY_DELTA
    assert delta.dra == pytest.approx(-0.0896)
    assert delta.ddec == pytest.approx(0.0768)


def test_an_unconfigured_stale_frame_falls_back_to_the_report_path():
    verdict, delta, why = ua.alignment_apply_plan(
        _hdr(-17.5, 0.5),
        ua.AlignmentShift(configured=False, table_present=False), 'x.fits')
    assert verdict == ua.SKIP_STALE
    assert delta is None
    assert 'no configured alignment' in why


def test_the_escape_hatch_restores_the_report_only_behaviour(monkeypatch):
    monkeypatch.setenv('REALIGN_DELTA_ON_DISAGREE', '0')
    verdict, _, why = ua.alignment_apply_plan(_hdr(1.9, 0.5), _shift(-17.5, 0.5))
    assert verdict == ua.SKIP_STALE
    assert 'REALIGN_DELTA_ON_DISAGREE=0' in why


def test_re_correcting_is_the_default():
    assert ua.realign_delta_enabled()


def test_the_component_split_decides_staleness_not_the_total():
    """A re-measured bulk masked by an opposite jitter change sums back to the
    same total; per-component it is still stale, and the delta is the total
    difference (zero here) -- so the frame is rewritten with the right split."""
    h = _hdr(1.0, 0.0)
    h[ua.BULK_RA_KEY] = 1.0
    h[ua.BULK_DEC_KEY] = 0.0
    h[ua.JITTER_RA_KEY] = 0.0
    h[ua.JITTER_DEC_KEY] = 0.0
    s = ua.AlignmentShift(bulk_ra=0.5, jitter_ra=0.5, source='T', reference_frame='V')
    assert s.total_ra == pytest.approx(1.0)
    verdict, delta, _ = ua.alignment_apply_plan(h, s)
    assert verdict == ua.APPLY_DELTA
    assert delta.dra == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# the wiring
# ---------------------------------------------------------------------------

_DRIVER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'PipelineRerunNIRCAM-LONG.py')


def _fix_alignment_source():
    with open(_DRIVER) as fh:
        tree = ast.parse(fh.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'fix_alignment':
            return node
    raise AssertionError('fix_alignment not found in ' + _DRIVER)


def test_the_driver_consults_the_plan():
    """Reverting the wiring -- going back to ``if RAOFFSET: warn`` -- fails
    here even though every policy test above still passes."""
    fn = _fix_alignment_source()
    called = {n.func.id for n in ast.walk(fn)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert 'alignment_apply_plan' in called


def test_the_driver_shifts_the_wcs_by_the_delta_not_the_total():
    """``adjust_wcs`` must consume the planned amount.  Passing ``rashift``
    there would re-apply the whole table value on top of the baked one, which
    is the double-count this issue's sickle case is."""
    fn = _fix_alignment_source()
    calls = [n for n in ast.walk(fn)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == 'adjust_wcs']
    assert calls, 'fix_alignment no longer calls adjust_wcs'
    for call in calls:
        kw = {k.arg: k.value for k in call.keywords}
        assert isinstance(kw['delta_ra'], ast.Name) and kw['delta_ra'].id == '_apply_ra'
        assert isinstance(kw['delta_dec'], ast.Name) and kw['delta_dec'].id == '_apply_dec'


def test_the_previous_total_is_recorded_so_the_step_is_reversible():
    with open(_DRIVER) as fh:
        src = fh.read()
    assert 'PREV_RA_KEY' in src and 'PREV_DEC_KEY' in src and 'NREALIGN_KEY' in src
