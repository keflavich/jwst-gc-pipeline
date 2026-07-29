"""Step 0: verify-don't-reapply, raise on disagreement, and only re-measure when
something that affects the answer has changed."""

import json
import os

import numpy as np
import pytest
from astropy.io import fits

from jwst_gc_pipeline.reduction import bulk_offset_step0 as s0
from jwst_gc_pipeline.reduction.bulk_offset_step0 import (
    BulkOffsetVerificationError,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _frame(path, cal_ver='1.14.1', crds_ctx='jwst_1253.pmap', dvacorr='COMPLETE'):
    hdu0 = fits.PrimaryHDU()
    hdu0.header['CAL_VER'] = cal_ver
    hdu0.header['CRDS_CTX'] = crds_ctx
    hdu1 = fits.ImageHDU(np.zeros((4, 4), dtype='f4'))
    hdu1.header['DVACORR'] = dvacorr
    fits.HDUList([hdu0, hdu1]).writeto(path, overwrite=True)
    return path


class _FakeTie:
    """Stand-in for measure_reference_tie, counting how often it is called."""

    def __init__(self, dra, ddec, apply_ok=True, bulk_source='same-star'):
        self.dra, self.ddec = dra, ddec
        self.apply_ok, self.bulk_source = apply_ok, bulk_source
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return {'dra_mas': self.dra, 'ddec_mas': self.ddec,
                'apply_ok': self.apply_ok, 'bulk_source': self.bulk_source}


@pytest.fixture
def field(tmp_path):
    bp = tmp_path / 'field'
    (bp / 'offsets').mkdir(parents=True)
    frames = [_frame(str(bp / f'jw02045001001_02101_0000{i}_nrcb3_destreak.fits'))
              for i in (1, 2)]
    return str(bp), frames


# ---------------------------------------------------------------------------
# verify_recorded_bulk
# ---------------------------------------------------------------------------

def test_agreeing_measurement_passes():
    sep = s0.verify_recorded_bulk((-90.0, -34.0), (-88.0, -35.5))
    assert sep == pytest.approx(np.hypot(2.0, -1.5))


def test_disagreement_raises_with_an_actionable_message():
    with pytest.raises(BulkOffsetVerificationError) as ei:
        s0.verify_recorded_bulk((-90.0, -34.0), (1800.0, -700.0),
                                context='3958/o007/F187N')
    msg = str(ei.value)
    assert 'BULK OFFSET VERIFICATION FAILED' in msg
    assert '3958/o007/F187N' in msg
    # it must say what to do, not just that something is wrong
    assert 'alignment_config.py' in msg
    assert 'Do NOT build products on this' in msg


def test_nonfinite_measurement_is_a_failure_not_a_pass():
    """A NaN separation must not slip through a `sep <= tol` comparison."""
    with pytest.raises(BulkOffsetVerificationError):
        s0.verify_recorded_bulk((-90.0, -34.0), (float('nan'), -34.0))


def test_override_env_downgrades_to_a_warning(monkeypatch):
    monkeypatch.setenv(s0.ALLOW_FAIL_ENV, '1')
    with pytest.warns(UserWarning, match='BULK OFFSET VERIFICATION FAILED'):
        sep = s0.verify_recorded_bulk((-90.0, -34.0), (1800.0, -700.0))
    assert sep > 100


def test_tolerance_is_env_overridable(monkeypatch):
    # 150 mas apart: fails at the default 100 mas ...
    with pytest.raises(BulkOffsetVerificationError):
        s0.verify_recorded_bulk((0.0, 0.0), (150.0, 0.0))
    # ... passes when the caller widens it deliberately
    monkeypatch.setenv('BULK_VERIFY_TOL_MAS', '200')
    assert s0.verify_recorded_bulk((0.0, 0.0), (150.0, 0.0)) == pytest.approx(150.0)


# ---------------------------------------------------------------------------
# generation hash
# ---------------------------------------------------------------------------

def test_hash_is_stable_when_nothing_changed(field):
    bp, frames = field
    a = s0.generation_hash(frames, 'VIRAC2', recorded=(-90.0, -34.0))
    b = s0.generation_hash(frames, 'VIRAC2', recorded=(-90.0, -34.0))
    assert a == b


@pytest.mark.parametrize('kw', [
    {'cal_ver': '1.15.0'}, {'crds_ctx': 'jwst_1300.pmap'}, {'dvacorr': 'SKIPPED'},
])
def test_hash_changes_when_the_wcs_generation_moves(field, kw):
    bp, frames = field
    before = s0.generation_hash(frames, 'VIRAC2', recorded=(-90.0, -34.0))
    _frame(frames[0], **kw)
    assert s0.generation_hash(frames, 'VIRAC2', recorded=(-90.0, -34.0)) != before


def test_hash_changes_with_the_reference_and_the_recorded_value(field):
    bp, frames = field
    base = s0.generation_hash(frames, 'VIRAC2', recorded=(-90.0, -34.0))
    assert s0.generation_hash(frames, 'Gaia', recorded=(-90.0, -34.0)) != base
    assert s0.generation_hash(frames, 'VIRAC2', recorded=(-91.0, -34.0)) != base


def test_unreadable_frame_does_not_produce_a_stable_hash(field, tmp_path):
    """A frame that cannot be read must not hash as if it had been checked --
    that would cache a verification which never really ran."""
    bp, frames = field
    good = s0.generation_hash(frames, 'VIRAC2')
    missing = os.path.join(bp, 'not_a_real_frame_destreak.fits')
    assert s0.generation_hash(frames + [missing], 'VIRAC2') != good


# ---------------------------------------------------------------------------
# the step: verify vs measure
# ---------------------------------------------------------------------------

def test_verify_mode_changes_nothing(field, monkeypatch):
    bp, frames = field
    fake = _FakeTie(-88.0, -35.0)
    monkeypatch.setattr(s0, 'measure_bulk_offset', fake)
    res = s0.step0_bulk_offset(None, None, None, frames, bp, '3958', '007',
                               'F187N', recorded_mas=(-90.0, -34.0))
    assert res.mode == 'verify'
    assert res.passed
    assert res.recorded_dra_mas == -90.0          # untouched
    assert res.measured_dra_mas == -88.0
    assert res.sep_mas == pytest.approx(np.hypot(2.0, -1.0))


def test_verify_mode_raises_when_the_field_has_moved(field, monkeypatch):
    bp, frames = field
    monkeypatch.setattr(s0, 'measure_bulk_offset', _FakeTie(1800.0, -700.0))
    with pytest.raises(BulkOffsetVerificationError):
        s0.step0_bulk_offset(None, None, None, frames, bp, '3958', '007',
                             'F187N', recorded_mas=(-90.0, -34.0))


def test_measure_mode_records_a_new_bulk_offset(field, monkeypatch):
    bp, frames = field
    monkeypatch.setattr(s0, 'measure_bulk_offset', _FakeTie(420.0, -130.0))
    res = s0.step0_bulk_offset(None, None, None, frames, bp, '9999', '001',
                               'F212N', recorded_mas=None)
    assert res.mode == 'measure'
    assert res.passed and res.apply_ok
    rec = json.load(open(s0.step0_record_path(bp, '9999', '001', 'F212N')))
    assert rec['measured_dra_mas'] == 420.0
    assert rec['passed'] is True
    assert rec['verified_at']


def test_measure_mode_refuses_to_record_an_unverified_tie(field, monkeypatch):
    """apply_ok=False means measure_reference_tie's own checks did not sign off
    (spurious peak, bad per-tile map, gross reference split).  Recording that
    would propagate a wrong frame into every downstream product."""
    bp, frames = field
    monkeypatch.setattr(s0, 'measure_bulk_offset',
                        _FakeTie(420.0, -130.0, apply_ok=False))
    with pytest.raises(BulkOffsetVerificationError, match='MEASUREMENT REJECTED'):
        s0.step0_bulk_offset(None, None, None, frames, bp, '9999', '001',
                             'F212N', recorded_mas=None)
    assert not os.path.exists(s0.step0_record_path(bp, '9999', '001', 'F212N'))


# ---------------------------------------------------------------------------
# cost control
# ---------------------------------------------------------------------------

def test_unchanged_generation_skips_the_expensive_remeasure(field, monkeypatch):
    bp, frames = field
    fake = _FakeTie(-88.0, -35.0)
    monkeypatch.setattr(s0, 'measure_bulk_offset', fake)
    kw = dict(recorded_mas=(-90.0, -34.0))
    first = s0.step0_bulk_offset(None, None, None, frames, bp, '3958', '007',
                                 'F187N', **kw)
    second = s0.step0_bulk_offset(None, None, None, frames, bp, '3958', '007',
                                  'F187N', **kw)
    assert fake.calls == 1, "step 0 re-measured despite nothing having changed"
    assert not first.from_cache and second.from_cache
    assert second.sep_mas == pytest.approx(first.sep_mas)


def test_a_new_generation_forces_a_remeasure(field, monkeypatch):
    bp, frames = field
    fake = _FakeTie(-88.0, -35.0)
    monkeypatch.setattr(s0, 'measure_bulk_offset', fake)
    kw = dict(recorded_mas=(-90.0, -34.0))
    s0.step0_bulk_offset(None, None, None, frames, bp, '3958', '007', 'F187N', **kw)
    _frame(frames[0], crds_ctx='jwst_1300.pmap')     # reprocessed
    res = s0.step0_bulk_offset(None, None, None, frames, bp, '3958', '007',
                               'F187N', **kw)
    assert fake.calls == 2
    assert not res.from_cache


def test_changing_the_recorded_value_forces_a_remeasure(field, monkeypatch):
    """Editing alignment_config.py must not be validated by a cache entry that
    was written for the OLD value."""
    bp, frames = field
    fake = _FakeTie(-88.0, -35.0)
    monkeypatch.setattr(s0, 'measure_bulk_offset', fake)
    s0.step0_bulk_offset(None, None, None, frames, bp, '3958', '007', 'F187N',
                         recorded_mas=(-90.0, -34.0))
    s0.step0_bulk_offset(None, None, None, frames, bp, '3958', '007', 'F187N',
                         recorded_mas=(-60.0, -20.0))
    assert fake.calls == 2


def test_force_bypasses_the_cache(field, monkeypatch):
    bp, frames = field
    fake = _FakeTie(-88.0, -35.0)
    monkeypatch.setattr(s0, 'measure_bulk_offset', fake)
    kw = dict(recorded_mas=(-90.0, -34.0))
    s0.step0_bulk_offset(None, None, None, frames, bp, '3958', '007', 'F187N', **kw)
    s0.step0_bulk_offset(None, None, None, frames, bp, '3958', '007', 'F187N',
                         force=True, **kw)
    assert fake.calls == 2


def test_a_failed_check_is_never_served_from_cache(field, monkeypatch):
    """Only a PASSING result may satisfy a later run; otherwise a single
    override would permanently silence the gate."""
    bp, frames = field
    monkeypatch.setenv(s0.ALLOW_FAIL_ENV, '1')
    fake = _FakeTie(1800.0, -700.0)
    monkeypatch.setattr(s0, 'measure_bulk_offset', fake)
    kw = dict(recorded_mas=(-90.0, -34.0))
    with pytest.warns(UserWarning):
        s0.step0_bulk_offset(None, None, None, frames, bp, '3958', '007',
                             'F187N', **kw)
    monkeypatch.delenv(s0.ALLOW_FAIL_ENV)
    with pytest.raises(BulkOffsetVerificationError):
        s0.step0_bulk_offset(None, None, None, frames, bp, '3958', '007',
                             'F187N', **kw)
    assert fake.calls == 2


# ---------------------------------------------------------------------------
# no ad-hoc measurement
# ---------------------------------------------------------------------------

def test_measurement_is_delegated_to_the_sanctioned_estimator():
    """This module must not grow its own offset estimator -- and in particular
    must never pair a nearest-neighbour match with a median/mean against a dense
    reference (CLAUDE.md ASTROMETRY RULE #1)."""
    src = open(s0.__file__.replace('.pyc', '.py')).read()
    assert 'measure_reference_tie' in src
    for banned in ('match_to_catalog_sky', 'search_around_sky'):
        assert banned not in src, (
            f"{banned} appeared in bulk_offset_step0.py; measurement belongs in "
            f"measure_reference_tie, not here")


# ---------------------------------------------------------------------------
# an estimator that refused to sign off cannot confirm anything
# ---------------------------------------------------------------------------

def test_verify_refuses_when_the_estimator_did_not_sign_off(field, monkeypatch):
    """apply_ok=False is usually a dirty per-tile map -- a rigid sub-field shift
    that the bulk number alone does not reveal, which is the single failure this
    step exists to catch.  A close separation must not pass it."""
    bp, frames = field
    monkeypatch.setattr(s0, 'measure_bulk_offset',
                        _FakeTie(-88.0, -35.0, apply_ok=False))
    with pytest.raises(BulkOffsetVerificationError, match='INCONCLUSIVE'):
        s0.step0_bulk_offset(None, None, None, frames, bp, '3958', '007',
                             'F187N', recorded_mas=(-90.0, -34.0))
    assert not os.path.exists(s0.step0_record_path(bp, '3958', '007', 'F187N'))


def test_widening_the_tolerance_does_not_cache_a_pass(field, monkeypatch):
    """A pass recorded under a widened BULK_VERIFY_TOL_MAS must not satisfy a
    later run at the normal tolerance."""
    bp, frames = field
    fake = _FakeTie(1800.0, -700.0)
    monkeypatch.setattr(s0, 'measure_bulk_offset', fake)
    kw = dict(recorded_mas=(-90.0, -34.0))
    monkeypatch.setenv('BULK_VERIFY_TOL_MAS', '5000')
    first = s0.step0_bulk_offset(None, None, None, frames, bp, '3958', '007',
                                 'F187N', **kw)
    assert first.passed
    monkeypatch.setenv('BULK_VERIFY_TOL_MAS', '100')
    with pytest.raises(BulkOffsetVerificationError):
        s0.step0_bulk_offset(None, None, None, frames, bp, '3958', '007',
                             'F187N', **kw)
    assert fake.calls == 2, "the widened-tolerance record was re-served"


# ---------------------------------------------------------------------------
# where a field's bulk tie lives -- three states, not two
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('proposal,field,expect', [
    ('1182', '004', s0.BULK_IN_TABLE),
    ('4147', '012', s0.BULK_IN_TABLE),
    ('6151', '001', s0.BULK_IN_TABLE),
    ('3958', '007', s0.BULK_RECORDED),
    ('2092', '002', s0.BULK_RECORDED),
    ('9999', '001', s0.BULK_NONE),
])
def test_bulk_tie_state(proposal, field, expect):
    """Fields tied by a TABLE must not look like fields with nothing on record;
    collapsing those sends a tied field into MEASURE."""
    assert s0.bulk_tie_state(proposal, field) == expect


def test_missing_table_is_not_reported_as_new_data(tmp_path):
    """A declared table that is absent means 'cannot tell', not 'new data'."""
    bp = str(tmp_path)
    os.makedirs(os.path.join(bp, 'offsets'))
    val, status = s0.recorded_bulk_mas(bp, '4147', '012', 'F212N',
                                       'jw04147012001', -29.4)
    assert val is None
    assert status == s0.BULK_TABLE_ABSENT


def test_table_without_a_matching_row_is_its_own_state(tmp_path):
    from astropy.table import Table
    bp = str(tmp_path)
    os.makedirs(os.path.join(bp, 'offsets'))
    Table(rows=[('jw01182004001', 'F200W', -17.5, 13.4)],
          names=('Visit', 'Filter', 'dra (arcsec)', 'ddec (arcsec)')).write(
        os.path.join(bp, 'offsets', 'Offsets_JWST_Brick1182_VIRAC2locked.csv'))
    val, status = s0.recorded_bulk_mas(bp, '1182', '004', 'F444W',
                                       'jw01182004001', -29.0)
    assert val is None and status == s0.BULK_NO_ROW


def test_a_genuine_zero_row_is_a_measurement_not_an_absence(tmp_path):
    """`(0, 0)` in the table is a real tie.  Treating it as 'nothing on record'
    would route an already-tied field to MEASURE."""
    from astropy.table import Table
    bp = str(tmp_path)
    os.makedirs(os.path.join(bp, 'offsets'))
    Table(rows=[('jw01182004001', 'F200W', 0.0, 0.0)],
          names=('Visit', 'Filter', 'dra (arcsec)', 'ddec (arcsec)')).write(
        os.path.join(bp, 'offsets', 'Offsets_JWST_Brick1182_VIRAC2locked.csv'))
    val, status = s0.recorded_bulk_mas(bp, '1182', '004', 'F200W',
                                       'jw01182004001', -29.0)
    assert status == s0.BULK_OK
    assert val == pytest.approx((0.0, 0.0))


def test_only_an_unconfigured_field_reads_as_new_data(tmp_path):
    val, status = s0.recorded_bulk_mas(str(tmp_path), '9999', '001', 'F212N',
                                       'jw09999001001', -29.0)
    assert val is None and status == s0.BULK_NOT_CONFIGURED


def test_recorded_bulk_uses_the_configured_visit_key(tmp_path):
    """2092 keys on the 3-character visit suffix; a hand-rolled
    basename.split('_')[0] misses it entirely."""
    v2, st2 = s0.recorded_bulk_mas(
        str(tmp_path), '2092', '002', 'F480M', 'ignored', 0.0,
        frame_name='jw02092002002_02101_00003_nrcb3_destreak.fits')
    assert st2 == s0.BULK_OK and v2[1] == pytest.approx(-171.0)
    v1, st1 = s0.recorded_bulk_mas(
        str(tmp_path), '2092', '002', 'F480M', 'ignored', 0.0,
        frame_name='jw02092002001_02101_00003_nrcb3_destreak.fits')
    assert st1 == s0.BULK_NO_ROW and v1 is None


# ---------------------------------------------------------------------------
# frame vs refcat
# ---------------------------------------------------------------------------

def test_gns_tie_measured_against_a_virac_refcat_is_refused():
    """sickle records a GNS tie; the default refcat search finds gaia_virac2.
    Comparing across frames yields a real separation that is NOT an astrometry
    error, and the failure message would send an operator after the wrong thing."""
    ok, detail = s0.reference_frame_matches_refcat(
        'GNS', '/orange/adamginsburg/jwst/sickle/catalogs/gaia_virac2_refcat_epoch2024.64.fits')
    assert not ok
    assert 'FRAME MISMATCH' in detail and 'GNS' in detail


def test_matching_frame_passes():
    ok, _ = s0.reference_frame_matches_refcat(
        'VIRAC2', '/x/catalogs/gaia_virac2_refcat_epoch2023.72.fits')
    assert ok


def test_unrecognised_refcat_does_not_block():
    """This check must not become a new way to fail a correct run."""
    ok, _ = s0.reference_frame_matches_refcat('VIRAC2', '/x/catalogs/mystery.fits')
    assert ok


# ---------------------------------------------------------------------------
# the unreadable-frame guard must be distinguishable from the basename hash
# ---------------------------------------------------------------------------

def test_unreadable_frame_changes_the_hash_beyond_its_name(field, tmp_path):
    """Mutation cover: hashing only the basename would give the same digest for
    two unreadable frames whose ERRORS differ.  A frame that cannot be read must
    not hash as though it had been checked."""
    bp, frames = field
    truncated = os.path.join(bp, 'jw02045001001_02101_00009_nrcb3_destreak.fits')
    with open(truncated, 'wb') as fh:
        fh.write(b'not a fits file at all')
    missing = os.path.join(bp, 'jw02045001001_02101_00009_nrcb3_destreak.fits.gone')
    h_trunc = s0.generation_hash([truncated], 'VIRAC2')
    h_missing = s0.generation_hash([missing], 'VIRAC2')
    # different failure modes -> different digests; a basename-only hash would
    # collide these two whenever the names matched
    assert h_trunc != h_missing
    # and neither may equal the digest of a readable frame
    assert h_trunc != s0.generation_hash([frames[0]], 'VIRAC2')
