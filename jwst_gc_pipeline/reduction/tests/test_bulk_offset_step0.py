"""Step 0: verify-don't-reapply, raise on disagreement, and only re-measure when
something that affects the answer has changed."""

import glob
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
# dense threading (Gaia-only reducer path) -- see fix/gaia-only-per-tile-gate
# ---------------------------------------------------------------------------

def test_step0_forwards_dense_to_the_estimator(field, monkeypatch):
    """step0_bulk_offset must thread ``dense`` down to measure_bulk_offset -- else
    a Gaia-only field silently runs the per-tile gate (noise) and its real,
    coherent bulk tie is rejected (apply_ok=False), stranding the reducer's only
    absolute Gaia tie."""
    bp, frames = field
    seen = {}

    def spy(*args, **kwargs):
        seen['dense'] = kwargs.get('dense')
        return {'dra_mas': 420.0, 'ddec_mas': -130.0,
                'apply_ok': True, 'bulk_source': 'same-star'}

    monkeypatch.setattr(s0, 'measure_bulk_offset', spy)
    s0.step0_bulk_offset(None, None, None, frames, bp, '9999', '001', 'F187N',
                         dense=False)
    assert seen['dense'] is False


def test_measure_bulk_offset_signs_off_on_a_gaia_only_reference():
    """Integration: the real measure_bulk_offset wrapper, given ``dense=False``
    for a Gaia-only reference (full == sparse), signs off a coherent small tie via
    the same-star check, and threads ``dense`` down so ``per_tile_source`` names
    the spatial estimator that was consulted.  This is the reducer-side mirror of
    test_gaia_only_reference_per_tile_does_not_gate -- see its docstring for why
    the dense arm no longer REJECTS this particular (unoffset, densely paired)
    synthetic field since issue #610."""
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    rng = np.random.default_rng(3)
    ra0, dec0 = 290.9, 14.5
    cosd = np.cos(np.radians(dec0))
    ra = ra0 + rng.uniform(0, 90.0, 400) / 3600.0 / cosd
    dec = dec0 + rng.uniform(0, 90.0, 400) / 3600.0
    # consensus sits 10 mas west of the (Gaia-only) reference
    cons = SkyCoord(ra=(ra - 10.0 / 3.6e6 / cosd) * u.deg, dec=dec * u.deg,
                    frame='icrs')
    ref = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')  # full == sparse

    tie_sparse = s0.measure_bulk_offset(cons, ref, ref, dense=False,
                                        context='gaia-only')
    tie_dense = s0.measure_bulk_offset(cons, ref, ref, dense=True,
                                       context='gaia-only-dense')
    assert tie_sparse['apply_ok']                       # same-star signs off
    # `dense` reaches measure_reference_tie and picks the spatial estimator
    assert tie_sparse['per_tile_source'] == 'same-star-bulk'
    assert tie_dense['per_tile_source'] == 'same-star-region'
    # the histogram grid over this sparse reference is starved either way
    assert not tie_dense['per_tile'].get('clean')
    assert tie_sparse['dra_mas'] == pytest.approx(10.0, abs=2.0)


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
    # sickle moved GNS/RECORDED_BULK -> VIRAC2/TABLE_LOCKED on 2026-08-04, so its
    # tie now lives in Offsets_JWST_Brick3958_VIRAC2locked.csv like the other GC
    # fields.  Kept in this list on the other side of the line precisely because
    # it crossed it.
    ('3958', '007', s0.BULK_IN_TABLE),
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


# ---------------------------------------------------------------------------
# one catalog per exposure -- a loose glob must not double the input
# ---------------------------------------------------------------------------

def _cat_names(stage):
    """The per-exposure catalog names the pipeline writes, for one stage tag."""
    return [f'f187n_nrcb{det}_visit001_vgroup03102_exp{exp:05d}{stage}'
            f'_daophot_basic.fits'
            for det in (1, 2) for exp in (1, 2, 3)]


@pytest.fixture
def catalog_dir(tmp_path):
    """A filter directory carrying BOTH the plain `_m1` and the `_group_m1`
    variant of every exposure, which is what sickle F187N actually looks like."""
    d = tmp_path / 'F187N'
    d.mkdir()
    for name in _cat_names('_m1') + _cat_names('_group_m1'):
        (d / name).write_text('')
    return d


def test_exposure_key_ignores_the_stage_tag(catalog_dir):
    """The plain and `_group_` catalogs of one exposure are the SAME exposure --
    that is why a glob matching both doubles the stars fed to the estimator."""
    plain = catalog_dir / _cat_names('_m1')[0]
    grouped = catalog_dir / _cat_names('_group_m1')[0]
    assert s0.exposure_key(plain) == s0.exposure_key(grouped)
    assert s0.exposure_key(plain) == ('f187n', 'nrcb1', '001', '03102', '00001')


def test_unparseable_catalog_names_are_left_alone():
    """A hand-supplied --catalog-glob with some other naming scheme must not be
    rejected by this guard."""
    assert s0.exposure_key('/x/hand_made_catalog.fits') is None
    assert s0.duplicate_exposure_catalogs(
        ['/x/a.fits', '/x/b.fits', '/x/a.fits']) == {}


def test_default_glob_cannot_pick_up_a_group_sibling_set(catalog_dir):
    """A `_group_` sibling set must not double the input.

    This exercises ``default_catalog_glob`` itself, not a copy of its pattern, so
    loosening the exposure number back to ``exp*`` fails here.  Every exposure in
    ``catalog_dir`` has both a plain and a ``_group_`` catalog, which is what
    sickle F187N carries: the greedy form matched 192 catalogs for 96 frames and
    moved the measured tie by ~60 mas.
    """
    base = str(catalog_dir.parent)
    matched = sorted(glob.glob(s0.default_catalog_glob(base, 'F187N', '_m1')))
    assert len(matched) == 6, (
        f"default_catalog_glob matched {len(matched)} catalogs for 6 exposures; "
        f"it is picking up a second stage of the same exposures")
    assert s0.duplicate_exposure_catalogs(matched) == {}
    assert all('_group_m1_' not in os.path.basename(p) for p in matched)
    # the greedy form is what the guard exists for
    greedy = sorted(glob.glob(str(
        catalog_dir / 'f187n_*_visit*_vgroup*_exp*_m1_daophot_basic.fits')))
    assert len(greedy) == 12


def test_duplicate_exposure_catalogs_names_the_doubled_exposures(catalog_dir):
    """The guard is what protects a hand-written --catalog-glob, which the pinned
    default cannot."""
    greedy = sorted(glob.glob(str(
        catalog_dir / 'f187n_*_visit*_vgroup*_exp*_m1_daophot_basic.fits')))
    dupes = s0.duplicate_exposure_catalogs(greedy)
    assert len(dupes) == 6, "every exposure was matched twice"
    key = ('f187n', 'nrcb1', '001', '03102', '00001')
    assert sorted(os.path.basename(p) for p in dupes[key]) == [
        'f187n_nrcb1_visit001_vgroup03102_exp00001_group_m1_daophot_basic.fits',
        'f187n_nrcb1_visit001_vgroup03102_exp00001_m1_daophot_basic.fits',
    ]


# ---------------------------------------------------------------------------
# a field spanning several visits has no single "the" bulk tie
# ---------------------------------------------------------------------------

def _locked_table(path, rows):
    """A VIRAC2locked offsets table with (Visit, Filter, dra, ddec) rows."""
    from astropy.table import Table
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Table({'Visit': [r[0] for r in rows], 'Exposure': [1] * len(rows),
           'Filter': [r[1] for r in rows],
           'dra (arcsec)': [r[2] for r in rows],
           'ddec (arcsec)': [r[3] for r in rows]}).write(path, overwrite=True)


def test_visits_tied_to_different_values_are_refused(tmp_path):
    """brick 1182 obs004 visit 001 sits ~17" from visit 002.  One measurement
    over both cannot verify either, so picking the first is a silent choice."""
    bp = str(tmp_path / 'brick')
    _locked_table(f'{bp}/offsets/Offsets_JWST_Brick1182_VIRAC2locked.csv',
                  [('jw01182004001', 'F200W', -17.484, 13.546),
                   ('jw01182004002', 'F200W', 1.900, 0.100)])
    frames = ['jw01182004001_02101_00001_nrca1_destreak.fits',
              'jw01182004002_02101_00001_nrca1_destreak.fits']
    value, status, per_visit = s0.recorded_bulk_over_visits(
        bp, '1182', '004', 'F200W', frames, -28.72)
    assert status == s0.BULK_VISITS_DISAGREE
    assert value is None
    assert len(per_visit) == 2
    assert all(st == s0.BULK_OK for _, st in per_visit.values())


def test_some_visits_tied_and_others_not_is_its_own_state(tmp_path):
    """cloudef 2092 obs002 is a two-visit mosaic where only visit 002 carries a
    recorded shift -- reporting that as 'no row' blamed the wrong thing."""
    frames = ['jw02092002001_02101_00001_nrcblong_destreak.fits',
              'jw02092002002_02101_00001_nrcblong_destreak.fits']
    value, status, per_visit = s0.recorded_bulk_over_visits(
        str(tmp_path), '2092', '002', 'F480M', frames, -28.7)
    assert status == s0.BULK_VISITS_MIXED
    assert value is None
    assert {st for _, st in per_visit.values()} == {s0.BULK_OK, s0.BULK_NO_ROW}


def test_visits_agreeing_within_tolerance_resolve_normally(tmp_path):
    """Two visits that share a tie are not a conflict."""
    bp = str(tmp_path / 'f')
    _locked_table(f'{bp}/offsets/Offsets_JWST_Brick1182_VIRAC2locked.csv',
                  [('jw01182004001', 'F200W', 0.100, -0.050),
                   ('jw01182004002', 'F200W', 0.100, -0.050)])
    frames = ['jw01182004001_02101_00001_nrca1_destreak.fits',
              'jw01182004002_02101_00001_nrca1_destreak.fits']
    value, status, _ = s0.recorded_bulk_over_visits(
        bp, '1182', '004', 'F200W', frames, -28.72)
    assert status == s0.BULK_OK
    assert value[1] == pytest.approx(-50.0)


def test_a_single_visit_keeps_its_own_status(tmp_path):
    """Scoping to one visit must behave exactly as the single-visit lookup does."""
    bp = str(tmp_path / 'f')
    value, status, per_visit = s0.recorded_bulk_over_visits(
        bp, '1182', '004', 'F200W',
        ['jw01182004001_02101_00001_nrca1_destreak.fits'], -28.72)
    assert status == s0.BULK_TABLE_ABSENT       # no table written above
    assert value is None and len(per_visit) == 1


# ---------------------------------------------------------------------------
# the default refcat search has to cover the frame the field is tied to
# ---------------------------------------------------------------------------

def test_refcat_search_finds_a_gaia_only_catalog(tmp_path):
    """W51, M4 and M92 carry gaia_refcat.fits and nothing else; searching only
    gaia_virac2_refcat*.fits meant those fields could not start."""
    cats = tmp_path / 'catalogs'
    cats.mkdir()
    (cats / 'gaia_refcat.fits').write_text('')
    path, cands = s0.refcat_for_frame(str(tmp_path), 'Gaia')
    assert path is not None and os.path.basename(path) == 'gaia_refcat.fits'
    assert len(cands) == 1


def test_refcat_search_prefers_the_matching_frame(tmp_path):
    """With both catalogs present, the choice must follow the declared frame --
    otherwise the frame check would block the default path."""
    cats = tmp_path / 'catalogs'
    cats.mkdir()
    (cats / 'gaia_refcat.fits').write_text('')
    (cats / 'gaia_virac2_refcat_epoch2023.72.fits').write_text('')
    gaia, _ = s0.refcat_for_frame(str(tmp_path), 'Gaia')
    virac, _ = s0.refcat_for_frame(str(tmp_path), 'VIRAC2')
    assert os.path.basename(gaia) == 'gaia_refcat.fits'
    assert os.path.basename(virac) == 'gaia_virac2_refcat_epoch2023.72.fits'


def test_refcat_search_reports_nothing_found(tmp_path):
    (tmp_path / 'catalogs').mkdir()
    path, cands = s0.refcat_for_frame(str(tmp_path), 'VIRAC2')
    assert path is None and cands == []


def test_frame_glob_is_scoped_to_one_observation(tmp_path):
    """A filter directory holds every observation of the proposal.  cloudef F480M
    carries obs002 (two visits) alongside obs005; an unscoped `*_destreak.fits`
    folded obs005 into obs002's generation hash and reported a third visit."""
    pipeline = tmp_path / 'F480M' / 'pipeline'
    pipeline.mkdir(parents=True)
    for stem in ('jw02092002001', 'jw02092002002', 'jw02092005001'):
        (pipeline / f'{stem}_02101_00001_nrcblong_destreak.fits').write_text('')
    matched = sorted(os.path.basename(p) for p in glob.glob(
        s0.default_frame_glob(str(tmp_path), 'F480M', '2092', '002')))
    assert [m[:13] for m in matched] == ['jw02092002001', 'jw02092002002'], (
        f"obs005 leaked into the obs002 frame set: {matched}")


def test_frame_glob_zero_pads_proposal_and_field(tmp_path):
    """The stem is jw + 5-digit proposal + 3-digit observation."""
    assert '/jw03958007' in s0.default_frame_glob('/b', 'F187N', '3958', '007')
    assert '/jw01182004' in s0.default_frame_glob('/b', 'F200W', '1182', '4')


# ---------------------------------------------------------------------------
# raw-WCS vs already-tied catalogs
# ---------------------------------------------------------------------------

def _frame_with_applied(path, raoffset=None, deoffset=None):
    """A destreak-shaped frame, optionally carrying an applied WCS offset."""
    hdu0 = fits.PrimaryHDU()
    hdu0.header['CAL_VER'] = '1.14.1'
    hdu1 = fits.ImageHDU(np.zeros((4, 4), dtype='f4'))
    if raoffset is not None:
        hdu1.header['RAOFFSET'] = raoffset      # coordinate arcsec
        hdu1.header['DEOFFSET'] = deoffset
    fits.HDUList([hdu0, hdu1]).writeto(path, overwrite=True)
    return path


def test_raw_frames_report_no_applied_offset(tmp_path):
    """sgrc and cloudef frames carry RAOFFSET = 0; nothing to subtract."""
    p = _frame_with_applied(str(tmp_path / 'a_destreak.fits'), 0.0, 0.0)
    dra, ddec, n_with, n_total = s0.applied_bulk_mas([p], -29.0)
    assert (dra, ddec) == (0.0, 0.0)
    assert (n_with, n_total) == (1, 1)


def test_frames_without_the_keyword_report_zero_and_no_count(tmp_path):
    p = _frame_with_applied(str(tmp_path / 'a_destreak.fits'))
    dra, ddec, n_with, n_total = s0.applied_bulk_mas([p], -29.0)
    assert (dra, ddec, n_with, n_total) == (0.0, 0.0, 0, 1)


def test_applied_offset_is_converted_to_on_sky_mas(tmp_path):
    """RAOFFSET is Delta-alpha COORDINATE arcsec; the recorded bulk it has to be
    compared with is ON-SKY mas, so the RA term picks up cos(dec)."""
    p = _frame_with_applied(str(tmp_path / 'a_destreak.fits'), -17.5968, 13.4529)
    dra, ddec, _, _ = s0.applied_bulk_mas([p], -28.72)
    cosd = np.cos(np.radians(-28.72))
    assert dra == pytest.approx(-17.5968 * cosd * 1000.0)
    assert ddec == pytest.approx(13452.9, abs=0.1)
    # brick F200W visit 001 records (-15434.9, +13452.9) mas on-sky; the applied
    # value matches it, so the residual the frames still owe is ~0 -- NOT the
    # 20500 mas "failure" that comparing against the recorded bulk produces.
    recorded = (-15434.9, 13452.9)
    owed = np.hypot(recorded[0] - dra, recorded[1] - ddec)
    assert owed < 100.0, f"residual still owed came out {owed:.1f} mas"


def test_an_unreadable_frame_does_not_fake_a_zero_offset(tmp_path):
    """A frame that cannot be read must not contribute a 0.0 that drags the
    median toward 'already raw'."""
    good = _frame_with_applied(str(tmp_path / 'good_destreak.fits'), -17.6, 13.4)
    missing = str(tmp_path / 'gone_destreak.fits')
    dra, ddec, n_with, n_total = s0.applied_bulk_mas([good, missing], -28.72)
    assert n_with == 1 and n_total == 2
    assert dra == pytest.approx(-17.6 * np.cos(np.radians(-28.72)) * 1000.0)


def test_applied_offset_uses_the_median_not_the_first_frame(tmp_path):
    paths = [_frame_with_applied(str(tmp_path / f'{i}_destreak.fits'), off, 0.0)
             for i, off in enumerate((-17.0, -17.6, -17.5))]
    dra, _, n_with, _ = s0.applied_bulk_mas(paths, 0.0)
    assert n_with == 3
    assert dra == pytest.approx(-17500.0)


def test_per_visit_runs_do_not_share_one_record(field, monkeypatch):
    """brick 1182 obs004's two visits verify to different residuals.  A shared
    record filename meant each run evicted the other's."""
    bp, frames = field
    fake = _FakeTie(-28.9, -72.7)
    monkeypatch.setattr(s0, 'measure_bulk_offset', fake)
    s0.step0_bulk_offset(None, None, None, frames, bp, '1182', '004', 'F200W',
                         recorded_mas=(0.0, 0.0), visit='001')
    s0.step0_bulk_offset(None, None, None, frames, bp, '1182', '004', 'F200W',
                         recorded_mas=(0.0, 0.0), visit='002')
    for vis in ('001', '002'):
        assert os.path.exists(
            s0.step0_record_path(bp, '1182', '004', 'F200W', visit=vis)), (
            f"visit {vis} has no record of its own")
    # and the unscoped name is not what a scoped run writes
    assert not os.path.exists(s0.step0_record_path(bp, '1182', '004', 'F200W'))


def test_unscoped_record_path_is_unchanged(field):
    """Existing records must keep their names, so a scoped run is additive."""
    bp, _ = field
    assert s0.step0_record_path(bp, '3958', '007', 'F187N').endswith(
        'step0_bulk_3958_o007_F187N.json')
    assert s0.step0_record_path(bp, '3958', '007', 'F187N', visit='001').endswith(
        'step0_bulk_3958_o007_F187N_v001.json')
