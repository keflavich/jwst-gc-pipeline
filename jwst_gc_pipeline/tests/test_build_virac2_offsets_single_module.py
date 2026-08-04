"""Builder behaviour on a SINGLE-MODULE field.

sickle (3958/007) is nrcb-only.  Three separate places in
``build_virac2_offsets`` assumed a two-module field and refused the whole build
for a field that is perfectly measurable:

* ``coarse_from_i2d`` required a ``-merged`` mosaic, which a single-module field
  never produces;
* ``_gather``'s ``exp*`` glob also matched the grouped-fit per-frame catalogs
  sickle carries, so two products claimed the same frame and the duplicate check
  aborted;
* ``lock_filter`` let the "no per-frame catalogs" refusal from the module the
  field never observed abort the build for the module it did.

These test the first two directly, and the third's guard rail: skipping empty
modules must not degrade into silently locking nothing.
"""
import os

import pytest

from jwst_gc_pipeline.reduction.build_virac2_offsets import (
    MOSAIC_PREFERENCE, REGION, WrongObservationError, mosaic_candidates,
    perframe_matches)


STEM = '/base/F210M/pipeline/jw03958-o007_t001_nircam_clear-f210m'


def test_merged_is_preferred_over_any_single_module():
    """A two-module field must tie on merged -- the seam lives nowhere else."""
    labels = [label for label, _ in mosaic_candidates(STEM)]
    assert labels[0] == 'merged'
    assert set(labels[1:]) == {'nrca', 'nrcb', 'nrcalong', 'nrcblong'}


def test_single_module_field_has_a_candidate_to_fall_back_to(tmp_path):
    """nrcb-only: no merged product exists, but the build is still measurable."""
    stem = str(tmp_path / 'jw03958-o007_t001_nircam_clear-f210m')
    (tmp_path / 'jw03958-o007_t001_nircam_clear-f210m-nrcb_i2d.fits').touch()
    present = [(label, path) for label, path in mosaic_candidates(stem)
               if os.path.exists(path)]
    assert [label for label, _ in present] == ['nrcb']


def test_no_mosaic_at_all_is_still_refused(tmp_path):
    stem = str(tmp_path / 'jw03958-o007_t001_nircam_clear-f210m')
    assert not [p for _, p in mosaic_candidates(stem) if os.path.exists(p)]


@pytest.mark.parametrize('mtag', ['_m3', '_m6'])
def test_grouped_variant_is_not_mistaken_for_the_plain_per_frame(mtag):
    """The grouped fit is a legitimate SECOND product of the same exposure.

    Matching both is what made the duplicate check report them as a stale
    pre-observation-token duplicate, which they are not.
    """
    plain = f'f210m_nrcb1_visit001_vgroup0310e_exp00001{mtag}_daophot_basic.fits'
    grouped = (f'f210m_nrcb1_visit001_vgroup0310e_exp00001_group{mtag}'
               f'_daophot_basic.fits')
    assert perframe_matches(plain, mtag)
    assert not perframe_matches(grouped, mtag)


def test_the_grouped_variant_is_matched_when_it_is_what_was_asked_for():
    """Nothing here should make the grouped products unreachable."""
    grouped = ('f210m_nrcb1_visit001_vgroup0310e_exp00001_group_m3'
               '_daophot_basic.fits')
    assert perframe_matches(grouped, '_group_m3')


def test_a_different_merge_tag_does_not_match():
    name = 'f210m_nrcb1_visit001_vgroup0310e_exp00001_m3_daophot_basic.fits'
    assert not perframe_matches(name, '_m6')


def test_sickle_region_is_registered_single_module_and_obs_007():
    """Step 0 refuses to measure a fresh tie for an already-tied field, so the
    route to VIRAC2 is to BUILD the table -- which needs a REGION entry."""
    rc = REGION['sickle']
    assert rc['proposal'] == '3958'
    # NIRCam is observation 007; the 3958 MIRI data are observation 001 and are
    # NOT tied by this entry.
    assert rc['field'] == '007'
    assert set(rc['filts']) == {'f187n', 'f210m', 'f335m', 'f470n', 'f480m'}
    # One epoch for the whole field: all five bands were taken together.
    assert {v[1] for v in rc['filts'].values()} == {2024.643}
    assert {v[2] for v in rc['filts'].values()} == {'_m3'}


def test_skipping_empty_modules_cannot_silently_lock_nothing():
    """The empty-module skip is only safe because a build that skipped EVERY
    module still raises.  Without this, a field whose catalogs were all missing
    would produce a zero-row table that reads as 'tied'."""
    import inspect

    from jwst_gc_pipeline.reduction import build_virac2_offsets as b
    src = inspect.getsource(b.lock_filter)
    assert 'no per-frame catalogs' in src, 'the empty-module skip is gone'
    assert 'if not rows:' in src, 'the skip lost its no-rows guard'
    assert WrongObservationError.__name__ in src
