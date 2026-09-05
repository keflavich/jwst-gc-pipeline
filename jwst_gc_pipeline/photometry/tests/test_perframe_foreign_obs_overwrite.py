"""A per-frame catalog write must not replace another observation's file.

cloudef is proposal 2092 observations 002 and 005 under one basepath, and the
per-frame name carries no observation token for that proposal, so both
observations spell
``f480m_nrcalong_visit001_vgroup02103_exp00001_m2_daophot_basic.fits``.  The
obs-005 recatalog on 2026-08-19 wrote over 528 of obs-002's catalogs; F480M
kept none of its own and its offsets table could not be rebuilt.  Nothing
raised (issue #718).

The end-to-end test below drives the real writer with cloudef's real numbers
and asserts the second observation's write now stops.  The unit tests around it
pin what the guard deliberately lets through, so the refusal cannot be widened
into "refuse to overwrite anything" -- a re-run of the SAME exposure must keep
overwriting its own output, which is how every iteration of every stage works.
"""
import os
import types

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from jwst_gc_pipeline.photometry.perframe_write_guard import (
    ForeignObservationOverwriteError,
    assert_no_foreign_observation_overwrite,
    exposure_observation,
    foreign_observation_conflict,
    recorded_source_exposure,
)

O002 = ('/orange/adamginsburg/jwst/cloudef/F480M/pipeline/'
        'jw02092002001_02103_00001_nrcalong_destreak_o002_crf.fits')
O005 = ('/orange/adamginsburg/jwst/cloudef/F480M/pipeline/'
        'jw02092005001_02103_00001_nrcalong_destreak_o005_crf.fits')


def _catalog(path, source_exposure):
    """A per-frame catalog on disk, stamped like the writers stamp one."""
    tbl = Table({'x_fit': [1.0], 'y_fit': [1.0]})
    if source_exposure is not None:
        tbl.meta['filename'] = source_exposure
    tbl.write(path, overwrite=True)
    return path


# --- the identity the guard keys on ---------------------------------------

def test_exposure_observation_reads_proposal_and_observation():
    assert exposure_observation(O002) == ('2092', '002')
    assert exposure_observation(O005) == ('2092', '005')


def test_exposure_observation_ignores_the_directory():
    """gc2211's per-observation trees put an observation in the PATH.

    The exposure's own is the one that matters; reading the path instead is
    the class of bug ``naming.frame_identity`` documents (issue #472).
    """
    assert exposure_observation(
        '/orange/adamginsburg/jwst/gc2211_o023/F200W/pipeline/'
        'jw02211046001_02101_00001_nrca1_cal.fits') == ('2211', '046')


def test_exposure_observation_declines_a_mosaic_and_a_stranger():
    """A name that does not identify one exposure yields no identity.

    The merged-module catalogs go through the same writer with a resampled
    mosaic as their source.  Merged-level naming is a separate concern with
    its own closed PR (#459); the guard must not start refusing there.
    """
    assert exposure_observation(
        'jw02221-o001_t001_nircam_clear-f410m_i2d.fits') is None
    assert exposure_observation('some_hand_made_table.fits') is None
    assert exposure_observation(None) is None


# --- what it refuses -------------------------------------------------------

def test_refuses_a_write_over_another_observations_catalog(tmp_path):
    out = str(tmp_path / 'f480m_nrcalong_visit001_vgroup02103_exp00001.fits')
    _catalog(out, O002)
    with pytest.raises(ForeignObservationOverwriteError) as err:
        assert_no_foreign_observation_overwrite(out, O005)
    assert '002' in str(err.value) and '005' in str(err.value)


# --- what it still lets through -------------------------------------------

def test_a_rerun_overwrites_its_own_output(tmp_path):
    """The same exposure re-fit at a later iteration must still overwrite."""
    out = str(tmp_path / 'f480m_nrcalong_visit001_vgroup02103_exp00001.fits')
    _catalog(out, O005)
    assert_no_foreign_observation_overwrite(out, O005)


def test_a_first_write_is_not_blocked(tmp_path):
    out = str(tmp_path / 'not_there_yet.fits')
    assert_no_foreign_observation_overwrite(out, O005)


def test_an_existing_file_with_no_provenance_is_not_blocked(tmp_path):
    """Nothing to compare: a table written before the FILENAME stamp existed.

    Refusing here would stop runs over files that are very likely their own.
    """
    out = str(tmp_path / 'f480m_nrcalong_visit001_vgroup02103_exp00001.fits')
    _catalog(out, None)
    assert recorded_source_exposure(out) is None
    assert_no_foreign_observation_overwrite(out, O005)


def test_a_truncated_existing_file_is_not_blocked(tmp_path):
    """A zero-length catalog a killed job left behind reads as unknown."""
    out = str(tmp_path / 'f480m_nrcalong_visit001_vgroup02103_exp00001.fits')
    open(out, 'wb').close()
    assert recorded_source_exposure(out) is None
    assert_no_foreign_observation_overwrite(out, O005)


def test_recorded_source_exposure_reads_the_ext1_stamp(tmp_path):
    out = str(tmp_path / 'stamped.fits')
    _catalog(out, O002)
    assert fits.getheader(out, ext=1)['FILENAME'] == O002
    assert recorded_source_exposure(out) == O002


# --- end to end, through the real writer ----------------------------------

def _wcs():
    w = WCS(naxis=2)
    w.wcs.crpix = [10, 10]
    w.wcs.cdelt = [-1.7e-5, 1.7e-5]
    w.wcs.crval = [266.5, -28.7]
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.cunit = ['deg', 'deg']
    return w


def _write_frame(basepath, source_exposure):
    """cloudef F480M exposure 1 of visit 001, vgroup 02103, through the writer."""
    from jwst_gc_pipeline.photometry import crowdsource_catalogs_long as L
    im1 = fits.HDUList([fits.PrimaryHDU(),
                        fits.ImageHDU(np.zeros((4, 4), dtype=float))])
    options = types.SimpleNamespace(each_exposure=True, proposal_id='2092',
                                    field='005', desaturated=False)
    result = Table({'x_fit': [1.0, 2.0], 'y_fit': [1.0, 2.0],
                    'flux_fit': [10.0, 20.0]})
    L.save_photutils_results(
        result, _wcs(), source_exposure, im1, 'nrcalong', basepath, 'F480M',
        'nrcalong', '', '', '_exp00001', '_visit001', '_vgroup02103',
        options=options, basic_or_iterative='basic', iteration_label='m2')


def test_writer_stops_the_cloudef_collision(tmp_path):
    """The 2026-08-19 event: obs-005's write lands on obs-002's file."""
    basepath = str(tmp_path)
    os.makedirs(os.path.join(basepath, 'F480M'))
    _write_frame(basepath, O002)
    written = os.listdir(os.path.join(basepath, 'F480M'))
    # The name carries no observation token, which is the whole problem.
    assert written == ['f480m_nrcalong_visit001_vgroup02103_exp00001_m2'
                       '_daophot_basic.fits']
    with pytest.raises(ForeignObservationOverwriteError):
        _write_frame(basepath, O005)
    # obs-002's catalog is still obs-002's.
    kept = os.path.join(basepath, 'F480M', written[0])
    assert recorded_source_exposure(kept) == O002


def test_writer_still_overwrites_its_own_frame(tmp_path):
    basepath = str(tmp_path)
    os.makedirs(os.path.join(basepath, 'F480M'))
    _write_frame(basepath, O005)
    _write_frame(basepath, O005)
    out = os.path.join(basepath, 'F480M',
                       'f480m_nrcalong_visit001_vgroup02103_exp00001_m2'
                       '_daophot_basic.fits')
    assert recorded_source_exposure(out) == O005


# --- the SKIP door ---------------------------------------------------------
#
# The write door refusing is only half of it.  `--skip-if-done` and
# `--list-missing-tasks` both ask `_expected_output_exists`, which asked
# `os.path.exists` and nothing else: a file the OTHER observation wrote answers
# "this frame is already done", the run skips every colliding frame, exits 0
# having measured nothing, and the foreign photometry stands in for it
# downstream.  That is the same data loss through a quieter door -- and it is
# reachable on cloudef today, where 134 exposure keys are spelled by both
# 2092/002 and 2092/005 (F162M 63, F210M 49, F360M 9, F480M 13).

def _sentinel_options(**kw):
    opts = types.SimpleNamespace(
        daophot=True, basic_only=True, each_exposure=True, desaturated=False,
        bgsub=False, epsf=False, blur=False, group=False, proposal_id='2092',
        field='005', use_iter3_residual_bg=False, manual_stop_after_phase='',
        iteration_label=None, each_suffix='destreak', modules='nrcalong')
    opts.__dict__.update(kw)
    return opts


def _sentinel_path(basepath, options, **kw):
    from jwst_gc_pipeline.photometry import crowdsource_catalogs_long as L
    return L._predict_tblfilename(
        basepath, 'F480M', 'nrcalong', options, '001', '02103', '00001',
        iteration_label=None, method='daophot', basic_or_iterative='basic',
        **kw)


def _expected(basepath, options, source_exposure=None):
    from jwst_gc_pipeline.photometry import crowdsource_catalogs_long as L
    return L._expected_output_exists(
        basepath, 'F480M', 'nrcalong', options, '001', '02103', '00001',
        source_exposure=source_exposure)


def test_skip_if_done_does_not_count_another_observations_catalog_as_done(tmp_path):
    """obs-002's file at obs-005's name is NOT obs-005's frame being done."""
    options = _sentinel_options()
    path = _sentinel_path(str(tmp_path), options)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _catalog(path, O002)
    # The plain existence test -- what the sentinel used to be -- says done.
    assert os.path.exists(path)
    assert _expected(str(tmp_path), options) is True
    # Told whose frame this run is measuring, it says NOT done.
    assert _expected(str(tmp_path), options, source_exposure=O005) is False


def test_skip_if_done_still_skips_this_observations_own_output(tmp_path):
    """The ordinary resume must keep skipping, or every restart refits all."""
    options = _sentinel_options()
    path = _sentinel_path(str(tmp_path), options)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _catalog(path, O005)
    assert _expected(str(tmp_path), options, source_exposure=O005) is True


@pytest.mark.parametrize('source', [None, 'jw02221-o001_t001_nircam_clear-f410m_i2d.fits'])
def test_skip_if_done_is_unchanged_without_a_usable_source(tmp_path, source):
    """No source, or a source that names no exposure: plain existence, as before."""
    options = _sentinel_options()
    path = _sentinel_path(str(tmp_path), options)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _catalog(path, O002)
    assert _expected(str(tmp_path), options, source_exposure=source) is True


def test_skip_if_done_is_unchanged_when_the_existing_file_has_no_stamp(tmp_path):
    """An unstamped or truncated file stays 'done' -- the documented fail-open."""
    options = _sentinel_options()
    path = _sentinel_path(str(tmp_path), options)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _catalog(path, None)
    assert _expected(str(tmp_path), options, source_exposure=O005) is True


def test_both_doors_ask_the_same_question(tmp_path):
    """The skip door and the write door must not disagree about one file.

    They read the same helper.  If they part, a frame is either skipped and
    never measured (skip says done, write would have refused) or refit into a
    refusal on every restart forever (skip says not-done, write allows).
    """
    out = str(tmp_path / 'f480m_nrcalong_visit001_vgroup02103_exp00001.fits')
    for existing, writing in ((O002, O005), (O005, O005), (None, O005)):
        _catalog(out, existing) if existing else _catalog(out, None)
        conflict = foreign_observation_conflict(out, writing)
        refused = False
        try:
            assert_no_foreign_observation_overwrite(out, writing)
        except ForeignObservationOverwriteError:
            refused = True
        assert refused == (conflict is not None)


# --- the LEGACY writer's call site ----------------------------------------
#
# The daophot writer is covered end to end above.  The legacy crowdsource
# writer had the same call added and no test at all: deleting its
# `assert_no_foreign_observation_overwrite` line left the whole selection green
# (reviewer's measurement, and reproduced here before this test existed).

def _crowdsource_results():
    stars = Table({'x': [1.0, 2.0], 'y': [1.0, 2.0], 'dx': [0.1, 0.1],
                   'dy': [0.1, 0.1], 'flux': [10.0, 20.0]})
    return (stars, np.zeros((4, 4)), np.zeros((4, 4)), None)


def _write_frame_legacy(basepath, source_exposure):
    from jwst_gc_pipeline.photometry.legacy import crowdsource_step as S
    im1 = fits.HDUList([fits.PrimaryHDU(),
                        fits.ImageHDU(np.zeros((4, 4), dtype=float))])
    options = types.SimpleNamespace(proposal_id='2092')
    return S.save_crowdsource_results(
        _crowdsource_results(), _wcs(), source_exposure, 'nsky0', im1,
        'nrcalong', basepath, 'F480M', 'nrcalong', '', '', '_exp00001',
        '_visit001', '_vgroup02103', options=options)


def test_legacy_writer_stops_the_cloudef_collision(tmp_path):
    basepath = str(tmp_path)
    os.makedirs(os.path.join(basepath, 'F480M'))
    _write_frame_legacy(basepath, O002)
    out = os.path.join(basepath, 'F480M',
                       'f480m_nrcalong_visit001_vgroup02103_exp00001'
                       '_crowdsource_nsky0.fits')
    assert os.path.exists(out)
    with pytest.raises(ForeignObservationOverwriteError):
        _write_frame_legacy(basepath, O005)
    assert recorded_source_exposure(out) == O002


def test_legacy_writer_still_overwrites_its_own_frame(tmp_path):
    basepath = str(tmp_path)
    os.makedirs(os.path.join(basepath, 'F480M'))
    _write_frame_legacy(basepath, O005)
    _write_frame_legacy(basepath, O005)
    out = os.path.join(basepath, 'F480M',
                       'f480m_nrcalong_visit001_vgroup02103_exp00001'
                       '_crowdsource_nsky0.fits')
    assert recorded_source_exposure(out) == O005


# --- one reader of the provenance stamp, two derivations of the observation -

def test_cataloging_reads_the_stamp_through_this_module(tmp_path):
    """`cataloging._catalog_source_frame` and the guard are one measurement.

    They were two character-for-character copies, one per door -- this one
    reads the stamp to decide whose catalog a file IS, the guard reads it to
    decide whose catalog it is about to replace.  Two copies behave alike
    until one of them is edited, so behaviour cannot pin this and the source
    has to: the body must DELEGATE and must not re-implement the ext-1 read.
    """
    import inspect
    from jwst_gc_pipeline.photometry import cataloging
    src = inspect.getsource(cataloging._catalog_source_frame)
    assert 'return recorded_source_exposure(fn)' in src
    assert 'getheader' not in src.split('"""')[-1]
    out = str(tmp_path / 'stamped.fits')
    _catalog(out, O002)
    assert cataloging._catalog_source_frame(out) == recorded_source_exposure(out)


@pytest.mark.parametrize('base,obs_token', [
    ('jw02092002001_02103_00001_nrcalong_destreak_o002_crf.fits', '_o002_'),
    ('jw02092005001_02103_00001_nrcalong_destreak_o005_crf.fits', '_o005_'),
    ('jw05365002001_02101_00001_mirimage_o002_crf.fits', '_o002_'),
    ('jw05365998001_02101_00001_mirimage_o998_crf.fits', '_o998_'),
    ('jw03958003001_02101_00001_mirimage_o003_crf.fits', '_o003_'),
    ('jw02211046001_02101_00001_nrca1_destreak_o046_crf.fits', '_o046_'),
])
def test_the_two_observation_derivations_agree_on_real_crf_names(base, obs_token):
    """The prefix and the `_oNNN_` suffix name the same observation.

    `cataloging._drop_foreign_obs_duplicates` derives the observation from the
    `_oNNN_` token in the crf basename (`_SRC_OBS_RE`); this guard derives it
    from the `jw<PPPPP><OOO>` prefix.  Both readings are of one source string,
    so on every spelling the pipeline actually writes they must agree, or one
    door drops a file the other keeps.
    """
    from jwst_gc_pipeline.photometry.cataloging import _SRC_OBS_RE
    assert _SRC_OBS_RE.search(base).group(0) == obs_token
    assert exposure_observation(base)[1] == obs_token.strip('_o')


def test_the_two_derivations_part_only_where_the_questions_differ():
    """Two spellings where they answer differently, and why that is right.

    * sgrb2's JOINT `_o002-998_`.  `_SRC_OBS_RE` reads the joint token, which
      `_drop_foreign_obs_duplicates` decomposes and membership-tests, so a
      joint-scoped merge KEEPS the file.  The guard reads the prefix and says
      002 -- the exposure it was measured on really is observation 002's, so a
      998 exposure writing over it really would destroy 002's catalog.  The
      merge asks "is this file part of my joint scope"; the guard asks "whose
      exposure produced it".  Different questions, both answered right.
    * a RESAMPLED crf (`jw05365-o002_t001_miri_...`).  The guard declines it,
      which is the documented merged-level fail-open (#459); `_SRC_OBS_RE`
      still finds `_o002_`.
    """
    from jwst_gc_pipeline.photometry.cataloging import _SRC_OBS_RE
    joint = 'jw05365002001_02101_00001_mirimage_o002-998_crf.fits'
    assert _SRC_OBS_RE.search(joint).group(0) == '_o002-998_'
    assert exposure_observation(joint) == ('5365', '002')

    resampled = 'jw05365-o002_t001_miri_f770w_0_o002_crf.fits'
    assert _SRC_OBS_RE.search(resampled).group(0) == '_o002_'
    assert exposure_observation(resampled) is None
