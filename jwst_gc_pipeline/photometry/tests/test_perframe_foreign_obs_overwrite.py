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
