"""A run must never publish an OLDER run's product crf as its per-exposure crf.

``outlier_detection`` names its CR-flagged crf after the asn PRODUCT while
cataloging globs per-exposure names, so the reduction copies one onto the other.
The eligibility test used to be ``not skip_outlier_detection`` -- "did this run
TRY to write product crf" -- which is a proxy for what actually matters, "are
these files this run's".

sickle (#270) is the case: 264 product crf from 2026-06-27, its last run with
outlier_detection enabled, were copied over the per-exposure names on every
iteration of the VIRAC2 re-tie. Byte-identical to the quarantined originals, and
246 mas from the member frames the run had just produced, against a 30 mas
tolerance.

These test BEHAVIOUR -- what ends up in the per-exposure crf -- rather than the
shape of the branch. The previous version of this file inspected the source with
``ast``, and the review demonstrated that it rejected a correct restructuring
while accepting two broken variants (guard kept with an empty body; guard kept
while the fallback ALSO copied the stale product crf).
"""
import importlib.util
import os
import pathlib

import numpy as np
import pytest
from astropy.io import fits

# The driver's filename carries a hyphen, so it cannot be imported by name.
_SRC = (pathlib.Path(__file__).resolve().parents[1]
        / "PipelineRerunNIRCAM-LONG.py")
_SPEC = importlib.util.spec_from_file_location("pipeline_rerun_nircam_long", _SRC)
pr = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pr)


EXPSTART = 60545.5031
DETECTOR = 'NRCB1'


def _frame(path, value, expstart=EXPSTART, detector=DETECTOR, mtime=None):
    """A one-pixel FITS frame whose SCI value identifies which generation it is."""
    hdu = fits.PrimaryHDU()
    hdu.header['EXPSTART'] = expstart
    hdu.header['DETECTOR'] = detector
    sci = fits.ImageHDU(np.full((2, 2), float(value), dtype='float32'), name='SCI')
    fits.HDUList([hdu, sci]).writeto(path, overwrite=True)
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return str(path)


def _value(path):
    with fits.open(path) as h:
        return float(h['SCI'].data[0, 0])


PROD = 'jw03958-o007_t001_nircam_clear-f187n-nrcb'
MEMBER = 'jw03958007001_03102_00001_nrcb1_destreak.fits'
TARGET = 'jw03958007001_03102_00001_nrcb1_destreak_o007_crf.fits'


def test_a_stale_product_crf_is_not_published(tmp_path):
    """The sickle failure. A product crf older than this Image3 call is a previous
    generation's and must not become this run's photometry input."""
    member = _frame(tmp_path / MEMBER, value=2.0, mtime=2_000_000)
    _frame(tmp_path / f'{PROD}_0_o007_crf.fits', value=1.0, mtime=1_000_000)

    n = pr.write_per_exposure_crf(
        output_dir=str(tmp_path), prod_name=PROD, members=[member],
        field='007', image3_started=1_900_000)

    assert n == 1
    assert _value(tmp_path / TARGET) == 2.0, "the stale product crf was published"


def test_a_fresh_product_crf_is_published(tmp_path):
    """The normal outlier_detection path still works: a product crf this call
    wrote IS the right source, because it carries the CR flags."""
    member = _frame(tmp_path / MEMBER, value=2.0, mtime=2_000_000)
    _frame(tmp_path / f'{PROD}_0_o007_crf.fits', value=3.0, mtime=2_000_100)

    n = pr.write_per_exposure_crf(
        output_dir=str(tmp_path), prod_name=PROD, members=[member],
        field='007', image3_started=2_000_000)

    assert n == 1
    assert _value(tmp_path / TARGET) == 3.0


def test_freshness_covers_the_outlier_detection_enabled_path(tmp_path):
    """The reason freshness beats the flag: with outlier detection ENABLED, a
    leftover from a run with different asn membership still maps onto a current
    member by (EXPSTART, DETECTOR).  The old `not skip_outlier_detection` test
    would have copied it; the mtime test declines it."""
    member = _frame(tmp_path / MEMBER, value=2.0, mtime=2_000_000)
    _frame(tmp_path / f'{PROD}_7_o007_crf.fits', value=9.0, mtime=1_000_000)

    pr.write_per_exposure_crf(
        output_dir=str(tmp_path), prod_name=PROD, members=[member],
        field='007', image3_started=1_900_000)

    assert _value(tmp_path / TARGET) == 2.0


def test_the_slack_tolerates_clock_skew(tmp_path):
    """A product crf written a few seconds BEFORE the recorded start (NFS mtime
    granularity, node/fileserver skew) is still this run's."""
    member = _frame(tmp_path / MEMBER, value=2.0, mtime=2_000_000)
    _frame(tmp_path / f'{PROD}_0_o007_crf.fits', value=3.0,
           mtime=2_000_000 - pr.CRF_FRESH_SLACK_S / 2)

    pr.write_per_exposure_crf(
        output_dir=str(tmp_path), prod_name=PROD, members=[member],
        field='007', image3_started=2_000_000)

    assert _value(tmp_path / TARGET) == 3.0


def test_the_stale_case_is_reported_as_a_WARNING(tmp_path, capsys):
    """Triage greps for WARNING.  A line saying a previous generation's data was
    found in the output directory must not be the one that is missed."""
    member = _frame(tmp_path / MEMBER, value=2.0, mtime=2_000_000)
    _frame(tmp_path / f'{PROD}_0_o007_crf.fits', value=1.0, mtime=1_000_000)

    pr.write_per_exposure_crf(
        output_dir=str(tmp_path), prod_name=PROD, members=[member],
        field='007', image3_started=1_900_000)

    out = capsys.readouterr().out
    assert 'WARNING' in out
    assert 'predate this Image3 call' in out
    assert '#270' in out


def test_every_written_crf_gets_a_provenance_stamp(tmp_path, monkeypatch):
    """crf carried no provenance at all, which is why a 246 mas
    image-vs-photometry split sat undetected until an unrelated loop failed."""
    stamped = []
    monkeypatch.setattr(pr, '_stamp_imaging_product', stamped.append)
    member = _frame(tmp_path / MEMBER, value=2.0, mtime=2_000_000)

    pr.write_per_exposure_crf(
        output_dir=str(tmp_path), prod_name=PROD, members=[member],
        field='007', image3_started=1_900_000)

    assert stamped == [str(tmp_path / TARGET)]


def test_a_missing_member_is_reported_not_silent(tmp_path, capsys):
    pr.write_per_exposure_crf(
        output_dir=str(tmp_path), prod_name=PROD,
        members=[str(tmp_path / 'absent_destreak.fits')],
        field='007', image3_started=1_900_000)
    assert 'WARNING' in capsys.readouterr().out


def test_a_fresh_product_crf_with_no_member_match_is_not_guessed(tmp_path, capsys):
    """Different EXPSTART: there is no member this belongs to, so nothing is
    written rather than something being written to the wrong name."""
    member = _frame(tmp_path / MEMBER, value=2.0, mtime=2_000_000)
    _frame(tmp_path / f'{PROD}_0_o007_crf.fits', value=3.0,
           expstart=EXPSTART + 1.0, mtime=2_000_100)

    n = pr.write_per_exposure_crf(
        output_dir=str(tmp_path), prod_name=PROD, members=[member],
        field='007', image3_started=2_000_000)

    assert n == 0
    assert not os.path.exists(tmp_path / TARGET)
    assert 'no per-exposure cal match' in capsys.readouterr().out
