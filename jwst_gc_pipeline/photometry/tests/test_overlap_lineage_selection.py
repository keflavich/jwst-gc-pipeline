"""The registration gate must read one copy of each exposure, not several.

A field-and-filter working directory normally holds the same physical exposure
several times over: the reduction has been re-run with different settings across
three years and each run wrote its own copy, under a name differing only by a
lineage token (``_destreak``, ``_align``, or none).  Those copies carry
different sky coordinates -- in cloudc/F405N up to 8.5 arcsec apart -- so
pooling them makes the gate compare an exposure against stale copies of itself.

Measured 2026-08-10: 43 of 127 working directories hold more than one copy of
some exposure.
"""
import importlib.util
import os
from pathlib import Path

import pytest

_SCRIPT = (Path(__file__).resolve().parents[3]
           / 'scripts' / 'release' / 'check_interframe_overlap.py')

if not _SCRIPT.exists():                                    # pragma: no cover
    pytest.skip('check_interframe_overlap.py not present', allow_module_level=True)

_spec = importlib.util.spec_from_file_location('_cio', _SCRIPT)
cio = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cio)


def _name(detector='nrcblong', lineage='', obs='002', exp='00001'):
    return (f'jw02221{obs}001_08201_{exp}_{detector}{lineage}'
            f'_o{obs}_crf.fits')


# ---------------------------------------------------------------------------
# Which exposure a name refers to
# ---------------------------------------------------------------------------

def test_the_lineage_copies_of_one_exposure_share_an_identity():
    """This is what makes them duplicates rather than different exposures."""
    identities = {cio.exposure_identity(_name(lineage=lin))
                  for lin in ('', '_align', '_destreak')}
    assert len(identities) == 1


def test_two_exposures_of_one_detector_are_not_one_identity():
    """Exposure number is part of the identity; without it, a whole dither
    sequence would collapse to a single frame."""
    assert (cio.exposure_identity(_name(exp='00001'))
            != cio.exposure_identity(_name(exp='00002')))


def test_the_same_exposure_number_in_two_observations_is_not_one_identity():
    assert (cio.exposure_identity(_name(obs='001'))
            != cio.exposure_identity(_name(obs='002')))


def test_a_name_that_is_not_a_crf_has_no_identity():
    assert cio.exposure_identity('jw02221002001_08201_00001_nrcblong_cal.fits') is None


# ---------------------------------------------------------------------------
# Choosing between copies
# ---------------------------------------------------------------------------

def _touch(directory, name, mtime=None):
    path = directory / name
    path.write_bytes(b'')
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return str(path)


def test_a_single_copy_is_never_dropped(tmp_path):
    """This resolves duplication.  Whether a lone frame is fit to release is a
    different question, and dropping one here would turn a whole MIRI directory
    into an empty frame set."""
    only = _touch(tmp_path, _name(detector='mirimage'))
    kept, dropped = cio.select_one_copy_per_exposure([only], 'brick', 'F2550W')
    assert kept == [only]
    assert dropped == []


def test_the_lineage_this_field_reduces_to_is_the_one_kept(tmp_path, monkeypatch):
    """cloudc destreaks, so its reduced frame is *_destreak_o002_crf.fits --
    and the cataloguing stage reads that same policy to pick its inputs, so
    following it here keeps the gate and the catalogue on the same files."""
    monkeypatch.setattr(cio, 'has_baked_alignment', lambda p: True)
    bare = _touch(tmp_path, _name(lineage=''))
    align = _touch(tmp_path, _name(lineage='_align'))
    destreak = _touch(tmp_path, _name(lineage='_destreak'))
    kept, dropped = cio.select_one_copy_per_exposure(
        [bare, align, destreak], 'cloudc', 'F405N')
    assert kept == [destreak]
    assert {p for p, _ in dropped} == {bare, align}


def test_a_field_that_does_not_destreak_keeps_the_align_copy(tmp_path, monkeypatch):
    """wd2 is an extended-emission field: destreaking is off, so its reduced
    frame is *_align_o002_crf.fits.  Both copies here carry an applied offset,
    so nothing but the recorded policy separates them -- which is 29 of the 43
    affected directories."""
    monkeypatch.setattr(cio, 'has_baked_alignment', lambda p: True)
    align = _touch(tmp_path, _name(detector='nrca1', lineage='_align'))
    destreak = _touch(tmp_path, _name(detector='nrca1', lineage='_destreak'))
    kept, _ = cio.select_one_copy_per_exposure([align, destreak], 'wd2', 'F200W')
    assert kept == [align]


def test_an_unaligned_copy_loses_to_an_aligned_one(tmp_path, monkeypatch):
    """When the policy's lineage is not on disk, the frame that had the bulk
    offset applied is the one whose sky coordinates mean anything.  This is the
    other 14 directories, where the stale copy is the bare 2023 one that
    alignment never reached."""
    monkeypatch.setattr(cio, '_reduction_lineage', lambda *a: '_destreak')
    bare = _touch(tmp_path, _name(lineage=''))
    align = _touch(tmp_path, _name(lineage='_align'))
    monkeypatch.setattr(cio, 'has_baked_alignment', lambda p: p == align)
    kept, dropped = cio.select_one_copy_per_exposure([bare, align], 'wd2', 'F150W')
    assert kept == [align]
    assert dropped[0][1] == 'the only copy carrying an applied RAOFFSET'


def test_when_nothing_is_aligned_a_copy_is_still_chosen_and_the_reason_says_so(tmp_path, monkeypatch):
    """The gate reports on the frames that exist; refusing to produce a verdict
    because none is aligned would replace a measurable answer with silence.
    The reason string is what carries that to the log."""
    monkeypatch.setattr(cio, '_reduction_lineage', lambda *a: '_destreak')
    monkeypatch.setattr(cio, 'has_baked_alignment', lambda p: False)
    old = _touch(tmp_path, _name(lineage=''), mtime=1000)
    new = _touch(tmp_path, _name(lineage='_align'), mtime=2000)
    kept, dropped = cio.select_one_copy_per_exposure([old, new], 'wd2', 'F150W')
    assert kept == [new]
    assert 'no copy carries an applied RAOFFSET' in dropped[0][1]


def test_an_unresolved_tie_falls_back_to_newest_and_says_it_did(tmp_path, monkeypatch):
    """Reaching this means the recorded settings did not decide it, which the
    reader has to be told rather than left to infer from a filename."""
    monkeypatch.setattr(cio, '_reduction_lineage', lambda *a: None)
    monkeypatch.setattr(cio, 'has_baked_alignment', lambda p: True)
    old = _touch(tmp_path, _name(lineage='_align'), mtime=1000)
    new = _touch(tmp_path, _name(lineage='_destreak'), mtime=2000)
    kept, dropped = cio.select_one_copy_per_exposure([old, new], 'w51', 'F444W')
    assert kept == [new]
    assert 'newest' in dropped[0][1]


def test_every_exposure_keeps_exactly_one_copy(tmp_path, monkeypatch):
    """The property the whole thing exists for, over a realistic directory:
    four dithers x two detectors, each present in three lineages."""
    monkeypatch.setattr(cio, 'has_baked_alignment', lambda p: True)
    paths = [_touch(tmp_path, _name(detector=det, lineage=lin, exp=f'0000{e}'))
             for det in ('nrca1', 'nrcb1')
             for e in range(1, 5)
             for lin in ('', '_align', '_destreak')]
    kept, dropped = cio.select_one_copy_per_exposure(paths, 'cloudc', 'F405N')
    assert len(kept) == 8
    assert len(dropped) == 16
    assert len({cio.exposure_identity(p) for p in kept}) == 8


def test_the_selection_is_stable_whatever_order_the_directory_lists(tmp_path, monkeypatch):
    """A verdict that depends on filesystem ordering is not reproducible."""
    monkeypatch.setattr(cio, 'has_baked_alignment', lambda p: True)
    paths = [_touch(tmp_path, _name(lineage=lin))
             for lin in ('', '_align', '_destreak')]
    first, _ = cio.select_one_copy_per_exposure(paths, 'cloudc', 'F405N')
    second, _ = cio.select_one_copy_per_exposure(list(reversed(paths)),
                                                 'cloudc', 'F405N')
    assert first == second


# ---------------------------------------------------------------------------
# Reading "was the bulk offset applied?" off the frame
# ---------------------------------------------------------------------------

def test_an_offset_of_zero_still_counts_as_applied(tmp_path):
    """RAOFFSET=0.0 means the field's correction for that exposure was zero, not
    that alignment never ran -- wd2 and w51 frames read exactly that."""
    from astropy.io import fits
    path = tmp_path / 'zero.fits'
    sci = fits.ImageHDU(data=None, name='SCI')
    sci.header['RAOFFSET'] = 0.0
    sci.header['DEOFFSET'] = 0.0
    fits.HDUList([fits.PrimaryHDU(), sci]).writeto(path)
    assert cio.has_baked_alignment(str(path))


def test_a_frame_alignment_never_reached_reads_as_unaligned(tmp_path):
    from astropy.io import fits
    path = tmp_path / 'raw.fits'
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(data=None, name='SCI')]).writeto(path)
    assert not cio.has_baked_alignment(str(path))


def test_an_unreadable_file_is_not_claimed_to_be_aligned(tmp_path):
    """Fail towards 'not aligned' so a truncated file loses to a readable one
    rather than winning on a header that could not be read."""
    path = tmp_path / 'truncated.fits'
    path.write_bytes(b'not a fits file')
    assert not cio.has_baked_alignment(str(path))
