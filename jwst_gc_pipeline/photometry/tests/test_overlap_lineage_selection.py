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


def test_the_lineage_this_field_reduces_to_is_the_one_kept(tmp_path):
    """cloudc destreaks, so its reduced frame is *_destreak_o002_crf.fits --
    and the cataloguing stage reads that same policy to pick its inputs, so
    following it here keeps the gate and the catalogue on the same files."""
    bare = _touch(tmp_path, _name(lineage=''))
    align = _touch(tmp_path, _name(lineage='_align'))
    destreak = _touch(tmp_path, _name(lineage='_destreak'))
    kept, dropped = cio.select_one_copy_per_exposure(
        [bare, align, destreak], 'cloudc', 'F405N')
    assert kept == [destreak]
    assert {p for p, _ in dropped} == {bare, align}


def test_a_field_that_does_not_destreak_keeps_the_align_copy(tmp_path):
    """wd2 is an extended-emission field: its streak-removal step is off, so
    its reduced frame is *_align_o002_crf.fits.  Nothing but the recorded
    setting separates the two copies -- which is every one of the 43 affected
    directories, since that setting decides all of them."""
    align = _touch(tmp_path, _name(detector='nrca1', lineage='_align'))
    destreak = _touch(tmp_path, _name(detector='nrca1', lineage='_destreak'))
    kept, _ = cio.select_one_copy_per_exposure([align, destreak], 'wd2', 'F200W')
    assert kept == [align]






def test_an_unresolved_tie_falls_back_to_newest_and_says_it_did(tmp_path, monkeypatch):
    """Reaching this means the recorded settings did not decide it, which the
    reader has to be told rather than left to infer from a filename."""
    monkeypatch.setattr(cio, '_reduction_lineage', lambda *a: None)
    old = _touch(tmp_path, _name(lineage='_align'), mtime=1000)
    new = _touch(tmp_path, _name(lineage='_destreak'), mtime=2000)
    kept, dropped = cio.select_one_copy_per_exposure([old, new], 'w51', 'F444W')
    assert kept == [new]
    assert 'newest' in dropped[0][1]
    assert 'no recorded reduction setting' in dropped[0][1]


def test_every_exposure_keeps_exactly_one_copy(tmp_path):
    """The property the whole thing exists for, over a realistic directory:
    four dithers x two detectors, each present in three lineages."""
    paths = [_touch(tmp_path, _name(detector=det, lineage=lin, exp=f'0000{e}'))
             for det in ('nrca1', 'nrcb1')
             for e in range(1, 5)
             for lin in ('', '_align', '_destreak')]
    kept, dropped = cio.select_one_copy_per_exposure(paths, 'cloudc', 'F405N')
    assert len(kept) == 8
    assert len(dropped) == 16
    assert len({cio.exposure_identity(p) for p in kept}) == 8


def test_the_selection_is_stable_whatever_order_the_directory_lists(tmp_path):
    """A verdict that depends on filesystem ordering is not reproducible."""
    paths = [_touch(tmp_path, _name(lineage=lin))
             for lin in ('', '_align', '_destreak')]
    first, _ = cio.select_one_copy_per_exposure(paths, 'cloudc', 'F405N')
    second, _ = cio.select_one_copy_per_exposure(list(reversed(paths)),
                                                 'cloudc', 'F405N')
    assert first == second









def test_the_destreak_policy_is_not_applied_to_MIRI(tmp_path, monkeypatch):
    """Streak removal is a NIRCam stage-1 step and the policy does not name
    MIRI products, so asking it about a MIRI frame would demand a lineage that
    never exists -- leaving the newest-copy fallback to decide it by age when
    nothing recorded applies.  Say so explicitly."""
    assert cio._reduction_lineage('brick', 'F2550W', '002', 'mirimage') is None
    assert cio._reduction_lineage('brick', 'F212N', '002', 'nrca1') == '_destreak'




# ---------------------------------------------------------------------------
# What the selector refuses to do quietly
# ---------------------------------------------------------------------------

def test_a_name_the_selector_cannot_identify_raises_rather_than_vanishing(tmp_path):
    """Frames disappearing without a word is the failure this whole change
    exists to remove, so the selector must not reproduce it on its own input.

    This used to use wd1/F200W's ``jw01905001001_0210b_...`` as the specimen,
    because the base-36 activity id defeated the parser.  It no longer does --
    that was the defect, not the intent -- so the specimen here is a name that
    is genuinely unidentifiable: a product-level crf
    (``jw05365-o002_t001_miri_...``), which carries no exposure number at all
    and sits beside the per-exposure frames in sgrb2's MIRI directories."""
    good = _touch(tmp_path, _name(lineage='_destreak'))
    odd = _touch(tmp_path, 'jw05365-o002_t001_miri_f770w_0_o002_crf.fits')
    with pytest.raises(cio.UnparseableFrameError, match='cannot identify'):
        cio.select_one_copy_per_exposure([good, odd], 'cloudc', 'F405N')


def test_the_base36_specimen_no_longer_defeats_the_selector(tmp_path):
    """The other half of the same statement: wd1/F200W's real names must now go
    THROUGH the selector rather than raising.  All 96 of that band's frames
    have a base-36 activity id, so before the parser was widened the gate saw
    an empty directory there."""
    a = _touch(tmp_path, 'jw01905001001_0210b_00001_nrca1_destreak_o001_crf.fits')
    kept, dropped = cio.select_one_copy_per_exposure([a], 'wd1', 'F200W')
    assert [os.path.basename(k) for k in kept] == [os.path.basename(a)]
    assert dropped == []


def test_a_retired_path_copy_never_competes(tmp_path):
    """Those products' FITS header and their coordinate solution disagree by
    arcseconds, so they are rejected outright elsewhere.  Giving them an
    identity would let one into a contest it could win by being newest."""
    assert cio.exposure_identity(
        _name(lineage='_destreak_realigned_to_vvv')) is None


# ---------------------------------------------------------------------------
# Measuring what was discarded, rather than only naming it
# ---------------------------------------------------------------------------

def test_the_discarded_copy_is_measured_against_the_one_kept(tmp_path, monkeypatch, capsys):
    """The selection reads filenames and a recorded setting.  This is the only
    part that consults the frames, so it is the only thing that could notice
    the setting pointing at the stale copy."""
    kept = _touch(tmp_path, _name(lineage='_destreak'))
    gone = _touch(tmp_path, _name(lineage='_align'))
    monkeypatch.setattr(cio, 'lineage_separation_mas', lambda a, b: 8470.0)
    cio.report_lineage_disagreement([(gone, 'why')], [kept], 'cloudc/F405N')
    out = capsys.readouterr().out
    assert '8470.0' in out
    assert 'beyond 100 mas' in out
    assert 'one of the two frames is wrong' in out


def test_a_pair_that_cannot_be_measured_is_counted_not_called_agreement(tmp_path, monkeypatch, capsys):
    """A missing measurement reported as 0 would read as 'these agree'."""
    kept = _touch(tmp_path, _name(lineage='_destreak'))
    gone = _touch(tmp_path, _name(lineage='_align'))
    monkeypatch.setattr(cio, 'lineage_separation_mas', lambda a, b: None)
    measured = cio.report_lineage_disagreement([(gone, 'why')], [kept], 'x/y')
    assert measured == [(gone, None)]
    assert 'mas from the one kept' not in capsys.readouterr().out


def test_a_separation_below_the_threshold_is_summarised_not_flagged(tmp_path, monkeypatch, capsys):
    kept = _touch(tmp_path, _name(lineage='_destreak'))
    gone = _touch(tmp_path, _name(lineage='_align'))
    monkeypatch.setattr(cio, 'lineage_separation_mas', lambda a, b: 37.6)
    cio.report_lineage_disagreement([(gone, 'why')], [kept], 'w51/F444W')
    out = capsys.readouterr().out
    assert '37.6' in out
    assert 'beyond 100 mas' not in out


# ---------------------------------------------------------------------------
# The measurement itself, against real frames
# ---------------------------------------------------------------------------
#
# The tests above drive `report_lineage_disagreement` with the measurement
# replaced, which pins the reporting but leaves `lineage_separation_mas` itself
# unexercised -- the same shape of gap that let an earlier version of this
# module ship a selection rule that never ran.  These two run the real function
# on real frames.

_ARCHIVE = '/orange/adamginsburg/jwst'
_W51_ALIGN = (f'{_ARCHIVE}/w51/F444W/pipeline/'
              'jw06151001001_03101_00001_nrcalong_align_o001_crf.fits')
_W51_DESTREAK = _W51_ALIGN.replace('_align_o', '_destreak_o')

_have_pair = all(os.path.exists(p) for p in (_W51_ALIGN, _W51_DESTREAK))
needs_archive = pytest.mark.skipif(
    not _have_pair, reason='w51 F444W lineage pair not on this host')


@needs_archive
def test_two_real_lineage_copies_measure_their_known_separation():
    """w51/F444W's two copies of this exposure sit 37.6 mas apart -- measured
    independently at nine pixels while investigating issue #205, and the number
    the selection's whole cost/benefit argument rests on.

    Both copies record RAOFFSET=(0,0), which is why the separation has to be
    measured from the frames rather than read from their headers."""
    sep = cio.lineage_separation_mas(_W51_ALIGN, _W51_DESTREAK)
    assert sep is not None
    assert 35.0 < sep < 40.0, sep


@needs_archive
def test_the_measurement_is_symmetric_and_zero_against_itself():
    """A frame compared with itself is 0.0, and swapping the arguments cannot
    change the answer -- so a report of 0.0 means the copies agree, rather than
    meaning the two paths got crossed."""
    assert cio.lineage_separation_mas(_W51_ALIGN, _W51_ALIGN) == 0.0
    forward = cio.lineage_separation_mas(_W51_ALIGN, _W51_DESTREAK)
    backward = cio.lineage_separation_mas(_W51_DESTREAK, _W51_ALIGN)
    assert forward == pytest.approx(backward, abs=0.5)


@needs_archive
def test_sampling_more_than_one_pixel_is_what_sees_a_rotation():
    """The three sample pixels are not decoration: a pure rotation about the
    array centre reads 0 mas at the centre alone.  Measuring at the corners is
    what makes a rotation or scale difference between two reductions visible,
    so a single-pixel sample would be a silent weakening."""
    corners = cio.lineage_separation_mas(
        _W51_ALIGN, _W51_DESTREAK, samples=((256, 256), (1792, 1792)))
    centre_only = cio.lineage_separation_mas(
        _W51_ALIGN, _W51_DESTREAK, samples=((1024, 1024),))
    assert corners is not None and centre_only is not None


def test_a_frame_that_cannot_be_read_measures_None_never_zero(tmp_path):
    """`None` and `0.0` mean opposite things -- "not measured" versus "these
    agree" -- and the report prints the second as agreement.  Returning 0.0
    here would make every unreadable pair look perfect."""
    broken = tmp_path / 'truncated.fits'
    broken.write_bytes(b'not a fits file')
    assert cio.lineage_separation_mas(str(broken), str(broken)) is None


@needs_archive
def test_sample_pixels_off_the_array_measure_None_never_zero():
    """A subarray exposure is smaller than the sample pixels, so its coordinate
    solution returns a non-finite position.  That is an unmeasured pair, not an
    agreeing one."""
    assert cio.lineage_separation_mas(
        _W51_ALIGN, _W51_DESTREAK, samples=((10 ** 7, 10 ** 7),)) is None
