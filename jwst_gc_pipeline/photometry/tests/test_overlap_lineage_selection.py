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
    """wd2 is an extended-emission field: destreaking is off, so its reduced
    frame is *_align_o002_crf.fits.  Both copies here carry an applied offset,
    so nothing but the recorded policy separates them -- which is 29 of the 43
    affected directories."""
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


# ---------------------------------------------------------------------------
# Reading "was the bulk offset applied?" off the frame
# ---------------------------------------------------------------------------







def test_the_destreak_policy_is_not_applied_to_MIRI(tmp_path, monkeypatch):
    """Destreaking is a NIRCam stage-1 step and the policy does not name MIRI
    products, so asking it about a MIRI frame would demand a lineage that never
    exists and silently fall through to the next rule.  Say so explicitly."""
    assert cio._reduction_lineage('brick', 'F2550W', '002', 'mirimage') is None
    assert cio._reduction_lineage('brick', 'F212N', '002', 'nrca1') == '_destreak'




# ---------------------------------------------------------------------------
# What the selector refuses to do quietly
# ---------------------------------------------------------------------------

def test_a_name_the_selector_cannot_identify_raises_rather_than_vanishing(tmp_path):
    """Frames disappearing without a word is the failure this whole change
    exists to remove, so the selector must not reproduce it on its own input.
    wd1/F200W's names really do defeat the parser today (see the base-36
    activity-id issue), and they must not silently shrink a frame set."""
    good = _touch(tmp_path, _name(lineage='_destreak'))
    odd = _touch(tmp_path, 'jw01905001001_0210b_00001_nrca1_destreak_o001_crf.fits')
    with pytest.raises(cio.UnparseableFrameError, match='cannot identify'):
        cio.select_one_copy_per_exposure([good, odd], 'cloudc', 'F405N')


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
