"""What the quarantine may and may not set aside.

The failure it prevents is cheap to describe and expensive to hit: a bare-module
LW per-frame catalog shadows its `long` twin, the m2 checkpoint ingests one
physical exposure twice, and `seed_offsets_table_from_consensus` refuses the
write at the END of a ~2 h m12 finalize -- on a field whose astrometry had
already passed.  cloudef 2092/002 lost three campaign cycles to it and sgrb2
5365/001 one.

The failure it could CAUSE is worse and quieter: setting aside a bare-module
file that is the only copy of a frame, which does not raise anything -- the
frame simply stops being in the catalog.  brick and cloudc both carry
bare-only LW frames, so this is not hypothetical.  Hence the shape of these
tests: one for the thing it must catch, and several for the things it must
refuse to touch.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'scripts', 'reduction'))

import quarantine_alias_perframe_catalogs as q      # noqa: E402


def _write(d, name, mtime):
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_text('x')
    os.utime(p, (mtime, mtime))
    return p


STAGE = '_m2_daophot_basic.fits'
FRAME = 'visit001_vgroup02101_exp00001'


def test_a_stale_shadow_is_set_aside(tmp_path):
    d = tmp_path / 'F360M'
    _write(d, f'f360m_nrcb_{FRAME}{STAGE}', 1_000_000)
    _write(d, f'f360m_nrcblong_{FRAME}{STAGE}', 2_000_000)
    quarantine, skipped = q.plan(str(tmp_path))
    assert len(quarantine) == 1
    assert 'f360m_nrcb_' in os.path.basename(quarantine[0][0])
    assert not skipped


def test_a_bare_module_file_with_no_twin_is_the_only_copy(tmp_path):
    """brick and cloudc carry bare-only LW frames.  Setting one aside removes
    the frame from the catalog and raises nothing."""
    d = tmp_path / 'F360M'
    _write(d, f'f360m_nrcb_{FRAME}{STAGE}', 1_000_000)
    quarantine, skipped = q.plan(str(tmp_path))
    assert not quarantine
    assert 'only copy' in skipped[0][1]


def test_a_newer_bare_copy_is_not_the_stale_shadow_shape(tmp_path):
    """Every collision seen so far had the `long` twin newer.  A bare copy that
    is newer means something else happened, and guessing is how the only copy
    gets removed."""
    d = tmp_path / 'F360M'
    _write(d, f'f360m_nrcb_{FRAME}{STAGE}', 3_000_000)
    _write(d, f'f360m_nrcblong_{FRAME}{STAGE}', 2_000_000)
    quarantine, skipped = q.plan(str(tmp_path))
    assert not quarantine
    assert 'NEWER' in skipped[0][1]


def test_short_wavelength_filters_are_not_touched(tmp_path):
    """For an SW filter the detector token is `nrca1`..`nrcb4` and a bare family
    token can be legitimate -- there is no `long` spelling to be shadowed by."""
    d = tmp_path / 'F212N'
    _write(d, f'f212n_nrcb_{FRAME}{STAGE}', 1_000_000)
    quarantine, skipped = q.plan(str(tmp_path))
    assert not quarantine and not skipped


def test_a_different_STAGE_is_a_different_file_not_a_twin(tmp_path):
    """`_m2_` and `_m4_` of the same frame are two products.  Pairing on the
    frame alone would set aside an m2 catalog because an m4 one exists."""
    d = tmp_path / 'F360M'
    _write(d, f'f360m_nrcb_{FRAME}_m2_daophot_basic.fits', 1_000_000)
    _write(d, f'f360m_nrcblong_{FRAME}_m4_daophot_basic.fits', 2_000_000)
    quarantine, skipped = q.plan(str(tmp_path))
    assert not quarantine
    assert 'only copy' in skipped[0][1]


def test_a_different_EXPOSURE_is_not_a_twin(tmp_path):
    d = tmp_path / 'F360M'
    _write(d, f'f360m_nrcb_visit001_vgroup02101_exp00001{STAGE}', 1_000_000)
    _write(d, f'f360m_nrcblong_visit001_vgroup02101_exp00002{STAGE}', 2_000_000)
    quarantine, _ = q.plan(str(tmp_path))
    assert not quarantine


def test_apply_renames_in_place_and_leaves_the_twin(tmp_path):
    d = tmp_path / 'F360M'
    stale = _write(d, f'f360m_nrcb_{FRAME}{STAGE}', 1_000_000)
    twin = _write(d, f'f360m_nrcblong_{FRAME}{STAGE}', 2_000_000)
    quarantine, _ = q.plan(str(tmp_path))
    receipt = tmp_path / 'r.json'
    moved = q.apply_plan(quarantine, str(receipt))
    assert not stale.exists()
    assert twin.exists()
    assert len(moved) == 1
    assert q.SUFFIX in moved[0]['to']
    assert json.loads(receipt.read_text())['moved'] == moved


def test_restore_puts_it_back(tmp_path):
    d = tmp_path / 'F360M'
    stale = _write(d, f'f360m_nrcb_{FRAME}{STAGE}', 1_000_000)
    _write(d, f'f360m_nrcblong_{FRAME}{STAGE}', 2_000_000)
    quarantine, _ = q.plan(str(tmp_path))
    receipt = tmp_path / 'r.json'
    q.apply_plan(quarantine, str(receipt))
    assert q.restore(str(receipt)) == 1
    assert stale.exists()


def test_restore_refuses_to_overwrite_a_file_written_since(tmp_path):
    """A re-run may have produced a real file at that name.  Restoring over it
    would replace current data with the copy that was set aside as stale."""
    d = tmp_path / 'F360M'
    _write(d, f'f360m_nrcb_{FRAME}{STAGE}', 1_000_000)
    _write(d, f'f360m_nrcblong_{FRAME}{STAGE}', 2_000_000)
    quarantine, _ = q.plan(str(tmp_path))
    receipt = tmp_path / 'r.json'
    q.apply_plan(quarantine, str(receipt))
    _write(d, f'f360m_nrcb_{FRAME}{STAGE}', 4_000_000)     # written since
    with pytest.raises(q.RestoreError):
        q.restore(str(receipt))


def test_the_renamed_file_no_longer_matches_the_per_frame_pattern(tmp_path):
    """The whole mechanism: the pipeline finds these by glob, so the rename has
    to take the file OUT of the glob rather than merely mark it."""
    d = tmp_path / 'F360M'
    _write(d, f'f360m_nrcb_{FRAME}{STAGE}', 1_000_000)
    _write(d, f'f360m_nrcblong_{FRAME}{STAGE}', 2_000_000)
    quarantine, _ = q.plan(str(tmp_path))
    moved = q.apply_plan(quarantine, str(tmp_path / 'r.json'))
    name = os.path.basename(moved[0]['to'])
    assert not name.endswith('.fits')
    again, _ = q.plan(str(tmp_path))
    assert not again, 'a second pass must find nothing left to do'


def test_an_already_quarantined_file_is_not_rediscovered(tmp_path):
    """The rename leaves the frame/stage part of the name intact, so without an
    explicit skip every later run reports its own past work as 1869 files
    'left alone' and the real report is lost in it."""
    d = tmp_path / 'F360M'
    _write(d, f'f360m_nrcb_{FRAME}{STAGE}{q.SUFFIX}20260814T000000Z', 1_000_000)
    quarantine, skipped = q.plan(str(tmp_path))
    assert not quarantine and not skipped
