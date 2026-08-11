"""Two ways an overlap pair can be unmeasurable, and only one should block.

Before a field is published, a check confirms that overlapping exposures agree
about where the stars are.  Some pairs overlap on a sliver too thin to compare
directly; for those, both sides are instead compared against a common list of
known star positions, which settles the question.

Two things were stopping w51 from ever reaching a verdict:

1. no such star list was configured for it, because the only registry available
   was the one feeding a *different*, denser-catalogue-requiring check; and
2. four of its five unmeasurable pairs are MIRI-to-MIRI, and a mid-infrared
   image of these fields holds so few point sources that two pointings share
   none at all -- a question no star list can answer.
"""
import importlib.util
import os
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]


def _load(relpath, name):
    path = _REPO / relpath
    if not path.exists():                                   # pragma: no cover
        pytest.skip(f'{relpath} not present', allow_module_level=True)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cio = _load('scripts/release/check_interframe_overlap.py', '_cio')


# ---------------------------------------------------------------------------
# A pair with no stars to share
# ---------------------------------------------------------------------------

def test_two_MIRI_pointings_are_not_blocked_for_sharing_no_stars():
    """The stars are missing from the DATA, not from the star list, so no
    external list could settle it.  Blocking on it parks the field forever on a
    question nothing can answer."""
    assert cio._pair_cannot_share_stars(
        dict(a='002001:mirimage', b='002002:mirimage'))


def test_a_near_infrared_pair_is_still_blocked():
    """NIRCam images of these fields are crowded, so two of them failing to
    share stars means something is wrong -- exactly what the check is for."""
    assert not cio._pair_cannot_share_stars(
        dict(a='001001:nrca', b='001001:nrcb'))


def test_a_mixed_pair_is_still_blocked():
    """One MIRI side does not excuse the pair: the near-infrared side has the
    sources, so the comparison remains possible in principle."""
    assert not cio._pair_cannot_share_stars(
        dict(a='002001:mirimage', b='001001:nrca'))


def test_a_malformed_label_is_not_quietly_excused():
    """Anything unrecognised must fall through to blocking rather than be
    waved past -- an unparseable label is not evidence of anything."""
    assert not cio._pair_cannot_share_stars(dict(a='', b=''))
    assert not cio._pair_cannot_share_stars(dict())


# ---------------------------------------------------------------------------
# Which star list arbitrates an unmeasurable pair
# ---------------------------------------------------------------------------

stage = _load('scripts/release/stage_release.py', '_stage')


def test_a_sparse_star_list_is_allowed_to_arbitrate():
    """Sparse is better than none.  With no list the pair is unmeasurable and
    the field cannot stage at all; with one it usually resolves."""
    path = stage.OVERLAP_ARBITER_REFCAT.get('w51')
    assert path, 'w51 has no arbiter star list'
    if os.path.exists(path):
        assert stage.overlap_arbiter_refcat('w51') == path


def test_the_sparse_list_is_kept_OUT_of_the_absolute_frame_check():
    """That check asks whether a catalogue sits on the right sky and needs a
    dense list; a sparse one gives a noisy answer and would refuse good data.
    The two registries exist because the requirements are opposite."""
    assert 'w51' not in stage.FRAME_REFCAT


def test_a_field_with_only_a_dense_list_still_uses_it_to_arbitrate():
    """A denser catalogue does the tie-break job too, so a field does not need
    an entry in both registries."""
    assert 'brick' not in stage.OVERLAP_ARBITER_REFCAT
    brick = stage.FRAME_REFCAT.get('brick')
    if brick and os.path.exists(brick):
        assert stage.overlap_arbiter_refcat('brick') == brick


def test_a_field_with_no_list_at_all_says_so_rather_than_pretending():
    assert stage.overlap_arbiter_refcat('not-a-field') is None
