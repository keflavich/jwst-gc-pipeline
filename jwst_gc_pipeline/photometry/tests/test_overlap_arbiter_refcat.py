"""Two ways an overlap pair can be unmeasurable, and only one should block.

Before a field is published, a check confirms that overlapping exposures agree
about where the stars are.  Some pairs overlap on a sliver too thin to compare
directly; for those, both sides are instead compared against a common list of
known star positions, which settles the question.

W51 had no such list configured, because the only registry available was the
one feeding a *different* check -- the blocking absolute-frame one, which needs
a dense catalogue and would refuse good data given a sparse one.  That is what
this module covers.

(An earlier version of this change also exempted MIRI-to-MIRI pairs from
blocking, on the argument that mid-infrared images hold too few sources for two
pointings to share any.  Review measured the opposite -- those pairs share
hundreds of detections, and one of them carries a real 66 mas offset the gate
had already measured -- so that half was withdrawn.  See #385.)
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
