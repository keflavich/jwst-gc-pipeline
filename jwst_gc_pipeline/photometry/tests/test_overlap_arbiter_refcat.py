"""Which star list tie-breaks an overlap too thin to measure directly.

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


def test_the_absolute_frame_check_reads_only_its_own_registry():
    """`stage_release.check_catalog_on_frame` asks whether a shipped catalogue
    sits on the right sky, and needs a dense list -- a sparse one gives a noisy
    bulk tie and would refuse good data.  So it must read FRAME_REFCAT and NOT
    the tie-break registry.

    Asserted on the MECHANISM rather than on registry contents: an earlier
    version of this test checked `'w51' not in FRAME_REFCAT`, which is a fact
    about a dict literal.  Re-pointing check_catalog_on_frame at
    `overlap_arbiter_refcat` -- wiring the sparse list straight into the
    blocking check -- left that test green.
    """
    import inspect
    src = inspect.getsource(stage.check_catalog_on_frame)
    assert 'FRAME_REFCAT' in src
    assert 'overlap_arbiter_refcat' not in src, (
        'the absolute-frame check must not resolve its catalogue through the '
        'tie-break registry, which may hold a sparse list')


def test_a_field_with_no_list_is_told_so_rather_than_left_to_wonder():
    """Without this line, a pair stays unmeasurable and the log gives no reason
    -- indistinguishable from the arbiter having run and found nothing."""
    import inspect
    src = inspect.getsource(stage.main) if hasattr(stage, 'main') else ''
    if not src:
        import pathlib as _p
        src = (_p.Path(stage.__file__).read_text()
               if getattr(stage, '__file__', None) else '')
    assert 'no overlap arbiter star list' in src


def test_a_field_with_only_a_dense_list_still_uses_it_to_arbitrate():
    """A denser catalogue does the tie-break job too, so a field does not need
    an entry in both registries."""
    assert 'brick' not in stage.OVERLAP_ARBITER_REFCAT
    brick = stage.FRAME_REFCAT.get('brick')
    if brick and os.path.exists(brick):
        assert stage.overlap_arbiter_refcat('brick') == brick


def test_a_field_with_no_list_at_all_says_so_rather_than_pretending():
    assert stage.overlap_arbiter_refcat('not-a-field') is None


def test_a_catalogue_without_a_source_column_is_not_called_VIRAC2():
    """The gating slot used to be labelled `VIRAC2` whatever was read.

    VIRAC2 is the VVV-based near-infrared catalogue, and W51 lies OUTSIDE the
    VVV footprint -- it does not exist there.  An operator debugging a W51
    block was told a catalogue had been used that cannot exist for the field.
    The split is by presence of a `source` column, not by what the file is, so
    the label has to report what was actually read.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        '_cio_label', _REPO / 'scripts' / 'release' / 'check_interframe_overlap.py')
    cio = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cio)

    w51 = stage.OVERLAP_ARBITER_REFCAT.get('w51')
    if not (w51 and os.path.exists(w51)):
        pytest.skip('w51 star list not on this host')
    _rc, _gaia, label = cio._refcat(w51)
    assert 'VIRAC2' not in label.split('NOT')[0]
    assert 'no `source` column' in label
