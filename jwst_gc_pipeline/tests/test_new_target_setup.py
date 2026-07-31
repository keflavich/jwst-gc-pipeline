"""Guards on the four things a brand-new target used to trip over.

Each of these was found by writing GETTING_STARTED.md and trying to follow it:

* the catalog and merge drivers each pick ``/orange`` or ``/blue`` from their
  own hand-maintained target list, and the two lists had drifted apart;
* both drivers wrote into ``psfs/`` and ``catalogs/`` without creating them,
  which only a target directory built from scratch ever noticed;
* ``GC_BASEPATH_OVERRIDE`` replaces the basepath wholesale, so appending the
  NIRISS sub-level before it discarded the sub-level;
* stage 1 downloaded the program's ramp files even under ``-s``, which reuses
  the calibrated frames already on disk and never opens a ramp.

They read source rather than running ``main()``, which needs the survey tree.
"""
import ast
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CATALOG = os.path.join(REPO, 'jwst_gc_pipeline/photometry/crowdsource_catalogs_long.py')
MERGE = os.path.join(REPO, 'jwst_gc_pipeline/photometry/merge_catalogs.py')
REDUCE = os.path.join(REPO, 'jwst_gc_pipeline/reduction/PipelineRerunNIRCAM-LONG.py')


def _source(path):
    with open(path) as fh:
        return fh.read()


def test_neither_driver_keeps_its_own_list_of_which_tree_a_target_lives_in():
    """The two lists disagreed on wd1/wd2 until 2026-07-31: the catalog stage
    wrote to /orange and the merge read from /blue.  One registry now answers
    for both, so a second copy cannot drift from it."""
    for path in (CATALOG, MERGE):
        source = _source(path)
        assert 'field_registry.basepath(' in source, path
        assert "'/orange/adamginsburg/jwst/" not in source, (
            f'{path} still builds a data root itself')


@pytest.mark.parametrize('target', ['wd1', 'wd2'])
def test_wd1_and_wd2_are_on_orange(target):
    from jwst_gc_pipeline import fields
    assert fields.basepath(target).startswith('/orange/')


def test_the_catalog_driver_creates_the_directories_it_writes_into():
    source = _source(CATALOG)
    assert "for _subdir in ('psfs', 'catalogs')" in source
    assert 'os.makedirs(os.path.join(basepath, _subdir), exist_ok=True)' in source


def test_the_merge_creates_its_catalog_directory():
    assert ("os.makedirs(os.path.join(basepath, 'catalogs'), exist_ok=True)"
            in _source(MERGE))


def test_the_niriss_sublevel_is_appended_after_the_override():
    source = _source(CATALOG)
    override = source.index('basepath = apply_basepath_override(basepath)')
    sublevel = source.index("basepath = f'{basepath}niriss/'")
    assert override < sublevel, (
        'the override replaces the basepath wholesale, so appending niriss/ '
        'first throws the sub-level away and a redirected NIRISS run writes '
        'where a redirected NIRCam run of the same target writes')


def test_skipping_stages_1_and_2_also_skips_the_ramp_download():
    """``-s`` reuses the *_cal frames on disk, so the ramps are dead weight
    (~1.2 GB for one filter of one program)."""
    tree = ast.parse(_source(REDUCE))
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        assigns_products_fits = any(
            isinstance(child, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == 'products_fits'
                    for t in child.targets)
            for child in node.body)
        if assigns_products_fits:
            test_src = ast.dump(node.test)
            assert 'skip_step1and2' in test_src, (
                'the ramp download runs unconditionally once MAST is queried')
            return
    raise AssertionError('no ramp-download branch found in the reduce driver')
