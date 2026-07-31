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


def _orange_target_list(path):
    """The tuple of targets a driver sends to /orange.

    Found by its membership test rather than by line number, so it survives
    edits elsewhere in the file.
    """
    for node in ast.walk(ast.parse(_source(path))):
        if not isinstance(node, ast.Compare) or len(node.ops) != 1:
            continue
        if not isinstance(node.ops[0], ast.In):
            continue
        comparator = node.comparators[0]
        if not isinstance(comparator, (ast.Tuple, ast.List)):
            continue
        values = [e.value for e in comparator.elts
                  if isinstance(e, ast.Constant) and isinstance(e.value, str)]
        if 'sickle' in values and 'sgrb2' in values:
            return set(values)
    raise AssertionError(f'no /orange target list found in {path}')


def test_the_catalog_and_merge_drivers_agree_on_which_tree_a_target_lives_in():
    catalog, merge = _orange_target_list(CATALOG), _orange_target_list(MERGE)
    assert catalog == merge, (
        'the catalog and merge drivers disagree, so one stage writes to a tree '
        f'the other never reads: only in catalog={sorted(catalog - merge)}, '
        f'only in merge={sorted(merge - catalog)}')


@pytest.mark.parametrize('target', ['wd1', 'wd2'])
def test_wd1_and_wd2_are_on_orange_in_both_drivers(target):
    assert target in _orange_target_list(CATALOG)
    assert target in _orange_target_list(MERGE)


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
