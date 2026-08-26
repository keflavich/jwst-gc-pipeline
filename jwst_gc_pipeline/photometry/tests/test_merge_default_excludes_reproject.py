"""``merge_catalogs`` must not read ``merged-reproject`` mosaics by default.

Issue #475.  ``merged-reproject`` is written once by
``reduction/align_to_catalogs.py`` when alignment is measured and is never
refreshed, so it carries a frame that is independent of the one the rest of
the pipeline is using.  Measured against ``gaia_refcat.fits`` with the swept
offset histogram, m92's four ``-merged-reproject`` mosaics read 1852-2212 mas
where their plain siblings read 10-32 mas; wd1's read ~11 mas where its plain
ones read ~40.  Neither direction is safe to take silently.

``scripts/reduction/submit_merge.sbatch`` invokes the driver with no
``--modules``, so the default is what production actually uses -- which is why
this is pinned as a test rather than left to the comment beside the constant.
"""
import ast
import pathlib

from jwst_gc_pipeline.photometry import merge_catalogs as MC


MERGE_SRC = pathlib.Path(MC.__file__)


def test_default_merge_modules_constant_has_no_reproject():
    assert 'reproject' not in MC.DEFAULT_MERGE_MODULES
    assert 'merged' in MC.DEFAULT_MERGE_MODULES.split(',')


def _modules_option_default():
    """The literal/name used as the ``--modules`` default in ``main()``.

    Read from the source rather than by running ``main()``: the driver does
    real work (globs, FITS reads) before anything is parseable.
    """
    tree = ast.parse(MERGE_SRC.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == 'add_option'):
            continue
        flags = [a.value for a in node.args
                 if isinstance(a, ast.Constant) and isinstance(a.value, str)]
        if '--modules' not in flags:
            continue
        for kw in node.keywords:
            if kw.arg == 'default':
                return kw.value
    raise AssertionError("no --modules add_option found in merge_catalogs.main")


def test_modules_option_default_is_the_constant():
    default = _modules_option_default()
    # A bare string literal here is how the stale product got into the default
    # in the first place; the constant is where the reasoning lives.
    assert isinstance(default, ast.Name), (
        "--modules default should be DEFAULT_MERGE_MODULES, not an inline "
        f"literal (got {ast.dump(default)})")
    assert default.id == 'DEFAULT_MERGE_MODULES'


def test_no_module_level_default_mentions_reproject():
    # Belt and braces: catches a re-introduction via any other add_option
    # default in the driver as well.
    tree = ast.parse(MERGE_SRC.read_text())
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == 'add_option'):
            continue
        for kw in node.keywords:
            if kw.arg != 'default':
                continue
            if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                if 'reproject' in kw.value.value:
                    offenders.append(kw.value.value)
    assert offenders == [], (
        f"add_option defaults naming a reproject product: {offenders}")
