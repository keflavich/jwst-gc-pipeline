"""The PSF-data path must come from the environment when the user set one.

`saturated_star_finding` raises unless STPSF_PATH is set, while importing
`reduction.filtering` used to overwrite it with a HiPerGator path -- so setting
it correctly off HiPerGator got you a directory that does not exist, and the
failure surfaced far from the cause.

Only the branch matching the installed package can be imported here, so the
other one is checked statically.
"""
import ast
import importlib
import importlib.util
import os
import pathlib

FILTERING = pathlib.Path(__file__).resolve().parent.parent / 'filtering.py'
INSTALLED = ('WEBBPSF_PATH' if importlib.util.find_spec('webbpsf')
             else 'STPSF_PATH')


def test_an_existing_setting_is_not_overwritten(monkeypatch, tmp_path):
    monkeypatch.setenv(INSTALLED, str(tmp_path))
    import jwst_gc_pipeline.reduction.filtering as F
    importlib.reload(F)
    assert os.environ[INSTALLED] == str(tmp_path)


def test_a_default_is_still_supplied_when_unset(monkeypatch):
    # Not a test of the fix -- plain assignment would pass this too.  It guards
    # against dropping the default, which HiPerGator runs rely on.
    monkeypatch.delenv('STPSF_PATH', raising=False)
    monkeypatch.delenv('WEBBPSF_PATH', raising=False)
    import jwst_gc_pipeline.reduction.filtering as F
    importlib.reload(F)
    assert os.environ.get(INSTALLED)


def test_neither_branch_assigns_the_psf_path():
    """Whichever branch is not installed cannot be imported, so read it.

    Without this, the uninstalled branch could go back to overwriting the
    user's setting and no test would notice -- which is exactly what the
    original review found: only one of the two parametrised cases ever ran.
    """
    tree = ast.parse(FILTERING.read_text())
    offenders = [ast.unparse(node) for node in ast.walk(tree)
                 if isinstance(node, ast.Assign)
                 and 'PSF_PATH' in ast.unparse(node)
                 and any(isinstance(t, ast.Subscript)
                         and isinstance(t.value, ast.Attribute)
                         and t.value.attr == 'environ'
                         for t in node.targets)]
    assert not offenders, f'assigns instead of setdefault: {offenders}'


def test_the_stpsf_guard_calls_a_function_that_exists():
    """It called ``os.get``, which is not a thing.  Reaching that line would
    have raised AttributeError instead of the message it means to give.

    Checked by reading the source: the module-level guard higher up the file
    raises first, so the branch cannot be exercised at run time.
    """
    import ast
    import os as _os
    path = _os.path.join(
        _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
        'saturated_star_finding.py')
    tree = ast.parse(open(path).read())
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Attribute)
             and isinstance(node.value, ast.Name) and node.value.id == 'os'
             and node.attr == 'get']
    assert not calls, 'os.get() does not exist; use os.environ.get()'
