"""Grep-guard: no tracked file names the retired /blue JWST data tree.

Issue #937: all field data lives under ``/orange/adamginsburg/jwst/<field>``.
The old ``/blue`` tree is being moved and must not be read or written again.
This test fails if a tracked source or doc file contains the literal /blue
data path.  A field's directory comes from ``jwst_gc_pipeline.fields``
(``basepath(field)`` or ``jwst_root()``), which reads ``fields.yaml``.

Not flagged, because they are not under the data tree: the Python interpreter
(``/blue/adamginsburg/adamginsburg/miniconda3``), the repo checkout
(``.../repos``) and scratch space (``.../tmp``).

Like the other grep-guards, this walks ``git ls-files``: a NEW file must be
``git add``ed before the guard can see it.
"""
import pathlib
import re
import subprocess

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

#: Built from pieces so this file does not contain the literal it forbids.
_BLUE = '/blue/adamginsburg/adamginsburg'
_BLUE_DATA = re.compile(re.escape(_BLUE + '/jwst') + r'(?![\w-])')

#: Reviewed exceptions, as ``(file, needle)``: a line matches an entry when the
#: needle appears in it.  Every entry names the retired tree on purpose, to
#: REFUSE it or to translate a path recorded before the move.
ALLOWED = (
    ("jwst_gc_pipeline/astrometry/catalog_roll_correction.py",
     "BLUE_JWST = "),
    ("jwst_gc_pipeline/astrometry_utils.py",
     "replace('" + _BLUE + "/jwst/'"),
    ("jwst_gc_pipeline/cmz/treasury_joint_catalog.py",
     "FORBIDDEN_OUTPUT_PREFIX = "),
    ("jwst_gc_pipeline/cmz/treasury_joint_catalog.py",
     "refuses to write under"),
    ("jwst_gc_pipeline/cmz/treasury_joint_catalog.py",
     "``" + _BLUE + "/jwst/``"),
)

_SKIP_SUFFIXES = {'.fits', '.png', '.jpg', '.jpeg', '.pdf', '.gz', '.npy',
                  '.npz', '.pkl', '.ecsv', '.asdf', '.zip', '.tar', '.ico'}


def _tracked_files():
    out = subprocess.run(['git', '-C', str(REPO_ROOT), 'ls-files', '-z'],
                         capture_output=True, check=True).stdout
    for name in out.decode().split('\0'):
        if name and pathlib.Path(name).suffix.lower() not in _SKIP_SUFFIXES:
            yield name


def _hits():
    for name in _tracked_files():
        path = REPO_ROOT / name
        if not path.is_file():
            continue
        text = path.read_text(errors='replace')
        if _BLUE not in text:
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            if _BLUE_DATA.search(line):
                yield name, number, line


def test_no_tracked_file_names_the_blue_data_tree():
    allowed = {}
    for name, needle in ALLOWED:
        allowed.setdefault(name, []).append(needle)
    offenders = [f"{name}:{number}: {line.strip()}"
                 for name, number, line in _hits()
                 if not any(n in line for n in allowed.get(name, ()))]
    assert not offenders, (
        "the retired /blue JWST data tree is named in tracked files (#937):\n  "
        + "\n  ".join(offenders)
        + "\n\nField data lives under /orange/adamginsburg/jwst.  Take the "
          "path from the registry: `jwst_gc_pipeline.fields.basepath(field)` "
          "or `fields.jwst_root()`.")


def test_allowed_entries_still_match():
    """Keep the exception list from rotting: each entry must still be present."""
    stale = [f"{name}: {needle}" for name, needle in ALLOWED
             if needle not in (REPO_ROOT / name).read_text(errors='replace')]
    assert not stale, ("ALLOWED names lines that are gone (remove them):\n  "
                       + "\n  ".join(stale))


@pytest.mark.parametrize('line, expected', [
    (_BLUE + '/jwst/brick/F200W', True),
    (_BLUE + '/jwst', True),
    (_BLUE + '/miniconda3/envs/python313/bin/python', False),
    (_BLUE + '/repos/jwst-gc-pipeline', False),
    (_BLUE + '/tmp/claude-1/jwst-scratch', False),
    ('/orange/adamginsburg/jwst/brick', False),
])
def test_pattern_separates_data_from_interpreter_and_scratch(line, expected):
    assert bool(_BLUE_DATA.search(line)) is expected
