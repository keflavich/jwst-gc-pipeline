"""Grep-guard: the pipeline drivers take a field's directory from the registry.

``fields.yaml`` opens by saying it is the single place a target is registered.
That is true only if the drivers ask it where the data are.  Each reduce driver
used to build the path itself::

    basepath = f'/orange/adamginsburg/jwst/{regionname}/'

which pins every field to one tree, so ``roots:`` and a field's ``root:`` reached
the catalog and merge stages but not the reduction.  On HiPerGator the two agreed
by accident: ``brick`` and ``cloudc`` are ``root: blue``, and
``/orange/adamginsburg/jwst/brick`` is a symlink into the blue tree.  Anywhere
else -- a second cluster, a laptop -- the reduction read a directory that does
not exist, which is why onboarding a new target needed
``GC_BASEPATH_OVERRIDE`` to point the drivers back at their own registered path.

Ask the registry instead::

    basepath = field_registry.basepath(regionname)

``GC_BASEPATH_OVERRIDE`` still applies on top, for scratch reductions.

This test FAILS if a driver contains a hard-coded survey root.  It polices the
drivers only: a default argument elsewhere in the package is overridden by its
caller, while a driver's ``basepath`` decides where a whole run reads and writes.
"""
import os
import re
from pathlib import Path

import pytest

from jwst_gc_pipeline import fields as field_registry

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The files that decide where a run reads and writes.
DRIVERS = (
    "jwst_gc_pipeline/reduction/PipelineRerunNIRCAM-LONG.py",
    "jwst_gc_pipeline/reduction/PipelineRerunNIRISS.py",
    "jwst_gc_pipeline/reduction/PipelineMIRI.py",
    "jwst_gc_pipeline/photometry/crowdsource_catalogs_long.py",
)

#: An absolute survey root written into the source.
_HARDCODED_ROOT = re.compile(
    r"""['"]/(?:orange/adamginsburg|blue/adamginsburg/adamginsburg)/jwst/""")

#: Reviewed exceptions, as ``file: reason``.  A line matches an entry when the
#: entry's ``needle`` appears in it.
ALLOWED = (
    # get_psf_model's `basepath` is the tree ABOVE the per-target directories:
    # it becomes `jwst_root`, which locates the shared PSF store (psfs_shared/)
    # that every target draws from.  One shared location, so no field's `root:`
    # applies.  (The `basepath = f'{basepath}/{target}'` line below it is
    # assigned and never read.)
    ("jwst_gc_pipeline/photometry/crowdsource_catalogs_long.py",
     "basepath='/blue/adamginsburg/adamginsburg/jwst/'"),
)


def _code_lines(path):
    """Source lines with whole-line comments dropped.

    A comment may name a path when it is describing history; only code that
    builds one is a finding.
    """
    for number, line in enumerate(path.read_text(errors="replace").splitlines(),
                                  start=1):
        if line.lstrip().startswith("#"):
            continue
        yield number, line


@pytest.mark.parametrize("driver", DRIVERS)
def test_driver_takes_basepath_from_registry(driver):
    path = REPO_ROOT / driver
    assert path.is_file(), f"{driver} has moved; update DRIVERS"
    allowed = tuple(needle for name, needle in ALLOWED if name == driver)
    offenders = [f"{driver}:{number}: {line.strip()}"
                 for number, line in _code_lines(path)
                 if _HARDCODED_ROOT.search(line)
                 and not any(needle in line for needle in allowed)]
    assert not offenders, (
        "hard-coded survey root in a pipeline driver:\n  "
        + "\n  ".join(offenders)
        + "\n\nThe field's directory comes from fields.yaml: "
          "`basepath = field_registry.basepath(regionname)`.  A hard-coded root "
          "pins every field to one tree and silently ignores the field's own "
          "`root:`."
    )


def test_allowed_entries_still_match():
    """Keep the exception list from rotting: each entry must still be present."""
    stale = [f"{name}: {needle}" for name, needle in ALLOWED
             if needle not in (REPO_ROOT / name).read_text(errors="replace")]
    assert not stale, ("ALLOWED names lines that are gone (remove them):\n  "
                       + "\n  ".join(stale))


#: Which tree each field lives on.  The `localdata` check below cannot notice a
#: blue<->orange flip for brick or cloudc, because /orange/.../brick is a symlink
#: into the blue tree -- so the two fields that have one are exactly the two it
#: is blind to.  Pin them here instead, so moving a field has to be deliberate.
EXPECTED_ROOTS = {
    'arches': 'orange', 'brick': 'blue', 'cloudc': 'blue',
    'cloudef': 'orange',
    # 2092 obs 005 (CLOUDEF-REFERENCE), split out of cloudef because it is a
    # control field 11.77' away sharing no sources with CLOUDEF-CENTER.
    'cloudef_controlfield': 'orange', 'gc-treasury': 'blue', 'gc2211': 'orange',
    'm4': 'orange', 'm92': 'orange',
    'ngc6334': 'orange', 'ngc6397': 'orange', 'omegacen': 'orange',
    'quintuplet': 'orange', 'sgra': 'orange', 'sgrb2': 'orange',
    'sgrc': 'orange', 'sickle': 'orange', 'w51': 'orange', 'wd1': 'orange',
    'wd2': 'orange',
}


def test_every_field_keeps_its_registered_tree():
    registered = {f.name: f.root for f in field_registry.FIELDS}
    assert registered == EXPECTED_ROOTS


def test_registry_basepath_names_the_field():
    """Every registered field resolves to a directory named for it."""
    for field in field_registry.FIELDS:
        basepath = field_registry.basepath(field.name)
        assert basepath.endswith(f"/{field.name}/"), (
            f"{field.name} resolves to {basepath}, which does not end in the "
            f"field's own name")


@pytest.mark.localdata
def test_registry_basepath_matches_the_tree_on_disk():
    """On HiPerGator the registered path and the retired hard-coded one agree.

    The hard-coded ``/orange/...`` path worked for ``root: blue`` fields only
    because those directories are symlinks into the blue tree.  Where both
    exist, they must still resolve to the same place -- otherwise this change
    moved where a field reduces.
    """
    checked = 0
    for field in field_registry.FIELDS:
        registered = field_registry.basepath(field.name)
        retired = f"/orange/adamginsburg/jwst/{field.name}/"
        if not (os.path.isdir(registered) and os.path.isdir(retired)):
            continue
        checked += 1
        assert os.path.realpath(registered) == os.path.realpath(retired), (
            f"{field.name}: registry says {registered}, the retired hard-coded "
            f"path was {retired}, and they are different directories")
    if checked == 0:
        pytest.skip("no field directory present on this machine")
