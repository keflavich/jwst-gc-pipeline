"""Grep-guard: forbid code that spells the JWST filename prefix ``jw0`` + proposal.

MAST zero-pads the proposal number to FIVE digits in every product name
(``jw02221...``, ``jw10678...``).  The literal ``f'jw0{proposal_id}'`` is
byte-identical to that for a 4-digit proposal and wrong for a 5-digit one
(10678, the GC Treasury program, and omegacen's 12587): against 10678 it
fabricates ``jw010678`` -- the MAST URI filter selects zero uncals, every glob
matches nothing, and the m2 visit token fails its own ``^jw\\d{11}$``
validator (issue #414).  Because the spelling WORKS on every 4-digit program,
nothing else catches it until a 5-digit run dies.

The same 4-digit assumption has a second spelling, in the reduction path: the
header slice ``header['PROGRAM'][1:5]``.  ``PROGRAM`` is MAST's five-character
padded form, so that slice drops the fifth digit of a 5-digit proposal.  It is
banned here too.

Build the prefix and read the proposal with the helpers instead::

    from jwst_gc_pipeline.mast_names import jw_prefix, proposal_id_from_program
    fn = f'{jw_prefix(proposal_id)}-o{field}_t001_...'
    proposal_id = proposal_id_from_program(header['PROGRAM'])

Vocabulary, for the failure messages above.  A JWST *visit token* is
``jw`` + proposal(5) + observation(3) + visit(3) = 11 digits, the shape
``^jw\\d{11}$`` counts (``jw02221001001``).  An *uncal* is the raw ramp
product MAST serves, the reduce's stage-1 input.  The *MAST URI filter* is the
substring test the reduce applies to each ``dataURI`` a MAST query returns, to
keep only this proposal-and-observation's uncals.  *m2* is the second merge
stage of cataloging, where per-exposure astrometry is re-verified.

Scope: every git-tracked file with a scanned extension (``.py`` including test
files, plus ``.md``, ``.sh``, ``.sbatch``, ``.slurm``), in any context --
f-strings, ``.format`` templates, ``%`` formatting, ``'jw0' +`` concatenation,
comments, docstrings and prose alike.  A comment or a README line documenting
the wrong spelling is the template the next f-string gets copied from, and the
one place a user reads the glob.  Files that quote the banned form on purpose
(this one, ``mast_names.py``, ``test_mast_names.py``) are allowlisted individually
rather than by excluding a whole tree.  Literal 4-digit product names like
``jw02221001001_..._cal.fits`` stay legal: they name real files on disk, and
none of the banned patterns matches a digit after ``jw0``.

Untracked working-tree files are deliberately out of scope: the guard polices
what the repository ships, and an untracked scratch script becomes covered the
moment it is ``git add``ed.

See ``jwst_gc_pipeline/mast_names.py`` and issue #414.
"""
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

#: File types scanned.  Prose and shell are included because both carry globs
#: a human copies from (``GETTING_STARTED.md`` documents the association glob).
SCANNED_SUFFIXES = {".py", ".md", ".sh", ".sbatch", ".slurm"}

#: Every spelling of "``jw0`` glued to a proposal that is not a literal digit".
#: ``jw0{`` f-string/format, ``jw0%`` percent-formatting, ``jw0<`` prose
#: placeholder, ``jw0$`` shell variable, and ``'jw0' +`` concatenation.  A
#: literal product name (``jw02221...``) has a DIGIT after ``jw0`` and matches
#: none of them.
BANNED_RE = re.compile(r"""jw0[{<%$]|['"]jw0['"]\s*\+|\+\s*['"]jw0['"]""")

#: The header-slice spelling of the same 4-digit assumption.
BANNED_PROGRAM_SLICE_RE = re.compile(r"""\[\s*['"]PROGRAM['"]\s*\]\s*\[""")

ALLOWLIST = {
    # document the banned spellings in the docstrings that explain the fix
    "jwst_gc_pipeline/mast_names.py",
    "jwst_gc_pipeline/tests/test_mast_names.py",
    "jwst_gc_pipeline/tests/test_no_jw0_prefix_literals.py",
}

#: A floor on the file count, so a guard that scans NOTHING fails instead of
#: passing over an empty offender list.  The repo tracks ~390 ``.py`` files
#: alone; 150 leaves room for a large deletion without tripping.
MIN_SCANNED_FILES = 150


def _iter_scanned_files():
    """Git-tracked files with a scanned extension.

    ``check=True`` on purpose: if ``git ls-files`` cannot run, this guard
    cannot do its job, and a guard that cannot do its job has to say so rather
    than report a clean tree.
    """
    out = subprocess.run(["git", "-C", str(REPO_ROOT), "ls-files"],
                         capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        rel = Path(line)
        if rel.suffix not in SCANNED_SUFFIXES:
            continue
        p = REPO_ROOT / rel
        if not p.is_file():
            continue
        yield rel, p


def _offending_lines(text):
    return [i for i, line in enumerate(text.splitlines(), 1)
            if BANNED_RE.search(line) or BANNED_PROGRAM_SLICE_RE.search(line)]


def test_the_guard_scans_the_repository():
    """A guard over an empty file list passes vacuously.  Pin the count."""
    scanned = list(_iter_scanned_files())
    assert len(scanned) >= MIN_SCANNED_FILES, (
        f"the guard found only {len(scanned)} files to scan; it is not "
        f"looking at the repository, so its 'no offenders' result means "
        f"nothing")
    names = {rel.as_posix() for rel, _ in scanned}
    # one file of each scanned kind that must be in the set
    for expected in ("jwst_gc_pipeline/mast_names.py",
                     "jwst_gc_pipeline/photometry/cataloging.py",
                     "GETTING_STARTED.md"):
        assert expected in names, expected


def test_git_failure_is_an_error_not_a_clean_tree():
    """Point the guard at a directory with no git tree: it raises."""
    with pytest.raises((subprocess.CalledProcessError, FileNotFoundError)):
        subprocess.run(["git", "-C", "/", "ls-files"],
                       capture_output=True, text=True, check=True)


def test_no_jw0_prefix_literal_anywhere():
    offenders = []
    for rel, path in _iter_scanned_files():
        if rel.as_posix() in ALLOWLIST:
            continue
        hits = _offending_lines(path.read_text(errors="replace"))
        if hits:
            offenders.append(f"{rel.as_posix()}:{','.join(map(str, hits[:5]))}")
    assert not offenders, (
        "4-digit-proposal filename spelling in non-allowlisted file(s):\n  "
        + "\n  ".join(sorted(offenders))
        + "\n\n'jw0' + proposal_id assumes a 4-digit proposal and breaks every "
        "filename, glob and visit token of a 5-digit program (jw010678 vs the "
        "real jw10678 -- issue #414), and header['PROGRAM'][1:5] drops that "
        "program's fifth digit. Use "
        "`from jwst_gc_pipeline.mast_names import jw_prefix, "
        "proposal_id_from_program`."
    )


def test_allowlist_entries_exist():
    """Keep the allowlist from rotting -- every entry must point at a real file."""
    missing = [rel for rel in ALLOWLIST if not (REPO_ROOT / rel).is_file()]
    assert not missing, ("ALLOWLIST references files that no longer exist "
                         "(remove them):\n  " + "\n  ".join(sorted(missing)))


@pytest.mark.parametrize('bad', [
    "glob(f'jw0{proposal_id}-o{field}_t001_*_i2d.fits')",
    'token = f"jw0{proposal_id}{field}{visit:03d}"',
    "ASN_GLOB = 'jw0{proposal}-o{obsid}*_asn.json'",
    "pat = 'jw0%s-o%s_t001.fits' % (proposal_id, field)",
    "pat = 'jw0' + str(proposal_id) + '-o' + field",
    'pat = "jw0" + str(proposal_id)',
    "stem = str(proposal_id) + 'x'  # built from 'jw0' + prop",
    "# the association glob is jw0<proposal>-o<obs>*_asn.json",
    "ls $DIR/jw0${PROPOSAL}-o${OBS}*_cal.fits",
    "proposal_id = hdu[0].header['PROGRAM'][1:5]",
    'project_id = header["PROGRAM"][1:5]',
])
def test_the_guard_matches_every_banned_spelling(bad):
    """A guard that cannot fire is not a guard."""
    assert _offending_lines(bad), bad


@pytest.mark.parametrize('good', [
    "prefix = jw_prefix(proposal_id)",
    "fn = 'jw02221001001_02101_00001_nrcb1_cal.fits'",
    "pat = f'{jw_prefix(proposal_id)}-o{field}*_asn.json'",
    "proposal_id = proposal_id_from_program(header['PROGRAM'])",
    "program = header['PROGRAM'].strip()",
    "ls $DIR/jw10678-o001*_cal.fits",
])
def test_the_guard_does_not_fire_on_the_correct_spellings(good):
    assert not _offending_lines(good), good
