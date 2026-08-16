"""Grep-guard: forbid NEW code that spells the JWST filename prefix ``jw0{...}``.

MAST zero-pads the proposal number to FIVE digits in every product name
(``jw02221...``, ``jw10678...``).  The literal ``f'jw0{proposal_id}'`` is
byte-identical to that for a 4-digit proposal and wrong for a 5-digit one
(10678, the GC Treasury program, and omegacen's 12587): against 10678 it
fabricates ``jw010678`` -- the MAST URI filter selects zero uncals, every glob
matches nothing, and the m2 visit token fails its own ``^jw\\d{11}$``
validator (issue #414).  Because the spelling WORKS on every 4-digit program,
nothing else catches it until a 5-digit run dies.

Build the prefix with the helper instead::

    from jwst_gc_pipeline.naming import jw_prefix
    fn = f'{jw_prefix(proposal_id)}-o{field}_t001_...'

This test FAILS if a git-tracked, non-test Python file contains the ``jw0{``
literal, in any context -- f-strings, ``.format`` templates, comments and
docstrings alike (a comment documenting the wrong spelling is the template the
next f-string gets copied from).  Literal 4-digit product names like
``jw02221001001_..._cal.fits`` stay legal: they name real files on disk.

See ``jwst_gc_pipeline/naming.py`` and issue #414.
"""
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

BANNED = "jw0{"

ALLOWLIST = {
    # documents the banned spelling in the docstrings that explain the fix
    "jwst_gc_pipeline/naming.py",
}


def _iter_py_files():
    """Git-tracked .py files only -- the guard polices committed code, not local
    scratch scripts in the working tree."""
    try:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), "ls-files", "*.py"],
                             capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return
    for line in out.splitlines():
        rel = Path(line)
        p = REPO_ROOT / rel
        if not p.is_file():
            continue
        if "tests" in rel.parts or p.name.startswith("test_"):
            continue
        yield rel, p


def _offending_lines(text):
    return [i for i, line in enumerate(text.splitlines(), 1) if BANNED in line]


def test_no_jw0_prefix_literal_anywhere():
    offenders = []
    for rel, path in _iter_py_files():
        if rel.as_posix() in ALLOWLIST:
            continue
        hits = _offending_lines(path.read_text(errors="replace"))
        if hits:
            offenders.append(f"{rel.as_posix()}:{','.join(map(str, hits[:5]))}")
    assert not offenders, (
        "literal 'jw0{' JWST filename prefix in non-allowlisted file(s):\n  "
        + "\n  ".join(sorted(offenders))
        + "\n\n'jw0' + proposal_id assumes a 4-digit proposal and breaks every "
        "filename, glob and visit token of a 5-digit program (jw010678 vs the "
        "real jw10678 -- issue #414). Use "
        "`from jwst_gc_pipeline.naming import jw_prefix` and spell the prefix "
        "`{jw_prefix(proposal_id)}`."
    )


def test_allowlist_entries_exist():
    """Keep the allowlist from rotting -- every entry must point at a real file."""
    missing = [rel for rel in ALLOWLIST if not (REPO_ROOT / rel).is_file()]
    assert not missing, ("ALLOWLIST references files that no longer exist "
                         "(remove them):\n  " + "\n  ".join(sorted(missing)))


def test_the_guard_actually_matches_the_bad_pattern():
    """A guard that cannot fire is not a guard."""
    for bad in ("glob(f'jw0{proposal_id}-o{field}_t001_*_i2d.fits')",
                'token = f"jw0{proposal_id}{field}{visit:03d}"',
                "ASN_GLOB = 'jw0{proposal}-o{obsid}*_asn.json'"):
        assert _offending_lines(bad), bad
    # and the padded spellings it must NOT fire on
    for good in ("prefix = jw_prefix(proposal_id)",
                 "fn = 'jw02221001001_02101_00001_nrcb1_cal.fits'",
                 "pat = f'{jw_prefix(proposal_id)}-o{field}*_asn.json'"):
        assert not _offending_lines(good), good
