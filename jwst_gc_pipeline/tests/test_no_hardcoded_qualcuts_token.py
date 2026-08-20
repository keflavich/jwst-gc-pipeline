"""Grep-guard: forbid spelling one program's oksep quality-cut token literally.

``oksep`` is a hand-written label from program **2221** ("the separations
between this source's detections in different exposures are small enough to
call it one real star").  The quality-cut table it names is written per field,
and its suffix carries that field's OWN proposal:
``merge_catalogs._qualcuts_oksep_suffix()`` builds ``_qualcuts_oksep1905`` for
wd1, ``_qualcuts_oksep6151`` for w51, ``_qualcuts_oksep10678`` for the GC
Treasury.

The old code stamped ``2221`` on every target, which is why 11 fields with no
connection to that program carry ``_qualcuts_oksep2221`` catalogs on disk.
That literal then spread to the readers, and each copy failed quietly rather
than loudly:

* ``stage_release.py``'s catalog regexes matched only ``_qualcuts_oksep2221``,
  and the loop that uses them does ``if m is None: continue`` -- so w51's and
  wd1's quality-filtered tables sat on disk and never reached a release, with
  nothing logged.
* the staged README told every field its quality subset was named
  ``_qualcuts_oksep2221``.

A literal works on brick and cloudc (which really are program 2221), so
nothing catches a new copy until some other field silently loses its table.

Ask the registry instead::

    from jwst_gc_pipeline.photometry.merge_catalogs import _qualcuts_oksep_suffix
    fn = f'{tablename}{_qualcuts_oksep_suffix(target)}.fits'

or, to MATCH any field's token, use ``stage_release.QUALCUTS_RE``.

Scope: every git-tracked ``.py``, ``.md``, ``.rst``, ``.sh``, ``.sbatch``,
``.yaml``/``.yml``, ``.json`` and ``.toml`` file.  Prose counts: a README line
naming one program's token is what the next glob gets copied from.

Two kinds of line stay legal, and both are opt-outs rather than accidents:

* ``merge_catalogs._qualcuts_oksep_suffix`` itself, which returns the literal
  for the fields that are genuinely 2221 (brick registers 1182+2221 and cloudc
  2221+2526, so the generic token would rename their 67 existing catalogs).
* a line carrying the ``noqa: qualcuts-token`` pragma -- used by the tests that
  assert the per-field mapping, and by prose that quotes the token in order to
  prohibit it.
"""
import pathlib
import re
import subprocess

REPO = pathlib.Path(__file__).resolve().parents[2]

SCANNED_SUFFIXES = {".py", ".md", ".rst", ".sh", ".sbatch", ".slurm",
                    ".json", ".yml", ".yaml", ".toml"}

#: The one file exempted wholesale: this module cannot state its own rule
#: without quoting what it forbids, and it quotes the token on ~11 lines.
#:
#: `merge_catalogs.py` is deliberately NOT here.  It owns the three lines that
#: legitimately spell the literal -- the two docstring sentences explaining the
#: 2221 carve-out and the `return` that implements it -- and those carry the
#: line pragma instead.  A whole-file exemption would blind the guard across
#: 3300 other lines of a module whose subject is catalog filenames, which is
#: exactly where the next hardcoded token would land.
ALLOWLIST = {
    "jwst_gc_pipeline/tests/test_no_hardcoded_qualcuts_token.py",
}

#: A floor on the file count, so a guard that scans NOTHING fails instead of
#: passing over an empty offender list (a `git ls-files` that returns nothing
#: from a detached worktree, say).  Borrowed from the jw0 guard, which carries
#: the same failure mode.
MIN_SCANNED_FILES = 350

PRAGMA = "noqa: qualcuts-token"

#: ``_qualcuts_oksep`` followed by a literal proposal number.  A `{`
#: (interpolation) or a regex character class is the correct, general form and
#: is not matched.
BANNED = re.compile(r"_qualcuts_oksep[0-9]")


def _tracked_files():
    out = subprocess.run(["git", "-C", str(REPO), "ls-files"],
                         capture_output=True, text=True, check=True).stdout
    for rel in out.splitlines():
        path = REPO / rel
        if path.suffix in SCANNED_SUFFIXES and path.is_file():
            yield rel, path


def _offending_lines(text):
    return [n for n, line in enumerate(text.splitlines(), 1)
            if BANNED.search(line) and PRAGMA not in line]


def test_the_guard_actually_scans_the_repository():
    """A guard that scans nothing passes for the wrong reason."""
    scanned = list(_tracked_files())
    assert len(scanned) >= MIN_SCANNED_FILES, (
        f"only {len(scanned)} files scanned (floor {MIN_SCANNED_FILES}); the "
        "file walk is broken, so an empty offender list means nothing")


def test_no_hardcoded_qualcuts_token():
    offenders = []
    for rel, path in _tracked_files():
        if rel in ALLOWLIST:
            continue
        hits = _offending_lines(path.read_text(errors="replace"))
        if hits:
            offenders.append(f"{rel}:{','.join(map(str, hits[:5]))}")
    assert not offenders, (
        "one program's oksep token spelled literally in:\n  "
        + "\n  ".join(sorted(offenders))
        + "\n\nThe quality-cut suffix carries each field's own proposal, so a "
        "literal silently drops every other field's table (issue: w51's "
        "_qualcuts_oksep6151 and wd1's _qualcuts_oksep1905 never reached a "
        "release).  Build it with "
        "`merge_catalogs._qualcuts_oksep_suffix(target)`, or match any field's "
        f"with `stage_release.QUALCUTS_RE`.  A line that must quote a literal "
        f"token opts out with `{PRAGMA}`."
    )


def test_the_guard_would_catch_the_defect_it_exists_for():
    """The pattern matches the spellings that actually shipped, and spares the
    general forms that replaced them."""
    caught = [
        'rf"(?P<qc>_qualcuts_oksep2221)?\\.(?P<ext>fits|ecsv)$"',
        "fn = f'{base}/catalogs/{stem}_qualcuts_oksep2221.fits'",
        "`_qualcuts_oksep2221` is the quality-filtered subset.",
        "_qualcuts_oksep1905",
    ]
    for line in caught:
        assert _offending_lines(line) == [1], line
    spared = [
        'QUALCUTS_RE = r"_qualcuts_oksep[0-9A-Za-z-]+"',
        "fn = f'{tablename}{_qualcuts_oksep_suffix(target)}.fits'",
        "the `_qualcuts_oksep<proposal>` variant is the quality-filtered subset",
        f"assert suffix == '_qualcuts_oksep2221'  # {PRAGMA}",
    ]
    for line in spared:
        assert _offending_lines(line) == [], line


def test_the_suffix_module_is_scanned_rather_than_exempted():
    """`merge_catalogs.py` owns the token, so it is where a new literal lands.

    Exempting the whole file would hide one anywhere in its ~3300 lines; the
    three lines that legitimately spell it carry the pragma instead.
    """
    rel = "jwst_gc_pipeline/photometry/merge_catalogs.py"
    assert rel not in ALLOWLIST
    scanned = {r for r, _ in _tracked_files()}
    assert rel in scanned, "the suffix module dropped out of the scanned set"
    text = (REPO / rel).read_text()
    assert _offending_lines(text) == [], (
        "unpragma'd literal in the module that owns the suffix")
    # and the pragma is doing real work there: without it the file offends
    assert [n for n, line in enumerate(text.splitlines(), 1)
            if BANNED.search(line)], (
        "no literal left in merge_catalogs.py -- if the carve-out is gone, "
        "drop this test with it")
