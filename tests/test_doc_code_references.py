"""Grep-guard: documentation must not reference code that does not exist.

Docs in this repo carry a lot of concrete pointers -- module paths, symbol names
and (historically) `file.py:NNN` line citations.  Line citations rot silently:
a 2026-07-30 audit found 37 of 41 checkable ones pointed at unrelated code after
refactors, including three that named the wrong function for a safety guard.

This test enforces two rules on tracked ``.md`` files:

1. **No new ``file.py:NNN`` line citations.**  Cite ``module.py::symbol`` or just
   the symbol; ``git grep`` finds it and it survives a refactor.  Files listed in
   ``LINE_CITATION_ALLOWLIST`` are historical snapshots that carry an explicit
   "line numbers are as-of <date>" banner and are exempt.
2. **A ``jwst_gc_pipeline.a.b`` dotted module path named in a doc must resolve**
   to a real module (or be a real symbol inside one).

Both are cheap textual checks: no imports, no data.
"""
import os
import re
import subprocess

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Historical snapshots: banner says the line numbers are not maintained.
LINE_CITATION_ALLOWLIST = {
    'REFACTOR_PLAN.md',
    'PSFPhotometryPlan2026-06-09.md',
    'jwst_gc_pipeline/astrometry_gdc/GDC_EXPERIMENT_REPORT.md',
    'scripts/release/CMZ_HIPS_AND_CATALOG_SHARING_PLAN.md',
    'docs/multiframe_keep_verification/REGRESSION_VERIFICATION.md',
    'docs/pr57_recovery_investigation/README.md',
    'docs/reports/CRDS_STALE_FILTEROFFSET_RMAP_INCIDENT.md',
    'docs/reports/DVA_INTERDETECTOR_REPORT.md',
    'docs/reports/SATSTAR_WING_CALIBRATION_REPORT.md',
    'docs/reports/SATURATED_STAR_PHOTOMETRY_ARTICLE.md',
    'docs/review/miri_satstar_2026-07-04/README.md',
}

# ``jwst/…`` etc: paths inside third-party packages, not ours.
_FOREIGN_PREFIXES = ('jwst/', 'photutils/', 'astropy/', 'crowdsource/',
                     'brick2221/', 'jwst_rgb/', 'peppar/')

_CITATION_RE = re.compile(r'\b([\w./\-]+\.py):(\d+)')
_MODULE_RE = re.compile(r'\bjwst_gc_pipeline(?:\.[a-zA-Z_]\w*)+')
_CODE_FENCE_RE = re.compile(r'```.*?```', re.S)


def _tracked(*globs):
    out = subprocess.run(['git', 'ls-files', *globs], cwd=REPO,
                         capture_output=True, text=True, check=True)
    return out.stdout.split()


@pytest.fixture(scope='module')
def docs():
    return _tracked('*.md')


def test_no_line_number_citations(docs):
    """A ``file.py:NNN`` citation is stale the moment the file is edited."""
    offenders = []
    for doc in docs:
        if doc in LINE_CITATION_ALLOWLIST:
            continue
        text = open(os.path.join(REPO, doc), errors='replace').read()
        # Traceback pastes inside fenced blocks legitimately carry line numbers.
        text = _CODE_FENCE_RE.sub('', text)
        for lineno, line in enumerate(text.splitlines(), 1):
            for path, num in _CITATION_RE.findall(line):
                if path.startswith(_FOREIGN_PREFIXES):
                    continue
                offenders.append(f'{doc}:{lineno}: {path}:{num}')
    assert not offenders, (
        'documentation must not cite code by line number (they go stale '
        'silently -- cite `module.py::symbol` or the bare symbol instead):\n  '
        + '\n  '.join(offenders))


def test_dotted_module_paths_resolve(docs):
    """``jwst_gc_pipeline.x.y`` named in a doc must be a real module or symbol."""
    py_files = set(_tracked('*.py'))
    symbols = set()
    for f in py_files:
        src = open(os.path.join(REPO, f), errors='replace').read()
        symbols.update(re.findall(r'^\s*(?:def|class)\s+(\w+)', src, re.M))
        symbols.update(re.findall(r'^([A-Z][A-Z0-9_]{2,})\s*[:=]', src, re.M))

    missing = []
    for doc in docs:
        for lineno, line in enumerate(
                open(os.path.join(REPO, doc), errors='replace').read().splitlines(), 1):
            for dotted in _MODULE_RE.findall(line):
                parts = dotted.split('.')
                candidates = ['/'.join(parts) + '.py',
                              '/'.join(parts) + '/__init__.py']
                if any(c in py_files for c in candidates):
                    continue
                # ``pkg.module.symbol`` form
                if parts[-1] in symbols and any(
                        c in py_files for c in ['/'.join(parts[:-1]) + '.py',
                                                '/'.join(parts[:-1]) + '/__init__.py']):
                    continue
                missing.append(f'{doc}:{lineno}: {dotted}')
    assert not missing, (
        'docs name jwst_gc_pipeline module paths that do not exist:\n  '
        + '\n  '.join(missing))


def test_alignment_config_table_covers_every_field():
    """ASTROMETRY_WCS_CORRECTION_FLOW.md lists one row per configured field.

    The registry is the thing a reducer consults to answer "what is this field
    tied to"; a field present in code but absent from the table reads as
    unconfigured.
    """
    from jwst_gc_pipeline.reduction import alignment_config as ac

    doc = open(os.path.join(
        REPO, 'jwst_gc_pipeline/reduction/ASTROMETRY_WCS_CORRECTION_FLOW.md'),
        errors='replace').read()
    seg = doc[doc.index('### The configured fields'):]
    seg = seg[:seg.index('### How a locked-table row is selected')]
    rows = [ln for ln in seg.splitlines()
            if ln.startswith('| ') and not ln.startswith('| proposal')
            and not set(ln) <= set('|-: ')]
    assert len(rows) == len(ac.ALIGNMENT_CONFIG), (
        f'the field table lists {len(rows)} rows but ALIGNMENT_CONFIG has '
        f'{len(ac.ALIGNMENT_CONFIG)} entries -- update '
        'ASTROMETRY_WCS_CORRECTION_FLOW.md when adding a field')
    documented = {(c[0].strip(), c[1].strip())
                  for c in (r.strip().strip('|').split('|') for r in rows)}
    for entry in ac.ALIGNMENT_CONFIG:
        props = {p for p, _ in documented}
        assert entry.proposal in props, (
            f'proposal {entry.proposal} is configured but not in the doc table')


def test_saturation_floor_table_matches_code():
    """SATURATED_PIXEL_HANDLING.md's per-filter floor table must match the dicts.

    Both floors are science-affecting (which pixels are masked / which sources
    are vetoed), and the table has been wrong before: it grouped F182M/F210M and
    F187N at the wrong severity floor.
    """
    from jwst_gc_pipeline.reduction.saturated_star_finding import (
        SAT_SEVERITY_FLOOR, _SATSTAR_DATA_FLOOR)

    doc = open(os.path.join(REPO, 'SATURATED_PIXEL_HANDLING.md'),
               errors='replace').read()
    seg = doc[doc.index('| filters | satstar-finder floor'):]
    seg = seg[:seg.index('\n\n')]

    def _num(cell):
        m = re.search(r'(\d+)', cell.replace('**', ''))
        return float(m.group(1)) if m else 0.0

    mismatches, checked = [], 0
    for line in seg.splitlines():
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if len(cells) != 3 or cells[0].startswith('filters'):
            continue
        filts = re.findall(r'F\d+[A-Z]', cells[0])
        if not filts:
            continue
        want = (_num(cells[1]), _num(cells[2]))
        for filt in filts:
            checked += 1
            got = (_SATSTAR_DATA_FLOOR.get(filt.lower(), 0.0),
                   SAT_SEVERITY_FLOOR.get(filt.lower(), 0.0))
            if got != want:
                mismatches.append(
                    f'{filt}: doc (finder={want[0]}, severity={want[1]}) '
                    f'vs code (finder={got[0]}, severity={got[1]})')

    doc_filters = {f.lower() for f in re.findall(r'F\d+[A-Z]', seg)}
    unlisted = sorted((set(SAT_SEVERITY_FLOOR) | set(_SATSTAR_DATA_FLOOR))
                      - doc_filters)
    assert checked, 'parsed no filter rows out of the floor table'
    assert not mismatches, 'floor table disagrees with the code:\n  ' + '\n  '.join(mismatches)
    assert not unlisted, f'filters with a code floor but no doc row: {unlisted}'
