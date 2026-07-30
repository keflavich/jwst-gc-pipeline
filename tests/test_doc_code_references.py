"""Grep-guard: documentation must not reference code that does not exist.

Docs in this repo carry a lot of concrete pointers -- module paths, symbol names
and (historically) ``file.py:NNN`` line citations.  Line citations rot silently:
a 2026-07-30 audit found 37 of 41 checkable ones pointing at unrelated code after
refactors, including three that named the wrong function for a safety guard.

Four rules on tracked ``.md`` files:

1. **No line citations into this repo's own files.**  Any ``file.py:NNN`` /
   ``file.py line NNN`` / ``file.py#LNNN`` form whose path resolves to a tracked
   file is rejected; cite ``module.py::symbol`` or the bare symbol instead.
   Third-party paths (``stdatamodels/...``, upstream URLs) are ignored because
   they do not resolve here, which is also why pasted upstream tracebacks are
   fine.  Paste *our* tracebacks inside a fenced block.
2. **``module.py::symbol`` must resolve** -- the symbol must be defined in THAT
   file.  This is the format rule 1 pushes authors towards, so it is the one that
   most needs checking.
3. **Dotted ``jwst_gc_pipeline.a.b`` paths must resolve**, and a trailing symbol
   must live in the module actually named.
4. **Two content tables must match the code they describe** (the alignment field
   registry and the saturation floor table) -- both have been wrong before, and
   both are science-affecting.

Rules 1-3 are pure text.  Rule 4 reads the floor dicts out of the AST (no
import: ``saturated_star_finding`` pulls in stpsf/jwst and needs ``STPSF_PATH``,
which a doc guard must not require -- same reasoning as
``test_manual_defaults_consistency.py``) and imports the import-light
``alignment_config`` for the registry.
"""
import ast
import os
import re
import subprocess

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Historical snapshots that carry an explicit "line numbers are as-of <date>"
#: banner.  Exempt from rules 1 and 3 (they legitimately name deleted modules).
#: ``test_allowlist_is_earned`` fails if an entry stops needing its exemption.
LINE_CITATION_ALLOWLIST = {
    'REFACTOR_PLAN.md',
    'jwst_gc_pipeline/astrometry_gdc/GDC_EXPERIMENT_REPORT.md',
}

#: Repo-relative paths whose basename also exists upstream would make the
#: "resolves to a tracked file" test ambiguous.  There are none today; assert it.
_AMBIGUOUS_BASENAMES = {'__init__.py'}

# ---------------------------------------------------------------------------
# citation surface forms.  Anchored on a source-file extension so prose like
# "Table 2:15" is not a citation.  ``L`` and the word "line" are both accepted
# because both appear in the wild.
_EXT = r'(?:py|sh|sbatch|slurm|ipynb|ya?ml|toml|cfg)'
_CITATION_RES = (
    re.compile(rf'([\w./\-]+\.{_EXT})\s*:\s*(\d+)', re.I),
    re.compile(rf'([\w./\-]+\.{_EXT})\s*#\s*L(\d+)', re.I),
    re.compile(rf'([\w./\-]+\.{_EXT})`?[,;]?\s+(?:at\s+)?[Ll](?:ine)?s?\.?\s*(\d+)'),
    re.compile(rf'\b[Ll](?:ine)?s?\.?\s*(\d+)\s+(?:of|in)\s+([\w./\-]+\.{_EXT})'),
    # `symbol:610` -- a line citation hung off a symbol name rather than a path.
    # This form slipped past the first version of this guard.
    re.compile(r'`([A-Za-z_]\w{2,}):(\d{3,})`|`([A-Za-z_]\w{2,})`:(\d{3,})\b'),
)
#: rule 4 of _CITATION_RES yields (lineno, path); the rest yield (path, lineno)
_REVERSED_CITATION_RE = _CITATION_RES[3]
#: ``symbol:NNN`` names no file, so ``_is_ours`` cannot judge it -- it is always ours.
_SYMBOL_CITATION_RE = _CITATION_RES[4]

_SYMBOL_REF_RE = re.compile(r'([\w./\-]+\.py)::([\w.]+)')
_MODULE_RE = re.compile(r'\bjwst_gc_pipeline(?:\.[a-zA-Z_]\w*)+')

#: A pasted traceback / pytest tail names real files at real lines and is not a
#: documentation pointer.  Fenced blocks are stripped outright; these patterns
#: catch the unfenced case.
_PASTED_OUTPUT_RES = (
    re.compile(r'^\s*File\s+"'),
    re.compile(r'^\s*(?:FAILED|ERROR|PASSED)\s'),
    re.compile(r'^\s*\S+\.py:\d+:\s*(?:in\b|[A-Z]\w*(?:Error|Warning|Exception)\b)'),
    re.compile(r'Traceback \(most recent call last\)'),
)


def _strip_code_blocks(text):
    """Drop fenced blocks (``` and ~~~).  Unpaired fences close at EOF."""
    out, fence = [], None
    for line in text.splitlines():
        stripped = line.lstrip()
        if fence is None:
            if stripped.startswith('```') or stripped.startswith('~~~'):
                fence = stripped[:3]
                out.append('')
                continue
            out.append(line)
        else:
            out.append('')
            if stripped.startswith(fence):
                fence = None
    return out


def _tracked(*globs):
    try:
        out = subprocess.run(['git', 'ls-files', *globs], cwd=REPO,
                             capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip('not a git checkout; nothing to scan')
    return out.stdout.splitlines()


@pytest.fixture(scope='module')
def docs():
    return _tracked('*.md')


@pytest.fixture(scope='module')
def tracked_paths():
    paths = set(_tracked())
    basenames = {}
    for path in paths:
        basenames.setdefault(os.path.basename(path), []).append(path)
    return paths, basenames


def _is_ours(path, tracked_paths):
    """Does this doc-written path point at a file in THIS repo?"""
    paths, basenames = tracked_paths
    if path in paths or any(p.endswith('/' + path) for p in paths):
        return True
    base = os.path.basename(path)
    return base in basenames and base not in _AMBIGUOUS_BASENAMES


@pytest.fixture(scope='module')
def defined_symbols():
    """``{repo_relative_path: {names defined at module level}}``."""
    per_file = {}
    for path in _tracked('*.py'):
        src = open(os.path.join(REPO, path), errors='replace').read()
        names = set(re.findall(r'^\s*(?:def|class)\s+(\w+)', src, re.M))
        names |= set(re.findall(r'^([A-Za-z_]\w*)\s*[:=]', src, re.M))
        per_file[path] = names
    return per_file


def _citations(line):
    """Yield ``(path, lineno)`` for every citation form on this line."""
    for regex in _CITATION_RES:
        for groups in regex.findall(line):
            if regex is _REVERSED_CITATION_RE:
                yield groups[1], groups[0]
            else:
                yield groups[0], groups[1]


# ---------------------------------------------------------------------------
# rule 1
# ---------------------------------------------------------------------------

def test_no_line_number_citations(docs, tracked_paths):
    """A ``file.py:NNN`` citation into our own tree is stale on the next edit."""
    offenders = []
    for doc in docs:
        if doc in LINE_CITATION_ALLOWLIST:
            continue
        text = open(os.path.join(REPO, doc), errors='replace').read()
        for lineno, line in enumerate(_strip_code_blocks(text), 1):
            if any(r.search(line) for r in _PASTED_OUTPUT_RES):
                continue
            for path, num in _citations(line):
                if _is_ours(path, tracked_paths):
                    offenders.append(f'{doc}:{lineno}: {path}:{num}')
            for groups in _SYMBOL_CITATION_RE.findall(line):
                symbol, num = [g for g in groups if g][:2]
                offenders.append(f'{doc}:{lineno}: `{symbol}:{num}` '
                                 '(line citation hung off a symbol name)')
    assert not offenders, (
        'documentation must not cite our code by line number (they go stale '
        'silently -- write `module.py::symbol` or the bare symbol instead; '
        'fence pasted tracebacks):\n  ' + '\n  '.join(offenders))


def test_the_citation_guard_actually_fires(tmp_path, tracked_paths):
    """Self-test: every surface form the guard claims to catch, and the exemptions."""
    caught = [
        'see `cataloging.py:761` for the veto',
        'see `cataloging.py` line 761',
        'cataloging.py, lines 761',
        'cataloging.py#L761',
        'L761 of cataloging.py',
        'line 761 in cataloging.py',
        'the runner submit_cataloging_chain.sh:44 sets it',
    ]
    caught_symbol_form = ['buffer (`compute_adaptive_mask_buffer:610`), so',
                          'defaults (`accept_satstar_fit`:1558)']
    ignored = [
        'upstream stdatamodels/jwst/datamodels/util.py:77 raises',
        'https://github.com/spacetelescope/gwcs/blob/master/gwcs/wcs.py:412',
        '  File "cataloging.py", line 761, in _filter_extended_emission',
        'FAILED tests/test_reduction_scripts.py:44 - boom',
        'cataloging.py:761: AssertionError',
        'see `cataloging.py::_filter_extended_emission`',
        'Table 2:15 lists the filters',
    ]
    missed = [t for t in caught
              if not any(_is_ours(p, tracked_paths) for p, _ in _citations(t))]
    false_pos = []
    for t in ignored:
        if any(r.search(t) for r in _PASTED_OUTPUT_RES):
            continue
        if any(_is_ours(p, tracked_paths) for p, _ in _citations(t)):
            false_pos.append(t)
    missed_symbol = [t for t in caught_symbol_form
                     if not _SYMBOL_CITATION_RE.search(t)]
    assert not missed, f'guard does not catch: {missed}'
    assert not missed_symbol, f'guard does not catch symbol:NNN form: {missed_symbol}'
    assert not false_pos, f'guard falsely flags: {false_pos}'


def test_fenced_blocks_are_stripped_not_the_prose():
    text = '```\nfake.py:1\n```\nreal prose cataloging.py:2\n~~~\nfake.py:3\n~~~\n'
    kept = [ln for ln in _strip_code_blocks(text) if ln]
    assert kept == ['real prose cataloging.py:2'], kept


# ---------------------------------------------------------------------------
# rule 2
# ---------------------------------------------------------------------------

def test_symbol_references_resolve(docs, tracked_paths, defined_symbols):
    """``module.py::symbol`` -- the format rule 1 mandates -- must be real."""
    paths, basenames = tracked_paths
    missing = []
    for doc in docs:
        for lineno, line in enumerate(
                open(os.path.join(REPO, doc), errors='replace').read().splitlines(), 1):
            for path, symbol in _SYMBOL_REF_RE.findall(line):
                if not _is_ours(path, tracked_paths):
                    continue  # a foreign repo's file; nothing to resolve against
                candidates = ([path] if path in paths
                              else basenames.get(os.path.basename(path), []))
                leaf = symbol.split('.')[-1]
                if not any(leaf in defined_symbols.get(c, ()) for c in candidates):
                    missing.append(f'{doc}:{lineno}: {path}::{symbol}')
    assert not missing, (
        'docs name `file.py::symbol` pairs where the symbol is not defined in '
        'that file:\n  ' + '\n  '.join(missing))


# ---------------------------------------------------------------------------
# rule 3
# ---------------------------------------------------------------------------

def test_dotted_module_paths_resolve(docs, defined_symbols):
    """``jwst_gc_pipeline.x.y`` must be a real module, or a symbol in one."""
    py_files = set(defined_symbols)
    missing = []
    for doc in docs:
        if doc in LINE_CITATION_ALLOWLIST:
            continue  # historical: legitimately names deleted modules
        text = open(os.path.join(REPO, doc), errors='replace').read()
        for lineno, line in enumerate(_strip_code_blocks(text), 1):
            for dotted in _MODULE_RE.findall(line):
                parts = dotted.split('.')
                if any('/'.join(parts) + s in py_files
                       for s in ('.py', '/__init__.py')):
                    continue
                parent = ['/'.join(parts[:-1]) + s for s in ('.py', '/__init__.py')]
                if any(parts[-1] in defined_symbols.get(p, ()) for p in parent):
                    continue
                missing.append(f'{doc}:{lineno}: {dotted}')
    assert not missing, (
        'docs name jwst_gc_pipeline paths that do not resolve (module missing, '
        'or the trailing symbol is not in the module named):\n  '
        + '\n  '.join(missing))


# ---------------------------------------------------------------------------
# rule 4a -- the alignment registry table
# ---------------------------------------------------------------------------

def _markdown_rows(path, start_heading, end_heading):
    text = open(os.path.join(REPO, path), errors='replace').read()
    for heading in (start_heading, end_heading):
        if heading not in text:
            pytest.fail(f'{path}: heading {heading!r} not found -- this guard '
                        f'needs updating along with the doc')
    seg = text[text.index(start_heading):text.index(end_heading)]
    rows = []
    for line in seg.splitlines():
        line = line.strip()
        if not line.startswith('|'):
            continue
        cells = [c.strip() for c in line.strip('|').split('|')]
        if set(''.join(cells)) <= set('-: '):
            continue  # separator
        rows.append(cells)
    return rows


def _plain(cell):
    return cell.replace('**', '').replace('`', '').strip()


_SOURCE_TOKEN = {'locked': 'TABLE_LOCKED', 'consensus': 'TABLE_CONSENSUS',
                 'recorded_bulk': 'RECORDED_BULK'}


def test_alignment_config_table_matches_code():
    """Every column of the field table must match ALIGNMENT_CONFIG.

    Counting rows is not enough: an earlier version of this test did exactly
    that, and a table with a deleted quintuplet row, an invented 1182/007 row
    and four falsified reference frames passed it.
    """
    from jwst_gc_pipeline.reduction import alignment_config as ac

    rows = _markdown_rows('jwst_gc_pipeline/reduction/ASTROMETRY_WCS_CORRECTION_FLOW.md',
                          '### The configured fields',
                          '### How a locked-table row is selected')
    documented = set()
    for cells in rows:
        if not re.fullmatch(r'\d{4}', _plain(cells[0])):
            continue  # header or legend row
        assert len(cells) >= 5, f'field-table row has {len(cells)} cells: {cells}'
        obs = ','.join(sorted(re.findall(r'\d{3}', cells[1]))) or _plain(cells[1]).lower()
        source = _plain(cells[3]).lower().replace(' ', '')
        documented.add((_plain(cells[0]), obs, _plain(cells[2]).lower(), source,
                        _plain(cells[4]).upper()))

    expected = set()
    for entry in ac.ALIGNMENT_CONFIG:
        obs = ','.join(sorted(entry.fields)) if entry.fields else 'all'
        source = _SOURCE_TOKEN[entry.source]
        if entry.consensus_jitter and entry.source == ac.RECORDED_BULK:
            source += '+jitter'
        expected.add((entry.proposal, obs, entry.reference_frame.lower(),
                      source.lower(), (entry.reference_filter or '—').upper()))

    assert documented == expected, (
        'the field table in ASTROMETRY_WCS_CORRECTION_FLOW.md disagrees with '
        'ALIGNMENT_CONFIG.\n  in code, not documented: '
        f'{sorted(expected - documented)}\n  in the doc, not in code: '
        f'{sorted(documented - expected)}')


# ---------------------------------------------------------------------------
# rule 4b -- the saturation floor table
# ---------------------------------------------------------------------------

def _module_level_dict(relpath, name):
    """Read a module-level dict literal out of the AST (no import)."""
    tree = ast.parse(open(os.path.join(REPO, relpath), errors='replace').read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    pytest.fail(f'{relpath}: no module-level dict named {name}')


def test_saturation_floor_table_matches_code():
    """The per-filter floor table must match the code dicts, row by row.

    Both floors decide which pixels are masked and which sources are vetoed, and
    the table has been wrong twice: it grouped F182M/F210M at the wrong severity
    floor, and F187N (8000, the highest) at 5000.  An earlier version of this
    test accepted a filter merely *mentioned* in prose as documented; every
    ``|`` row in the table block must now parse.
    """
    src = 'jwst_gc_pipeline/reduction/saturated_star_finding.py'
    finder_floor = _module_level_dict(src, '_SATSTAR_DATA_FLOOR')
    severity_floor = _module_level_dict(src, 'SAT_SEVERITY_FLOOR')

    doc = 'SATURATED_PIXEL_HANDLING.md'
    text = open(os.path.join(REPO, doc), errors='replace').read()
    head = '| filters | finder wing floor'
    if head not in text:
        pytest.fail(f'{doc}: floor-table header not found -- update this guard')
    seg = text[text.index(head):]
    seg = seg[:seg.index('\n\n')]

    def _num(cell):
        m = re.search(r'(\d[\d,]*)', cell.replace('**', ''))
        return float(m.group(1).replace(',', '')) if m else 0.0

    mismatches, unparsed, documented_filters, checked = [], [], set(), 0
    for line in seg.splitlines():
        line = line.strip()
        if not line.startswith('|'):
            continue
        cells = [c.strip() for c in line.strip('|').split('|')]
        if set(''.join(cells)) <= set('-: ') or cells[0].startswith('filters'):
            continue
        filts = re.findall(r'F\d+[A-Z]', cells[0])
        if len(cells) != 3 or (not filts and 'else' not in cells[0].lower()):
            unparsed.append(line)
            continue
        if not filts:
            continue  # the explicit "anything else" row
        want = (_num(cells[1]), _num(cells[2]))
        for filt in filts:
            checked += 1
            documented_filters.add(filt.lower())
            got = (finder_floor.get(filt.lower(), 0.0),
                   severity_floor.get(filt.lower(), 0.0))
            if got != want:
                mismatches.append(
                    f'{filt}: doc (finder={want[0]}, severity={want[1]}) '
                    f'vs code (finder={got[0]}, severity={got[1]})')

    unlisted = sorted((set(severity_floor) | set(finder_floor)) - documented_filters)
    assert checked, 'parsed no filter rows out of the floor table'
    assert not unparsed, (
        'unparseable row(s) in the floor table -- a filter hidden in prose or a '
        'malformed row would otherwise pass unchecked:\n  ' + '\n  '.join(unparsed))
    assert not mismatches, ('floor table disagrees with the code:\n  '
                            + '\n  '.join(mismatches))
    assert not unlisted, (
        f'filters with a code floor but no row in the doc table: {unlisted}')


# ---------------------------------------------------------------------------
# allowlist hygiene
# ---------------------------------------------------------------------------

def test_allowlist_is_earned(tracked_paths):
    """Every exemption must be a tracked file that still needs exempting.

    An entry with no citations left is silently un-guardable forever, so it must
    be removed rather than kept "just in case".
    """
    paths, _ = tracked_paths
    problems = []
    for doc in sorted(LINE_CITATION_ALLOWLIST):
        if doc not in paths:
            problems.append(f'{doc}: not a tracked file')
            continue
        text = open(os.path.join(REPO, doc), errors='replace').read()
        n = sum(len(list(_citations(line))) for line in _strip_code_blocks(text))
        if not n:
            problems.append(f'{doc}: no line citations left -- drop the exemption')
        if not re.search(r'(as[- ]of|HISTORICAL|not maintained|snapshot)', text[:2000], re.I):
            problems.append(f'{doc}: no "as-of/historical" banner near the top')
    assert not problems, 'LINE_CITATION_ALLOWLIST has rotted:\n  ' + '\n  '.join(problems)


# ---------------------------------------------------------------------------
# rule 4c -- documented photometry defaults vs MANUAL_DEFAULTS
# ---------------------------------------------------------------------------

def _flag_to_dest():
    """``{'--manual-x': 'manual_x'}`` from the parser source (no import)."""
    src = os.path.join(REPO, 'jwst_gc_pipeline/photometry/crowdsource_catalogs_long.py')
    tree = ast.parse(open(src, errors='replace').read())
    out = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or getattr(node.func, 'attr', None) != 'add_option':
            continue
        kw = {k.arg: k.value for k in node.keywords}
        dest = kw.get('dest')
        if not isinstance(dest, ast.Constant):
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and str(arg.value).startswith('--'):
                out[arg.value] = dest.value
    return out


def _as_number(cell):
    cell = cell.replace('**', '').replace('`', '').strip()
    cell = cell.replace('−', '-').replace('–', '-')
    m = re.fullmatch(r'([+-]?\d+(?:\.\d+)?)', cell)
    return float(m.group(1)) if m else None


def test_documented_photometry_defaults_match_manual_defaults():
    """Every Table-B default cell that names a MANUAL_DEFAULTS knob must match.

    The 2026-07-30 audit compared only numeric cells in flag-keyed rows and
    reported "0 mismatches"; the daofind window was documented as +-0.3 while
    MANUAL_DEFAULTS had carried -+1.0 since 2026-07-07, in a PHASE-keyed table the
    checker never looked at.  Both shapes are covered here.
    """
    from jwst_gc_pipeline.photometry.manual_defaults import MANUAL_DEFAULTS

    flag_to_dest = _flag_to_dest()
    doc = 'PHOTOMETRY_PIPELINE.md'
    text = open(os.path.join(REPO, doc), errors='replace').read()

    mismatches, checked = [], 0
    for lineno, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line.startswith('|'):
            continue
        cells = [c.strip() for c in line.strip('|').split('|')]
        m = re.match(r'^`?(--[\w\-]+)`?', cells[0])
        if not m or len(cells) < 2:
            continue
        dest = flag_to_dest.get(m.group(1))
        if dest not in MANUAL_DEFAULTS:
            continue
        want = MANUAL_DEFAULTS[dest]
        if not isinstance(want, (int, float)) or isinstance(want, bool):
            continue
        # A row may document a pair ("-1.0 / 1.0") for two knobs; accept a match
        # against any number in the default cell.
        nums = [_as_number(tok) for tok in re.split(r'[/,]', cells[1])]
        nums = [n for n in nums if n is not None]
        if not nums:
            continue
        checked += 1
        if not any(abs(n - float(want)) < 1e-9 for n in nums):
            mismatches.append(f'{doc}:{lineno}: {m.group(1)} doc={cells[1]!r} '
                              f'code={want} (MANUAL_DEFAULTS[{dest!r}])')

    assert checked, 'parsed no flag-keyed default rows out of Table B'
    assert not mismatches, ('documented defaults disagree with MANUAL_DEFAULTS:\n  '
                            + '\n  '.join(mismatches))


def test_daofind_phase_table_matches_code():
    """The PHASE-keyed daofind table (m1 / m2 / m3..m7) must match the code.

    m1's window is hard-coded in ``cataloging.py``; m2+ come from
    MANUAL_DEFAULTS.  This is the table that carried the wrong roundness window.
    """
    from jwst_gc_pipeline.photometry.manual_defaults import MANUAL_DEFAULTS

    text = open(os.path.join(REPO, 'PHOTOMETRY_PIPELINE.md'), errors='replace').read()
    head = '| pass | round lo/hi | sharp lo/hi |'
    if head not in text:
        pytest.fail('daofind phase table not found -- update this guard')
    seg = text[text.index(head):]
    seg = seg[:seg.index('\n\n')]

    rows = {}
    for line in seg.splitlines():
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if len(cells) < 4 or cells[0].startswith('pass') or set(''.join(cells)) <= set('-: '):
            continue
        def pair(cell):
            vals = [_as_number(t) for t in re.split(r'[/,]', cell)]
            return [v for v in vals if v is not None]
        rows[cells[0]] = (pair(cells[1]), pair(cells[2]), pair(cells[3]))

    m1 = next((v for k, v in rows.items() if k.startswith('m1')), None)
    assert m1, f'no m1 row parsed from {sorted(rows)}'
    assert m1[0] == [-1.0, 1.0], f'm1 roundness documented {m1[0]}, code hard-codes [-1.0, 1.0]'
    assert m1[1] == [0.30, 1.40], f'm1 sharpness documented {m1[1]}, code hard-codes [0.30, 1.40]'

    want_round = [MANUAL_DEFAULTS['manual_resid_roundlo'], MANUAL_DEFAULTS['manual_resid_roundhi']]
    want_sharp = [MANUAL_DEFAULTS['manual_resid_sharplo'], MANUAL_DEFAULTS['manual_resid_sharphi']]
    want_snr = MANUAL_DEFAULTS['manual_iter2_local_snr']
    later = {k: v for k, v in rows.items() if not k.startswith('m1')}
    assert later, 'no m2+ rows parsed'
    for name, (rnd, shp, snr) in later.items():
        assert rnd == want_round, f'{name}: roundness doc={rnd} code={want_round}'
        assert shp == want_sharp, f'{name}: sharpness doc={shp} code={want_sharp}'
        assert snr == [want_snr], f'{name}: local-S/N doc={snr} code=[{want_snr}]'


def test_no_leaked_tool_markup_in_docs(docs):
    """`</content>` / `</invoke>` pasted into a doc is a broken artifact."""
    offenders = []
    for doc in docs:
        for lineno, line in enumerate(
                open(os.path.join(REPO, doc), errors='replace').read().splitlines(), 1):
            if re.search(r'</?(?:content|invoke|antml:\w+)\b', line):
                offenders.append(f'{doc}:{lineno}: {line.strip()[:60]}')
    assert not offenders, 'leaked tool-call markup in docs:\n  ' + '\n  '.join(offenders)
