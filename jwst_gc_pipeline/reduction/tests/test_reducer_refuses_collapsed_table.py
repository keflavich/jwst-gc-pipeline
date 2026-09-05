"""The reducer must STOP on a collapsed offsets table, not bake it in (#705).

The brick-1182 curation overwrote visit-001's offset with visit-002's -- both
~+1.9" for a visit truly ~20" off -- and the collapsed table is the one that
reached ``fix_alignment``, so half the F200W mosaic stayed ~20" out of place.
``flag_collapsed_visits`` names that signature, and every writer of a locked
table already refuses on it; the APPLY path warned and shifted the pixels
anyway.

These tests pin the refusal at the apply path, and pin that it still lets a
correctly-built per-visit table through.
"""
import os

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from jwst_gc_pipeline.reduction import unified_alignment as ua
from jwst_gc_pipeline.reduction.validate_offsets_table import (
    BroadcastProvenanceError, CollapsedOffsetsTableError)

#: A configured TABLE_LOCKED field, so ``resolve_shift`` dispatches to the
#: locked reader rather than to consensus.
PROPOSAL, FIELD = '1182', '004'
FILTER = 'F200W'


@pytest.fixture(autouse=True)
def _clear_validate_memo():
    ua._VALIDATED_OFFSETS_TABLES.clear()
    yield
    ua._VALIDATED_OFFSETS_TABLES.clear()


def _rows(dra_by_visit):
    """A minimal per-visit locked table: one row per visit, one filter."""
    visits = sorted(dra_by_visit)
    return Table({
        'Visit': visits,
        'Vgroup': [2101] * len(visits),
        'Exposure': [1] * len(visits),
        'Filter': [FILTER] * len(visits),
        'Module': ['nrcb'] * len(visits),
        'dra': [dra_by_visit[v][0] for v in visits],
        'ddec': [dra_by_visit[v][1] for v in visits],
        'dra (arcsec)': [dra_by_visit[v][0] for v in visits],
        'ddec (arcsec)': [dra_by_visit[v][1] for v in visits],
    })


def _write_table(tmp_path, tbl):
    offsets = tmp_path / 'offsets'
    offsets.mkdir(exist_ok=True)
    path = offsets / f'Offsets_JWST_Brick{PROPOSAL}_VIRAC2locked.csv'
    tbl.write(str(path), overwrite=True)
    return str(path)


def _frame(tmp_path, visit):
    """A frame whose basename the locked reader can parse to ``visit``."""
    path = tmp_path / f'{visit}_02101_00001_nrcb1_cal.fits'
    hdu1 = fits.ImageHDU(np.zeros((4, 4)), name='SCI')
    fits.HDUList([fits.PrimaryHDU(), hdu1]).writeto(path, overwrite=True)
    return str(path)


def _resolve(tmp_path, visit):
    return ua.resolve_shift(_frame(tmp_path, visit), PROPOSAL, FIELD, FILTER,
                            'nrcb', str(tmp_path) + '/')


# The collapse signature: two DISTINCT visits carrying one value.  The numbers
# are brick-1182's -- v001's real tie was -17.5", and curation gave it v002's.
COLLAPSED = {'jw01182004001': (+1.9021, -0.4402),
             'jw01182004002': (+1.9024, -0.4400)}

# A correctly built table: the same two visits, tied independently.
DISTINCT = {'jw01182004001': (-17.5031, +10.2277),
            'jw01182004002': (+1.9024, -0.4400)}


def test_reducer_refuses_a_collapsed_table(tmp_path):
    _write_table(tmp_path, _rows(COLLAPSED))
    with pytest.raises(CollapsedOffsetsTableError) as ex:
        _resolve(tmp_path, 'jw01182004001')
    assert 'COLLAPSED OFFSETS TABLE' in str(ex.value)
    assert 'Offsets_JWST_Brick1182_VIRAC2locked.csv' in str(ex.value)


def test_a_correctly_built_table_still_applies(tmp_path):
    _write_table(tmp_path, _rows(DISTINCT))
    shift = _resolve(tmp_path, 'jw01182004001')
    assert shift.source == ua.TABLE_LOCKED
    assert shift.total_ra == pytest.approx(-17.5031)
    assert shift.total_dec == pytest.approx(+10.2277)


def test_the_refusal_is_not_spent_by_the_first_frame(tmp_path):
    """The memo reports the WARNING once; it must not swallow the STOP.

    ``fix_alignment`` is called once per exposure with the same table.  If the
    table is recorded as validated before the check runs, only the first frame
    is checked and every later one is shifted by the collapsed value with no
    check at all -- the fail-open ``_check_generation`` already fixed for
    GENLOCK_STRICT.
    """
    _write_table(tmp_path, _rows(COLLAPSED))
    for _ in range(3):
        with pytest.raises(CollapsedOffsetsTableError):
            _resolve(tmp_path, 'jw01182004001')


def test_broadcast_provenance_also_stops_the_reducer(tmp_path):
    """One correction smeared over every visit changes the applied shift too.

    gc2211 (#284): five observations 0.3-17.6 arcmin apart all carried one
    ``prov_*`` pair, so the correction landed on all five products.  It rides
    ``raise_on_issue`` for the same reason the collapse does.
    """
    tbl = _rows(DISTINCT)
    tbl['prov_dra_onsky_mas'] = [-2470.1, -2470.1]
    tbl['prov_ddec_onsky_mas'] = [+2825.4, +2825.4]
    _write_table(tmp_path, tbl)
    with pytest.raises(BroadcastProvenanceError):
        _resolve(tmp_path, 'jw01182004001')


def test_a_broken_audit_trail_alone_does_not_stop_the_reducer(tmp_path):
    """Divergence keeps its own switch, and stays a warning here.

    ``dra (arcsec)`` is what the reducer applies and it is still
    self-consistent; what a divergence costs is the record of how it got there.
    Escalating it on the shared switch would stop reductions over an audit
    trail.
    """
    tbl = _rows(DISTINCT)
    # `(arcsec)` moved and the plain pair did not, with no provenance saying so.
    tbl['dra (arcsec)'] = [d + 0.030 for d in tbl['dra']]
    tbl['prov_dra_onsky_mas'] = [0.0, 0.0]
    tbl['prov_ddec_onsky_mas'] = [0.0, 0.0]
    _write_table(tmp_path, tbl)
    with pytest.warns(UserWarning, match='OFFSETS TABLE PROVENANCE BROKEN'):
        shift = _resolve(tmp_path, 'jw01182004001')
    assert shift.total_ra == pytest.approx(-17.5031 + 0.030)


def test_every_live_locked_table_still_passes(tmp_path):
    """The gate stops no field that is currently correct.

    Guards against a refusal that is right in principle and unshippable in
    practice five days before the 10678 window.
    """
    from glob import glob
    from jwst_gc_pipeline.reduction.validate_offsets_table import (
        flag_broadcast_provenance, flag_collapsed_visits)
    live = sorted(glob('/orange/adamginsburg/jwst/*/offsets/'
                       'Offsets_JWST_Brick*_VIRAC2locked.csv'))
    if not live:
        pytest.skip('live offsets tables not mounted')
    for path in live:
        t = Table.read(path)
        assert flag_collapsed_visits(t) == [], os.path.basename(path)
        assert flag_broadcast_provenance(t) == [], os.path.basename(path)


# ---------------------------------------------------------------------------
# The docs that describe this call site (review of PR #770, B1)
# ---------------------------------------------------------------------------
#
# Both docs described the apply path as warning by default and raising only
# under ``OFFSETS_TABLE_COLLAPSE_RAISE=1``.  Once this call site passes
# ``raise_on_issue=True`` that reads backwards, and an operator who trusted it
# would set an env var expecting a behaviour change and get none.  Each test
# below pins BOTH sides -- the prose and the code it describes -- so neither
# can drift alone, following
# ``photometry/tests/test_astrometry_docs_match_code.py``.

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
AUDIT_MD = os.path.join(REPO, 'jwst_gc_pipeline', 'reduction',
                        'ASTROMETRY_REDUNDANCY_AUDIT.md')
CHECKPOINTS_MD = os.path.join(REPO, 'jwst_gc_pipeline', 'photometry',
                              'ASTROMETRY_CHECKPOINTS.md')


def _production_sane_calls():
    """Every non-test ``assert_offsets_table_sane(...)`` call in the package.

    Returns ``{"<module>:<lineno>": raises_on_issue}``.  Read with ``ast`` so a
    call spanning lines, or one whose keyword moves, is still seen.
    """
    import ast
    from glob import glob
    out = {}
    for path in glob(os.path.join(REPO, 'jwst_gc_pipeline', '**', '*.py'),
                     recursive=True):
        if os.sep + 'tests' + os.sep in path:
            continue
        with open(path) as fh:
            src = fh.read()
        if 'assert_offsets_table_sane' not in src:
            continue
        for node in ast.walk(ast.parse(src)):
            if not isinstance(node, ast.Call):
                continue
            fname = getattr(node.func, 'id', None) or getattr(
                node.func, 'attr', None)
            if fname != 'assert_offsets_table_sane':
                continue
            raises = any(
                kw.arg == 'raise_on_issue'
                and isinstance(kw.value, ast.Constant) and kw.value.value is True
                for kw in node.keywords)
            key = f'{os.path.relpath(path, REPO)}:{node.lineno}'
            out[key] = raises
    return out


def test_the_apply_path_call_site_raises():
    """The code side of both doc claims below."""
    calls = _production_sane_calls()
    apply_calls = {k: v for k, v in calls.items()
                   if 'unified_alignment.py' in k}
    assert apply_calls, calls
    assert all(apply_calls.values()), (
        f'the reducer apply path no longer raises on a collapse: {apply_calls}')


def test_the_audit_doc_does_not_call_the_apply_path_warn_only():
    with open(AUDIT_MD) as fh:
        section = fh.read().split('## Collapse safeguard')[1].split('\n## ')[0]
    flat = ' '.join(section.split())
    assert 'fires a warning by default' not in flat, (
        'ASTROMETRY_REDUNDANCY_AUDIT.md still describes the fix_alignment '
        'apply path as warning by default; it raises (PR #770)')
    assert 'RAISES' in section or 'raises `CollapsedOffsetsTableError`' in flat


def test_the_env_switch_row_matches_whether_any_caller_still_needs_it():
    """``OFFSETS_TABLE_COLLAPSE_RAISE`` is a no-op exactly when every
    production caller already passes ``raise_on_issue=True``.

    Pinned as an equivalence, so reverting EITHER side turns this red: revert
    the code and the doc's "no-op" becomes a lie; revert the doc and a switch
    that changes nothing is still advertised as changing something.
    """
    with open(CHECKPOINTS_MD) as fh:
        table = fh.read().split('## Environment switches')[1]
    row = [ln for ln in table.splitlines()
           if ln.startswith('| `OFFSETS_TABLE_COLLAPSE_RAISE')]
    assert len(row) == 1, table
    doc_says_noop = 'no-op' in row[0].lower()
    calls = _production_sane_calls()
    assert calls, 'no production call site found'
    code_is_noop = all(calls.values())
    assert doc_says_noop == code_is_noop, (
        f'env-table row says no-op={doc_says_noop} while the production call '
        f'sites give no-op={code_is_noop}: {calls}')
