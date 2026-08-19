"""No code path may reach an offsets table tied to a frame we left.

The GC fields moved from VVV/GNS to VIRAC2, but the old tables are still on disk:

    Offsets_JWST_Brick1182_VVV.csv           Offsets_JWST_Brick1182_VVV_average.csv
    Offsets_JWST_Brick2221_VVV.csv           Offsets_JWST_Brick2221_VVV_average.csv
    Offsets_JWST_Brick3958_GNS.csv           Offsets_JWST_Brick2211_GNS_perexp_*.csv
    Offsets_JWST_Brick1182_F200ref_average.csv   (and F405ref, F444ref)

`_astrom_offsets_channel` == 'locked' resolved to `_VIRAC2locked.csv` and, when
that was absent, fell back to `glob('..._*locked.csv')` then
`glob('..._*_average.csv')` with `sorted(...)[0]`.  Those globs match the list
above and the sort is alphabetical, so brick/1182 would have selected
`F200ref_average` -- a table tied to a different frame -- and said nothing.

Nothing is exposed today (every locked field has its VIRAC2locked table), which
is exactly why this needs a test rather than an observation.

The sibling fallback in `unified_alignment` is NOT the same hazard: it builds
`Offsets_JWST_Brick{prop}_{refname}_average.csv` from the field's own declared
frame token, so it can only ever name a table of the frame the field is tied to,
and it raises when no token is supplied.  That is pinned below too, because the
distinction is the whole reason one was removed and the other kept.
"""
import os

import pytest

from jwst_gc_pipeline.photometry import cataloging as _cat


LEGACY = [
    'Offsets_JWST_Brick1182_VVV_average.csv',
    'Offsets_JWST_Brick1182_VVV_average_lockmodules.csv',
    'Offsets_JWST_Brick1182_F200ref_average.csv',
    'Offsets_JWST_Brick1182_F405ref_average.csv',
    'Offsets_JWST_Brick1182_F444ref_average.csv',
    'Offsets_JWST_Brick1182_VIRAC2_average.csv',
]


def _offsets(tmp_path, names):
    d = tmp_path / 'offsets'
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        (d / n).write_text('Visit,Filter,dra (arcsec),ddec (arcsec)\n')
    return str(tmp_path)


def _resolve(base, proposal='1182', field='004'):
    return _cat._astrom_find_offsets_table(base, proposal, field)


def test_the_locked_table_is_used_when_present(tmp_path):
    base = _offsets(tmp_path, LEGACY + ['Offsets_JWST_Brick1182_VIRAC2locked.csv'])
    got = _resolve(base)
    assert got and os.path.basename(got) == 'Offsets_JWST_Brick1182_VIRAC2locked.csv'


def test_a_missing_locked_table_resolves_to_NOTHING(tmp_path):
    """Not to the alphabetically-first legacy table sitting beside it.

    Every name in LEGACY matches one of the two globs the old fallback used, and
    `sorted(...)[0]` picked `F200ref_average` -- a different frame, chosen by
    filename order.
    """
    base = _offsets(tmp_path, LEGACY)
    assert _resolve(base) is None


@pytest.mark.parametrize('name', LEGACY)
def test_no_single_legacy_table_can_be_selected_alone(tmp_path, name):
    """One at a time, so a future sort or glob change cannot make any of them
    reachable without this failing."""
    base = _offsets(tmp_path, [name])
    assert _resolve(base) is None, f'{name} was selected'


def test_a_consensus_field_is_unaffected(tmp_path):
    """The consensus channel names its table exactly and never globbed."""
    d = tmp_path / 'offsets'
    d.mkdir(parents=True)
    (d / 'Offsets_JWST_Brick2045_consensus.csv').write_text('Visit\n')
    got = _cat._astrom_find_offsets_table(str(tmp_path), '2045', '001')
    assert got and os.path.basename(got) == 'Offsets_JWST_Brick2045_consensus.csv'


def test_the_resolver_has_no_glob_left_in_the_locked_branch():
    """A behavioural test cannot see a glob that is added back with a pattern
    matching nothing in the fixtures above."""
    import inspect
    src = inspect.getsource(_cat._astrom_find_offsets_table)
    _, _, locked = src.partition("if channel == 'locked':")
    # `glob.glob(` -- the CALL, not the word, which appears in the comment
    # explaining why the call is gone.
    code = '\n'.join(ln for ln in locked.splitlines()
                     if not ln.lstrip().startswith('#'))
    assert 'glob.glob(' not in code, (
        'the locked branch globs again; a glob here can reach the VVV/GNS-era '
        'tables that are still on disk')


# ---------------------------------------------------------------------------
# the sibling fallback, which is keyed on the FRAME and is therefore safe
# ---------------------------------------------------------------------------

def test_the_legacy_alignment_fallback_is_named_from_the_declared_frame():
    """`unified_alignment` builds `..._{refname}_average.csv` where refname is
    the field's own frame token, so it cannot name a frame the field is not tied
    to.  Every GC proposal's token is VIRAC2; only the non-GC ones are Gaia."""
    from jwst_gc_pipeline import fields
    gc = ('1182', '2092', '2211', '2045', '3958', '4147', '5365', '1939', '10678')
    for prop in gc:
        assert fields.reference_frame(prop) == 'VIRAC2', prop
    for prop in ('6151', '1334', '1979'):
        assert fields.reference_frame(prop) == 'Gaia', prop


def test_no_proposal_declares_a_VVV_or_GNS_frame():
    """The frame move is the premise of removing the fallback; if a proposal
    ever declares VVV or GNS again, the `unified_alignment` path would build a
    VVV/GNS table name legitimately and this file's reasoning stops holding."""
    from jwst_gc_pipeline import fields
    seen = set()
    for prop in ('1182', '2221', '2092', '2211', '2045', '3958', '4147',
                 '5365', '1939', '6151', '1334', '1979', '10678'):
        tok = fields.reference_frame(prop)
        if tok:
            seen.add(tok)
    assert seen <= {'VIRAC2', 'Gaia'}, seen
