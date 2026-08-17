"""The names the pipeline builds for a 5-digit proposal, pinned by VALUE.

``test_mast_names.py`` pins the helper; the grep guard refuses the old literal.
Neither says what a CALL SITE produces, so reverting ``jw_prefix`` to the
4-digit spelling used to leave the sites issue #414 names -- the MAST URI
filter, the association and uncal globs, the association-member provenance
assert, the product name, and m1 frame discovery -- reporting nothing.

Those sites are expressions inside three reduce drivers and the cataloging
module, and each family repeats across all of them, so this pins one test per
FAMILY rather than one per site: the f-string is read out of the driver source
by an anchor that is part of the name it builds, and evaluated with
``proposal_id`` set to a 5-digit proposal and to a 4-digit one.  A site
A site respelled with the old
4-digit-only prefix -- as concatenation, as percent-formatting, or as an
f-string -- fails here: the first two leave no f-string for the anchor to
find, and the third renders ``jw010678``.

m1 frame discovery is pinned end to end instead, on a tmp tree, because
``get_filenames`` is callable.
"""
import ast
import warnings
from pathlib import Path

import pytest

from jwst_gc_pipeline.mast_names import jw_prefix

REDUCTION = Path(__file__).resolve().parents[1] / 'reduction'

DRIVERS = {
    'nircam': REDUCTION / 'PipelineRerunNIRCAM-LONG.py',
    'miri': REDUCTION / 'PipelineMIRI.py',
    'niriss': REDUCTION / 'PipelineRerunNIRISS.py',
}

#: Anchors that identify each family, by a fragment of the name it builds or
#: of the statement it appears in.  Every driver has a site of every family.
FAMILIES = {
    # the substring test applied to each dataURI a MAST query returns, to keep
    # only this proposal-and-observation's raw ramp (uncal) products
    'mast_uri_filter': "in uri",
    # the image3 association the reduce consumes
    'asn_glob': "_image3_*0[0-9][0-9]_asn.json",
    # the whole-asn provenance check on every association member
    'asn_member_provenance': "in member['expname']",
    # the name Image3 writes its mosaic under
    'product_name': "['products'][0]['name'] =",
}

#: What a non-prefix interpolation renders as.  The families here are pinned on
#: the PREFIX, so the rest of the name only has to be present, not real.
PLACEHOLDER = '001'


def _render(node, proposal_id):
    """Render an f-string node with ``jw_prefix(proposal_id)`` evaluated.

    Only the prefix call is evaluated; every other interpolation becomes a
    placeholder, so no name from the driver's own scope is needed and a nested
    ``os.path.join(...)`` cannot derail the render.
    """
    out = []
    for part in node.values:
        if isinstance(part, ast.Constant):
            out.append(str(part.value))
            continue
        expr = ast.unparse(part.value)
        if expr == 'jw_prefix(proposal_id)':
            out.append(jw_prefix(proposal_id))
        else:
            out.append(PLACEHOLDER)
    return ''.join(out)


def _prefix_fstrings_for(path, anchor, proposal_id):
    """Every f-string that STARTS with the proposal prefix, on a line carrying
    ``anchor``, rendered."""
    text = path.read_text()
    lines = text.splitlines()
    found = []
    for node in ast.walk(ast.parse(text)):
        if not isinstance(node, ast.JoinedStr) or not node.values:
            continue
        span = '\n'.join(lines[node.lineno - 1:node.end_lineno])
        if anchor not in span:
            continue
        head = node.values[0]
        if not isinstance(head, ast.FormattedValue):
            continue
        if not ast.unparse(head.value).startswith('jw_prefix('):
            continue
        found.append((span.strip(), _render(node, proposal_id)))
    return found


@pytest.mark.parametrize('driver', sorted(DRIVERS))
@pytest.mark.parametrize('family', sorted(FAMILIES))
def test_each_name_family_carries_the_whole_five_digit_proposal(driver, family):
    """Every name a reduce driver builds for 10678 starts ``jw10678``."""
    hits = _prefix_fstrings_for(DRIVERS[driver], FAMILIES[family], '10678')
    assert hits, (
        f'{DRIVERS[driver].name}: no f-string found for the {family} family '
        f'(anchor {FAMILIES[family]!r}).  Either the site moved, or it was '
        f'respelled as concatenation or %-formatting, which is exactly the '
        f'regression this pins.')
    for source, name in hits:
        assert name.startswith('jw10678'), f'{source}\n  -> {name}'
        assert 'jw010678' not in name, f'{source}\n  -> {name}'


@pytest.mark.parametrize('driver', sorted(DRIVERS))
@pytest.mark.parametrize('family', sorted(FAMILIES))
def test_each_name_family_is_unchanged_for_a_four_digit_proposal(driver, family):
    """The same sites reproduce the historical spelling for 2221, so nothing
    on disk is renamed."""
    for source, name in _prefix_fstrings_for(DRIVERS[driver], FAMILIES[family], '2221'):
        assert name.startswith('jw02221'), f'{source}\n  -> {name}'


def test_the_anchors_still_find_something_in_every_driver():
    """A stale anchor would make every assertion above vacuous."""
    for driver, path in DRIVERS.items():
        for family, anchor in FAMILIES.items():
            assert _prefix_fstrings_for(path, anchor, '2221'), (driver, family)


# ---------------------------------------------------------------------------
# m1 frame discovery, end to end
# ---------------------------------------------------------------------------

def _crf(tmp_path, filtername, name):
    d = tmp_path / filtername / 'pipeline'
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text('')
    return d


def test_m1_frame_discovery_finds_a_five_digit_proposals_frames(tmp_path):
    """``get_filenames`` globs the per-exposure crf files cataloging starts
    from.  Against 10678 the old spelling globbed ``jw010678...`` and raised
    'No matches found', after the reduce had already produced the frames."""
    from jwst_gc_pipeline.photometry.crowdsource_catalogs_long import get_filenames

    name = 'jw10678001001_02101_00001_nrcalong_destreak_o001_crf.fits'
    _crf(tmp_path, 'F212N', name)
    found = get_filenames(str(tmp_path), 'F212N', '10678', '001',
                          'destreak_o001_crf', 'nrcalong')
    assert [Path(f).name for f in found] == [name]


def test_m1_frame_discovery_does_not_accept_the_over_padded_name(tmp_path):
    """The converse: frames written under the wrong prefix are not this
    proposal's frames."""
    from jwst_gc_pipeline.photometry.crowdsource_catalogs_long import get_filenames

    _crf(tmp_path, 'F212N',
         'jw010678001001_02101_00001_nrcalong_destreak_o001_crf.fits')
    with pytest.raises(ValueError, match='No matches found'):
        get_filenames(str(tmp_path), 'F212N', '10678', '001',
                      'destreak_o001_crf', 'nrcalong')


def test_m1_frame_discovery_is_unchanged_for_a_four_digit_proposal(tmp_path):
    from jwst_gc_pipeline.photometry.crowdsource_catalogs_long import get_filenames

    name = 'jw02221001001_02101_00001_nrcalong_destreak_o001_crf.fits'
    _crf(tmp_path, 'F212N', name)
    found = get_filenames(str(tmp_path), 'F212N', '2221', '001',
                          'destreak_o001_crf', 'nrcalong')
    assert [Path(f).name for f in found] == [name]


# ---------------------------------------------------------------------------
# structural: no f-string anywhere in the package builds `jw` + interpolation
# by hand
# ---------------------------------------------------------------------------

PACKAGE = Path(__file__).resolve().parents[1]

#: The helper is where the pad is spelled out; it is the one place `jw` may be
#: glued to a value directly.
STRUCTURAL_ALLOWLIST = {'jwst_gc_pipeline/mast_names.py'}


def _joined_strings(tree):
    return [n for n in ast.walk(tree) if isinstance(n, ast.JoinedStr)]


def test_every_product_name_fstring_interpolates_the_helper():
    """Structural companion to the value pins: in every production module, an
    f-string whose literal text runs up to ``jw`` and then interpolates must
    interpolate ``jw_prefix(...)``.  This is the shape the sweep produced, and
    it covers the sites no anchor above names."""
    offenders = []
    for path in sorted(PACKAGE.rglob('*.py')):
        rel = path.relative_to(PACKAGE.parent).as_posix()
        if '/tests/' in rel or path.name.startswith('test_'):
            continue
        if rel in STRUCTURAL_ALLOWLIST:
            continue
        with warnings.catch_warnings():
            # plotting/plot_tools.py:1083 carries LaTeX ($\mu$, $\sigma$) in a
            # non-raw f-string, which makes ast.parse warn.  Pre-existing and
            # unrelated to naming; silenced so this test adds no noise.
            warnings.simplefilter('ignore', SyntaxWarning)
            tree = ast.parse(path.read_text())
        for node in _joined_strings(tree):
            parts = node.values
            for i, part in enumerate(parts[:-1]):
                if not isinstance(part, ast.Constant):
                    continue
                if not str(part.value).endswith('jw'):
                    continue
                nxt = parts[i + 1]
                if not isinstance(nxt, ast.FormattedValue):
                    continue
                expr = ast.unparse(nxt.value)
                if not expr.startswith('jw_prefix('):
                    offenders.append(f'{rel}:{node.lineno}: jw{{{expr}}}')
    assert not offenders, (
        "f-string(s) gluing a value straight onto 'jw' without jw_prefix:\n  "
        + "\n  ".join(offenders))
