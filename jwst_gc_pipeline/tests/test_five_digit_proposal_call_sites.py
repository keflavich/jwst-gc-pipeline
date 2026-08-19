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
respelled with the old 4-digit-only prefix -- as concatenation, as
percent-formatting, or as an f-string -- fails here: the first two leave no
f-string for the anchor to find, and the third renders ``jw010678``.

m1 frame discovery is pinned end to end instead, on a tmp tree, because
``get_filenames`` is callable.

Vocabulary.  *m1* is the first merge stage of cataloging, the one that globs
the per-exposure ``crf`` frames a reduce produced and starts the photometry
from them; *m2* is the second, where per-exposure astrometry is re-verified.
A *crf* is the per-exposure calibrated frame ``Image3`` writes after outlier
detection, and an *asn* is the association file listing the frames a stage
consumes.  An *uncal* is the raw ramp product MAST serves.  The *MAST URI
filter* is the substring test the reduce applies to each ``dataURI`` a MAST
query returns, to keep only this proposal-and-observation's uncals.  A JWST
*visit token* is ``jw`` + proposal(5) + observation(3) + visit(3).
"""
import ast
import re
import subprocess
import warnings
from pathlib import Path

import pytest

from jwst_gc_pipeline import fields as FIELDS
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
#: the PREFIX, so a placeholder in every other position is enough.
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
# structural: no f-string anywhere in the repo glues `jw` to an unpadded value
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The two files that build the prefix from an already-five-digit value, with
#: the reason each is exempt.  ``test_the_structural_allowlist_is_live`` fails
#: if either stops carrying such an f-string, so an entry cannot outlive its
#: site.
STRUCTURAL_ALLOWLIST = {
    # the helper itself: the one place the pad is spelled out
    'jwst_gc_pipeline/mast_names.py':
        'defines jw_prefix; the pad is written here',
    # rebuilds the visit token out of `(?P<prop>\d{5})` matched off a filename
    # already on disk, so the five digits come back exactly as MAST wrote them
    'scripts/reduction/expand_offsets_granularity.py':
        'round-trips a 5-digit regex group off an existing product name',
}

#: The format spec that pads to the same five digits ``jw_prefix`` produces.
FIVE_DIGIT_SPEC = '05d'

#: A module-level constant already holding the padded proposal.
FIVE_DIGIT_CONSTANT_RE = re.compile(r'\A[0-9]{5}\Z')


def _joined_strings(tree):
    return [n for n in ast.walk(tree) if isinstance(n, ast.JoinedStr)]


def _five_digit_constants(tree):
    """Module-level names bound to a five-digit string literal.

    ``scripts/carta/make_sickle_snippets.py`` sets ``PROG = '03958'`` and
    spells ``f'jw{PROG}-...'``; the value carries the pad already, so the site
    is right and the check follows the name to see it.
    """
    names = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not isinstance(node.value, ast.Constant):
            continue
        if not isinstance(node.value.value, str):
            continue
        if FIVE_DIGIT_CONSTANT_RE.match(node.value.value) is None:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                names.add(target.id)
    return names


def _format_spec_text(node):
    """The literal text of a ``FormattedValue`` format spec.

    The spec is itself a ``JoinedStr``, so a constant one (``:05d``) is a
    single ``Constant`` child.  A computed spec (``f'{width}d'``) has an
    interpolation among its children and returns ``''``, so it never reads as
    the five-digit pad.
    """
    if node.format_spec is None:
        return ''
    parts = getattr(node.format_spec, 'values', [])
    if len(parts) == 1 and isinstance(parts[0], ast.Constant):
        return str(parts[0].value)
    return ''


def jw_glue_offenders(source, label='<source>'):
    """Sites where an f-string glues a value onto ``jw`` with no five-digit pad.

    Three spellings carry the pad and are accepted: a ``jw_prefix(...)`` call,
    an explicit ``:05d`` format spec, and a name bound at module level to a
    five-digit string literal.  ``jw_prefix`` is the one to reach for -- it
    also validates its argument, where ``f'jw{pid:05d}'`` renders ``jw100000``
    for a six-digit input -- and the other two cover the sites that cannot
    import it (a stdlib-only script) or already hold the padded token.  Every
    other spelling leaves the width to the value, so it renders ``jw2221``
    where MAST wrote ``jw02221`` and ``jw010678`` where MAST wrote ``jw10678``.
    """
    with warnings.catch_warnings():
        # plotting/plot_tools.py:1083 carries LaTeX ($\mu$, $\sigma$) in a
        # non-raw f-string, which makes ast.parse warn.  Pre-existing and
        # unrelated to naming; silenced so this test adds no noise.
        warnings.simplefilter('ignore', SyntaxWarning)
        tree = ast.parse(source)
    padded_names = _five_digit_constants(tree)
    offenders = []
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
            if expr.startswith('jw_prefix('):
                continue
            if _format_spec_text(nxt) == FIVE_DIGIT_SPEC:
                continue
            if isinstance(nxt.value, ast.Name) and nxt.value.id in padded_names:
                continue
            offenders.append(f'{label}:{node.lineno}: jw{{{expr}}}')
    return offenders


def _scanned_sources():
    """Every git-tracked production ``.py`` in the repo, tests excluded.

    Repo-wide rather than package-wide: this PR respelled 10 call sites under
    ``scripts/``, and a scan that stopped at ``jwst_gc_pipeline/`` left all of
    them free to drift back.
    """
    out = subprocess.run(['git', '-C', str(REPO_ROOT), 'ls-files'],
                         capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        rel = Path(line)
        if rel.suffix != '.py':
            continue
        if '/tests/' in line or rel.name.startswith('test_'):
            continue
        path = REPO_ROOT / rel
        if not path.is_file():
            continue
        yield line, path


def test_the_structural_scan_reaches_the_scripts_tree():
    """A scan that stops at the package boundary passes over ten of the sites
    this PR swept.  Pin the count and one file from each tree."""
    scanned = {rel for rel, _ in _scanned_sources()}
    assert len(scanned) >= 180, len(scanned)
    for expected in ('jwst_gc_pipeline/photometry/cataloging.py',
                     'jwst_gc_pipeline/reduction/PipelineMIRI.py',
                     'scripts/reduction/preflight_reduce_inputs.py',
                     'scripts/miri_reduction/miri_tile_homogenize.py',
                     'scripts/carta/make_sickle_snippets.py'):
        assert expected in scanned, expected


def test_every_product_name_fstring_interpolates_the_helper():
    """Structural companion to the value pins: across the whole repo, an
    f-string whose literal text runs up to ``jw`` and then interpolates has to
    carry the five-digit pad.  This is the shape the sweep produced, and it
    covers the sites no anchor above names."""
    offenders = []
    for rel, path in _scanned_sources():
        if rel in STRUCTURAL_ALLOWLIST:
            continue
        offenders += jw_glue_offenders(path.read_text(), rel)
    assert not offenders, (
        "f-string(s) gluing a value onto 'jw' with no five-digit pad:\n  "
        + "\n  ".join(offenders)
        + "\n\nUse jwst_gc_pipeline.mast_names.jw_prefix(...), which pads and "
          "validates; ':05d' is accepted where the helper cannot be imported.")


def test_the_structural_allowlist_is_live():
    """Every exempt file must still exist and still carry the shape it is
    exempt from, so an entry cannot outlive its site."""
    for rel, reason in STRUCTURAL_ALLOWLIST.items():
        path = REPO_ROOT / rel
        assert path.is_file(), f'{rel} ({reason}) no longer exists'
        source = path.read_text()
        assert jw_glue_offenders(source, rel) or 'jw_prefix' in source, (
            f'{rel} no longer glues anything onto jw; drop its exemption '
            f'({reason})')


@pytest.mark.parametrize('source', [
    "pat = f'jw{proposal_id}-o{field}_asn.json'",
    "pat = f'jw{int(proposal_id)}-o{field}_asn.json'",
    "pat = f'jw{proposal_id:04d}-o{field}_asn.json'",
    "PROG = '3958'\npat = f'jw{PROG}-o{field}_asn.json'",
    "pat = f'{base}/jw{proposal_id}{obs}{visit}_cal.fits'",
])
def test_the_structural_check_rejects_an_unpadded_glue(source):
    assert jw_glue_offenders(source), source


@pytest.mark.parametrize('source', [
    # the sanctioned spelling: the helper renders the whole prefix
    "pat = f'{jw_prefix(proposal_id)}-o{field}_asn.json'",
    # the explicit pad, for a site that cannot import the helper
    "pat = f'jw{proposal_id:05d}-o{field}_asn.json'",
    "pat = f'jw{int(proposal_id):05d}-o{observation_number(field)}'",
    # a name already holding the padded proposal
    "PROG = '03958'\npat = f'jw{PROG}-o{field}_asn.json'",
    # a literal product name: nothing is glued onto 'jw' at all
    "name = f'jw10678001001_{vgroup}_cal.fits'",
])
def test_the_structural_check_accepts_every_five_digit_spelling(source):
    assert not jw_glue_offenders(source), source


def test_the_padded_format_spec_renders_what_the_helper_renders():
    """``:05d`` is accepted because it produces the same prefix.

    Over every proposal in the registry the two spellings agree.  They part on
    an input no JWST filename can hold, which is why ``jw_prefix`` stays the
    one to call: it refuses, and the bare format spec renders a six-digit
    prefix.
    """
    proposals = {o.proposal for f in FIELDS.FIELDS for o in f.observations}
    assert len(proposals) >= 10, sorted(proposals)
    for pid in sorted(proposals):
        assert jw_prefix(pid) == f'jw{int(pid):05d}', pid
    assert f'jw{100000:05d}' == 'jw100000'
    with pytest.raises(ValueError):
        jw_prefix(100000)
