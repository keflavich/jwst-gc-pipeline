"""The registry holds values the pipeline can act on, never tripwire strings.

``fields.yaml`` carried ``reference_frame: THIS_IS_A_BUG_IF_YOU_USE_THIS`` for
proposal 2221.  It was put there to make a wrong path fail loudly: the token
names a legacy offsets-table filename, and for 2221 that filename is the retired
F405ref table.  ``resolve_shift`` does refuse it -- but only on the path that
builds that filename.  Everywhere else the string was an ordinary value, so it
reached ``provenance_header_cards`` and was stamped into released products::

    brick/F410M/pipeline/jw02221001001_07101_00001_nrcalong_destreak_o001_crf.fits
    APROVRF = 'THIS_IS_A_BUG_IF_YOU_USE_THIS'

A tripwire belongs in code, where it stops a run.  In data it becomes the
answer to "what frame is this tied to?" and outlives the run that wrote it.

These tests fail if a placeholder returns to the registry.  The complementary
guard is in ``provenance_header_cards``, which refuses to stamp one.
"""
import pytest
import yaml

from jwst_gc_pipeline import fields as field_registry
from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
    looks_like_placeholder, provenance_header_cards)


def _strings(node, trail=()):
    """Every string value in the loaded YAML, with the path that reaches it."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _strings(value, trail + (str(key),))
    elif isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            yield from _strings(value, trail + (str(index),))
    elif isinstance(node, str):
        yield trail, node


def test_registry_holds_no_placeholder_values():
    with open(field_registry.REGISTRY_PATH) as fh:
        registry = yaml.safe_load(fh)
    offenders = [f"{'.'.join(trail)}: {value!r}"
                 for trail, value in _strings(registry)
                 if looks_like_placeholder(value)]
    assert not offenders, (
        "placeholder value(s) in fields.yaml:\n  " + "\n  ".join(offenders)
        + "\n\nThe registry is read as data and its values are written into "
          "products.  To make a path fail, raise on that path; to say a field "
          "has no value for a key, leave the key out.")


def test_2221_has_no_frame_token():
    """2221's frame comes from alignment_config, so the registry has no token.

    Leaving it out is what stops the legacy filename being built: ``refname is
    None`` raises in ``resolve_shift`` before it can name the retired F405ref
    table.
    """
    assert field_registry.reference_frame('2221') is None


def test_provenance_refuses_a_placeholder_frame():
    with pytest.raises(ValueError, match='placeholder'):
        provenance_header_cards(stage='fix_alignment', dra_onsky_mas=1.0,
                                ddec_onsky_mas=1.0, method='offsets-table',
                                references='THIS_IS_A_BUG_IF_YOU_USE_THIS',
                                table_name='offsets.csv')


def test_provenance_accepts_a_real_frame():
    cards = dict((name, value) for name, value, _ in provenance_header_cards(
        stage='fix_alignment', dra_onsky_mas=1.0, ddec_onsky_mas=1.0,
        method='offsets-table', references='VIRAC2',
        table_name='offsets.csv'))
    assert cards['APROVRF'] == 'VIRAC2'


@pytest.mark.parametrize('value,placeholder', [
    ('THIS_IS_A_BUG_IF_YOU_USE_THIS', True),   # the string this guard exists for
    ('FIXME_MEASURE_THIS', True),
    ('XXX_PLACEHOLDER_XXX', True),
    ('unknown', True),
    ('VIRAC2', False),
    ('Gaia', False),
    ('GNS', False),
    ('n/a', False),
    ('DEBUG', False),                          # one word, not the word BUG
])
def test_what_counts_as_a_placeholder(value, placeholder):
    """Sentinels are written between underscores, so a word-boundary match
    would find none of them: `_` is a word character."""
    assert looks_like_placeholder(value) is placeholder


def test_the_guard_catches_the_string_it_was_written_for():
    """The predicate, run over the registry as it stood when the guard was
    written.  Both 2221 rows have to be caught.

    Inline rather than read from git: a test that reads `origin/main` passes on
    this branch and fails on the branch it creates, the moment the sentinel is
    gone from main -- and skips silently in a shallow checkout, which turns a
    broken assertion into no test at all.
    """
    as_it_was = yaml.safe_load("""
    fields:
      brick:
        observations:
          '2221':
            reference_frame: THIS_IS_A_BUG_IF_YOU_USE_THIS
          '1182':
            reference_frame: VIRAC2
      cloudc:
        observations:
          '2221':
            reference_frame: THIS_IS_A_BUG_IF_YOU_USE_THIS
    """)
    caught = [value for _, value in _strings(as_it_was)
              if looks_like_placeholder(value)]
    assert caught == ['THIS_IS_A_BUG_IF_YOU_USE_THIS'] * 2
