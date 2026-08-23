"""``measure_offset``'s caller must be told the dense-reference caveat.

CLAUDE.md states it: the offset-histogram peak is immune to NN-collapse and is
NOT unbiased against a DENSE reference.  Two catalogs tracing the same clustered
field make a correlated wrong-pair background that pulls the peak by several
mas; on brick (2026-07-16) that read ~9-10 mas against dense VIRAC2 with a
+-6.5 mas RA term whose sign flipped between filters, against a same-star tie of
~0 in RA and ~-5 mas in Dec.

The knowledge lived in ``visit_consensus`` and in CLAUDE.md, and in neither
docstring a caller of ``measure_offset`` reads -- both of which said
"density-immune" with no qualifier.  ``bulk_astrometry_update.measure_bulk_offset``
then wrote the raw peak into released coordinates (issue #397), which is what a
reader of those docstrings would think was correct.

A docstring is the only mechanism available here: ``measure_offset`` has no way
to know whether the reference it was handed is dense, and the sanctioned refine
step lives in a different module.  So the text is pinned, the way the NN-median
ban and the SIP-header ban are pinned by their own grep guards.
"""
import pytest

from jwst_gc_pipeline.photometry import astrometry_offsets as ao


#: Phrases that carry the three things a caller has to come away with: that the
#: bias exists, that it is specific to a DENSE reference, and where the precise
#: value comes from instead.
REQUIRED = (
    'dense',
    'same-star',
    'local_residual_map',
    'measure_reference_tie',
    'bulk_source',
)


def test_the_module_docstring_states_the_dense_reference_caveat():
    doc = (ao.__doc__ or '').lower()
    assert doc, 'astrometry_offsets lost its module docstring'
    missing = [w for w in REQUIRED if w.lower() not in doc]
    assert not missing, (
        'the module docstring says "density-immune" without the dense-reference '
        'caveat; missing: ' + ', '.join(missing))


def test_measure_offsets_docstring_states_it_where_a_caller_reads():
    """A caller reads the function, not the module header."""
    doc = (ao.measure_offset.__doc__ or '').lower()
    assert doc, 'measure_offset lost its docstring'
    for word in ('dense', 'same-star', 'local_residual_map',
                 'measure_reference_tie'):
        assert word.lower() in doc, (
            f"measure_offset's docstring does not mention {word!r}: a caller "
            f"reading it would take dra/ddec as the precise bulk against a "
            f"dense reference")


def test_the_refine_route_the_docstrings_name_exists():
    """The pointer has to point somewhere, or it decays into folklore."""
    assert callable(ao.local_residual_map)
    vc = pytest.importorskip('jwst_gc_pipeline.photometry.visit_consensus')
    assert callable(vc.measure_reference_tie)
