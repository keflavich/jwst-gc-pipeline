"""Per-field diagnostic measurement figures and LaTeX write-ups.

This package builds a small, fixed, publication-quality figure set for one
field at a time and emits a LaTeX document that discusses it.  It is
deliberately *not* a QA gate: the release gates live in
``scripts/release/`` and the initial-data-product checks live in the
separate ``JWST-GC/data-qa`` repository.  What lives here is the
comprehensive *analysis* layer -- the measurements you would put in a paper
appendix to characterise how well the astrometry, the photometry and the
diffuse-emission ("background") measurement actually behave.

The figure budget is intentionally tight (order ten figures per field, each
multi-panel over the field's filters) so that the resulting document stays
readable and every panel earns its place.

Entry point: ``scripts/analysis/make_diagnostic_writeup.py``.
"""

from jwst_gc_pipeline.diagnostics.inventory import FieldInventory

# Deliberately NOT re-exporting the ``inventory()`` function: it would bind
# the package attribute ``diagnostics.inventory`` to the function and shadow
# the submodule of the same name.  Import it from its module.
__all__ = ['FieldInventory']
