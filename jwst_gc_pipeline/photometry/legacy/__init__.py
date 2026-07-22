"""Legacy photometry pipeline — BENCHMARKS ONLY, not part of active runs.

The crowdsource-based and "iter2"/"iter3" seeded-daophot per-exposure pipeline
(``do_photometry_step``) is superseded by the manual-iteration pipeline
(``cataloging.do_photometry_step_manual`` / ``run_manual_pipeline``, the m12/m3..m7
phases).  It is retained here only so its results can be reproduced for
benchmark comparisons; do not wire it into active reductions.

Sequestration is in progress (see REFACTOR_PLAN.md "Sequestration"): the legacy
entrypoint is being moved out of ``catalog_long.py`` into this
subpackage, while the shared helpers (get_psf_model, load_data, save_*,
satstar loaders, naming tokens) and the active CLI ``main`` stay in the
active tree.
"""
