"""CI grep-guard: every production stage entry must call the run guard.

If a stage entry point stops calling ``assert_runnable_version``, the "pipeline
runs only on tagged versions" guarantee silently lapses.  This static check
fails CI in that case (mirrors the repo's other grep-guard tests).
"""
import os

import jwst_gc_pipeline

_PKG = os.path.dirname(os.path.abspath(jwst_gc_pipeline.__file__))

# (relative path under the package, human label) for each guarded stage entry.
STAGE_ENTRY_FILES = [
    ('reduction/PipelineRerunNIRCAM-LONG.py', 'imaging'),
    ('photometry/crowdsource_catalogs_long.py', 'cataloging'),
]


def test_stage_entries_call_run_guard():
    missing = []
    for rel, label in STAGE_ENTRY_FILES:
        path = os.path.join(_PKG, rel)
        with open(path) as fh:
            src = fh.read()
        if 'assert_runnable_version(' not in src:
            missing.append(f'{label} ({rel})')
    assert not missing, (
        'stage entry points missing assert_runnable_version() call -- the '
        'production run guard has lapsed for: ' + ', '.join(missing))
