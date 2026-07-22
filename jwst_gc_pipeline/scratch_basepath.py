"""Non-destructive experimental-run basepath redirect.

A single, testable mechanism used by BOTH entry points (the cataloging driver
``crowdsource_catalogs_long`` and the reduction driver ``PipelineRerunNIRCAM``):
when ``GC_BASEPATH_OVERRIDE`` is set to a non-empty path, the whole pipeline
basepath is redirected to that scratch tree.  Staged (via
``scripts/reduction/stage_scratch_basepath.sh``) with symlinks/copies of the
real INPUT frames only, this lets a full-frame reduce/catalog run write all of
its OUTPUTS under the scratch dir and never overwrite released products.

Process-global by design: the one-process-one-(target, filter) invariant
documented in CLAUDE.md means a single process only ever resolves one basepath,
so a process-global env override is safe.  Empty/unset -> normal in-place
behaviour.

If/when the ``paths.py`` / ``JWST_GC_DATAROOT`` layer (PR #98) lands, resolve
the default basepath through it FIRST, then apply this override as the explicit
final word -- they compose (dataroot picks the tree; this override replaces it
wholesale for an experimental run).
"""
import os

ENV_VAR = 'GC_BASEPATH_OVERRIDE'


def apply_basepath_override(basepath):
    """Return the scratch basepath if ``GC_BASEPATH_OVERRIDE`` is set, else
    ``basepath`` unchanged.

    The returned override is normalised to a single trailing slash so it is a
    drop-in for the ``f'/orange/.../{regionname}/'`` strings the callers build.
    Pure (reads the environment, mutates nothing) so it is unit-testable.
    """
    override = os.environ.get(ENV_VAR, '').strip()
    if override:
        return override.rstrip('/') + '/'
    return basepath
