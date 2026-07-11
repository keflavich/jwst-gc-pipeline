"""Grep-guard: forbid NEW ad-hoc dense-nearest-neighbour-median astrometry.

The in-code guard (``measure_offsets.assert_sparse_reference_for_nn_median``) only
protects the pipeline call sites that import it.  An agent (or human) who writes a
standalone script that does ``match_to_catalog_sky(...)`` and then takes
``np.median`` of the separations/offsets bypasses that guard entirely -- which is
exactly how the brick-1182 / prop-2221 4" astrometry errors kept recurring.

This test is the language-level shield: it FAILS if any Python file in the repo
pairs a nearest-neighbour match with a median/mean reduction, UNLESS the file is on
the reviewed allowlist below.  A new file that trips it must either

  (a) switch to offset-histogram stacking -- use
      ``jwst_gc_pipeline.photometry.astrometry_offsets.measure_offset`` -- or

  (b) be added to ``ALLOWLIST`` with a one-line justification, after a human
      confirms its match+median usage is source-association or histogram-refinement,
      NOT a dense-NN-median astrometric correction.

See CLAUDE.md and reduction/ASTROMETRY_WCS_CORRECTION_FLOW.md.
"""
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

# Key on match_to_catalog_sky specifically -- that is NEAREST-neighbour (nthneighbor
# defaults to 1), the method that collapses to ~0 in a crowded field.  We deliberately
# do NOT flag search_around_sky: it returns ALL pairs within a radius, which is the
# BASIS of the sanctioned offset-histogram stacking (which then medians only to refine
# the peak).  Flagging it would fire on the correct method.  File-level co-occurrence
# of a NN match with a median/mean is a strong tripwire for a NEW ad-hoc NN-median
# astrometry script; the allowlist carries the already-reviewed legitimate users.
_MATCH = re.compile(r"\bmatch_to_catalog_sky\b")
_REDUCE = re.compile(r"\b(np\.n?median|np\.n?mean|\.median\(|\.mean\()")

# Reviewed files where match+median is legitimate (source association for merging /
# dedup, guarded NN, or histogram-stacking refinement -- NOT dense-NN-median
# correction). Keep this list SHORT and justified; do not add to it to silence the
# guard on a genuine violation.
ALLOWLIST = {
    # sanctioned histogram-stacking helpers (median only refines the peak)
    "jwst_gc_pipeline/photometry/astrometry_offsets.py",
    "scripts/reduction/astrometry_audit.py",
    "scripts/release/registration_failsafes.py",  # per-cell agreement-fraction, not a correction
    "scripts/miri_reduction/apply_measured_miri_wcs_offsets.py",  # histogram refine_offset
    "scripts/miri_reduction/check_visit_registration.py",
    "scripts/miri_reduction/miri_f2550w_image3_rerun_v2.py",
    # the guard itself (nthneighbor=2 self-spacing measurement)
    "jwst_gc_pipeline/photometry/measure_offsets.py",
    # reference-catalog builders (guarded internally / sparse Gaia tie)
    "jwst_gc_pipeline/reduction/build_gaia_virac2_refcat.py",
    "jwst_gc_pipeline/reduction/build_gaia_virac2_refcat_byquery.py",
    "jwst_gc_pipeline/reduction/align_to_catalogs.py",  # guarded realign_to_catalog
    "jwst_gc_pipeline/photometry/generate_offsets_table.py",  # guarded voff()
    "jwst_gc_pipeline/photometry/make_reference_from_pipeline_catalogs.py",  # guarded bootstrap
    # cross-band source association for catalog merging / dedup (NOT astrometry)
    "jwst_gc_pipeline/photometry/merge_catalogs.py",
    "jwst_gc_pipeline/photometry/crowdsource_catalogs_long.py",
    "jwst_gc_pipeline/photometry/dedup_catalog.py",
    "jwst_gc_pipeline/photometry/cataloging.py",
    "jwst_gc_pipeline/photometry/legacy/crowdsource_step.py",
    "scripts/reduction/combine_brick_allband.py",  # cross-band merge
    # PR #57 diagnostic: nearest-neighbour SEPARATION histogram (median only for the
    # figure-title label + a caveat plot) -- NOT an astrometric correction.
    "docs/pr57_recovery_investigation/make_caveat_figs.py",
}

def _iter_py_files():
    """Only GIT-TRACKED .py files -- the guard polices committed code, not local
    scratch scripts in the working tree (which would false-positive on CI runners
    that never see them and annoy locally)."""
    try:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), "ls-files", "*.py"],
                             capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return  # not a git checkout (e.g. sdist install) -> nothing to police
    for line in out.splitlines():
        rel = Path(line)
        p = REPO_ROOT / rel
        if not p.is_file():
            continue
        # tests reference the forbidden tokens in strings on purpose
        if "tests" in rel.parts or p.name.startswith("test_"):
            continue
        yield rel, p


def test_no_adhoc_nn_median_astrometry():
    offenders = []
    for rel, path in _iter_py_files():
        text = path.read_text(errors="replace")
        if _MATCH.search(text) and _REDUCE.search(text):
            if rel.as_posix() not in ALLOWLIST:
                offenders.append(rel.as_posix())
    assert not offenders, (
        "FORBIDDEN dense-NN-median astrometry pattern (a nearest-neighbour match "
        "reduced by median/mean) found in non-allowlisted file(s):\n  "
        + "\n  ".join(sorted(offenders))
        + "\n\nUse offset-HISTOGRAM stacking instead: "
        "jwst_gc_pipeline.photometry.astrometry_offsets.measure_offset. "
        "If this usage is genuinely source-association or histogram-refinement (NOT "
        "a dense-NN-median astrometric correction), add the file to ALLOWLIST in "
        "this test with a justification. See CLAUDE.md."
    )


def test_allowlist_entries_exist():
    """Keep the allowlist from rotting -- every entry must point at a real file."""
    missing = [rel for rel in ALLOWLIST if not (REPO_ROOT / rel).is_file()]
    assert not missing, (
        "ALLOWLIST references files that no longer exist (remove them):\n  "
        + "\n  ".join(sorted(missing)))
