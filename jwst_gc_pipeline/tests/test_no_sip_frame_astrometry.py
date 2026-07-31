"""Grep-guard: forbid NEW code that reads a detector frame's SIP header for astrometry.

Every JWST detector-frame product carries an authoritative **GWCS** (ASDF
extension) and a fitted ``RA---TAN-SIP`` approximation of it (SCI header).
Building an ``astropy.wcs.WCS`` from a per-exposure SCI header therefore silently
substitutes the approximation for the truth:

* the fit residual -- 5-8 mas, position-dependent and different per detector
  *and* per filter, on every frame written before the tight-fit change.  (In
  pixels that same residual is up to ~165 millipixels; SIP's own forward->inverse
  round trip closes to 0.000 mpix, so it is one error seen in two units, not
  two.)
* off-footprint, the iterative SIP inverse either **raises ``NoConvergence``**
  (the failure that aborted the whole W51 m8 forced fill) or, with
  ``quiet=True``, returns **finite garbage with no warning** -- which propagates
  into a catalog instead of stopping the run.  The GWCS returns ``NaN``.

Read the GWCS instead::

    from jwst_gc_pipeline.frame_wcs import frame_wcs
    ww = frame_wcs(filename_or_hdulist)

This test FAILS if a git-tracked Python file builds a ``WCS`` from something that
looks like a per-exposure SCI header, unless it is on the reviewed ``ALLOWLIST``.

``i2d`` mosaics are exempt and deliberately not matched: ``resample`` writes a
rectified plain ``RA---TAN`` grid with no SIP, so ``WCS(i2d_header)`` is exact.

See ``jwst_gc_pipeline/frame_wcs.py`` and
``reduction/ASTROMETRY_WCS_CORRECTION_FLOW.md``.
"""
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# WCS(...) built from something named like a SCI-extension header of a frame.
# Deliberately narrow: it must mention 'SCI' (or ext=1 / ext=('SCI',1)), which is
# the detector-frame case.  i2d readers usually also say 'SCI' -- hence the
# allowlist for reviewed i2d call sites.
_SIP_WCS = re.compile(
    r"""(?:astropy_?)?(?:wcs\.)?WCS\(\s*[^)\n]*(?:
          \['SCI'\]|\["SCI"\]|                       # hdul['SCI'].header
          ext\s*=\s*\(\s*['"]SCI['"]|                # getheader(fn, ext=('SCI',1))
          ext\s*=\s*['"]SCI['"]|                     # getheader(fn, 'SCI')
          ,\s*['"]SCI['"]\s*\)                       # getheader(fn, 'SCI')
        )""",
    re.VERBOSE)

ALLOWLIST = {
    # the GWCS-first reader itself: it *is* the sanctioned SIP fallback
    "jwst_gc_pipeline/frame_wcs.py",
    # load_frame_wcs() is GWCS-first via stdatamodels (prefer_gwcs=True) and the
    # WCS(hdul['SCI'].header) below it is that reader's own warned fallback, the
    # same shape as frame_wcs.py's.  NB the fallback is NOT harmless on the
    # products this package globs: on _crf/_destreak the SIP fit disagrees with
    # the GWCS by up to ~5.5 mas, which exceeds the ~0.6-3.8 mas CRDS-vs-STDGDC
    # delta the experiment measures.  Tracked on #154; allowlisted here because
    # the call site is a deliberate fallback, not a SIP-first astrometry read.
    "jwst_gc_pipeline/astrometry_gdc/gdc_wcs.py",
    # writes/verifies the SIP header against the GWCS -- reads SIP on purpose
    "jwst_gc_pipeline/reduction/fits_wcs_sync.py",
    "scripts/release/audit_fits_gwcs_agreement.py",
    # all astrometry here goes through frame_wcs; the one remaining
    # WCS(fh['SCI'].header, relax=True) is the no-GWCS FALLBACK of the
    # SCI->PRIMARY header copy, whose primary path is sync_header_to_gwcs.
    "jwst_gc_pipeline/reduction/saturated_star_finding.py",
    # reviewed i2d (rectified, no SIP) readers
    "jwst_gc_pipeline/photometry/cataloging.py",
    "jwst_gc_pipeline/photometry/crowdsource_catalogs_long.py",
    "scripts/release/make_preview_rgb.py",
    "scripts/satstar_deblend/run_satstar_compare.py",
    # frozen legacy path (see manual-defaults consolidation); not used for
    # release products
    "jwst_gc_pipeline/photometry/legacy/crowdsource_step.py",
    # DISPLAY only: the WCS is handed to WCSAxes, which needs a real
    # astropy.wcs.WCS.  No catalog position is derived from it.
    "jwst_gc_pipeline/plotting/plot_tools.py",
    # one-off historical MIRI reduction scripts, kept for provenance.  They are
    # not on any live pipeline path; converting them cannot be validated
    # because the runs they belong to are finished.  Do NOT copy their pattern.
    "scripts/miri_reduction/apply_measured_miri_wcs_offsets.py",
    "scripts/miri_reduction/check_visit_registration.py",
    "scripts/miri_reduction/miri_f2550w_colprofile_v5.py",
    "scripts/miri_reduction/miri_f2550w_framereg_v13.py",
    "scripts/miri_reduction/miri_f2550w_skyconst_v12.py",
    "scripts/miri_reduction/miri_f2550w_tile_homogenize_v3.py",
    "scripts/miri_reduction/miri_f2550w_visitfix_v14.py",
    "scripts/miri_reduction/miri_tile_homogenize.py",
    "scripts/miri_reduction/patch_nan_holes_from_clean_frames.py",
    # frozen diagnostics/figures for a closed investigation
    "docs/pr57_recovery_investigation/make_caveat_figs.py",
    "docs/pr57_recovery_investigation/make_figs.py",
    "scripts/satstar_deblend/batch_validate.py",
    # satstar fit-shape analysis scripts: they COPY the SCI WCS (relax=True on
    # both read and write, so the SIP terms survive) into a working header for
    # the satstar fitter, exactly as saturated_star_finding.py:4617 does.  No
    # position is derived from the SIP WCS -- it is carried, not evaluated.
    "scripts/satstar_deblend/adaptive_fit_shape_ab.py",
    "scripts/satstar_deblend/fit_footprint_sweep.py",
}


def _iter_py_files():
    """Git-tracked .py files only -- the guard polices committed code, not local
    scratch scripts in the working tree."""
    try:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), "ls-files", "*.py"],
                             capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return
    for line in out.splitlines():
        rel = Path(line)
        p = REPO_ROOT / rel
        if not p.is_file():
            continue
        if "tests" in rel.parts or p.name.startswith("test_"):
            continue
        yield rel, p


def test_no_sip_wcs_for_frame_astrometry():
    offenders = []
    for rel, path in _iter_py_files():
        if rel.as_posix() in ALLOWLIST:
            continue
        if _SIP_WCS.search(path.read_text(errors="replace")):
            offenders.append(rel.as_posix())
    assert not offenders, (
        "SIP-header WCS built from a detector-frame SCI header in "
        "non-allowlisted file(s):\n  " + "\n  ".join(sorted(offenders))
        + "\n\nSIP is a fitted approximation of the GWCS: 5-8 mas of "
        "position-dependent forward error on legacy frames, and an inverse that "
        "raises NoConvergence off-footprint. Use "
        "`from jwst_gc_pipeline.frame_wcs import frame_wcs; ww = frame_wcs(...)`. "
        "If the file genuinely reads an i2d mosaic (rectified, no SIP) or must "
        "read the SIP header on purpose, add it to ALLOWLIST with a justification."
    )


def test_allowlist_entries_exist():
    """Keep the allowlist from rotting -- every entry must point at a real file."""
    missing = [rel for rel in ALLOWLIST if not (REPO_ROOT / rel).is_file()]
    assert not missing, ("ALLOWLIST references files that no longer exist "
                         "(remove them):\n  " + "\n  ".join(sorted(missing)))


def test_the_guard_actually_matches_the_bad_pattern(tmp_path):
    """A guard that cannot fire is not a guard."""
    for bad in ("ww = wcs.WCS(hdul['SCI'].header)",
                'ww = WCS(h["SCI"].header, relax=True)',
                "w = wcs.WCS(fits.getheader(fn, ext=('SCI', 1)))",
                "w = WCS(fits.getheader(fn, 'SCI'))"):
        assert _SIP_WCS.search(bad), bad
    for ok in ("ww = frame_wcs(filename)",
               "ww = wcs.WCS(header)",
               "ww = WCS(hdr, naxis=2)"):
        assert not _SIP_WCS.search(ok), ok
