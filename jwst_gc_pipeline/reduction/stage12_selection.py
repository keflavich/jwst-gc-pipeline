"""Which asn members the NIRCam stage-1/2 loop processes, and when to skip one.

``PipelineRerunNIRCAM-LONG.py``'s ``__main__`` runs ``main()`` once per module
(nrca, nrcb, merged) with the same ``skip_step1and2``, and the stage-1/2 block
sits inside every pass.  Before #417 that loop filtered the asn members only on
``'_nrc'``, so a fresh SKIP=0 run ramp-fitted and Image2-processed every uncal
three times (and wrote each ~126 MB ramp three times).  The 3x is entirely
intra-process, so it is removed intra-process:

* **Module scoping** (`member_in_stage12_pass`): the nrca/nrcb passes each
  claim only their own module's members, with the same substring semantics as
  the per-module asn trim in the driver's tweakreg block
  (``f'{module}' in expname``): 'nrca' claims nrca1-4 AND nrcalong, 'nrcb'
  claims nrcb1-4 AND nrcblong.  Every NIRCam detector name contains exactly
  one of 'nrca'/'nrcb', so the two module passes partition the members.  The
  merged pass keeps every NIRCam member: a merged-only or single-module run
  must still be able to produce every _cal it needs.

* **In-process memo** (`note_stage12_processed` / `stage12_already_processed`):
  the merged pass skips the members THIS interpreter already ran stage 1+2 on
  in an earlier module pass.  That covers the remaining third of #417 without
  changing what a run means: a new process starts with an empty memo, so
  ``SKIP=0`` still re-fits every ramp from the ``_uncal``, exactly as
  ``scripts/reduction/submit_reduction.sbatch`` and ``docs/HIPERGATOR.md``
  describe it and as ``PipelineMIRI.py`` / ``PipelineRerunNIRISS.py`` behave.

* **Opt-in resume** (`stage12_products_fresh`, reached only when
  ``STAGE12_RESUME=1``): with the operator asking for it, a member whose
  ``_cal.fits`` (and ``_ramp.fits``, when ramps are being saved) are newer
  than its ``_uncal.fits`` **and carry this run's ``CRDS_CTX``/``CAL_VER``**
  is left alone, so a retry after a partial failure reprocesses exactly the
  missing/stale members.  The stamp half is #433: every member a resume
  reprocesses is written with the current context and ``jwst`` version, so
  keeping a member stamped with an older one would leave ONE filter directory
  -- one Image3 input set -- straddling two calibrations.  A CRDS repin or a
  version bump therefore makes a resume reprocess the whole directory, which
  is what bare ``SKIP=0`` gives.  Reading the header also refuses a
  ``_cal.fits`` truncated when a job was killed, which has a newer mtime and
  used to read as current on mtimes alone.  It remains off by default: a
  Detector1/Image2 *parameter* change leaves every product in place, newer
  than its input and stamped with the current context, and no header read
  can see it.

`stage12_skip_reason` is the one entry point the driver calls: it returns the
message to log, or None to process the member.

The predicates are pure (the memo is explicit state with a documented reset)
so they unit-test with a tmpdir
(``reduction/tests/test_stage12_selection.py``).  Ramp retention itself
(``save_calibrated_ramp``) is a separate concern, tracked in #421; pass
``require_ramp`` to match whatever the driver is doing so the resume path
cannot silently go permanently False if ramps stop being written.
"""
import os

#: Environment variable that turns the on-disk resume skip on.
STAGE12_RESUME_ENV = 'STAGE12_RESUME'

#: Uncal files this interpreter has already run stage 1+2 on, absolute paths.
#: The driver chdirs to the per-filter output_dir before the loop, so the
#: bare asn expnames resolve consistently across the module passes.
_PROCESSED_UNCALS = set()


def member_in_stage12_pass(expname, module):
    """Whether this module pass's stage-1/2 loop claims this asn member.

    ``module`` is one of 'nrca', 'nrcb', 'merged' (``main()`` receives the
    ``_module_group`` family).  For nrca/nrcb this mirrors the later
    per-module member trim exactly (substring containment, so 'nrca' matches
    nrca1-4 and nrcalong).  Anything else -- 'merged', and any module name a
    future caller invents -- claims every NIRCam member, so an unrecognised
    module over-processes rather than silently dropping exposures.
    """
    if '_nrc' not in expname:
        return False
    if module in ('nrca', 'nrcb'):
        return module in expname
    return True


def reset_stage12_processed():
    """Forget which uncals this interpreter has processed (tests, long-lived hosts)."""
    _PROCESSED_UNCALS.clear()


def note_stage12_processed(uncal_path):
    """Record that stage 1+2 just ran on ``uncal_path`` in this interpreter."""
    _PROCESSED_UNCALS.add(os.path.abspath(uncal_path))


def stage12_already_processed(uncal_path):
    """Whether this interpreter already ran stage 1+2 on ``uncal_path``."""
    return os.path.abspath(uncal_path) in _PROCESSED_UNCALS


def stage12_resume_enabled(env=None):
    """Whether the opt-in on-disk resume skip is turned on (``STAGE12_RESUME``)."""
    env = os.environ if env is None else env
    return str(env.get(STAGE12_RESUME_ENV, '0')).strip().lower() in (
        '1', 'true', 'yes', 'on')


#: Primary-header keywords stamping the calibration a stage-1/2 product was
#: made with: the CRDS context that supplied its reference files, and the
#: ``jwst`` version of the code that ran.  Both are written by every
#: Detector1/Image2 product (verified on ``_cal.fits`` and ``_ramp.fits``).
CALIBRATION_STAMP_KEYWORDS = ('CRDS_CTX', 'CAL_VER')

#: Memo for :func:`current_calibration_stamp`, which is otherwise a CRDS call
#: per asn member.  A process's answer cannot change under it: the pipeline
#: resolves the context once and calibrates the whole run against it.
_CURRENT_STAMP = []


def current_calibration_stamp():
    """``(CRDS context, jwst version)`` the NEXT stage-1/2 call would stamp.

    ``crds.get_context_name('jwst')`` is the same resolution the pipeline
    itself does (operational context, or ``CRDS_CONTEXT`` when the run pins
    one), and reads from the local ``CRDS_PATH`` cache.  Memoized per process.
    """
    if not _CURRENT_STAMP:
        import crds
        import jwst
        _CURRENT_STAMP.append((str(crds.get_context_name('jwst')),
                               str(jwst.__version__)))
    return _CURRENT_STAMP[0]


def reset_current_calibration_stamp():
    """Forget the memoized current stamp (tests)."""
    _CURRENT_STAMP.clear()


def product_calibration_stamp(path):
    """``(CRDS_CTX, CAL_VER)`` from a product's primary header, or None.

    None means "this file does not answer the question": it is not readable as
    FITS (the truncated ``_cal.fits`` a killed job leaves behind, which has a
    newer mtime than its uncal and so reads as current on mtimes alone), or it
    is missing either keyword.  Callers treat None as "reprocess".
    """
    from astropy.io import fits
    try:
        header = fits.getheader(path, ext=0)
    except (OSError, ValueError, IndexError):
        # OSError covers truncated/empty/non-FITS; astropy raises ValueError or
        # IndexError on some malformed-header and no-HDU cases.
        return None
    values = tuple(header.get(key) for key in CALIBRATION_STAMP_KEYWORDS)
    if any(value is None or str(value).strip() == '' for value in values):
        return None
    return tuple(str(value).strip() for value in values)


def stage12_products_fresh(uncal_path, require_ramp=True, stamp=None):
    """Whether ``uncal_path``'s stage-1/2 products are current.

    True when the sibling ``_cal.fits`` (and ``_ramp.fits``, unless
    ``require_ramp`` is False because the driver is not saving ramps) exist,
    have strictly newer mtimes than the ``_uncal.fits``, AND carry the same
    ``CRDS_CTX``/``CAL_VER`` this process would stamp on them if it reprocessed
    them.  With the uncal itself absent, present products count as current on
    mtimes: reprocessing is impossible without the uncal, and running
    Detector1 on it would only fail loudly.  With a required product missing
    the member is reprocessed, so the caller gets that same loud failure on a
    missing uncal.  The stamp check applies either way -- a product that cannot
    be reprocessed is still not allowed to carry a foreign calibration into the
    kept set.

    The stamp comparison is what keeps a resume from leaving one Image3 input
    set straddling two CRDS contexts (#433).  Every member this run reprocesses
    is written with the CURRENT context and ``jwst`` version, so the set is
    homogeneous exactly when every member it KEEPS already carries those --
    which is the comparison made here, not a separate strictness policy.  A
    repin or a version bump therefore makes a resume reprocess the whole
    filter directory, which is the behaviour bare ``SKIP=0`` exists to give.

    ``stamp`` overrides the current ``(CRDS_CTX, CAL_VER)`` (tests, and callers
    that resolve it once for a whole asn).  ``stamp=False`` skips the
    comparison and restores the mtime-only test.
    """
    if not uncal_path.endswith('_uncal.fits'):
        raise ValueError(
            f"stage12_products_fresh expects an _uncal.fits path, got {uncal_path}")
    required = [uncal_path.replace('_uncal.fits', '_cal.fits')]
    if require_ramp:
        required.append(uncal_path.replace('_uncal.fits', '_ramp.fits'))
    if not all(os.path.exists(path) for path in required):
        return False
    if stamp is not False:
        if stamp is None:
            stamp = current_calibration_stamp()
        stamp = tuple(str(value).strip() for value in stamp)
        if any(product_calibration_stamp(path) != stamp for path in required):
            return False
    if not os.path.exists(uncal_path):
        return True
    uncal_mtime = os.path.getmtime(uncal_path)
    return all(os.path.getmtime(path) > uncal_mtime for path in required)


def stage12_skip_reason(uncal_path, resume=None, require_ramp=True, stamp=None):
    """Why the stage-1/2 loop should skip ``uncal_path``, or None to process it.

    Two reasons, in order: this interpreter already ran stage 1+2 on the
    member in an earlier module pass (#417's 3x), or ``STAGE12_RESUME=1`` and
    its products are current -- newer than the uncal and carrying this run's
    CRDS context and ``jwst`` version (#433).  Without the resume opt-in, a
    fresh process reprocesses every member, which is what ``SKIP=0`` means.
    """
    if stage12_already_processed(uncal_path):
        return ("this run already ran stage 1+2 on it in an earlier module pass")
    if resume is None:
        resume = stage12_resume_enabled()
    if resume and stage12_products_fresh(uncal_path, require_ramp=require_ramp,
                                         stamp=stamp):
        return (f"{STAGE12_RESUME_ENV} is set and its stage-1/2 products are "
                "newer than the _uncal.fits and carry this run's "
                f"{'/'.join(CALIBRATION_STAMP_KEYWORDS)}")
    return None
