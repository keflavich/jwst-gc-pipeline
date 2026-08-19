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
  than its ``_uncal.fits`` is left alone, so a retry after a partial failure
  reprocesses exactly the missing/stale members.  It is off by default because
  on-disk products are current only with respect to the ``_uncal`` mtime: a
  CRDS repin, a ``jwst`` version bump or a Detector1/Image2 parameter change
  leaves every product in place and newer than its input while making all of
  them wrong, and that is the case bare ``SKIP=0`` exists to fix.  It also
  reads whole-file states only -- a ``_cal.fits`` truncated when a job was
  killed has a newer mtime and reads as current (it surfaces as a read error
  downstream rather than silently).

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


def stage12_products_fresh(uncal_path, require_ramp=True):
    """Whether ``uncal_path``'s stage-1/2 products are newer than it.

    True when the sibling ``_cal.fits`` (and ``_ramp.fits``, unless
    ``require_ramp`` is False because the driver is not saving ramps) exist
    and have strictly newer mtimes than the ``_uncal.fits``.  With the uncal
    itself absent, present products count as current: reprocessing is
    impossible without the uncal, and running Detector1 on it would only fail
    loudly.  With a required product missing the member is reprocessed, so the
    caller gets that same loud failure on a missing uncal.

    "Newer than the uncal" is the only thing this can see; it says nothing
    about the code, CRDS context or step parameters the products were made
    with, which is why the caller reaches it only under ``STAGE12_RESUME=1``.
    """
    if not uncal_path.endswith('_uncal.fits'):
        raise ValueError(
            f"stage12_products_fresh expects an _uncal.fits path, got {uncal_path}")
    required = [uncal_path.replace('_uncal.fits', '_cal.fits')]
    if require_ramp:
        required.append(uncal_path.replace('_uncal.fits', '_ramp.fits'))
    if not all(os.path.exists(path) for path in required):
        return False
    if not os.path.exists(uncal_path):
        return True
    uncal_mtime = os.path.getmtime(uncal_path)
    return all(os.path.getmtime(path) > uncal_mtime for path in required)


def stage12_skip_reason(uncal_path, resume=None, require_ramp=True):
    """Why the stage-1/2 loop should skip ``uncal_path``, or None to process it.

    Two reasons, in order: this interpreter already ran stage 1+2 on the
    member in an earlier module pass (#417's 3x), or ``STAGE12_RESUME=1`` and
    its products are newer than the uncal.  Without the resume opt-in, a fresh
    process reprocesses every member, which is what ``SKIP=0`` means.
    """
    if stage12_already_processed(uncal_path):
        return ("this run already ran stage 1+2 on it in an earlier module pass")
    if resume is None:
        resume = stage12_resume_enabled()
    if resume and stage12_products_fresh(uncal_path, require_ramp=require_ramp):
        return (f"{STAGE12_RESUME_ENV} is set and its stage-1/2 products are "
                "newer than the _uncal.fits")
    return None
