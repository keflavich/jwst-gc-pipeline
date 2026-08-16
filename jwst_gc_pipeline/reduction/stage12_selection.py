"""Which asn members the NIRCam stage-1/2 loop processes, and when to skip one.

``PipelineRerunNIRCAM-LONG.py``'s ``__main__`` runs ``main()`` once per module
(nrca, nrcb, merged) with the same ``skip_step1and2``, and the stage-1/2 block
sits inside every pass.  Before #417 that loop filtered the asn members only on
``'_nrc'``, so a fresh SKIP=0 run ramp-fitted and Image2-processed every uncal
three times (and wrote each ~126 MB ramp three times).  Two pure predicates
remove the 3x; the driver consults both before ``Detector1Pipeline.call``:

* **Module scoping** (`member_in_stage12_pass`): the nrca/nrcb passes each
  claim only their own module's members, with the same substring semantics as
  the per-module asn trim in the driver's tweakreg block
  (``f'{module}' in expname``): 'nrca' claims nrca1-4 AND nrcalong, 'nrcb'
  claims nrcb1-4 AND nrcblong.  Every NIRCam detector name contains exactly
  one of 'nrca'/'nrcb', so the two module passes partition the members.  The
  merged pass keeps every NIRCam member: a merged-only or single-module run
  must still be able to produce every _cal it needs, and on the default
  nrca,nrcb,merged sequence the freshness predicate below skips each member
  the earlier passes already produced.

* **Idempotence** (`stage12_products_fresh`): a member whose ``_cal.fits``
  AND ``_ramp.fits`` both exist and are both newer than its ``_uncal.fits``
  is already done, and re-running Detector1+Image2 on it reproduces the same
  outputs.  A retry after a partial failure therefore reprocesses exactly the
  missing/stale members (Detector1 done but Image2 crashed leaves a ramp with
  no cal -> reprocessed; a re-downloaded uncal is newer than both products ->
  reprocessed).

Both are pure functions so they unit-test with a tmpdir
(``reduction/tests/test_stage12_selection.py``).  Ramp retention itself
(``save_calibrated_ramp``) is a separate concern, tracked in #421.
"""
import os


def member_in_stage12_pass(expname, module):
    """Whether this module pass's stage-1/2 loop claims this asn member.

    ``module`` is one of 'nrca', 'nrcb', 'merged' (``main()`` receives the
    ``_module_group`` family).  For nrca/nrcb this mirrors the later
    per-module member trim exactly (substring containment, so 'nrca' matches
    nrca1-4 and nrcalong); 'merged' claims every NIRCam member.
    """
    if '_nrc' not in expname:
        return False
    if module in ('nrca', 'nrcb'):
        return module in expname
    return True


def stage12_products_fresh(uncal_path):
    """Whether ``uncal_path``'s stage-1/2 products are current, so it can be skipped.

    True when the sibling ``_cal.fits`` and ``_ramp.fits`` both exist and both
    have a strictly newer mtime than the ``_uncal.fits``.  With the uncal
    itself absent, two present products count as current: reprocessing is
    impossible without the uncal, and running Detector1 on it would only fail
    loudly.  With either product missing the member is reprocessed, so the
    caller gets that same loud failure on a missing uncal.
    """
    if not uncal_path.endswith('_uncal.fits'):
        raise ValueError(
            f"stage12_products_fresh expects an _uncal.fits path, got {uncal_path}")
    cal_path = uncal_path.replace('_uncal.fits', '_cal.fits')
    ramp_path = uncal_path.replace('_uncal.fits', '_ramp.fits')
    if not (os.path.exists(cal_path) and os.path.exists(ramp_path)):
        return False
    if not os.path.exists(uncal_path):
        return True
    uncal_mtime = os.path.getmtime(uncal_path)
    return (os.path.getmtime(cal_path) > uncal_mtime
            and os.path.getmtime(ramp_path) > uncal_mtime)
