"""Requested filters that resolve to zero frames.

The manual cataloging pipeline is handed a filter list (``--filternames``,
``FILTERS=`` in the submit scripts) and globs each filter's per-exposure ``crf``
frames under one input lineage suffix (``--each-suffix``, per-filter overrides
via ``--each-suffix-overrides``).  A filter whose glob matched nothing used to
contribute an empty candidate list and nothing else: every fan-out shard printed
``0 of 0 frames ... nothing to fit`` and exited 0, and the single finalize job
raised hours later, after the whole array had run (issue #592, wd1 F150W --
96 frames on disk under ``o001_crf`` while the run asked for
``destreak_o001_crf``).

Zero candidate frames has four causes and they are not the same thing, so this
module makes the distinction explicit instead of leaving it to a reader:

1. **The lineage suffix is wrong.**  Frames for the filter ARE on disk, under a
   different suffix.  The request and the disk disagree; raise, and name the
   suffix that would have worked so the operator can set
   ``--each-suffix-overrides``.
2. **The observation took this filter and its frames are absent.**  The field
   registry (``fields.yaml``) declares the filter for this proposal/observation,
   so the reduction that should have produced the frames has not run (or wrote
   them somewhere else).  Raise: cataloging the rest and reporting success ships
   a product short a band.
3. **The requested suffix is right and the frames are out of scope.**  They
   are on disk under exactly the suffix asked for, so the run's own ``--modules``
   or visit range is what missed them.  Raise, with wording that does not send
   the reader to ``--each-suffix-overrides``.
4. **The observation never took this filter.**  It is not declared for this
   observation and nothing is on disk under any lineage.  A caller sweeping one
   filter list across several fields legitimately names bands a given field does
   not have, so this is reported and skipped, not raised.

Cases 1-3 are the failure; case 4 is the legitimate one.
"""
import glob
import os

from jwst_gc_pipeline.mast_names import jw_prefix


class RequestedFilterHasNoFramesError(ValueError):
    """A requested filter resolved to zero candidate frames but should not have.

    Raised for causes 1-3 above.  A filter the observation never took (cause 4)
    does not raise.
    """


def frame_lineages_on_disk(basepath, filtername, proposal_id, field,
                           each_suffix='crf'):
    """``{lineage: n_frames}`` for every input lineage of ``filtername`` on disk.

    The lineage is the part of a per-exposure filename after the detector token:
    ``jw01905001001_02101_00001_nrca1_o001_crf.fits`` -> ``o001_crf``, and
    ``..._nrca1_destreak_o001_crf.fits`` -> ``destreak_o001_crf``.  Those are
    exactly the strings ``--each-suffix`` / ``--each-suffix-overrides`` take, so
    the answer can be quoted straight back into the error message.

    The glob is scoped to the proposal and observation by the filename prefix
    (``jw{proposal}{obs}``) -- a hyphen-joined joint field (``'001-002'``) is
    split and each observation globbed -- and to the product KIND by the last
    token of ``each_suffix`` (``crf``), so the per-frame catalogs and satstar
    products that share the stem are not counted as frames.  Modules and visits
    are left wildcard on purpose: the question this answers is "does this filter
    have frames at all", not "does it have the ones this run asked for".
    """
    stem = str(each_suffix).split('_')[-1].strip()
    tail = f'*_{stem}.fits' if stem else '*.fits'
    subfields = str(field).split('-') if '-' in str(field) else [str(field)]
    lineages = {}
    for sf in subfields:
        pattern = (f'{basepath}/{filtername}/pipeline/'
                   f'{jw_prefix(proposal_id)}{sf}{tail}')
        for path in glob.glob(pattern):
            parts = os.path.basename(path)[:-len('.fits')].split('_')
            # `jw<PPPPP><OOO><VVV>_<vgroup>_<exp>_<detector>_<lineage>` -- the
            # same four leading tokens naming.frame_identity counts.
            if len(parts) < 5:
                continue
            lineage = '_'.join(parts[4:])
            lineages[lineage] = lineages.get(lineage, 0) + 1
    return lineages


def declared_for_observation(target, proposal_id, field):
    """Upper-case filters ``fields.yaml`` declares for this observation.

    ``filters_for_observation`` is the precise answer -- the proposal's filters
    restricted to the instrument that took this observation -- but it returns
    ``[]`` for an observation it cannot resolve to a single instrument, and the
    Treasury is exactly that case: 10678 registers ``obsids: {nircam: '*'}``, a
    wildcard that owns all 139 tiles without naming one, so no obsid matches by
    name and every tile reads ``[]``.  Falling back to the PROPOSAL's declared
    list keeps the check working there (10678 declares F212N/F480M/F770W for
    every tile).  The fallback is proposal-scoped on purpose: a field-wide union
    would hand brick/1182's bands to a brick/2221 run.

    Empty only when the proposal itself is unregistered, which is the "cannot
    tell" answer: the caller then falls back to what is on disk.
    """
    from jwst_gc_pipeline import fields as field_registry
    per_obs = field_registry.filters_for_observation(target, proposal_id, field)
    if per_obs:
        return {str(f).upper() for f in per_obs}
    declared = set()
    for instrument in ('nircam', 'niriss'):
        by_proposal = field_registry.obs_filters(instrument).get(target) or {}
        declared |= {str(f).upper()
                     for f in (by_proposal.get(str(proposal_id)) or ())}
    return declared


def classify_requested_filter(filtername, n_candidates, *, target, proposal_id,
                              field, basepath, each_suffix, declared=None,
                              lineages=None):
    """``(verdict, message)`` for one requested filter.

    ``verdict`` is ``'ok'`` (frames found), ``'wrong-lineage'``,
    ``'outside-this-run'``, ``'declared-but-absent'`` or ``'not-observed'``.
    Everything but ``'ok'`` and ``'not-observed'`` is a failure;
    ``'not-observed'`` carries a message for the log.  ``declared`` and
    ``lineages`` are injectable so a test does not need a data tree.
    """
    if n_candidates > 0:
        return 'ok', ''
    if lineages is None:
        lineages = frame_lineages_on_disk(basepath, filtername, proposal_id,
                                          field, each_suffix=each_suffix)
    if declared is None:
        declared = declared_for_observation(target, proposal_id, field)
    other = {k: v for k, v in lineages.items() if k != each_suffix}
    if lineages:
        _found = ', '.join(f'{k} (n={v})' for k, v in sorted(lineages.items()))
        if other:
            _pick = sorted(other, key=lambda k: (-other[k], k))[0]
            return 'wrong-lineage', (
                f"requested filter {filtername} resolved to 0 candidate frames "
                f"with each_suffix={each_suffix!r}, but "
                f"{sum(lineages.values())} frame(s) are on disk under: {_found}. "
                f" Set --each-suffix-overrides={filtername.upper()}:{_pick} "
                f"(EACH_SUFFIX_OVERRIDES in the submit scripts) to read that "
                f"lineage for this filter.")
        # The requested lineage IS the only one on disk, so the suffix is right
        # and the run's own scan still found nothing: the frames sit outside the
        # modules or the visit range this run swept.
        return 'outside-this-run', (
            f"requested filter {filtername} resolved to 0 candidate frames "
            f"although {sum(lineages.values())} frame(s) are on disk under the "
            f"requested each_suffix={each_suffix!r}: none of them fall in this "
            f"run's modules or visit range.  Check --modules and the field's "
            f"nvisits.")
    if str(filtername).upper() in (declared or set()):
        return 'declared-but-absent', (
            f"requested filter {filtername} resolved to 0 candidate frames and "
            f"has no per-exposure frames on disk under ANY lineage, yet "
            f"fields.yaml declares it for {target}/{proposal_id} obs {field}.  "
            f"The reduction that writes {filtername} has not produced frames "
            f"under {basepath}/{filtername}/pipeline/ (looked for "
            f"each_suffix={each_suffix!r}).")
    return 'not-observed', (
        f"requested filter {filtername} has no frames on disk and is not "
        f"declared for {target}/{proposal_id} obs {field} in fields.yaml; "
        f"treating it as a band this observation never took and skipping it.")


def assert_requested_filters_have_frames(candidate_counts, *, target,
                                         proposal_id, field, basepath,
                                         each_suffix_for, declared=None,
                                         lineages_for=None, label='manual'):
    """Refuse a run whose requested filters resolve to zero frames.

    ``candidate_counts`` maps each REQUESTED filter to the number of candidate
    frames this run's own glob found, summed over modules and visits (zero for
    a module or visit a filter does not cover is normal; zero everywhere is
    not).  Returns the filters skipped as not-observed, and raises
    ``RequestedFilterHasNoFramesError`` naming every failing filter at once --
    one message, so an operator fixes all of them in one resubmission rather
    than discovering them one run at a time.

    ``declared`` and ``lineages_for`` (``filter -> {lineage: n}``) are injection
    points for tests.
    """
    if declared is None:
        declared = declared_for_observation(target, proposal_id, field)
    failures, skipped = [], []
    for filtername, n_candidates in candidate_counts.items():
        each_suffix = each_suffix_for(filtername)
        lineages = (lineages_for(filtername) if lineages_for is not None
                    else None)
        verdict, message = classify_requested_filter(
            filtername, n_candidates, target=target, proposal_id=proposal_id,
            field=field, basepath=basepath, each_suffix=each_suffix,
            declared=declared, lineages=lineages)
        if verdict == 'ok':
            continue
        if verdict == 'not-observed':
            skipped.append(filtername)
            print(f"[{label}] {message}", flush=True)
            continue
        failures.append(f"  {filtername}: {message}")
    if failures:
        raise RequestedFilterHasNoFramesError(
            f"[{label}] {len(failures)} requested filter(s) resolved to zero "
            f"candidate frames.  A filter named on the command line is an "
            f"assertion that it exists, so this run would have cataloged a "
            f"band short and reported success:\n" + '\n'.join(failures))
    return skipped
