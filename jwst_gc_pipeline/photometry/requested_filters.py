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

1. **The lineage suffix is wrong.**  NONE of the frames on disk are under a
   suffix this run asked for; they are all under a different one.  The request
   and the disk disagree; raise, and name the suffix that would have worked so
   the operator can set ``--each-suffix-overrides``.
2. **The observation took this filter and its frames are absent.**  The field
   registry (``fields.yaml``) declares the filter for this proposal/observation,
   so the reduction that should have produced the frames has not run (or wrote
   them somewhere else).  Raise: cataloging the rest and reporting success ships
   a product short a band.
3. **The requested suffix is right and the frames are out of scope.**  At least
   some frames are on disk under a suffix this run DID ask for, so the suffix is
   not what missed them -- the run's own ``--modules`` or visit range is.  Raise,
   with wording that does not send the reader to ``--each-suffix-overrides``.
   This case WINS over case 1 whenever both could apply, because switching
   lineage on a field that has both on disk is the ~106 mas two-lineage mix the
   project treats as a hazard (``sickle-two-lineage-catalog-mixing``): brick,
   cloudc, cloudef and sickle all carry ``o001_crf`` AND ``destreak_o001_crf``
   for the same filter, and on those fields case 1 used to fire and recommend
   the mix.
4. **The observation never took this filter.**  It is not declared for this
   observation and nothing is on disk under any lineage.  A caller sweeping one
   filter list across several fields legitimately names bands a given field does
   not have, so this is reported and DROPPED FROM THE RUN, not raised.  Dropping
   is what makes the skip real: a filter merely logged and left in
   ``filternames`` walks into the phase loop and dies there on
   ``no {filt}/{module} frames produced output in phase {phase}``
   (``cataloging.py``), which is the raise this verdict exists to avoid.

Cases 1-3 are the failure; case 4 is the legitimate one.

Case 2 -- and ONLY case 2 -- has an operator override,
``ALLOW_ABSENT_REQUESTED_FILTER=1`` with a written
``ALLOW_ABSENT_REQUESTED_FILTER_REASON`` (the ``<VAR>_REASON`` convention
``astrometry_checkpoint`` uses).  It downgrades the raise to a loud warning and
drops the band, so the rest of the run's bands still produce.  It exists for a
scheduling fact the operator cannot fix by editing the command line -- a
Treasury tile whose F480M reduction still lags its F212N on 2026-09-10 -- and it
deliberately does NOT cover cases 1 and 3, which are a mis-specified command
line that a resubmission fixes in seconds.
"""
import glob
import os
import re

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


#: Env override for the ``declared-but-absent`` verdict ONLY, with the
#: ``<VAR>_REASON`` companion ``astrometry_checkpoint.override_reason_env``
#: defines.  Named like ``ALLOW_LATE_STAGE_ASTROM_SHIFT`` /
#: ``ALLOW_REGISTRATION_FAIL``: an explicit, greppable decision.
ALLOW_ABSENT_ENV = 'ALLOW_ABSENT_REQUESTED_FILTER'


def requested_lineages(each_suffix, field):
    r"""Every lineage string this run's own glob actually asks for.

    A hyphen-joined joint field (``'002-998'``) is not one request but one per
    subfield: ``crowdsource_catalogs_long.get_filenames`` rewrites the
    observation token of ``each_suffix`` per subfield --
    ``re.sub(r'o\d{3}_crf', f'o{sf}_crf', each_suffix)`` -- so a run asking for
    ``o002_crf`` on field ``002-998`` globs ``o002_crf`` AND ``o998_crf``.  Both
    are the requested lineage; comparing the raw string against what is on disk
    reads the second one as somebody else's and calls a joint field
    wrong-lineage on every call.  The rewrite here is the same expression, kept
    deliberately identical to the globber's.

    Two registered joint fields go through this: sickle 3958 MIRI ``001-002``
    and 5365 MIRI ``002-998``.
    """
    subfields = str(field).split('-') if '-' in str(field) else [str(field)]
    if len(subfields) < 2:
        return {str(each_suffix)}
    return {re.sub(r'o\d{3}_crf', f'o{sf}_crf', str(each_suffix))
            for sf in subfields}


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

    NIRISS is deliberately NOT in the fallback.  ``fields.declared_filters``
    defaults to ``NIRCAM_MIRI`` for the reason its docstring gives: NIRCam and
    MIRI share the per-observation ``filters`` list AND the
    ``{FILTER}/pipeline/`` tree this module globs, while NIRISS carries its own
    ``niriss_filters`` and its own layout.  sgrc/4147 declares F158M/F200W/F356W
    for NIRISS, and none of them can ever have a NIRCam pipeline directory --
    unioning NIRISS in would turn a NIRCam run that names F356W on sgrc from a
    ``not-observed`` skip into a ``declared-but-absent`` refusal no reduction
    could clear.  ``obs_filters('nircam')`` and ``obs_filters('miri')`` return
    the same flat list, so one call covers both.

    Empty only when the proposal itself is unregistered, which is the "cannot
    tell" answer: the caller then falls back to what is on disk.
    """
    from jwst_gc_pipeline import fields as field_registry
    per_obs = field_registry.filters_for_observation(target, proposal_id, field)
    if per_obs:
        return {str(f).upper() for f in per_obs}
    by_proposal = field_registry.obs_filters('nircam').get(target) or {}
    return {str(f).upper()
            for f in (by_proposal.get(str(proposal_id)) or ())}


def classify_requested_filter(filtername, n_candidates, *, target, proposal_id,
                              field, basepath, each_suffix, declared=None,
                              lineages=None):
    """``(verdict, message)`` for one requested filter.

    ``verdict`` is ``'ok'`` (frames found), ``'wrong-lineage'``,
    ``'outside-this-run'``, ``'declared-but-absent'`` or ``'not-observed'``.
    Everything but ``'ok'`` and ``'not-observed'`` is a failure;
    ``'not-observed'`` carries a message for the log.  ``declared`` and
    ``lineages`` are injectable so a test does not need a data tree.

    ``'outside-this-run'`` WINS over ``'wrong-lineage'``: any frame under a
    lineage this run globs (``requested_lineages`` -- plural, because a joint
    field asks for one per subfield) means the suffix is right.  Only when NONE
    of the requested lineages is on disk is the suffix the thing to change.
    """
    if n_candidates > 0:
        return 'ok', ''
    if lineages is None:
        lineages = frame_lineages_on_disk(basepath, filtername, proposal_id,
                                          field, each_suffix=each_suffix)
    if declared is None:
        declared = declared_for_observation(target, proposal_id, field)
    wanted = requested_lineages(each_suffix, field)
    asked_for = {k: v for k, v in lineages.items() if k in wanted}
    other = {k: v for k, v in lineages.items() if k not in wanted}
    if lineages:
        # ORDER MATTERS.  `asked_for` is tested FIRST: on a field that carries
        # BOTH lineages for the same filter (brick/2221 obs 001 F410M is
        # {'o001_crf': 48, 'destreak_o001_crf': 48}, and cloudc/cloudef/sickle
        # are the same shape) the earlier `other`-first order returned
        # wrong-lineage and recommended --each-suffix-overrides -- i.e. it told
        # the operator to switch to the OTHER lineage while 48 frames sat under
        # the one requested.  That recommendation is the ~106 mas two-lineage
        # mix (`sickle-two-lineage-catalog-mixing`), and the two-lineage fields
        # are precisely where it fired.  Frames under a requested suffix mean
        # the suffix is right, whatever else shares the directory.
        if asked_for:
            _also = ('' if not other else
                     f"  ({', '.join(f'{k} (n={v})' for k, v in sorted(other.items()))} "
                     f"also sit(s) in the same directory under a lineage this "
                     f"run did not ask for; that is not the cause and switching "
                     f"to it would MIX lineages.)")
            return 'outside-this-run', (
                f"requested filter {filtername} resolved to 0 candidate frames "
                f"although {sum(asked_for.values())} frame(s) are on disk under "
                f"the requested each_suffix={each_suffix!r} "
                f"(lineage(s) {sorted(asked_for)}): none of them fall in this "
                f"run's modules or visit range.  Check --modules and the field's "
                f"nvisits.{_also}")
        _found = ', '.join(f'{k} (n={v})' for k, v in sorted(lineages.items()))
        _pick = sorted(other, key=lambda k: (-other[k], k))[0]
        return 'wrong-lineage', (
            f"requested filter {filtername} resolved to 0 candidate frames "
            f"with each_suffix={each_suffix!r}, and NO frame is on disk under "
            f"any lineage this run globs ({sorted(wanted)}), but "
            f"{sum(lineages.values())} frame(s) are on disk under: {_found}. "
            f" Set --each-suffix-overrides={filtername.upper()}:{_pick} "
            f"(EACH_SUFFIX_OVERRIDES in the submit scripts) to read that "
            f"lineage for this filter.")
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
    not).  Raises ``RequestedFilterHasNoFramesError`` naming every failing
    filter at once -- one message, so an operator fixes all of them in one
    resubmission rather than discovering them one run at a time.

    RETURNS THE FILTERS THE CALLER MUST DROP from its filter list: the
    not-observed ones, plus any ``declared-but-absent`` one waived by
    ``ALLOW_ABSENT_REQUESTED_FILTER=1``.  The caller has to act on the return
    value; a filter that is only logged stays in ``filternames`` and dies later
    in the phase loop on ``no {filt}/{module} frames produced output``, which is
    exactly the raise the not-observed verdict exists to avoid.

    ``declared`` and ``lineages_for`` (``filter -> {lineage: n}``) are injection
    points for tests.
    """
    if declared is None:
        declared = declared_for_observation(target, proposal_id, field)
    waive_absent = os.environ.get(ALLOW_ABSENT_ENV, '').strip() == '1'
    waive_reason = os.environ.get(f'{ALLOW_ABSENT_ENV}_REASON', '').strip()
    failures, drop = [], []
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
            drop.append(filtername)
            print(f"[{label}] {message}", flush=True)
            continue
        if verdict == 'declared-but-absent' and waive_absent:
            # The one waivable verdict.  A band the schedule has not delivered
            # yet is not something the operator can fix by editing the command
            # line, and refusing the whole run costs the bands that ARE ready
            # (a Treasury tile whose F480M lags its F212N).  Loud, greppable,
            # and the justification is printed beside it.
            drop.append(filtername)
            print(f"[{label}] WARNING (override {ALLOW_ABSENT_ENV}=1): "
                  f"{message}  Dropping {filtername} from this run; its "
                  f"products will be absent.", flush=True)
            if waive_reason:
                print(f"[{label}]   override justification: {waive_reason}",
                      flush=True)
            else:
                print(f"[{label}]   NO JUSTIFICATION RECORDED.  Set "
                      f"{ALLOW_ABSENT_ENV}_REASON to say why this band is "
                      f"absent.", flush=True)
            continue
        failures.append(f"  {filtername}: {message}")
    if failures:
        _waivable = [f for f in failures if 'fields.yaml declares it' in f]
        _hint = ('' if not _waivable else
                 f"\n\nA declared-but-absent band whose reduction is still "
                 f"running can be waived with {ALLOW_ABSENT_ENV}=1 plus a "
                 f"written {ALLOW_ABSENT_ENV}_REASON; the band is then dropped "
                 f"and the run's other bands still produce.")
        raise RequestedFilterHasNoFramesError(
            f"[{label}] {len(failures)} requested filter(s) resolved to zero "
            f"candidate frames.  A filter named on the command line is an "
            f"assertion that it exists, so this run would have cataloged a "
            f"band short and reported success.  THE WHOLE RUN STOPS, including "
            f"the filters that did resolve and including a fan-out shard, which "
            f"used to print `nothing to fit` and let the array finish the other "
            f"bands:\n" + '\n'.join(failures) + _hint)
    return drop
