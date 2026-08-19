"""The two call sites that decide which observation a merged catalog covers.

``merge_individual_frames`` globs the per-frame catalogs of one observation and
writes one merged catalog; which observation it covers comes from its ``field``
argument, and the rule for filling that argument is
``naming.merge_field_for_proposal`` -- the field for a per-obs-merged proposal
(10678), ``None`` for every other, whose merged names stay as they are on disk.

The rule used to be spelled inline at both call sites: the manual pipeline's
per-stage merge (``cataloging.run_manual_pipeline``) and the cutout run's
across-exposure merge (``crowdsource_catalogs_long.main``).  Both sit thousands
of lines inside functions no test drives, so the argument was pinned by reading
the source rather than by running it, and a dropped ``field=`` kwarg passed
every test while the cutout run silently produced no merged catalog: the
refusal it triggers lands in the per-method print-and-continue handler.

So the two invocations live here, where a test calls them with a recording
``merge`` and reads back the ``field`` and ``progid`` that reached the merge.
"""
from .naming import merge_field_for_proposal

__all__ = ['merge_frames_for_observation', 'merge_cutout_catalogs',
           'CUTOUT_MERGE_ERRORS']

#: Failures of ONE cutout merge method that leave the other method worth
#: running: a missing or unreadable input (``OSError``), a merge that finds no
#: frames or refuses its arguments (``ValueError``), a column or row the
#: combine step expected (``KeyError``, ``IndexError``).  Anything else --
#: a signature drift (``TypeError``), an interrupt, an import failure -- stops
#: the cutout run, as it did before this handler existed.
CUTOUT_MERGE_ERRORS = (OSError, ValueError, KeyError, IndexError)


def merge_frames_for_observation(proposal_id, field, *, merge=None, **kwargs):
    """``merge_individual_frames`` scoped to the observation under reduction.

    ``proposal_id`` and ``field`` are this run's, ``field`` being the RESOLVED
    value (the registry default where ``--field`` was omitted).  The merge is
    handed ``progid=proposal_id`` and the field ``merge_field_for_proposal``
    allows: the tile for a per-obs-merged proposal, ``None`` for every other,
    which keeps gc2211's all-obs pooling and the single-obs targets' untokened
    merged names byte-identical.

    ``merge`` is the callable to run, for tests; production passes nothing and
    gets ``merge_catalogs.merge_individual_frames``.  Every other keyword goes
    through untouched.
    """
    if merge is None:
        from .merge_catalogs import merge_individual_frames as merge
    return merge(progid=proposal_id,
                 field=merge_field_for_proposal(proposal_id, field),
                 **kwargs)


def merge_cutout_catalogs(*, proposal_id, field, target, module, filtername,
                          basepath, fwhm_basepath, options, merge=None):
    """Merge a cutout run's per-exposure catalogs, one call per dao method.

    A cutout run writes its per-exposure catalogs under its own tree, so it
    merges them there (``basepath``); only the per-filter
    ``reduction/fwhm_table.ecsv`` is read from the real target tree
    (``fwhm_basepath``), which a cutout tree does not carry.

    The observation is resolved ONCE, before the loop: a per-obs-merged
    proposal with no field raises here, where the raise stops the run.  Inside
    the loop it would reach the per-method handler below and end the run with
    one printed line and no merged catalog -- the failure this pass-through
    exists to prevent.
    """
    methods = [('dao', '_basic')]
    if not options.basic_only:
        methods.append(('daoiterative', '_iterative'))
    merge_field = merge_field_for_proposal(proposal_id, field)
    if merge is None:
        from .merge_catalogs import merge_individual_frames as merge
    for method, suffix in methods:
        try:
            merge(module=module, filtername=filtername.lower(),
                  progid=proposal_id, method=method, suffix=suffix,
                  target=target, basepath=basepath,
                  iteration_label=options.iteration_label or None,
                  bgsub=options.bgsub, desat=options.desaturated,
                  epsf=options.epsf, blur=options.blur,
                  resbgsub=getattr(options, 'use_iter3_residual_bg', False),
                  field=merge_field, fwhm_basepath=fwhm_basepath)
            print(f"cutout: wrote merged {method} catalog under "
                  f"{basepath}/catalogs/", flush=True)
        except CUTOUT_MERGE_ERRORS as ex:
            print(f"cutout: merge_individual_frames({method}) failed: {ex}",
                  flush=True)
