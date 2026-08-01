"""Run the whole pipeline on one observation, with one command.

    python -m jwst_gc_pipeline.run_pipeline --proposal 2221 --obsid 001

That reduces, catalogs and merges every filter the observation has, submitting
each stage to SLURM so that each one waits for the last. From Python:

    from jwst_gc_pipeline.run_pipeline import run_pipeline
    run_pipeline(proposal=2221, obsid=1)

To try the whole chain in minutes rather than a day, give it a cutout:

    python -m jwst_gc_pipeline.run_pipeline --proposal 2221 --obsid 001 \\
           --cutout-region cutout.reg

A cutout run runs here in this shell rather than going to the queue.

**What the observation is** comes from ``fields.yaml`` — the target, its
filters, its data directory, its reference catalog. A proposal that is not
registered yet stops with the block to add. **Where it runs** comes from
``config.yaml``, which ships with HiPerGator's settings; see
:mod:`jwst_gc_pipeline.config`.

Add ``--dry-run`` to print the commands and submit nothing.
"""
import argparse
import os
import subprocess
import sys

from jwst_gc_pipeline import config as pipeline_config
from jwst_gc_pipeline import fields
from jwst_gc_pipeline.scratch_basepath import apply_basepath_override
from jwst_gc_pipeline.photometry.naming import MIRI_FILTERS
from jwst_gc_pipeline.reduction import destreak_policy

#: The stages, in the order they must run.
STAGES = ('reduce', 'catalog', 'merge')

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class NotRegisteredError(KeyError):
    """The observation is absent from fields.yaml."""


class CutoutStageError(ValueError):
    """A cutout was asked for alongside stages it cannot restrict.

    Caught in ``main`` and printed on its own, because the message says which
    stages to ask for instead and a traceback buries that.
    """


def _normalise_obsid(obsid):
    """``1`` and ``'1'`` and ``'001'`` all mean observation 001."""
    text = str(obsid).strip()
    return text if '-' in text else text.zfill(3)


def resolve(proposal, obsid, instrument='nircam', filters=None):
    """What the registry knows about one observation.

    Returns target, filters, basepath and the reference catalog. Raises
    :class:`NotRegisteredError` with the YAML to add when the observation is
    new — which is what a first run on fresh data hits.
    """
    proposal = str(proposal).strip()
    obsid = _normalise_obsid(obsid)
    try:
        target = fields.target_for_obsid(proposal, obsid, instrument)
    except KeyError:
        raise NotRegisteredError(
            f'proposal {proposal} observation {obsid} ({instrument}) is not in '
            f'fields.yaml.\n\n'
            f'Add it, and every stage picks the target up:\n\n'
            f'  <target-name>:\n'
            f'    root: orange              # or blue\n'
            f'    observations:\n'
            f"      '{proposal}':\n"
            f'        nvisits: 1\n'
            f'        reference_frame: VIRAC2\n'
            f'        obsids:\n'
            f"          {instrument}: ['{obsid}']\n"
            f'        reference_catalog:\n'
            f"          '{obsid}': catalogs/<your-refcat>.fits\n"
            f'        filters: [f200w, f405n]\n\n'
            f'See docs/FIELDS.md.')
    per_proposal = fields.obs_filters(instrument).get(target, {})
    filternames = per_proposal.get(proposal, [])
    if not filternames:
        raise NotRegisteredError(
            f'{target} proposal {proposal} has no {instrument} filters in '
            f'fields.yaml, so there is nothing to run.  Add a `filters:` list '
            f'to that observation.')
    # A NIRCam run must not carry F2550W, and a MIRI run must not carry F410M:
    # each instrument has its own driver and its own products.
    if instrument == 'miri':
        filternames = [f for f in filternames if f.lower() in MIRI_FILTERS]
    elif instrument == 'nircam':
        filternames = [f for f in filternames if f.lower() not in MIRI_FILTERS]
    if not filternames:
        raise NotRegisteredError(
            f'{target} proposal {proposal} has no {instrument} filters '
            f'registered, so there is nothing to run for that instrument.')
    if filters:
        wanted = [f.lower() for f in filters]
        unknown = [f for f in wanted if f not in [x.lower() for x in filternames]]
        if unknown:
            raise NotRegisteredError(
                f'{target} proposal {proposal} has no {unknown} in fields.yaml; '
                f'it has {sorted(filternames)}')
        filternames = [f for f in filternames if f.lower() in wanted]
    basepath = apply_basepath_override(fields.basepath(target))
    return {
        'target': target,
        'proposal': proposal,
        'obsid': obsid,
        'instrument': instrument,
        'filters': [f.upper() for f in filternames],
        'basepath': basepath,
        'reference_catalog': fields.reference_catalog_path(
            proposal, obsid, target=target, instrument=instrument,
            basepath=basepath),
        'each_suffix': destreak_policy.crf_suffix(
            target, filternames[0], obsid),
        # sickle destreaks its short filters and not its long ones, so no one
        # suffix is right for the observation.
        'suffix_by_filter': destreak_policy.suffixes_by_filter(
            target, filternames, obsid),
    }


def _array_bounds(plan, stage_name, config):
    """`--array=0-N` for a stage, from its fan-out and the filter list."""
    fan = pipeline_config.stage(config, stage_name).get('fan_out', 'none')
    if fan == 'filter':
        return len(plan['filters'])
    if fan == 'program-filter':
        return len(fields.merge_jobs(plan['target'], plan['instrument']))
    return 1


#: The stage-1 driver for each instrument.
REDUCE_DRIVERS = {
    'nircam': 'jwst_gc_pipeline/reduction/PipelineRerunNIRCAM-LONG.py',
    'miri': 'jwst_gc_pipeline/reduction/PipelineMIRI.py',
    'niriss': 'jwst_gc_pipeline/reduction/PipelineRerunNIRISS.py',
}

#: What stage 2 calls a detector for each instrument.
INSTRUMENT_MODULES = {'miri': 'mirimage', 'niriss': 'nis'}


def _modules_for(plan, stage):
    """The `--modules` value: NIRCam names a detector group, the others one
    detector."""
    return INSTRUMENT_MODULES.get(plan['instrument'],
                                  str(stage.get('modules', 'merged')))


def _suffix_overrides(plan):
    """``FILTER:suffix`` pairs, for an observation whose filters differ."""
    per_filter = plan.get('suffix_by_filter') or {}
    odd = {f: s for f, s in per_filter.items() if s != plan['each_suffix']}
    return ','.join(f'{f}:{s}' for f, s in sorted(odd.items()))


def _job_environment(plan, stage_name, config):
    """The variables a stage's job needs, for the submitting environment.

    Passed by exporting them here and giving sbatch a bare ``--export=ALL``.
    Listing them inside ``--export=ALL,K=V,...`` instead truncates any value
    containing a comma, and FILTERS contains spaces.
    """
    stage = pipeline_config.stage(config, stage_name)
    environ = dict(pipeline_config.environment(config))
    # The submit scripts prepend this to PYTHONPATH, so the jobs run this
    # checkout rather than the installed package.
    environ['PIPE_ROOT'] = REPO_ROOT
    environ['PROPOSAL'] = plan['proposal']
    environ['FIELD'] = plan['obsid']
    environ['TARGET'] = plan['target']
    environ['FILTERS'] = ' '.join(plan['filters'])
    if stage_name in ('reduce', 'catalog'):
        environ['MODULES'] = _modules_for(plan, stage)
    if plan['instrument'] != 'nircam':
        # Stage 2 and the merge read this; stage 1 uses its own driver.
        environ['GC_INSTRUMENT_OVERRIDE'] = plan['instrument']
    if stage_name == 'catalog':
        # These five travel together; the submit script exits 64 on a partial set.
        environ['EACH_SUFFIX'] = plan['each_suffix']
        overrides = _suffix_overrides(plan)
        if overrides:
            environ['EACH_SUFFIX_OVERRIDES'] = overrides
    if stage_name == 'reduce':
        environ['SKIP'] = '1' if stage.get('skip_step1and2', True) else '0'
    if stage_name == 'merge' and stage.get('fan_out') == 'program-filter':
        environ['MERGE_SINGLEFIELDS'] = '1'
    if config.get('python'):
        environ['PYTHON'] = config['python']
    return environ


def _sbatch_command(plan, stage_name, config, dependency=None):
    """One `sbatch` invocation, as a list of arguments."""
    stage = pipeline_config.stage(config, stage_name)
    slurm = config.get('slurm') or {}
    script = pipeline_config.submit_script(config, stage_name,
                                           plan['instrument'])
    name = _job_name(plan, stage_name)
    count = _array_bounds(plan, stage_name, config)

    command = ['sbatch', '--parsable']
    if count > 1:
        command.append(f'--array=0-{count - 1}')
    command += [f'--account={slurm.get("account")}',
                f'--qos={slurm.get("qos")}',
                f'--partition={slurm.get("partition")}',
                f'--cpus-per-task={stage["cpus"]}',
                f'--mem={stage["memory"]}',
                f'--time={stage["walltime"]}',
                f'--job-name={name}']
    command += _log_arguments(slurm, stage_name, array=count > 1)
    if dependency:
        command.append(f'--dependency=afterok:{dependency}')
    command += ['--export=ALL', script]
    return command


def _log_arguments(slurm, stage_name, array=False):
    """``--output`` for one job, from ``slurm.log_dir``.

    The submit scripts carry a log path in an ``#SBATCH`` directive, which SLURM
    reads before any shell runs, so it cannot expand a variable.  Passing
    ``--output`` on the command line overrides the directive, which is how a
    machine that is not HiPerGator gets its own log directory.

    ``%a`` is the array task, and SLURM writes ``4294967294`` for it on a job
    that is not an array, so it is used only when there is one.

    Builds the argument and nothing else; :func:`_make_log_dir` does the
    creating, once, at submission.
    """
    log_dir = (slurm.get('log_dir') or '').strip()
    if not log_dir:
        return []
    task = '%A_%a' if array else '%j'
    return [f'--output={os.path.join(log_dir, stage_name)}_%x_{task}.out']


def _make_log_dir(slurm):
    """Create ``slurm.log_dir`` before the first submission.

    SLURM refuses a job whose log has nowhere to go, and it refuses it at
    submit, so this is worth failing on early and by name.  Called only when
    something is really being submitted: building a command must not touch the
    filesystem, or ``--dry-run`` would create directories on a machine that has
    none of this.
    """
    log_dir = (slurm.get('log_dir') or '').strip()
    if not log_dir:
        return
    try:
        os.makedirs(log_dir, exist_ok=True)
    except OSError as problem:
        raise pipeline_config.ConfigError(
            f'slurm.log_dir {log_dir!r} cannot be created ({problem}).  '
            f'SLURM refuses a job whose log has nowhere to go; point '
            f'slurm.log_dir somewhere writable.')
    # makedirs(exist_ok=True) succeeds on a directory that exists and is not
    # writable, which is the off-site case: the shipped path is there and
    # belongs to someone else.
    if not os.access(log_dir, os.W_OK):
        raise pipeline_config.ConfigError(
            f'slurm.log_dir {log_dir!r} exists but is not writable.  SLURM '
            f'refuses a job whose log has nowhere to go; point slurm.log_dir '
            f'somewhere writable.')


def _job_name(plan, stage_name):
    """`<target><program>-o<obsid>-<stage>`, the naming convention.

    The merge is the exception: it covers every proposal of the target, so it
    is named for the target alone.
    """
    if stage_name == 'merge':
        return f"{plan['target']}-merge"
    return (f"{plan['target']}{plan['proposal']}-o{plan['obsid']}-{stage_name}")


def _merge_all_command(plan, config, dependency):
    """The single job that combines the filters, after the merge array."""
    stage = pipeline_config.stage(config, 'merge')
    slurm = config.get('slurm') or {}
    return ['sbatch', '--parsable',
            f'--account={slurm.get("account")}',
            f'--qos={slurm.get("qos")}',
            f'--partition={slurm.get("partition")}',
            f'--cpus-per-task={stage["cpus"]}',
            f'--mem={stage["memory"]}',
            f'--time={stage["walltime"]}',
            f"--job-name={plan['target']}-mergeall",
            *_log_arguments(slurm, 'mergeall'),
            f'--dependency=afterok:{dependency}',
            '--export=ALL',
            pipeline_config.submit_script(config, 'merge',
                                          plan['instrument'])]


def _local_commands(plan, config, cutout_region=None):
    """The three stages as plain commands, for a cutout or a local run."""
    python = config.get('python') or sys.executable
    filters = ','.join(plan['filters'])
    reduce_stage = pipeline_config.stage(config, 'reduce')
    reduce_command = [
        python, os.path.join(REPO_ROOT, REDUCE_DRIVERS[plan['instrument']]),
        '-p', plan['proposal'], '-d', plan['obsid'], '-f', filters]
    if plan['instrument'] == 'nircam':
        # Only NIRCam has modules to choose between.  The MIRI and NIRISS
        # drivers are single-detector and reject -m.
        reduce_command += ['-m', _modules_for(plan, reduce_stage)]
    if reduce_stage.get('skip_step1and2', True):
        reduce_command.append('-s')

    catalog_command = [
        python, '-m', 'jwst_gc_pipeline.photometry.crowdsource_catalogs_long',
        f"--proposal_id={plan['proposal']}", f"--field={plan['obsid']}",
        f"--target={plan['target']}", f'--filternames={filters}',
        f"--modules={_modules_for(plan, pipeline_config.stage(config, 'catalog'))}",
        '--each-exposure', f"--each-suffix={plan['each_suffix']}"]
    if plan['instrument'] != 'nircam':
        catalog_command.append(f"--instrument={plan['instrument']}")
    overrides = _suffix_overrides(plan)
    if overrides:
        catalog_command.append(f'--each-suffix-overrides={overrides}')
    if cutout_region:
        catalog_command.append(f'--cutout-region={cutout_region}')

    commands = [('reduce', reduce_command), ('catalog', catalog_command)]
    if not cutout_region:
        # A cutout writes under cutouts/<label>/, which the merge does not read,
        # and the all-filter merge needs every filter anyway.
        commands.append(('merge', [
            python, '-m', 'jwst_gc_pipeline.photometry.merge_catalogs',
            f"--target={plan['target']}", '--merge-singlefields']))
    return commands


def run_pipeline(proposal, obsid, cutout_region=None, instrument='nircam',
                 stages=STAGES, dry_run=False, config_path=None, project=None,
                 filters=None):
    """Reduce, catalog and merge one observation.

    Parameters
    ----------
    proposal : str or int
        The proposal (program) id. ``project`` is accepted as an alias.
    obsid : str or int
        The observation number; ``1`` and ``'001'` mean the same thing.
    filters : list of str, optional
        Restrict to these filters. A cutout is usually worth running on one.
    cutout_region : str, optional
        A DS9 region file or ``'ra,dec,size'``. Runs here rather than on the
        queue, and stops after cataloging.
    dry_run : bool
        Print the commands and submit nothing.

    Returns a dict describing what was run or would be.
    """
    if project is not None:
        proposal = project
    config = pipeline_config.load(config_path)
    plan = resolve(proposal, obsid, instrument, filters=filters)

    scheduler = config.get('scheduler')
    if cutout_region:
        # The cutout reaches the cataloging command only.  Anywhere else it
        # would be accepted and then ignored: the run would reduce the whole
        # observation -- hours, on the queue, when minutes were asked for.
        if 'catalog' not in stages:
            raise CutoutStageError(
                f"--cutout-region needs the catalog stage: it restricts the "
                f"cataloging, and stage 1 always reduces the whole "
                f"observation.  Asked for stages {list(stages)}.  Reduce first "
                f"(--stages reduce), then run the cutout "
                f"(--stages catalog --cutout-region ...).")
        if 'merge' in stages and tuple(stages) != tuple(STAGES):
            # Asked for by name.  A cutout writes under cutouts/<label>/, which
            # the merge does not read, so it would run on nothing.  (Left in the
            # default set, it is dropped with a note below -- the documented
            # one-liner passes no stages at all.)
            raise CutoutStageError(
                f"--cutout-region cannot run the merge: a cutout writes under "
                f"cutouts/<label>/, which the merge does not read.  Asked for "
                f"stages {list(stages)}.  Use --stages reduce,catalog (or "
                f"--stages catalog if the frames are already reduced).")
        scheduler = (config.get('cutout') or {}).get('scheduler', 'local')
        if scheduler != 'local':
            # Only the local path threads cutout_region into the command; a
            # submitted job would run the whole observation.
            raise CutoutStageError(
                f"--cutout-region needs cutout.scheduler: local in "
                f"{config['source']}, which says {scheduler!r}.  A submitted "
                f"job carries no cutout and would catalog the whole "
                f"observation.")

    print(f"{plan['target']} proposal {plan['proposal']} observation "
          f"{plan['obsid']} ({plan['instrument']})")
    print(f"  filters:   {' '.join(plan['filters'])}")
    print(f"  directory: {plan['basepath']}")
    print(f"  refcat:    {plan['reference_catalog']}")
    print(f"  config:    {config['source']} (scheduler: {scheduler})")

    if 'merge' in stages and scheduler == 'slurm':
        every = fields.merge_jobs(plan['target'], plan['instrument'])
        proposals = sorted({proposal for proposal, _ in every})
        if len(proposals) > 1:
            print(f"  note: the merge covers every proposal of "
                  f"{plan['target']} ({', '.join(proposals)}), so it runs "
                  f"{len(every)} tasks rather than one per filter of this "
                  f"observation")

    if cutout_region and 'merge' in stages:
        print("  note: a cutout stops after cataloging.  It writes under "
              "cutouts/<label>/, which the merge does not read.")

    if cutout_region and len(plan['filters']) > 1:
        print(f"  note: this cutout runs {len(plan['filters'])} filters one "
              f"after another in this shell.  --filters F410M keeps it to "
              f"minutes.")

    if scheduler == 'local':
        return _run_local(plan, config, cutout_region, stages, dry_run)
    return _submit_slurm(plan, config, stages, dry_run)


def _run_local(plan, config, cutout_region, stages, dry_run):
    commands = [(name, command) for name, command in
                _local_commands(plan, config, cutout_region) if name in stages]
    environ = dict(os.environ)
    environ.update(pipeline_config.environment(config))
    # Stage 1 is a script path, so Python puts ITS directory on sys.path and
    # `import jwst_gc_pipeline` finds whatever is pip-installed.  Stage 2 uses
    # `python -m` and finds this checkout.  Without this the two stages run
    # different versions of the package.
    environ['PYTHONPATH'] = os.pathsep.join(
        [REPO_ROOT] + ([environ['PYTHONPATH']] if environ.get('PYTHONPATH') else []))
    for name, command in commands:
        print(f'\n=== {name} ===\n{" ".join(command)}', flush=True)
        if dry_run:
            continue
        result = subprocess.run(command, env=environ, cwd=REPO_ROOT)
        if result.returncode != 0:
            raise SystemExit(f'{name} exited {result.returncode}')
    return {'plan': plan, 'mode': 'local',
            'commands': [command for _, command in commands]}


def _submit_slurm(plan, config, stages, dry_run):
    if not dry_run:
        _make_log_dir(config.get('slurm') or {})
    submitted, dependency = {}, None
    for name in STAGES:
        if name not in stages:
            continue
        command = _sbatch_command(plan, name, config, dependency)
        environ = dict(os.environ)
        environ.update(_job_environment(plan, name, config))
        print(f'\n=== {name} ===\n{" ".join(command)}', flush=True)
        print('  ' + '  '.join(f'{k}={v!r}' for k, v
                               in sorted(_job_environment(plan, name, config).items())),
              flush=True)
        if dry_run:
            dependency = f'<{name}-job-id>'
        else:
            dependency = subprocess.run(
                command, capture_output=True, text=True, env=environ,
                check=True).stdout.strip()
            print(f'  submitted {dependency}')
        submitted[name] = dependency
        if name == 'merge' and pipeline_config.stage(
                config, 'merge').get('fan_out') == 'program-filter':
            tail = _merge_all_command(plan, config, dependency)
            tail_env = dict(os.environ)
            tail_env.update(_job_environment(plan, 'merge', config))
            tail_env.pop('MERGE_SINGLEFIELDS', None)
            print(f'\n=== merge (all filters) ===\n{" ".join(tail)}', flush=True)
            if not dry_run:
                submitted['merge_all'] = subprocess.run(
                    tail, capture_output=True, text=True, env=tail_env,
                    check=True).stdout.strip()
                print(f"  submitted {submitted['merge_all']}")
    return {'plan': plan, 'mode': 'slurm', 'jobs': submitted}


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='python -m jwst_gc_pipeline.run_pipeline',
        description='Reduce, catalog and merge one observation.')
    parser.add_argument('--proposal', '--project', required=True,
                        help='proposal (program) id, e.g. 2221')
    parser.add_argument('--obsid', '--field', required=True,
                        help="observation number, e.g. 001 (or 1)")
    parser.add_argument('--instrument', default='nircam',
                        choices=('nircam', 'miri', 'niriss'))
    parser.add_argument('--cutout-region', default=None,
                        help='DS9 region file or ra,dec,size -- runs here, in '
                             'minutes, and stops after cataloging')
    parser.add_argument('--stages', default=','.join(STAGES),
                        help='comma-separated subset of reduce,catalog,merge')
    parser.add_argument('--config', default=None,
                        help='config file (default: GC_PIPELINE_CONFIG, else '
                             'the packaged one)')
    parser.add_argument('--filters', default=None,
                        help='comma-separated subset of the observation\'s '
                             'filters; a cutout is usually worth one filter')
    parser.add_argument('--dry-run', action='store_true',
                        help='print the commands and submit nothing')
    args = parser.parse_args(argv)
    chosen = tuple(s.strip() for s in args.stages.split(','))
    unknown = [s for s in chosen if s not in STAGES]
    if unknown:
        parser.error(f'--stages names {unknown}; the stages are '
                     f'{", ".join(STAGES)}')
    try:
        run_pipeline(proposal=args.proposal, obsid=args.obsid,
                     cutout_region=args.cutout_region,
                     instrument=args.instrument,
                     filters=([f.strip() for f in args.filters.split(',')]
                              if args.filters else None),
                     stages=chosen,
                     dry_run=args.dry_run, config_path=args.config)
    except (NotRegisteredError, fields.FieldRegistryError,
            pipeline_config.ConfigError, CutoutStageError) as problem:
        # These say what to add to which file; a traceback buries that.
        message = problem.args[0] if problem.args else str(problem)
        print(f'\n{message}\n', file=sys.stderr)
        raise SystemExit(2)


if __name__ == '__main__':
    main()
