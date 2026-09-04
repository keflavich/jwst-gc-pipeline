"""The one-command runner: does it plan the right work, and say so clearly?

These check the plan and the commands rather than submitting anything.
"""
import fnmatch
import os

import pytest

from jwst_gc_pipeline import config as pipeline_config
from jwst_gc_pipeline import run_pipeline as rp
from jwst_gc_pipeline.reduction import destreak_policy


# --------------------------------------------------------------------------
# What the registry is asked for.
# --------------------------------------------------------------------------

@pytest.mark.parametrize('given', [1, '1', '001'])
def test_an_observation_number_is_accepted_in_any_form(given):
    assert rp.resolve('2221', given)['obsid'] == '001'


def test_a_joint_observation_number_is_left_alone():
    """Sgr B2's MIRI observations 002 and 998 are cataloged as '002-998'."""
    assert rp._normalise_obsid('002-998') == '002-998'


def test_the_plan_names_the_target_filters_and_reference_catalog():
    plan = rp.resolve('2221', '001')
    assert plan['target'] == 'brick'
    assert 'F410M' in plan['filters']
    assert plan['each_suffix'] == 'destreak_o001_crf'
    assert plan['reference_catalog'].endswith(
        'catalogs/gaia_virac2_refcat_epoch2022.70.fits')


def test_the_same_observation_number_resolves_per_instrument():
    """Proposal 2221 observation 001 is brick under NIRCam and cloudc under
    MIRI, and each has its own reference catalog."""
    assert rp.resolve('2221', '001')['target'] == 'brick'
    assert rp.resolve('2221', '001', instrument='miri')['target'] == 'cloudc'


def test_an_unregistered_proposal_says_what_to_add():
    """The first thing new data hits.  A bare KeyError would leave the user
    guessing which of six files to edit."""
    with pytest.raises(rp.NotRegisteredError) as problem:
        rp.resolve('99999', '001')
    message = str(problem.value)
    assert 'fields.yaml' in message
    assert "'99999':" in message
    assert 'reference_catalog:' in message
    assert 'docs/FIELDS.md' in message


def test_asking_for_a_filter_the_observation_lacks_says_which_it_has():
    with pytest.raises(rp.NotRegisteredError, match='f999w'):
        rp.resolve('2221', '001', filters=['F999W'])


def test_restricting_filters_keeps_only_those():
    plan = rp.resolve('2221', '001', filters=['F410M'])
    assert plan['filters'] == ['F410M']


# --------------------------------------------------------------------------
# What gets submitted.
# --------------------------------------------------------------------------

def test_each_stage_waits_for_the_one_before(capsys):
    rp.run_pipeline('2221', '001', dry_run=True)
    out = capsys.readouterr().out
    assert out.count('--dependency=afterok:') == 3   # catalog, merge, merge-all
    assert out.index('=== reduce ===') < out.index('=== catalog ===')
    assert out.index('=== catalog ===') < out.index('=== merge ===')


def test_the_array_size_comes_from_the_configured_fan_out():
    config = pipeline_config.load()
    plan = rp.resolve('2221', '001')
    # reduce and catalog fan out per filter; the merge per (program, filter).
    assert rp._array_bounds(plan, 'reduce', config) == len(plan['filters'])
    assert rp._array_bounds(plan, 'merge', config) == 11    # brick, both programs


def test_the_job_environment_carries_the_five_catalog_variables():
    """submit_cataloging.sbatch exits 64 unless all five are set."""
    config = pipeline_config.load()
    environ = rp._job_environment(rp.resolve('2221', '001'), 'catalog', config)
    for name in ('PROPOSAL', 'FIELD', 'TARGET', 'EACH_SUFFIX', 'MODULES'):
        assert environ.get(name), name


def test_filters_go_through_the_environment_rather_than_the_export_list():
    """FILTERS contains spaces and --export splits on commas, so listing
    variables inside --export=ALL,K=V truncates them."""
    config = pipeline_config.load()
    command = rp._sbatch_command(rp.resolve('2221', '001'), 'reduce', config)
    assert '--export=ALL' in command
    assert not any(part.startswith('--export=ALL,') for part in command)
    assert ' ' in rp._job_environment(
        rp.resolve('2221', '001'), 'reduce', config)['FILTERS']


def test_job_names_carry_target_program_and_observation():
    """The naming convention: several reduce jobs are usually queued at once."""
    config = pipeline_config.load()
    command = rp._sbatch_command(rp.resolve('2221', '001'), 'reduce', config)
    assert '--job-name=brick2221-o001-reduce' in command


# --------------------------------------------------------------------------
# Cutouts.
# --------------------------------------------------------------------------

def test_a_cutout_runs_here_rather_than_on_the_queue(capsys):
    rp.run_pipeline('2221', '001', filters=['F410M'],
                    cutout_region='266.535,-28.705,20', dry_run=True)
    out = capsys.readouterr().out
    assert 'scheduler: local' in out
    # No command submitted.  Matching the bare word would also match a checkout
    # path that happens to contain it.
    assert not any(line.strip().startswith('sbatch')
                   for line in out.splitlines())
    assert '--cutout-region=266.535,-28.705,20' in out


def test_a_cutout_stops_after_cataloging(capsys):
    """The merge reads <basepath>/catalogs/ and needs every filter; a cutout
    writes under cutouts/<label>/ and is usually one filter."""
    rp.run_pipeline('2221', '001', filters=['F410M'],
                    cutout_region='266.535,-28.705,20', dry_run=True)
    out = capsys.readouterr().out
    assert '=== merge ===' not in out


def test_a_full_run_does_include_the_merge(capsys):
    rp.run_pipeline('2221', '001', dry_run=True)
    assert '=== merge ===' in capsys.readouterr().out


# --------------------------------------------------------------------------
# Configuration.
# --------------------------------------------------------------------------

def test_the_packaged_config_is_hipergator():
    config = pipeline_config.load()
    assert config['slurm']['account'] == 'astronomy-dept'
    assert config['slurm']['qos'] == 'astronomy-dept-b'
    assert config['scheduler'] == 'slurm'


def test_a_copy_need_only_contain_what_differs(tmp_path, monkeypatch):
    mine = tmp_path / 'mine.yaml'
    mine.write_text('slurm:\n  qos: my-qos\n')
    monkeypatch.setenv(pipeline_config.ENV_VAR, str(mine))
    config = pipeline_config.load()
    assert config['slurm']['qos'] == 'my-qos'
    assert config['slurm']['account'] == 'astronomy-dept'   # kept
    assert config['stages']['reduce']['cpus'] == 16          # kept


def test_an_environment_variable_already_set_wins(monkeypatch):
    monkeypatch.setenv('CRDS_PATH', '/somewhere/else')
    assert pipeline_config.environment(
        pipeline_config.load())['CRDS_PATH'] == '/somewhere/else'


def test_a_config_naming_a_missing_file_says_so(monkeypatch):
    monkeypatch.setenv(pipeline_config.ENV_VAR, '/no/such/config.yaml')
    with pytest.raises(pipeline_config.ConfigError, match='not there'):
        pipeline_config.load()


@pytest.mark.parametrize('bad,match', [
    ({'scheduler': 'condor'}, 'scheduler'),
    ({'stages': {'reduce': {'fan_out': 'everything'}}}, 'fan_out'),
])
def test_a_config_the_runner_cannot_act_on_is_rejected(tmp_path, monkeypatch,
                                                       bad, match):
    import yaml
    mine = tmp_path / 'bad.yaml'
    mine.write_text(yaml.safe_dump(bad))
    monkeypatch.setenv(pipeline_config.ENV_VAR, str(mine))
    with pytest.raises(pipeline_config.ConfigError, match=match):
        pipeline_config.load()


def test_every_submit_script_the_config_names_exists():
    """A missing script would surface as an sbatch failure at submit time."""
    config = pipeline_config.load()
    for stage_name in ('reduce', 'catalog', 'merge'):
        for instrument in (pipeline_config.stage(config, stage_name)
                           ['submit_script']):
            assert os.path.exists(
                pipeline_config.submit_script(config, stage_name, instrument))


def test_an_instrument_with_no_submit_script_says_so():
    """Falling back to another instrument's script would run the wrong stage-1
    driver."""
    config = pipeline_config.load()
    with pytest.raises(pipeline_config.ConfigError, match='no submit script'):
        pipeline_config.submit_script(config, 'reduce', 'nirspec')


def test_every_instrument_the_runner_drives_can_be_submitted():
    """Each instrument with a stage-1 driver has a reduce submit script.

    MIRI had a driver and no script, so `run_pipeline --instrument miri` got as
    far as stage 1 and then refused.
    """
    config = pipeline_config.load()
    for instrument in rp.REDUCE_DRIVERS:
        assert pipeline_config.submit_script(config, 'reduce', instrument)


def test_a_submit_script_that_is_not_there_is_caught(tmp_path, monkeypatch):
    mine = tmp_path / 'c.yaml'
    mine.write_text('stages:\n  reduce:\n    submit_script:\n'
                    '      nircam: scripts/reduction/nope.sbatch\n')
    monkeypatch.setenv(pipeline_config.ENV_VAR, str(mine))
    with pytest.raises(pipeline_config.ConfigError, match='is not there'):
        pipeline_config.submit_script(pipeline_config.load(), 'reduce')


# --------------------------------------------------------------------------
# Findings from the onboarding review.
# --------------------------------------------------------------------------

def test_a_nircam_run_leaves_out_the_miri_filters():
    """brick/2221 registers F2550W, which is MIRI; array task 6 would have run
    the NIRCam driver on it."""
    assert 'F2550W' not in rp.resolve('2221', '001')['filters']
    assert rp.resolve('2221', '002', instrument='miri')['filters'] == ['F2550W']


def test_the_suffix_follows_the_destreak_policy():
    """Four registered targets destreak off and write *_align_o<obs>_crf; a
    hard-coded destreak_ suffix would find no files."""
    assert rp.resolve('2221', '001')['each_suffix'] == 'destreak_o001_crf'
    assert rp.resolve('6778', '001')['each_suffix'] == 'align_o001_crf'


def test_a_miri_plan_asks_for_the_lineage_miri_writes():
    """MIRI's driver has no destreak step and tags nothing.

    ``PipelineMIRI`` names the per-exposure frame straight off the ``_cal``
    stem (line 663) -- ``..._mirimage_o002_crf.fits`` -- so a plan asking for
    ``destreak_o002_crf`` sends ``get_filenames`` after a name no MIRI frame on
    disk carries, and the cataloging stage refuses with zero candidate frames.
    """
    plan = rp.resolve('2221', '002', instrument='miri')
    assert plan['each_suffix'] == 'o002_crf'
    assert plan['suffix_by_filter'] == {'F2550W': 'o002_crf'}
    # The glob crowdsource_catalogs_long.get_filenames builds from it has to
    # match the name PipelineMIRI writes -- and the ``_align_o<obs>_crf``
    # variant an older Image3 crf-naming branch left on brick/F2550W.
    for written in ('jw02221002001_02101_00001_mirimage_o002_crf.fits',
                    'jw02221002001_02101_00001_mirimage_align_o002_crf.fits'):
        assert fnmatch.fnmatch(written, f"*mirimage*{plan['each_suffix']}.fits")


def test_niriss_has_no_lineage_token_either():
    """``PipelineRerunNIRISS`` writes ``_o<obs>_crf`` too (line 525), and its
    filter names are NIRCam's, so only the instrument argument can say so."""
    plan = rp.resolve('4147', '012', instrument='niriss')
    assert plan['each_suffix'] == 'o012_crf'
    assert set(plan['suffix_by_filter'].values()) == {'o012_crf'}


def test_nircam_keeps_its_lineage_token():
    """What the MIRI/NIRISS split must NOT loosen.

    brick, cloudc, cloudef and sickle carry BOTH lineages in one directory
    (``o001_crf`` and ``destreak_o001_crf``).  A NIRCam plan that dropped the
    token would glob both and mix them -- the ~106 mas two-lineage catalog.
    """
    assert rp.resolve('2221', '001')['each_suffix'] == 'destreak_o001_crf'
    assert rp.resolve('6778', '001')['each_suffix'] == 'align_o001_crf'
    # An untokened suffix would match the destreaked frame as well; the
    # tokened one is what keeps the two lineages apart.
    assert not fnmatch.fnmatch(
        'jw02221001001_02101_00001_nrcb3_destreak_o001_crf.fits',
        '*nrcb3*align_o001_crf.fits')


def test_destreaking_is_off_for_every_non_nircam_instrument():
    """The policy answers the reduction driver, not just the cataloger:
    ``destreaks`` gates NIRCam stage 1's streak removal, and neither
    ``PipelineMIRI`` nor ``PipelineRerunNIRISS`` has that step."""
    assert destreak_policy.destreaks('brick', 'F2550W') is False
    assert destreak_policy.destreaks('brick', 'F2550W', instrument='miri') is False
    assert destreak_policy.destreaks('sgrb2', 'F770W') is False
    # NIRISS shares F200W with NIRCam, so it has to be named.
    assert destreak_policy.destreaks('m4-1979', 'F200W', instrument='niriss') is False
    assert destreak_policy.destreaks('brick', 'F410M') is True


def test_sickle_gets_per_filter_suffixes():
    """Its short filters destreak and its long ones do not, so no single
    --each-suffix is right."""
    config = pipeline_config.load()
    environ = rp._job_environment(rp.resolve('3958', '007'), 'catalog', config)
    assert 'F470N:align_o007_crf' in environ['EACH_SUFFIX_OVERRIDES']
    assert environ['EACH_SUFFIX'] == 'destreak_o007_crf'


def test_a_non_nircam_run_carries_its_instrument_and_modules():
    config = pipeline_config.load()
    plan = rp.resolve('4147', '012', instrument='niriss')
    environ = rp._job_environment(plan, 'catalog', config)
    assert environ['GC_INSTRUMENT_OVERRIDE'] == 'niriss'
    assert environ['MODULES'] == 'nis'


def test_each_instrument_reduces_with_its_own_driver():
    config = pipeline_config.load()
    for instrument, driver in (('nircam', 'PipelineRerunNIRCAM-LONG.py'),
                               ('niriss', 'PipelineRerunNIRISS.py')):
        assert rp.REDUCE_DRIVERS[instrument].endswith(driver)


def test_a_typo_in_stages_stops_rather_than_running_nothing():
    with pytest.raises(SystemExit):
        rp.main(['--proposal', '2221', '--obsid', '001',
                 '--stages', 'redcue', '--dry-run'])


def test_a_typo_in_the_config_stops(tmp_path, monkeypatch):
    """fields.yaml validates; this used to accept `sceduler:` in silence."""
    mine = tmp_path / 'typo.yaml'
    mine.write_text('sceduler: local\n')
    monkeypatch.setenv(pipeline_config.ENV_VAR, str(mine))
    with pytest.raises(pipeline_config.ConfigError, match='unknown top-level'):
        pipeline_config.load()


def test_the_merge_job_is_named_for_the_target():
    """It covers every proposal of the target, so an -o001- name would lie."""
    plan = rp.resolve('2221', '001')
    assert rp._job_name(plan, 'merge') == 'brick-merge'
    assert rp._job_name(plan, 'reduce') == 'brick2221-o001-reduce'


def test_every_registry_attribute_the_drivers_name_resolves():
    """A rename in fields.py stranded two driver call sites and no test saw
    it.  This walks the source for `field_registry.<attr>` and checks each."""
    import ast
    from jwst_gc_pipeline import fields as registry
    drivers = ['jwst_gc_pipeline/reduction/PipelineRerunNIRCAM-LONG.py',
               'jwst_gc_pipeline/reduction/PipelineMIRI.py',
               'jwst_gc_pipeline/reduction/PipelineRerunNIRISS.py',
               'jwst_gc_pipeline/photometry/merge_catalogs.py',
               'jwst_gc_pipeline/photometry/crowdsource_catalogs_long.py']
    missing = []
    for driver in drivers:
        tree = ast.parse(open(os.path.join(rp.REPO_ROOT, driver)).read())
        for node in ast.walk(tree):
            if (isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id in ('field_registry', 'fields')
                    and not hasattr(registry, node.attr)):
                missing.append(f'{driver}: fields.{node.attr}')
    assert not missing, 'drivers name registry attributes that do not exist:\n  ' + '\n  '.join(missing)


# --------------------------------------------------------------------------
# Findings from the end-to-end cutout run.
# --------------------------------------------------------------------------

def test_the_local_run_puts_this_checkout_on_the_path(monkeypatch, tmp_path):
    """Stage 1 is a script path, so Python puts ITS directory on sys.path and
    `import jwst_gc_pipeline` finds the pip-installed package.  Stage 2 uses
    `python -m` and finds the checkout.  The two stages ran different
    versions."""
    captured = {}

    def fake_run(command, env=None, cwd=None, **kwargs):
        captured['env'] = env
        class Result:
            returncode = 0
        return Result()

    monkeypatch.setattr(rp.subprocess, 'run', fake_run)
    rp.run_pipeline('2221', '001', filters=['F410M'],
                    cutout_region='1,2,3', dry_run=False)
    assert captured['env']['PYTHONPATH'].split(os.pathsep)[0] == rp.REPO_ROOT


def test_the_submitted_jobs_carry_pipe_root():
    """submit_*.sbatch prepend PIPE_ROOT to PYTHONPATH, for the same reason."""
    config = pipeline_config.load()
    environ = rp._job_environment(rp.resolve('2221', '001'), 'reduce', config)
    assert environ['PIPE_ROOT'] == rp.REPO_ROOT


def test_the_banner_reports_where_output_actually_goes(monkeypatch, capsys):
    """GC_BASEPATH_OVERRIDE is a safety variable; the one line naming the
    output directory has to reflect it."""
    monkeypatch.setenv('GC_BASEPATH_OVERRIDE', '/tmp/demo/brick/')
    rp.run_pipeline('2221', '001', filters=['F410M'], dry_run=True)
    out = capsys.readouterr().out
    assert 'directory: /tmp/demo/brick/' in out
    assert 'refcat:    /tmp/demo/brick/catalogs/' in out


def test_a_multi_filter_cutout_says_it_is_not_minutes(capsys):
    rp.run_pipeline('2221', '001', cutout_region='1,2,3', dry_run=True)
    assert '--filters F410M keeps it to minutes' in capsys.readouterr().out


def test_a_cutout_without_the_catalog_stage_is_refused():
    """The cutout reaches the cataloging command only.

    Accepted alongside `--stages reduce` it would be silently ignored, and the
    run would reduce the whole observation -- hours on the queue when minutes
    were asked for.
    """
    with pytest.raises(rp.CutoutStageError, match='needs the catalog stage'):
        rp.run_pipeline('2221', '001', filters=['F410M'],
                        cutout_region='266.535,-28.705,20',
                        stages=('reduce',), dry_run=True)

# --------------------------------------------------------------------------
# Where the logs go.
# --------------------------------------------------------------------------

def test_log_dir_reaches_the_submitted_job(capsys, monkeypatch):
    # A site copy is exported by anyone following GETTING_STARTED, and this
    # test is about the wiring, not about one site's path.
    monkeypatch.delenv(pipeline_config.ENV_VAR, raising=False)
    configured = pipeline_config.load()['slurm']['log_dir']
    rp.run_pipeline('2221', '001', filters=['F410M'], dry_run=True)
    out = capsys.readouterr().out
    assert f'--output={configured}/reduce_%x_' in out


def test_an_array_job_logs_per_task_and_a_single_job_does_not():
    """SLURM writes 4294967294 for %a on a job that is not an array."""
    slurm = {'log_dir': '/tmp'}
    array = rp._log_arguments(slurm, 'catalog', array=True)[0]
    single = rp._log_arguments(slurm, 'mergeall', array=False)[0]
    assert array.endswith('_%x_%A_%a.out')
    assert single.endswith('_%x_%j.out')


def test_no_log_dir_leaves_the_scripts_own_directive_alone():
    assert rp._log_arguments({}, 'reduce') == []


def test_building_a_command_creates_nothing(tmp_path):
    """--dry-run prints commands, so building one must not touch the disk.

    It also runs on machines that have none of these directories."""
    absent = tmp_path / 'not-yet' / 'logs'
    rp._log_arguments({'log_dir': str(absent)}, 'reduce')
    assert not absent.exists()


def test_an_uncreatable_log_dir_says_so():
    with pytest.raises(pipeline_config.ConfigError, match='log_dir'):
        rp._make_log_dir({'log_dir': '/proc/definitely/not/writable'})


def test_the_log_dir_is_made_at_submission(tmp_path):
    made = tmp_path / 'logs'
    rp._make_log_dir({'log_dir': str(made)})
    assert made.is_dir()


def test_a_misspelled_slurm_key_is_rejected(tmp_path, monkeypatch):
    """`logdir:` would be ignored, and the logs would silently go elsewhere."""
    mine = tmp_path / 'c.yaml'
    mine.write_text('slurm:\n  logdir: /tmp/somewhere\n')
    monkeypatch.setenv(pipeline_config.ENV_VAR, str(mine))
    with pytest.raises(pipeline_config.ConfigError, match='logdir'):
        pipeline_config.load()


def test_only_nircam_gets_a_modules_argument():
    """The MIRI and NIRISS stage-1 drivers reject -m."""
    config = pipeline_config.load()
    for instrument in ('miri', 'niriss'):
        plan = rp.resolve('2221' if instrument == 'miri' else '4147',
                          '002' if instrument == 'miri' else '012',
                          instrument)
        reduce_command = dict(rp._local_commands(plan, config))['reduce']
        assert '-m' not in reduce_command
    nircam = dict(rp._local_commands(rp.resolve('2221', '001', 'nircam'),
                                     config))['reduce']
    assert '-m' in nircam


def test_a_cutout_alongside_an_explicit_merge_is_refused():
    with pytest.raises(rp.CutoutStageError, match='cannot run the merge'):
        rp.run_pipeline('2221', '001', filters=['F410M'],
                        cutout_region='266.535,-28.705,20',
                        stages=('catalog', 'merge'), dry_run=True)


def test_a_cutout_on_a_queue_scheduler_is_refused(tmp_path, monkeypatch):
    """Only the local path threads the region into the command."""
    mine = tmp_path / 'c.yaml'
    mine.write_text('cutout:\n  scheduler: slurm\n')
    monkeypatch.setenv(pipeline_config.ENV_VAR, str(mine))
    with pytest.raises(rp.CutoutStageError, match='cutout.scheduler'):
        rp.run_pipeline('2221', '001', filters=['F410M'],
                        cutout_region='266.535,-28.705,20', dry_run=True)


def test_the_refusals_print_a_message_rather_than_a_traceback(capsys,
                                                              monkeypatch):
    monkeypatch.setattr('sys.argv',
                        ['run_pipeline', '--proposal', '2221', '--obsid',
                         '001', '--stages', 'reduce', '--cutout-region',
                         '266.535,-28.705,20', '--dry-run'])
    with pytest.raises(SystemExit):
        rp.main()
    assert 'needs the catalog stage' in capsys.readouterr().err


def test_a_cutout_keeps_the_default_stage_set(capsys):
    """The documented one-liner passes no stages, so merge is in the set; it is
    dropped with a note rather than refused."""
    rp.run_pipeline('2221', '001', filters=['F410M'],
                    cutout_region='266.535,-28.705,20', dry_run=True)
    out = capsys.readouterr().out
    assert 'a cutout stops after cataloging' in out.lower()
    assert '=== merge ===' not in out
