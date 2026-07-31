"""The one-command runner: does it plan the right work, and say so clearly?

These check the plan and the commands rather than submitting anything.
"""
import os

import pytest

from jwst_gc_pipeline import config as pipeline_config
from jwst_gc_pipeline import run_pipeline as rp


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
    assert 'sbatch' not in out
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


def test_the_shipped_config_and_the_submit_scripts_agree():
    """The runner passes these on the sbatch line, so a drift here silently
    changes what the jobs ask for."""
    config = pipeline_config.load()
    for stage_name in ('reduce', 'catalog', 'merge'):
        stage = pipeline_config.stage(config, stage_name)
        script = os.path.join(rp.REPO_ROOT, stage['submit_script'])
        assert os.path.exists(script), script
