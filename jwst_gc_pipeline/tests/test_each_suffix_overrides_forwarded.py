"""The per-filter suffix override has to survive the sbatch path.

``cataloging._resolve_each_suffix`` lets specific filters read a different
per-exposure crf than the global ``--each-suffix``
(``--each-suffix-overrides=F187N:destreak_o007_crf,F210M:destreak_o007_crf``).
Sickle is the case it exists for: SW destreaks and LW stays on the aligned copy
(``reduction/destreak_policy.py``), so no single ``--each-suffix`` serves the
filter list, and the filters on the other side of the split glob ZERO inputs at
m1.

``run_pipeline`` reaches the mechanism two ways.  The direct call appends the
flag; the sbatch path exports ``EACH_SUFFIX_OVERRIDES`` and used to stop there,
because no submit script read it -- so a sickle observation submitted through
``submit_cataloging.sbatch`` photometered every filter with the one
``EACH_SUFFIX`` (issue #432).

Pinned here:

  * each multi-filter cataloging submitter reads the variable and builds the
    flag from it -- executed, from the lines as shipped, not read for a
    substring;
  * the flag reaches the ``crowdsource_catalogs_long`` invocation in the same
    script;
  * the chain EXPORTS it instead of listing it in ``--export``: the value is
    comma-separated and SLURM's ``--export`` list would truncate it at the
    first pair;
  * the name is the one ``run_pipeline`` writes, so a rename on either side
    fails here rather than silently dropping the override again.
"""
import os
import re
import subprocess

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS = os.path.join(REPO, 'scripts', 'reduction')

#: every submitter that runs SEVERAL filters off one EACH_SUFFIX
SUBMITTERS = [
    'submit_cataloging.sbatch',
    'submit_cataloging_m7.sbatch',
    'submit_cataloging_perframe_phase.sbatch',
]

CHAIN = 'submit_cataloging_chain.sh'
ENV_VAR = 'EACH_SUFFIX_OVERRIDES'
FLAG = '--each-suffix-overrides'
SICKLE = 'F187N:destreak_o007_crf,F210M:destreak_o007_crf'


def _text(name):
    with open(os.path.join(SCRIPTS, name)) as fh:
        return fh.read()


def _override_block(name):
    """The shipped lines that turn the variable into the flag."""
    text = _text(name)
    match = re.search(
        r'^' + ENV_VAR + r'=\$\{' + ENV_VAR + r':-\}\n(?:.*\n)*?.*' +
        re.escape(FLAG) + r'=\$' + ENV_VAR + r'"\n',
        text, re.MULTILINE)
    assert match, f'{name} builds no {FLAG} argument from {ENV_VAR}'
    return match.group(0)


def _run(block, env):
    full = dict(os.environ)
    full.pop(ENV_VAR, None)
    full.update(env)
    return subprocess.run(
        ['bash', '-c', block + f'\necho "ARG=[${ENV_VAR}_ARG]"'],
        capture_output=True, text=True, env=full, timeout=60)


@pytest.mark.parametrize('name', SUBMITTERS)
def test_the_submitter_builds_the_flag_from_the_environment(name):
    block = _override_block(name)
    got = _run(block, {ENV_VAR: SICKLE})
    assert got.returncode == 0, got.stderr
    assert f'ARG=[{FLAG}={SICKLE}]' in got.stdout, got.stdout


@pytest.mark.parametrize('name', SUBMITTERS)
def test_an_unset_override_adds_no_argument(name):
    """The variable is optional: a field with one suffix for every filter must
    invoke exactly as before."""
    got = _run(_override_block(name), {})
    assert got.returncode == 0, got.stderr
    assert 'ARG=[]' in got.stdout, got.stdout


@pytest.mark.parametrize('name', SUBMITTERS)
def test_the_flag_reaches_the_cataloging_invocation(name):
    text = _text(name)
    invocation = re.search(
        r'crowdsource_catalogs_long \\\n(?:.*\\\n)*.*\n', text)
    assert invocation, f'{name}: no crowdsource_catalogs_long invocation found'
    body = invocation.group(0)
    assert '--each-suffix=' in body, f'{name}: invocation lost --each-suffix'
    assert f'${ENV_VAR}_ARG' in body, (
        f'{name}: {ENV_VAR}_ARG is built but never passed to the catalog run')


def test_the_chain_exports_the_override_rather_than_listing_it():
    """A comma-separated value in SLURM's ``--export`` list truncates at the
    first comma, which would hand the job one filter's override and drop the
    rest."""
    text = _text(CHAIN)
    assert re.search(r'^export ' + ENV_VAR + r'=', text, re.MULTILINE), (
        f'{CHAIN} does not export {ENV_VAR}, so the array tasks never see it')
    for line in text.splitlines():
        if line.startswith('COMMON_EXPORT'):
            assert ENV_VAR not in line, (
                f'{CHAIN} lists {ENV_VAR} in --export; its commas would '
                f'truncate the value')


def test_run_pipeline_writes_the_same_variable_name():
    with open(os.path.join(REPO, 'jwst_gc_pipeline', 'run_pipeline.py')) as fh:
        source = fh.read()
    assert f"'{ENV_VAR}'" in source, (
        f'run_pipeline no longer sets {ENV_VAR}; the submitters read that name')
