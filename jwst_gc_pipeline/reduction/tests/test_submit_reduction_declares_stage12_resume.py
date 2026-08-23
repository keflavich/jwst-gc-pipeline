"""`submit_reduction.sbatch` has to declare and report STAGE12_RESUME.

The mechanism works either way -- the reduce driver reads the environment
directly (`stage12_selection.stage12_resume_enabled`), so
`--export=ALL,STAGE12_RESUME=1` reaches it whether or not the batch script
mentions the name.  What was missing is observability (issue #434): the flag
was explained in a comment, had no default beside `PROPOSAL`/`FIELD`/`MODULES`/
`SKIP`/`FILTERS`, and was absent from the `REDUCE start:` provenance line, so a
job log did not say whether a run re-fit every ramp or resumed off products
already on disk.

Pinned here, by executing the shipped lines rather than reading them for a
substring:

  * the default resolves to `0` when unset and passes `1` through;
  * the variable is EXPORTED, so the value the log reports is the one the
    driver reads;
  * the provenance echo carries it;
  * the name matches `stage12_selection.STAGE12_RESUME_ENV`.
"""
import os
import re
import subprocess

import pytest

from jwst_gc_pipeline.reduction.stage12_selection import STAGE12_RESUME_ENV

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
SCRIPT = os.path.join(REPO, 'scripts', 'reduction', 'submit_reduction.sbatch')


def _text():
    with open(SCRIPT) as fh:
        return fh.read()


def _declaration():
    """The shipped line that gives STAGE12_RESUME its default."""
    match = re.search(
        r'^export ' + STAGE12_RESUME_ENV + r'=\$\{' + STAGE12_RESUME_ENV +
        r':-0\}$', _text(), re.MULTILINE)
    assert match, (
        f'{os.path.basename(SCRIPT)} does not declare and export '
        f'{STAGE12_RESUME_ENV} with a default')
    return match.group(0)


def _echo_line():
    match = re.search(r'^echo "REDUCE start:.*"$', _text(), re.MULTILINE)
    assert match, 'the REDUCE start provenance line is gone'
    return match.group(0)


def _run(script, env=None):
    full = dict(os.environ)
    full.pop(STAGE12_RESUME_ENV, None)
    full.update(env or {})
    return subprocess.run(['bash', '-c', script], capture_output=True,
                          text=True, env=full, timeout=60)


@pytest.mark.parametrize('given,expected', [(None, '0'), ('1', '1'),
                                            ('0', '0')])
def test_the_default_resolves_and_an_explicit_value_passes_through(given,
                                                                   expected):
    env = {} if given is None else {STAGE12_RESUME_ENV: given}
    got = _run(_declaration() + f'\necho "VALUE=[${STAGE12_RESUME_ENV}]"', env)
    assert got.returncode == 0, got.stderr
    assert f'VALUE=[{expected}]' in got.stdout, got.stdout


def test_the_declaration_exports_it_so_the_driver_reads_what_the_log_reports():
    got = _run(_declaration() +
               f'\nbash -c \'echo "CHILD=[${STAGE12_RESUME_ENV}]"\'',
               {STAGE12_RESUME_ENV: '1'})
    assert got.returncode == 0, got.stderr
    assert 'CHILD=[1]' in got.stdout, got.stdout


def test_the_provenance_line_reports_the_resume_state():
    script = '\n'.join([
        'PROPOSAL=2221 FIELD=001 FILT=F212N MODULES=nrca SKIP=0',
        _declaration(),
        _echo_line(),
    ])
    for given, expected in (('1', 'resume=1'), (None, 'resume=0')):
        env = {} if given is None else {STAGE12_RESUME_ENV: given}
        got = _run(script, env)
        assert got.returncode == 0, got.stderr
        assert expected in got.stdout, got.stdout
        assert 'skip=0' in got.stdout, got.stdout


def test_the_parameter_block_still_documents_the_flag():
    """The comment that explains what the flag does stays with the default."""
    text = _text()
    assert f'Set {STAGE12_RESUME_ENV}=1 to' in text
