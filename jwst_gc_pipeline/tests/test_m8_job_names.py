"""m8 jobs must carry target+program+obsid+stage at SUBMIT time.

CLAUDE.md standing rule: ``<target><program>-o<obsid>-<stage>[-FILTER]``, passed
with ``sbatch --job-name`` -- because the in-script runtime rename only fires
when a job STARTS, and a quota-bound job sits PENDING under the placeholder for
hours, which is exactly when the queue is being watched.

The m8 fan had both halves of that defect.  Submitted for wd2 on 2026-08-22, all
18 jobs sat pending as::

    39950426 catalog_m8p   PD
    39950425 catalog_m8p   PD
    ... x17, plus catalog_m8merge

and the runtime name they would eventually have taken,
``${TARGET}-catalog-m8p-${FILT}``, carries no program and no obsid -- so two
fields' m8 runs in flight together are indistinguishable, and it is not the
required shape either.

Same defect PR #177 fixed for the retie / per-frame jobs.
"""
import os
import re

import pytest


REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SUBMITTER = os.path.join(REPO, 'scripts', 'reduction', 'submit_cataloging_m8.sh')
PARTIAL = os.path.join(REPO, 'scripts', 'reduction',
                       'submit_cataloging_m8_partial.sbatch')
MERGE = os.path.join(REPO, 'scripts', 'reduction',
                     'submit_cataloging_m8_merge.sbatch')


def _sbatch_calls(path):
    """Each real `sbatch ...` invocation in a driver, one joined line each.

    Comments are dropped: the driver's header names the sbatch files it
    launches, and matching those would fail the check against prose.
    """
    src = open(path).read()
    src = re.sub(r'\\\n\s*', ' ', src)          # join continuations
    out = []
    for ln in src.split('\n'):
        stripped = ln.strip()
        if stripped.startswith('#'):
            continue
        if re.search(r'(^|[\s=(`$])sbatch\s', stripped):
            out.append(stripped)
    return out


def test_every_m8_sbatch_call_passes_a_job_name():
    calls = _sbatch_calls(SUBMITTER)
    assert calls, 'no sbatch calls found -- did the driver move?'
    for c in calls:
        assert '--job-name=' in c, (
            'an m8 job is submitted with no --job-name, so it sits PENDING under '
            f'the #SBATCH placeholder:\n  {c}')


def test_the_submit_time_name_has_target_program_obsid_stage():
    src = open(SUBMITTER).read()
    m = re.search(r'JOB_PREFIX="([^"]+)"', src)
    assert m, 'no JOB_PREFIX'
    prefix = m.group(1)
    for piece in ('${TARGET}', '${PROPOSAL}', '-o${FIELD}', 'm8'):
        assert piece in prefix, f'{piece} missing from job name prefix {prefix!r}'


def test_the_rendered_names_match_the_standing_convention():
    """Render the prefix for wd2 and check the actual shape, not just that the
    variables appear."""
    src = open(SUBMITTER).read()
    prefix = re.search(r'JOB_PREFIX="([^"]+)"', src).group(1)
    rendered = (prefix.replace('${TARGET}', 'wd2')
                      .replace('${PROPOSAL}', '3523')
                      .replace('${FIELD}', '005'))
    assert rendered == 'wd23523-o005-m8'
    assert re.fullmatch(r'[a-z0-9_]+\d{4}-o\d{3}-m8', rendered), rendered
    # and with a filter appended, the full form
    assert re.fullmatch(r'[a-z0-9_]+\d{4}-o\d{3}-m8-F\d{3}[A-Z]',
                        f'{rendered}-F115W')


@pytest.mark.parametrize('path,placeholder', [
    (PARTIAL, 'catalog_m8p'),
    (MERGE, 'catalog_m8merge'),
])
def test_the_runtime_rename_only_fires_on_a_bare_submission(path, placeholder):
    """A submit-time name must not be overwritten at runtime.

    The runtime form carries no program and no obsid, so an unconditional rename
    DEGRADES a correctly-named job -- and does it minutes-to-hours after the
    name mattered.  Renaming stays only as the fallback for a job submitted
    bare (the same idiom submit_cataloging_perframe_phase.sbatch uses).
    """
    src = open(path).read()
    assert 'scontrol update' in src
    guard = re.search(
        r'if \[ "\$\{SLURM_JOB_NAME:-' + re.escape(placeholder) + r'\}" = '
        r'"' + re.escape(placeholder) + r'" \]', src)
    assert guard, (
        f'{os.path.basename(path)} renames unconditionally; a submit-time name '
        'would be overwritten by the weaker runtime one')


@pytest.mark.parametrize('path', [PARTIAL, MERGE])
def test_the_runtime_fallback_also_uses_the_full_convention(path):
    """When it DOES fire, it must produce the required shape rather than the old
    `${TARGET}-catalog-m8p-${FILT}`."""
    src = open(path).read()
    m = re.search(r'JobName="([^"]+)"', src)
    assert m, 'no JobName assignment'
    name = m.group(1)
    for piece in ('${TARGET}', '${PROPOSAL}', '-o${FIELD}', 'm8'):
        assert piece in name, f'{piece} missing from runtime name {name!r}'
    assert 'catalog-m8' not in name, (
        f'runtime name {name!r} still uses the old program-less form')


@pytest.mark.parametrize('path', [SUBMITTER, PARTIAL, MERGE])
def test_scripts_stay_syntactically_valid(path):
    import subprocess

    r = subprocess.run(['bash', '-n', path], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_the_monitor_can_parse_the_rendered_names():
    """A name the monitor cannot bucket is only half a fix -- it would show up
    in the queue but not on the dashboard."""
    from jwst_gc_pipeline.monitoring import jobs as mj

    for name, want_stage in [('wd23523-o005-m8-F115W', 'm8'),
                             ('wd23523-o005-m8-merge', 'm8'),
                             ('gc2211_o0232211-o023-m8-F200W', 'm8')]:
        parsed = mj.parse_job_name(name)
        assert parsed is not None, f'monitor cannot parse {name!r}'
        assert parsed.get('stage') == want_stage, (name, parsed)
