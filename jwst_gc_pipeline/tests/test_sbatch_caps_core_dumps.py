"""Every batch script must disable core dumps before it runs anything.

A python ``SIGBUS`` in a peppar extract wrote a **43 GiB** ``core.*`` into a
checkout on 2026-08-06 (JWST-GC/pipeline-runners#5).  These jobs ask for 64-128
GB of RAM, so a core file is that size; ``/orange/adamginsburg`` runs at 97%
full (805T of 834T, 2026-08-22) and the treasury window multiplies the number of
jobs in flight.  A crashy configuration can therefore fill the filesystem, which
stalls the MAST monitor's own ``--min-free-tb`` disk gate and kills unrelated
reductions with ENOSPC.

The cap has to live **inside** the batch script.  HiPerGator runs
``PropagateResourceLimits = NONE``, so a ``ulimit -c 0`` in the submitting shell
does not reach the compute node -- measured on two probe jobs submitted from a
``ulimit -c 0`` shell: plain ``sbatch`` gave ``RLIMIT_CORE = (-1, -1)`` on the
node, ``sbatch --propagate=CORE`` gave ``(0, -1)``.  Neither the submitter's
shell nor a wrapper script can be trusted to carry it, and the batch script is
the one place every submission path passes through.

See jwst-gc-pipeline#428.
"""
import os
import re

import pytest


REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS = os.path.join(REPO, 'scripts')


def _sbatch_scripts():
    found = []
    for dirpath, _dirnames, filenames in os.walk(SCRIPTS):
        for name in sorted(filenames):
            if name.endswith('.sbatch'):
                found.append(os.path.join(dirpath, name))
    return sorted(found)


def _rel(path):
    return os.path.relpath(path, REPO)


ALL_SBATCH = _sbatch_scripts()


def test_there_are_batch_scripts_to_check():
    """Guard the guard: an empty glob would make every case below vacuous."""
    assert len(ALL_SBATCH) >= 20, (
        f'only {len(ALL_SBATCH)} *.sbatch found under {SCRIPTS}; the discovery '
        'walk is broken, so the core-dump check below tests nothing')


@pytest.mark.parametrize('path', ALL_SBATCH, ids=_rel)
def test_batch_script_disables_core_dumps(path):
    with open(path) as fh:
        lines = fh.read().split('\n')
    assert any(ln.strip() == 'ulimit -c 0' for ln in lines), (
        f'{_rel(path)} does not set `ulimit -c 0`, so a crash on the compute '
        'node dumps a core file the size of the job (43 GiB measured) onto a '
        'filesystem that is 97% full.  PropagateResourceLimits=NONE means the '
        'submitting shell cannot set this for you -- put it in the script.')


@pytest.mark.parametrize('path', ALL_SBATCH, ids=_rel)
def test_the_cap_precedes_every_command(path):
    """``ulimit`` after the first command leaves that command uncapped.

    It also has to sit after the ``#SBATCH`` block: sbatch stops reading
    directives at the first line that is neither blank nor a comment, so a
    ``ulimit`` placed above them would silently void every directive below it.
    """
    with open(path) as fh:
        lines = fh.read().split('\n')

    cap = None
    first_command = None
    last_directive = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('#SBATCH'):
            last_directive = i
            continue
        if not stripped or stripped.startswith('#'):
            continue
        if first_command is None:
            first_command = i
        if stripped == 'ulimit -c 0' and cap is None:
            cap = i

    assert cap is not None, f'{_rel(path)} does not set `ulimit -c 0`'
    assert cap == first_command, (
        f'{_rel(path)}: `ulimit -c 0` is at line {cap + 1} but the first '
        f'command runs at line {first_command + 1}; everything before the cap '
        'can still dump a core file')
    if last_directive is not None:
        assert cap > last_directive, (
            f'{_rel(path)}: `ulimit -c 0` at line {cap + 1} sits above the '
            f'#SBATCH directive at line {last_directive + 1}, which sbatch '
            'would then never read')


# ``sbatch --wrap "<body>"`` runs the body as the batch script, so it has no
# file of its own to carry the cap -- the body itself must set it.
_WRAP = re.compile(r'--wrap\s+"(.*?)(?<!\\)"', re.DOTALL)


def _wrapped_bodies():
    out = []
    for dirpath, _dirnames, filenames in os.walk(SCRIPTS):
        for name in sorted(filenames):
            if not name.endswith(('.sh', '.sbatch')):
                continue
            path = os.path.join(dirpath, name)
            with open(path) as fh:
                text = fh.read()
            for match in _WRAP.finditer(text):
                line = text[:match.start()].count('\n') + 1
                out.append((f'{_rel(path)}:{line}', match.group(1)))
    return out


WRAPPED = _wrapped_bodies()


@pytest.mark.parametrize('where,body', WRAPPED, ids=[w for w, _ in WRAPPED])
def test_wrapped_submission_disables_core_dumps(where, body):
    first = body.strip().split('\n')[0].strip()
    assert first.startswith('ulimit -c 0'), (
        f'{where}: the `sbatch --wrap` body starts with {first!r}.  A wrapped '
        'submission has no batch script file to carry `ulimit -c 0`, and the '
        'submitting shell cannot pass its own limit through '
        '(PropagateResourceLimits=NONE), so this job can dump a core file the '
        'size of its memory footprint.')
