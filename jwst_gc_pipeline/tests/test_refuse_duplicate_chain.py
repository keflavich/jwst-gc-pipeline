"""One field must not get two concurrent cataloging chains.

Two full chains were queued for sgrc (2026-08-21T20:35 and 2026-08-22T01:34),
neither with a dependency, so both would have started on priority and run 12
jobs each over the SAME per-frame products and the same merged catalogs -- two
writers, one output tree.  Nothing noticed: the collision guards in
``cataloging`` protect one run's frames from each other, not one field from two
concurrent runs.

The check is per PHASE rather than per field, because adding phases to a field
that already has a chain is the normal way to extend one -- cloudef's m4-m7 went
on behind its m3, and sgrb2's m3-m7 behind eleven per-filter m12 finalizes.
Only re-submitting a phase that is ALREADY queued is the mistake.

The submitter is driven with a fake ``squeue`` on PATH and a fake ``sbatch``, so
these assert what the script does rather than what the live queue happens to
hold.
"""
import os
import stat
import subprocess

import pytest


REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SUBMITTER = os.path.join(REPO, 'scripts', 'reduction',
                         'submit_cataloging_perframe.sh')
M8_SUBMITTER = os.path.join(REPO, 'scripts', 'reduction',
                            'submit_cataloging_m8.sh')
HELPER = os.path.join(REPO, 'scripts', 'reduction',
                      '_refuse_duplicate_chain.sh')

BASE_ENV = dict(
    PROPOSAL='4147', FIELD='012', TARGET='sgrc',
    FILTERS='F212N F405N', EACH_SUFFIX='destreak_o012_crf',
    NSHARDS='2', MODULES='nrcb',
)


def _fake_bin(tmp_path, queued_names):
    """A PATH dir whose `squeue` prints `queued_names` and whose `sbatch`
    records rather than submits."""
    d = tmp_path / 'bin'
    d.mkdir(exist_ok=True)

    (d / 'squeue').write_text(
        '#!/bin/bash\n' + ''.join(f'echo "{n}"\n' for n in queued_names))
    # --parsable callers read a job id off stdout; anything numeric will do.
    (d / 'sbatch').write_text(
        '#!/bin/bash\n'
        f'echo "$@" >> {tmp_path}/submitted.txt\n'
        'echo 111111\n')
    (d / 'scontrol').write_text('#!/bin/bash\nexit 0\n')
    for f in ('squeue', 'sbatch', 'scontrol'):
        (d / f).chmod((d / f).stat().st_mode | stat.S_IEXEC)
    return d


def _run(tmp_path, queued, **over):
    d = _fake_bin(tmp_path, queued)
    env = {**os.environ, **BASE_ENV, **over}
    env['PATH'] = f"{d}:{env['PATH']}"
    return subprocess.run(['bash', SUBMITTER], capture_output=True, text=True,
                          env=env, cwd=REPO, timeout=300)


def _submitted(tmp_path):
    f = tmp_path / 'submitted.txt'
    return f.read_text().splitlines() if f.exists() else []


# --------------------------------------------------------------------------
# the duplicate that happened
# --------------------------------------------------------------------------

def test_a_second_full_chain_is_refused(tmp_path):
    """The sgrc case: an m12 chain is queued and another is submitted."""
    r = _run(tmp_path, ['sgrc4147-o012-m12-fanout',
                        'sgrc4147-o012-m12-finalize'])
    assert r.returncode == 3, r.stdout + r.stderr
    assert 'REFUSING' in r.stderr
    assert 'm12' in r.stderr
    assert not _submitted(tmp_path), 'it submitted anyway'


def test_an_empty_queue_submits(tmp_path):
    r = _run(tmp_path, [])
    assert r.returncode == 0, r.stdout + r.stderr
    assert _submitted(tmp_path), 'nothing was submitted'


def test_another_fields_chain_is_not_a_duplicate(tmp_path):
    """The guard keys on target+proposal+obsid, so a busy queue full of other
    fields must not block this one."""
    r = _run(tmp_path, ['brick1182-o004-m12-fanout',
                        'gc2211_o0232211-o023-m12-fanout',
                        'sgrc4147-o099-m12-fanout'])
    assert r.returncode == 0, r.stdout + r.stderr
    assert _submitted(tmp_path)


# --------------------------------------------------------------------------
# extending a chain, which is normal and must keep working
# --------------------------------------------------------------------------

def test_adding_later_phases_behind_a_queued_m12_is_allowed(tmp_path):
    """sgrb2's m3-m7 went on behind eleven per-filter m12 finalizes; cloudef's
    m4-m7 behind its m3.  A per-FIELD check would have refused both."""
    r = _run(tmp_path,
             ['sgrc4147-o012-m12-finalize-F212N',
              'sgrc4147-o012-m12-finalize-F405N'],
             PHASES='m3 m4 m5 m6 m7')
    assert r.returncode == 0, r.stdout + r.stderr
    assert _submitted(tmp_path)


def test_a_phase_overlap_within_a_continuation_is_still_refused(tmp_path):
    """Extending is fine; extending with a phase already queued is not."""
    r = _run(tmp_path, ['sgrc4147-o012-m5-fanout'], PHASES='m4 m5 m6')
    assert r.returncode == 3, r.stdout + r.stderr
    assert 'm5' in r.stderr
    assert 'm4' not in r.stderr.split('phase(s):')[1].split('\n')[0]


# --------------------------------------------------------------------------
# the m1 / m12 prefix trap
# --------------------------------------------------------------------------

def test_m1_does_not_match_m12(tmp_path):
    """Without the trailing '-' in the pattern, phase 'm1' matches
    'm12-fanout' and every m1 submission would be refused by an m12 chain."""
    r = _run(tmp_path, ['sgrc4147-o012-m12-fanout'], PHASES='m1')
    assert r.returncode == 0, r.stdout + r.stderr
    assert _submitted(tmp_path)


def test_m12_does_match_m12(tmp_path):
    r = _run(tmp_path, ['sgrc4147-o012-m12-fanout'], PHASES='m12')
    assert r.returncode == 3


# --------------------------------------------------------------------------
# the override
# --------------------------------------------------------------------------

def test_the_override_submits_and_says_so(tmp_path):
    """For a queued chain known to be dead and about to be cancelled."""
    r = _run(tmp_path, ['sgrc4147-o012-m12-fanout'],
             GC_ALLOW_DUPLICATE_CHAIN='1')
    assert r.returncode == 0, r.stdout + r.stderr
    assert 'submitting anyway' in r.stderr
    assert _submitted(tmp_path)


def test_the_override_is_not_on_by_default(tmp_path):
    """A truthy-looking value that is not exactly 1 must not open the gate."""
    for val in ('0', '', 'no', 'false'):
        r = _run(tmp_path, ['sgrc4147-o012-m12-fanout'],
                 GC_ALLOW_DUPLICATE_CHAIN=val)
        assert r.returncode == 3, f'{val!r} opened the gate'


def test_the_refusal_names_the_jobs_it_found(tmp_path):
    """An operator who is refused has to be able to see WHICH chain blocked
    them without going hunting."""
    r = _run(tmp_path, ['sgrc4147-o012-m12-fanout'])
    assert 'sgrc4147-o012' in r.stderr


# --------------------------------------------------------------------------
# the m8 driver, which fans 18 jobs per field and was unguarded (#483 review)
# --------------------------------------------------------------------------

M8_ENV = dict(TARGET='wd2', PROPOSAL='3523', FIELD='005', MODULES='merged',
              FILTERS='F115W F150W', EACH_SUFFIX='align_o005_crf')


def _run_m8(tmp_path, queued, **over):
    d = _fake_bin(tmp_path, queued)
    env = {**os.environ, **M8_ENV, **over}
    env['PATH'] = f"{d}:{env['PATH']}"
    return subprocess.run(['bash', M8_SUBMITTER], capture_output=True,
                          text=True, env=env, cwd=REPO, timeout=300)


def test_a_second_m8_fan_is_refused(tmp_path):
    """The exposure the #483 review named: 18 m8 jobs for one field with no
    duplicate check.  Two fans write the same per-filter partials and the same
    merge."""
    r = _run_m8(tmp_path, ['wd23523-o005-m8-F115W', 'wd23523-o005-m8-merge'])
    assert r.returncode == 3, r.stdout + r.stderr
    assert 'REFUSING' in r.stderr
    assert not _submitted(tmp_path), 'it submitted anyway'


def test_an_m8_fan_submits_when_the_queue_is_clear(tmp_path):
    r = _run_m8(tmp_path, [])
    assert r.returncode == 0, r.stdout + r.stderr
    assert _submitted(tmp_path)


def test_an_m8_fan_is_not_blocked_by_the_fields_m7(tmp_path):
    """Arming m8 behind a running m7 is the normal flow -- that is exactly how
    a field is re-armed after a failed finalize.  Only a queued m8 blocks."""
    r = _run_m8(tmp_path, ['wd23523-o005-m7-finalize',
                           'wd23523-o005-m7-fanout'])
    assert r.returncode == 0, r.stdout + r.stderr
    assert _submitted(tmp_path)


def test_another_fields_m8_does_not_block(tmp_path):
    r = _run_m8(tmp_path, ['brick1182-o004-m8-F200W'])
    assert r.returncode == 0, r.stdout + r.stderr


def test_the_m8_override_works(tmp_path):
    r = _run_m8(tmp_path, ['wd23523-o005-m8-F115W'],
                GC_ALLOW_DUPLICATE_CHAIN='1')
    assert r.returncode == 0, r.stdout + r.stderr
    assert 'submitting anyway' in r.stderr


# --------------------------------------------------------------------------
# one implementation, not two copies
# --------------------------------------------------------------------------

def test_both_drivers_source_the_shared_helper():
    """A second copy is how the trailing-'-' bug and the `set -e` exit-code bug
    would come back in only one of them."""
    for path in (SUBMITTER, M8_SUBMITTER):
        src = open(path).read()
        assert '_refuse_duplicate_chain.sh' in src, (
            f'{os.path.basename(path)} does not source the shared guard')


def test_neither_driver_hand_rolls_the_check():
    for path in (SUBMITTER, M8_SUBMITTER):
        src = open(path).read()
        assert 'GC_ALLOW_DUPLICATE_CHAIN:-0' not in src, (
            f'{os.path.basename(path)} has its own copy of the guard')


def test_the_helper_keeps_the_trailing_dash():
    """The bug that made 'm1' match 'm12-fanout'.  Pinned in the helper now
    that it is the single implementation."""
    src = open(HELPER).read()
    assert '${_ph}-' in src, 'phase pattern lost its trailing dash'
