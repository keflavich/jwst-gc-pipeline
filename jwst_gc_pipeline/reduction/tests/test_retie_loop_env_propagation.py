"""The retie loop must judge the full chain by the same rules it converged under.

``run_field_retie_loop.sh`` runs two cataloging invocations: the m12-only loop
that closes the astrometry checkpoint, and the full m3..m7 chain once it has
converged.  The full chain **re-runs m12**, so m2 runs inside it too.

``ASTROM_M2_CORRECTION_FLOOR_MAS`` was named only on the first.  It reached the
second anyway -- ``submit_cataloging_perframe.sh``'s ``--export`` list begins
with ``ALL`` and the loop exports the variable -- so this is about legibility,
not propagation: a value that decides whether a gate raises should be visible at
the call site, and the two invocations should be symmetric.

The floor is now forwarded through the ``floor_env`` array, which is EMPTY when
the caller set nothing so that the per-field table answers instead of a driver
default (see ``test_m2_floor_not_defaulted_by_drivers.py``).  Symmetry is what
these tests pin, so they follow the mechanism rather than the literal name.

These tests pin the naming.  They cannot tell whether the value ARRIVES; that was
settled by probe (``sbatch --export="ALL,..." --wrap='echo $VAR'`` returns the
exported value) and is recorded in the script's comment.
"""
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
LOOP = REPO_ROOT / 'scripts' / 'reduction' / 'run_field_retie_loop.sh'

#: What must reach BOTH cataloging invocations.  Each one changes what the m2
#: checkpoint decides, so passing it to one and not the other means the two runs
#: judge the same frames by different rules.
#:
#: The correction floor is carried by the ``floor_env`` ARRAY rather than a bare
#: ``VAR=$VAR`` assignment.  The loop must not DEFAULT the floor -- the env is
#: resolved before the per-field table, so any default overrides every entry in
#: ``PER_FIELD_FLOOR_MAS``, and a default of 0 disables the floor outright.  So
#: the value is only forwarded when the caller set one, and an empty array
#: forwards nothing (a bare unset expansion would also abort under ``set -u``).
#: The invariant is unchanged: whatever the floor is, both invocations get it.
MUST_REACH_BOTH = ('"${floor_env[@]}"',)


#: The line that actually runs the submit script -- not the mentions of its name
#: in comments.
_CALL = re.compile(r'^\s*bash "\$HERE/submit_cataloging_perframe\.sh"', re.M)


def _invocations(text):
    """Each `bash "$HERE/submit_cataloging_perframe.sh"` call, together with the
    env assignments that precede it (the backslash-continued command)."""
    calls = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if not _CALL.match(line):
            continue
        # walk back over the continued command AND its leading comment block, so
        # the reason a variable is passed counts as part of the call site
        j = i - 1
        while j >= 0 and (lines[j].rstrip().endswith('\\')
                          or lines[j].lstrip().startswith('#')
                          or not lines[j].strip()):
            if not lines[j].strip() and not lines[j - 1].lstrip().startswith('#'):
                break
            j -= 1
        calls.append('\n'.join(lines[j + 1:i + 1]))
    return calls


def test_both_cataloging_invocations_exist():
    """If this drops to one, the loop has stopped running a full chain."""
    assert len(_invocations(LOOP.read_text())) == 2


def test_the_correction_floor_reaches_both_invocations():
    text = LOOP.read_text()
    calls = _invocations(text)
    for name in MUST_REACH_BOTH:
        missing = [i for i, c in enumerate(calls) if name not in c]
        assert not missing, (
            f"{name} is passed to invocation(s) {sorted(set(range(len(calls))) - set(missing))} "
            f"but not {missing}. The full m3..m7 chain re-runs m12, so m2 runs "
            f"there too -- a field that converged under this floor gets "
            f"re-judged without it and raises on residuals it already passed.")


def test_the_full_chain_does_not_re_enable_auto_apply():
    """m3+ is a FROZEN solution.  The floor must propagate; the apply flag must
    not -- otherwise the full chain would start correcting a frozen tie."""
    text = LOOP.read_text()
    assert 'unset ASTROM_CHECKPOINT_APPLY' in text
    full_chain = _invocations(text)[-1]
    assert 'ASTROM_CHECKPOINT_APPLY=1' not in full_chain


def test_the_reason_is_recorded_at_the_call_site():
    """Whoever trims this env list next needs to know why it is not redundant
    with the `export` at the top of the script."""
    full_chain = _invocations(LOOP.read_text())[-1]
    assert 'DependencyNeverSatisfied' in full_chain or 'export' in full_chain


# ---------------------------------------------------------------------------
# The tests above pin the NAMING and say so: "They cannot tell whether the value
# ARRIVES."  On 2026-08-27 it did not.  gc2211_o049 was relaunched with an
# explicit ASTROM_M2_CORRECTION_FLOOR_MAS=4.0 and the loop died at iteration 1:
#
#   run_field_retie_loop.sh: line 521: ASTROM_M2_CORRECTION_FLOOR_MAS=4.0: command not found
#   [iter 1] the cataloging submission FAILED (rc=127) -- STOPPING.
#
# bash fixes the boundary of an assignment prefix at PARSE time, before
# "${floor_env[@]}" expands, so a quoted array expansion after literal VAR=value
# tokens is the COMMAND WORD.  An EMPTY array expands to nothing and parses fine,
# which is why every field taking its floor from PER_FIELD_FLOOR_MAS was
# unaffected -- the bug only fires on the explicit-override path, which is the
# one the manual-restart note tells operators to use.
# ---------------------------------------------------------------------------
import subprocess


def test_bash_treats_a_quoted_array_after_assignments_as_the_COMMAND():
    """The mechanism, demonstrated rather than asserted.

    If this ever stops failing, bash changed and the `env` prefixes below became
    optional -- but they are still correct, so the fix does not depend on it."""
    broken = subprocess.run(
        ['bash', '-c', 'fe=(FOO=1); BAR=2 "${fe[@]}" echo hi'],
        capture_output=True, text=True)
    assert broken.returncode == 127, broken
    assert 'command not found' in broken.stderr

    fixed = subprocess.run(
        ['bash', '-c', 'fe=(FOO=1); env BAR=2 "${fe[@]}" echo hi'],
        capture_output=True, text=True)
    assert fixed.returncode == 0, fixed
    assert fixed.stdout.strip() == 'hi'


def test_an_empty_floor_env_hides_the_bug():
    """Why this survived: with no override the array is empty and the prefix
    parses normally, so the 11 fields on PER_FIELD_FLOOR_MAS never saw it."""
    ok = subprocess.run(
        ['bash', '-c', 'fe=(); BAR=2 "${fe[@]}" echo hi'],
        capture_output=True, text=True)
    assert ok.returncode == 0, ok
    assert ok.stdout.strip() == 'hi'


def _floor_env_sites(text):
    """Every line expanding the floor array, with the command it belongs to.

    The expansion sits on its own continuation line, so walk BACK to the start
    of the command -- the first preceding line that is not itself a continuation.
    """
    lines = text.splitlines()
    out = []
    for i, ln in enumerate(lines):
        # comments discuss the array too (including the one explaining this bug)
        if '"${floor_env[@]}"' not in ln or ln.lstrip().startswith('#'):
            continue
        j = i
        while j > 0 and lines[j - 1].rstrip().endswith('\\'):
            j -= 1
        out.append(lines[j].strip())
    return out


def test_every_floor_env_forwarding_site_is_prefixed_with_env():
    """The fix itself.  Both cataloging invocations forward the array, and a
    forwarding site without `env` is rc=127 the moment anyone overrides."""
    sites = _floor_env_sites(LOOP.read_text())
    assert len(sites) == 2, f'expected 2 forwarding sites, found {len(sites)}: {sites}'
    for s in sites:
        head = s.split('=', 1)[0]
        assert re.match(r'^(chain_out=\$\(env |env )', s), (
            f'floor_env forwarded without an `env` prefix -- bash will run the '
            f'assignment as a command (rc 127). Offending command starts: {head}')


def test_the_loop_script_still_parses():
    """A syntax error here is a dead campaign, not a failed test."""
    r = subprocess.run(['bash', '-n', str(LOOP)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
