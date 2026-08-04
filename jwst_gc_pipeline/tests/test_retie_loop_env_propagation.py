"""The retie loop must judge the full chain by the same rules it converged under.

``run_field_retie_loop.sh`` runs two cataloging invocations: the m12-only loop
that closes the astrometry checkpoint, and the full m3..m7 chain once it has
converged.  The full chain **re-runs m12**, so m2 runs inside it too.

``ASTROM_M2_CORRECTION_FLOOR_MAS`` was passed only to the first.  A field that
converged at 4.0 mas was therefore re-judged at the default 0 (strict 2 mas),
found the same sub-floor residuals the loop had deliberately declined, and
raised -- with ``ASTROM_CHECKPOINT_APPLY`` unset (correctly, m3+ is frozen) it
could not even correct them.  Every downstream phase then went
``DependencyNeverSatisfied``.

cloudef obs005 was the first field to reach that line, and it failed on 3
F162M/nrca residuals it had passed at 4.0 mas one job earlier.

``export`` at the top of the loop does not cover this: the submit script builds
an explicit ``--export`` list for sbatch, so only variables named at the call
site reach the job.
"""
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LOOP = REPO_ROOT / 'scripts' / 'reduction' / 'run_field_retie_loop.sh'

#: Variables that must reach BOTH cataloging invocations.  Each one changes what
#: the m2 checkpoint decides, so passing it to one and not the other means the
#: two runs judge the same frames by different rules.
MUST_REACH_BOTH = ('ASTROM_M2_CORRECTION_FLOOR_MAS',)


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
