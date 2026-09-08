"""The --allow-registration-fail override path must reach stage().

`withheld` is assigned by `gate_by_instrument`, which runs INSIDE the block
`override` skips.  So on the override path the name was unbound and `main`
raised UnboundLocalError at the `stage()` call -- before a single file was
copied.  The override had therefore never worked: every use of
`--allow-registration-fail` with `ALLOW_REGISTRATION_FAIL=1` died there.

That failure is silent about its cause: an operator who deliberately overrode
a red gate, with a written justification, got a traceback naming an internal
variable and no release.
"""
import ast
import os


_SRC = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..',
    'scripts', 'release', 'stage_release.py'))


def _main_fn():
    tree = ast.parse(open(_SRC).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'main':
            return node
    raise AssertionError('no main() in stage_release.py')


def _assigned_names(node):
    """Every name bound by an assignment, including tuple unpacking.

    `withheld` arrives as `items, withheld, refusal = gate_by_instrument(...)`,
    so a scanner that only understands `ast.Name` targets sees no assignment
    at all -- which is how this bug survived review in the first place.
    """
    out = set()
    for n in ast.walk(node):
        if not isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            continue
        targets = n.targets if isinstance(n, ast.Assign) else [n.target]
        for t in targets:
            for sub in ast.walk(t):
                if isinstance(sub, ast.Name):
                    out.add(sub.id)
    return out


def _guarded_by_not_override(fn):
    """Names assigned ONLY inside an `if not override:` body."""
    inside = set()
    for n in ast.walk(fn):
        if not isinstance(n, ast.If):
            continue
        t = n.test
        if isinstance(t, ast.UnaryOp) and isinstance(t.op, ast.Not) \
           and isinstance(t.operand, ast.Name) and t.operand.id == 'override':
            for stmt in n.body:
                inside |= _assigned_names(stmt)
    return inside


def test_withheld_is_bound_before_the_skippable_gate_block():
    fn = _main_fn()
    guarded = _guarded_by_not_override(fn)
    # everything assigned inside the skipped block must also be assigned
    # outside it, or the override path references an unbound local
    outside = set()
    for stmt in fn.body:
        if isinstance(stmt, ast.If):
            t = stmt.test
            if isinstance(t, ast.UnaryOp) and isinstance(t.op, ast.Not) \
               and isinstance(t.operand, ast.Name) and t.operand.id == 'override':
                continue
        outside |= _assigned_names(stmt)
    unbound = {n for n in guarded if n not in outside}
    # `withheld` is the one main() reads after the block; assert it explicitly
    # so a rename cannot quietly drop the guarantee
    assert 'withheld' in guarded, 'the gate block should still assign withheld'
    assert 'withheld' not in unbound, (
        'withheld is assigned only inside the block `override` skips, so the '
        'override path raises UnboundLocalError before staging anything')


def test_withheld_defaults_to_empty_not_none():
    """`stage()` takes `withheld_instruments=withheld or None`, and the manifest
    distinguishes 'nothing withheld' from 'gate never ran'.  An empty mapping
    is the honest value when no gate ran -- nothing was withheld."""
    src = open(_SRC).read()
    i = src.index('continuity_gate = "skipped(override)"')
    window = src[i:i + 900]
    assert 'withheld = {}' in window, (
        'withheld should be bound to an empty mapping next to the override '
        'decision, before the block that would otherwise assign it')
