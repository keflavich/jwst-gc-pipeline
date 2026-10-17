"""A function-local import of a module-level name makes that name local to the
WHOLE function, so any use of it outside the import's branch raises
UnboundLocalError.

``get_saturated_stars`` imported ``get_filtername`` inside its forced/MIRI
branch while #1187 added a later, unconditional ``get_filtername(header)``
call: every NIRCam frame without forced sources crashed the satstar stage
(quintuplet o003 m12, 2026-10-17, all 16 shards).
"""
import ast
from pathlib import Path

SOURCE = (Path(__file__).resolve().parents[1]
          / 'saturated_star_finding.py')


def _module_level_imports(tree):
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split('.')[0])
    return names


def test_no_function_reimports_a_module_level_name():
    tree = ast.parse(SOURCE.read_text())
    module_names = _module_level_imports(tree)
    shadowed = []
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(func):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    name = alias.asname or alias.name.split('.')[0]
                    if name in module_names:
                        shadowed.append(f'{func.name}:{node.lineno} {name}')
    assert not shadowed, (
        'function-local import re-binds a module-level name, which makes it '
        'unbound everywhere else in that function: ' + ', '.join(shadowed))
