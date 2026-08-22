#!/bin/bash
# Pin the checkout a batch job imports from.  Source this, do not execute it:
#
#     . "$(dirname "${BASH_SOURCE[0]}")/_pipe_root.sh"
#
# Reads PIPE_ROOT (and PYTHON) from the environment; a no-op when PIPE_ROOT is
# unset, which is the normal production case.
#
# Why this is not just PYTHONPATH
# -------------------------------
# Exporting PYTHONPATH="$PIPE_ROOT:$PYTHONPATH" does NOT pin the checkout.
# SLURM starts a job in the SUBMIT directory, and `python -m` puts the cwd at
# the FRONT of sys.path -- ahead of everything PYTHONPATH contributes.  So
# submitting from the main checkout with PIPE_ROOT=<worktree> imports MAIN's
# code while every log line names the worktree.  Nothing errors, because both
# checkouts import fine; the run just quietly executes the code you were trying
# to bypass.  On 2026-08-22 that cost three fields a full m12 fan-out against a
# fix that was sitting unused in the worktree.
#
# So: cd first, then PYTHONPATH, then VERIFY by importing and comparing paths.
# The verification is the part that matters -- it converts a silent wrong-code
# run into a refusal in the log's first lines.

if [ -n "${PIPE_ROOT:-}" ]; then
    if [ ! -d "$PIPE_ROOT/jwst_gc_pipeline" ]; then
        echo "[pipe-root] PIPE_ROOT=$PIPE_ROOT has no jwst_gc_pipeline/ -- refusing" >&2
        exit 2
    fi
    cd "$PIPE_ROOT" || exit 2
    export PYTHONPATH="$PIPE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
fi

"${PYTHON:-/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python}" - <<'_PIPE_ROOT_PY' || exit 2
import os

import jwst_gc_pipeline as _p

root = os.path.dirname(os.path.dirname(os.path.abspath(_p.__file__)))
print(f"[pipe-root] jwst_gc_pipeline resolves to {root}", flush=True)
want = os.environ.get('PIPE_ROOT') or ''
if want and os.path.realpath(want) != os.path.realpath(root):
    raise SystemExit(
        f"[pipe-root] PIPE_ROOT={want} but jwst_gc_pipeline imports from "
        f"{root} -- refusing to run the wrong checkout")
_PIPE_ROOT_PY
