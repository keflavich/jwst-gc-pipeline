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

_PR_ROOT=$("${PYTHON:-/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python}" - <<'_PIPE_ROOT_PY'
import os

import jwst_gc_pipeline as _p

print(os.path.dirname(os.path.dirname(os.path.abspath(_p.__file__))))
_PIPE_ROOT_PY
) || exit 2

# Report the COMMIT, not just the path.  A checkout's path says which tree ran;
# it does not say which code.  Two jobs that both printed
# `.../jwst-gc-pipeline-wt-excl` this week ran different code, because the
# worktree had main merged into it between their submissions -- sgrc job
# 39941339 tripped on a sub-floor residual that the merged per-field floor
# (#478) would have filtered, and answering "did that job have the fix?"
# required comparing its submit time against a merge time.  With the commit in
# the log it is one line.
#
# Not fatal if git is unavailable or the tree is not a repo: this is provenance
# for the log, and it must never be the reason a reduction does not start.
_PR_DESC="$_PR_ROOT"
if _PR_SHA=$(git -C "$_PR_ROOT" rev-parse --short HEAD 2>/dev/null); then
    if [ -n "$(git -C "$_PR_ROOT" status --porcelain 2>/dev/null)" ]; then
        _PR_DESC="$_PR_ROOT @ $_PR_SHA (DIRTY)"
    else
        _PR_DESC="$_PR_ROOT @ $_PR_SHA"
    fi
fi
echo "[pipe-root] jwst_gc_pipeline resolves to $_PR_DESC"

"${PYTHON:-/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python}" - <<'_PIPE_ROOT_PY' || exit 2
import os

import jwst_gc_pipeline as _p

root = os.path.dirname(os.path.dirname(os.path.abspath(_p.__file__)))
want = os.environ.get('PIPE_ROOT') or ''
if want and os.path.realpath(want) != os.path.realpath(root):
    raise SystemExit(
        f"[pipe-root] PIPE_ROOT={want} but jwst_gc_pipeline imports from "
        f"{root} -- refusing to run the wrong checkout")
_PIPE_ROOT_PY
