"""Configurable data-root paths for the JWST GC pipeline.

Historically the pipeline hardcoded the HiPerGator data tree at
``/orange/adamginsburg/jwst`` (and a scratch tree at
``/blue/adamginsburg/adamginsburg/jwst``).  Those absolute paths make the
pipeline impossible to run anywhere else.  This module centralizes the roots
and lets them be overridden with environment variables so the same code can
run on HiPerGator or on a laptop:

    export JWST_GC_DATAROOT=/Users/me/work/jwst      # replaces /orange/.../jwst
    export JWST_GC_BLUEROOT=/Users/me/work/jwst      # replaces /blue/.../jwst

Both default to the original HiPerGator locations, so existing HiPerGator jobs
are unaffected.
"""
import os

#: Primary data tree (was ``/orange/adamginsburg/jwst``).
DATA_ROOT = os.environ.get("JWST_GC_DATAROOT", "/orange/adamginsburg/jwst").rstrip("/")

#: Scratch/secondary tree (was ``/blue/adamginsburg/adamginsburg/jwst``).
BLUE_ROOT = os.environ.get("JWST_GC_BLUEROOT", "/blue/adamginsburg/adamginsburg/jwst").rstrip("/")
