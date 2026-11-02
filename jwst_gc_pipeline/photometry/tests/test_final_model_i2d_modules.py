"""The final phase writes the merged-catalog MODEL i2d for the merged module only.

The model i2d is a display-only full resample, and the sgrb2 m7 finalize
(2026-10) spent ~6.5 h of its 72 h wall writing one per module per filter.  A
release stages only the merged module's (``stage_release.IMAGE_RE``), so the
nrca/nrcb model mosaics are written only with ``--manual-model-i2d-all-modules``
(#1225).  A run with no merged module -- a disjoint field, or a MIRI run --
writes every module's, since those are its only mosaics.
"""
import importlib.util
import inspect
import pathlib
import sys

import pytest

from jwst_gc_pipeline.photometry import cataloging
from jwst_gc_pipeline.photometry.manual_defaults import MANUAL_DEFAULTS

ROOT = pathlib.Path(__file__).resolve().parents[3]


def _stage_release():
    spec = importlib.util.spec_from_file_location(
        '_sr_model_i2d', ROOT / 'scripts' / 'release' / 'stage_release.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules['_sr_model_i2d'] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize('modules, all_modules, want', [
    (['nrca', 'nrcb', 'merged'], False, ['merged']),
    (('nrca', 'nrcb', 'merged'), False, ['merged']),
    (['merged'], False, ['merged']),
    (['nrca', 'nrcb', 'merged'], True, ['nrca', 'nrcb', 'merged']),
    (['nrca', 'nrcb'], False, ['nrca', 'nrcb']),     # disjoint field
    (['nrcb'], False, ['nrcb']),
    (['mirimage'], False, ['mirimage']),             # MIRI
])
def test_which_modules_write_the_final_model_i2d(modules, all_modules, want):
    assert cataloging.final_phase_model_i2d_modules(modules, all_modules) == want


def test_the_nrca_nrcb_model_i2d_is_opt_in():
    assert MANUAL_DEFAULTS['manual_model_i2d_all_modules'] is False
    assert MANUAL_DEFAULTS['manual_keep_intermediate_model_i2d'] is False


def test_run_manual_pipeline_gates_the_model_i2d_by_module():
    src = ' '.join(inspect.getsource(cataloging.run_manual_pipeline).split())
    assert ("_final_model_i2d_modules = final_phase_model_i2d_modules( modules, "
            "getattr(options, 'manual_model_i2d_all_modules', False))") in src
    assert ("write_model_i2d=(_keep_all_model_i2d or (phase == _final_phase and "
            "module in _final_model_i2d_modules))") in src


@pytest.mark.parametrize('module, staged', [
    ('merged', True), ('nrca', False), ('nrcb', False)])
def test_the_release_stages_only_the_merged_model_i2d(module, staged):
    """The default rests on this: if the release starts staging a per-module
    model i2d, the nrca/nrcb mosaics must be written again by default."""
    name = (f'jw05365-o001_t001_nircam_clear-f182m-{module}_resbgsub_m7_'
            f'daophot_basic_mergedcat_model_i2d.fits')
    assert bool(_stage_release().IMAGE_RE.search(name)) is staged
