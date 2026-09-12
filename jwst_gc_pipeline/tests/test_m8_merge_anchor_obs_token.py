"""The m8 merge has to find a per-OBSERVATION field's m7 anchor.

``m8_merge_partials`` derives its anchor from ``--catdir`` + ``--module`` when
no explicit ``--m7`` is given, and it only ever built the whole-field name
``basic_<module>_indivexp_photometry_tables_merged_resbgsub_m7.fits``.  A
per-observation field appends the obs token -- ``..._resbgsub_m7_o049.fits`` --
so every such field's merge died at the argument check:

    m8_merge_partials.py: error: m7 anchor not found:
    .../basic_merged_indivexp_photometry_tables_merged_resbgsub_m7.fits

gc2211_o049, job 41879980 (2026-09-12).  The cost is the ordering: the merge is
``afterok`` on the per-band partials, so both bands had already been fitted (9
and 8.5 minutes) before the name was tested.  brick o001/o004, the other gc2211
observations, gc1266 and m4 all carry the same token.
"""
import importlib.util
import os
import pathlib

import pytest

_SCRIPT = (pathlib.Path(__file__).resolve().parents[2]
           / 'scripts' / 'reduction' / 'm8_merge_partials.py')
_STEM = 'basic_merged_indivexp_photometry_tables_merged_resbgsub_m7'


def _load():
    spec = importlib.util.spec_from_file_location('m8_merge_partials', _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_obs_token_anchor_is_found_when_the_obs_is_named(tmp_path):
    (tmp_path / f'{_STEM}_o049.fits').touch()
    m7 = _load()._default_m7(str(tmp_path), 'merged', '049')
    assert os.path.basename(m7) == f'{_STEM}_o049.fits'


def test_the_obs_is_zero_padded(tmp_path):
    """FIELD reaches the sbatch unpadded often enough to matter."""
    (tmp_path / f'{_STEM}_o049.fits').touch()
    m7 = _load()._default_m7(str(tmp_path), 'merged', '49')
    assert os.path.basename(m7) == f'{_STEM}_o049.fits'


def test_a_lone_obs_token_anchor_is_found_without_being_named(tmp_path):
    """One token'd anchor is unambiguous, so the caller need not repeat it."""
    (tmp_path / f'{_STEM}_o049.fits').touch()
    m7 = _load()._default_m7(str(tmp_path), 'merged')
    assert os.path.basename(m7) == f'{_STEM}_o049.fits'


def test_the_whole_field_name_still_wins_when_it_exists(tmp_path):
    """The un-token'd fields must keep resolving exactly as before."""
    (tmp_path / f'{_STEM}.fits').touch()
    (tmp_path / f'{_STEM}_o049.fits').touch()
    assert os.path.basename(
        _load()._default_m7(str(tmp_path), 'merged')) == f'{_STEM}.fits'


def test_a_named_obs_beats_the_whole_field_name(tmp_path):
    (tmp_path / f'{_STEM}.fits').touch()
    (tmp_path / f'{_STEM}_o049.fits').touch()
    assert os.path.basename(
        _load()._default_m7(str(tmp_path), 'merged', '049')) == f'{_STEM}_o049.fits'


def test_two_obs_token_anchors_refuse_rather_than_guess(tmp_path):
    """brick holds o001 and o004 side by side; picking one silently would
    overlay one observation's bands onto the other's row order."""
    (tmp_path / f'{_STEM}_o001.fits').touch()
    (tmp_path / f'{_STEM}_o004.fits').touch()
    with pytest.raises(SystemExit) as ei:
        _load()._default_m7(str(tmp_path), 'merged')
    assert '--m7' in str(ei.value) and '--obs' in str(ei.value)


def test_an_empty_tree_reports_the_whole_field_name(tmp_path):
    """So the caller's 'anchor not found' names a path a human recognises."""
    m7 = _load()._default_m7(str(tmp_path), 'merged')
    assert os.path.basename(m7) == f'{_STEM}.fits'
