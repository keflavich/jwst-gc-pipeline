"""The satstar-deblend scripts address the post-#469 gc2211 trees.

#469 split ``/orange/adamginsburg/jwst/gc2211`` into one tree per observation
and moved 69,815 frame products out of the shared tree.  The shared tree still
exists, so a frame path built from it raises nothing and globs to zero matches
-- ten scripts under ``scripts/satstar_deblend/`` sat in exactly that state
(issue #470), and the registry guard
(``test_every_region_basepath_matches_the_registry``) does not cover them,
because these scripts address the data directly rather than through
``jwst_gc_pipeline.fields``.

This is the guard for that corner.  It refuses the drained FRAME root in any
script under that directory, and pins the two-root split
``scripts/satstar_deblend/gc2211_paths.py`` encodes: frames per observation,
pooled catalogues and the PSF-grid cache on the shared tree, which is where
those files are.

No disk access -- the paths are pinned as strings, so the test says the same
thing on a machine with no ``/orange`` mounted.
"""
import ast
import importlib.util
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts' / 'satstar_deblend'

#: The frame layout that the split emptied: ``<shared>/<FILTER>/...``.  The
#: shared root itself is still legitimate for catalogues and PSF grids, so the
#: filter directory is what makes a reference a frame reference.
DRAINED_FRAME_PREFIX = '/orange/adamginsburg/jwst/gc2211/F'

#: The module-level constant every one of the ten scripts carried.
DRAINED_ROOT_ASSIGNMENT = "= '/orange/adamginsburg/jwst/gc2211'"


def _load_paths_module():
    path = SCRIPTS / 'gc2211_paths.py'
    spec = importlib.util.spec_from_file_location('gc2211_paths', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gc = _load_paths_module()

SCRIPT_FILES = sorted(p for p in SCRIPTS.glob('*.py') if p.name != 'gc2211_paths.py')


@pytest.mark.parametrize('script', SCRIPT_FILES, ids=lambda p: p.name)
def test_no_script_addresses_the_drained_frame_tree(script):
    text = script.read_text()
    assert DRAINED_FRAME_PREFIX not in text, (
        f'{script.name} builds a frame path under the shared gc2211 tree, '
        f'which #469 emptied; frames live in gc2211_o<obs> (issue #470)')
    assert DRAINED_ROOT_ASSIGNMENT not in text, (
        f'{script.name} re-introduces the single shared-tree constant; use '
        f'gc2211_paths.obs_root/frame/pipeline for frames and '
        f'gc2211_paths.CATALOGS/PSFS for what stayed behind')


@pytest.mark.parametrize('script', SCRIPT_FILES, ids=lambda p: p.name)
def test_a_script_naming_a_gc2211_exposure_imports_the_path_module(script):
    """Anything spelling a ``jw02211`` product name has a tree to choose."""
    text = script.read_text()
    if 'jw02211' not in text:
        pytest.skip(f'{script.name} names no gc2211 product')
    imports = {node.names[0].name for node in ast.walk(ast.parse(text))
               if isinstance(node, ast.Import)}
    assert 'gc2211_paths' in imports, (
        f'{script.name} spells a jw02211 product name without importing '
        f'gc2211_paths, so its tree is hardcoded somewhere')


# ---------------------------------------------------------------------------
# gc2211_paths itself
# ---------------------------------------------------------------------------

def test_frames_are_rooted_in_the_observations_own_tree():
    exposure = 'jw02211023001_02201_00001_nrca1'
    assert gc.obs_root(exposure) == '/orange/adamginsburg/jwst/gc2211_o023'
    assert gc.frame(exposure, 'F200W') == (
        '/orange/adamginsburg/jwst/gc2211_o023/F200W/'
        'jw02211023001_02201_00001_nrca1_cal.fits')
    assert gc.pipeline(exposure, 'F200W') == (
        '/orange/adamginsburg/jwst/gc2211_o023/F200W/pipeline')


def test_the_observation_is_read_from_a_glob_pattern_too():
    """``batch_validate`` addresses four observations by pattern alone."""
    assert gc.frame_glob('jw02211046*_nrca3_cal.fits', 'F200W') == (
        '/orange/adamginsburg/jwst/gc2211_o046/F200W/jw02211046*_nrca3_cal.fits')
    assert gc.observation_of('jw02211049001_02201_00001_nrcb2_cal.fits') == '049'
    assert gc.observation_of('028') == '028'
    assert gc.observation_of('o028') == '028'


def test_a_name_with_no_observation_token_raises():
    with pytest.raises(ValueError, match='jw02211'):
        gc.obs_root('some_frame_cal.fits')


def test_an_observation_free_pattern_expands_over_every_tree():
    patterns = gc.all_obs_frame_globs('jw02211*nrca1_cal.fits', 'F200W')
    assert len(patterns) == len(gc.OBSERVATIONS)
    assert all('/gc2211_o' in p for p in patterns)
    assert not any(p.startswith('/orange/adamginsburg/jwst/gc2211/') for p in patterns)


def test_catalogues_and_psfs_stay_on_the_shared_tree():
    """The pooled pre-split catalogues and the PSF cache did not move, and
    there is no per-observation equivalent of either."""
    assert gc.catalog('f200w_merged_indivexp_merged_dao_basic.fits') == (
        '/orange/adamginsburg/jwst/gc2211/catalogs/'
        'f200w_merged_indivexp_merged_dao_basic.fits')
    assert gc.PSFS == '/orange/adamginsburg/jwst/gc2211/psfs/'
