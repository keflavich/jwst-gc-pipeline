"""What the reduction actually hands MAST's product list (issue #416).

``observation_scope_mask`` is unit-tested next door, but its one production
wiring is a single line inside ``PipelineRerunNIRCAM-LONG.main``, and a
source-text guard on that line passes through both regressions that matter:
masking ``obs_table['filters']`` instead of ``obs_table['obs_id']`` (no row
then carries ``-o``, every row counts as unattributed, the mask goes all-True
and the download is obs-blind again), and swapping ``proposal_id`` with
``field`` (the prefix becomes ``jw00001-o10678``, the mask empties, the
reduction downloads nothing).  Neither edits that line's text.

``main`` takes ``Observations`` by injection, so drive it with a fake whose
``query_criteria`` returns a small table of real ``obs_id`` spellings and whose
``get_product_list`` records the rows it was handed and stops the run.  The
field's basepath is redirected to a tmpdir (``GC_BASEPATH_OVERRIDE``), so this
touches no data tree.

The module's hyphenated filename blocks a normal import, and importing it pulls
the jwst pipeline (~17 s), so it is loaded once for the module.
"""
import importlib.util
import os
import pathlib

import pytest
from astropy.table import Table

DRIVER = (pathlib.Path(__file__).resolve().parents[1]
          / 'PipelineRerunNIRCAM-LONG.py')

#: brick (o001) and cloudc (o002/o003) share proposal 2221, and the obs table
#: is queried per proposal, so a brick reduce sees all of them.  ``-c1001`` is
#: a candidate association: it attributes itself to no observation and is kept.
OBS_TABLE = Table({
    'obs_id': ['jw02221-o001_t001_nircam_clear-f212n',
               'jw02221-o002_t001_nircam_clear-f212n',
               'jw02221-o003_t001_nircam_clear-f212n',
               'jw02221-c1001_t001_nircam_clear-f212n'],
    'filters': ['F212N', 'F212N', 'F212N', 'F212N'],
})


@pytest.fixture(scope='module')
def driver():
    spec = importlib.util.spec_from_file_location('_prn_long', DRIVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _StopAfterProductList(Exception):
    """Stop ``main`` once the download has been handed its table."""


class _FakeObservations:
    """The astroquery surface ``main``'s MAST block uses."""

    cache_location = None
    TIMEOUT = None

    def __init__(self):
        self.handed = None

    def query_criteria(self, **criteria):
        self.criteria = criteria
        return OBS_TABLE.copy()

    def get_product_list(self, obs_table):
        self.handed = [str(o) for o in obs_table['obs_id']]
        raise _StopAfterProductList


def _run(driver, tmp_path, monkeypatch, regionname='brick', field='001',
         proposal_id='2221'):
    fake = _FakeObservations()
    monkeypatch.setenv('GC_BASEPATH_OVERRIDE', str(tmp_path))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(_StopAfterProductList):
        driver.main('F212N', 'nrcblong', Observations=fake,
                    regionname=regionname, field=field,
                    proposal_id=proposal_id, skip_step1and2=True)
    return fake


def test_the_download_is_handed_only_this_observations_rows(driver, tmp_path,
                                                            monkeypatch):
    """brick's o001 rows plus the unattributed candidate; cloudc's o002/o003
    stay behind."""
    fake = _run(driver, tmp_path, monkeypatch)
    assert fake.criteria == {'proposal_id': '2221'}
    assert fake.handed == ['jw02221-o001_t001_nircam_clear-f212n',
                           'jw02221-c1001_t001_nircam_clear-f212n']


def test_a_second_observation_of_the_same_proposal_gets_its_own_rows(
        driver, tmp_path, monkeypatch):
    """The same table, reduced as cloudc (proposal 2221 observation 002): the
    mask follows ``field``, so swapping ``proposal_id`` and ``field`` at the
    call site cannot pass."""
    fake = _run(driver, tmp_path, monkeypatch, regionname='cloudc',
                field='002')
    assert fake.handed == ['jw02221-o002_t001_nircam_clear-f212n',
                           'jw02221-c1001_t001_nircam_clear-f212n']


def test_the_mask_reads_the_obs_id_column(driver, tmp_path, monkeypatch):
    """A mask over any column that carries no ``-o`` token reads every row as
    unattributed and keeps the lot -- the obs-blind download, silently back."""
    fake = _run(driver, tmp_path, monkeypatch)
    assert len(fake.handed) < len(OBS_TABLE), (
        'every row was handed to get_product_list; the obs mask is not '
        'reading the obs_id column')
    assert not any('-o002' in o or '-o003' in o for o in fake.handed)


def test_the_download_environment_is_left_alone(driver, tmp_path, monkeypatch):
    """The run wrote only under the redirected basepath."""
    _run(driver, tmp_path, monkeypatch)
    assert (tmp_path / 'F212N' / 'pipeline').is_dir()
    assert os.path.isdir(tmp_path)
