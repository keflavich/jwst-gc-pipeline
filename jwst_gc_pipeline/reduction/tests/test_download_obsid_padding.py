"""`download_mast_observation --obsid` must mean the same thing however it is padded.

MAST names observation 8 of program 10678 ``jw10678-o008`` (products) and
``jw10678008...`` (exposures).  The script padded the obsid for the obs-table
scope but used the raw string for the uncal selector and the on-disk report:

* ``'08'`` built the needle ``jw1067808``, a substring of ``jw10678080...`` and
  not of ``jw10678008...``, so the run selected no uncal for observation 008;
* ``startswith('jw10678-o8')`` matched no file on disk, so the report said
  "0 uncal, 0 image3 asn" and then "MAST has not published level 3".

The selector and the counter now live in ``mast_obs_scope`` and the script calls
them with the padded number, so these tests run the production code.
"""
import ast
import importlib.util
import os
import pathlib
import sys
import types

import numpy as np
import pytest
from astropy.table import Table

from jwst_gc_pipeline.reduction.mast_obs_scope import (
    observation_number, on_disk_counts, uncal_uri_mask)

URI = 'mast:JWST/product/{}_02101_00001_nrcalong_uncal.fits'
O008 = URI.format('jw10678008001')
O080 = URI.format('jw10678080001')
O088 = URI.format('jw10678088002')
# A sibling that shares the first nine characters of observation 008's needle
# (`jw1067800`): a needle cut short by one character would take it.
O009 = URI.format('jw10678009001')


@pytest.mark.parametrize('obsid', ['008', '8', '08'])
def test_uncal_selector_picks_the_requested_observation(obsid):
    obs = observation_number(obsid)
    mask = uncal_uri_mask([O008, O080, O088, O009], '10678', obs, '_nrc')
    assert list(mask) == [True, False, False, False]


@pytest.mark.parametrize('obsid', ['08', '8'])
def test_uncal_selector_pads_an_unpadded_obsid_itself(obsid):
    """The helper pads too, so a caller that forgets cannot reintroduce the bug."""
    mask = uncal_uri_mask([O008, O080], '10678', obsid, '_nrc')
    assert list(mask) == [True, False]


def test_uncal_selector_checks_suffix_and_detector_token():
    uris = [O008,
            O008.replace('_uncal', '_rate'),
            O008.replace('_nrcalong', '_mirimage')]
    assert list(uncal_uri_mask(uris, '10678', '008', '_nrc')) == [True, False, False]


def test_uncal_selector_on_an_empty_list_is_boolean():
    mask = uncal_uri_mask([], '10678', '8', '_nrc')
    assert mask.dtype == bool and len(mask) == 0


@pytest.mark.parametrize('obsid', ['008', '8', '08'])
def test_on_disk_counts_agree_for_every_spelling(tmp_path, obsid):
    names = [
        'jw10678008001_02101_00001_nrcalong_uncal.fits',
        'jw10678-o008_20260912t000000_image3_00001_asn.json',
        # decoys from observation 080 and from another suffix
        'jw10678080001_02101_00001_nrcalong_uncal.fits',
        'jw10678-o080_20260912t000000_image3_00001_asn.json',
        'jw10678008001_02101_00001_nrcalong_rate.fits',
        # a stage-2 association of the same observation is not an image3 asn
        'jw10678-o008_20260912t000000_image2_00001_asn.json',
        # a sibling observation that shares the prefix `jw1067800`
        'jw10678009001_02101_00001_nrcalong_uncal.fits',
        'jw10678-o009_20260912t000000_image3_00001_asn.json',
    ]
    for name in names:
        (tmp_path / name).touch()
    assert on_disk_counts(os.listdir(tmp_path), '10678', obsid) == (1, 1)


@pytest.mark.parametrize('obsid', ['', '   ', '\t\n'])
def test_an_empty_obsid_is_an_error_not_match_all(obsid):
    """`observation_number('')` is '' and the needle would be the bare proposal
    prefix, which every URI of the program contains."""
    with pytest.raises(ValueError, match='names no observation'):
        uncal_uri_mask([O008, O080], '10678', obsid, '_nrc')
    with pytest.raises(ValueError, match='names no observation'):
        on_disk_counts(['jw10678008001_02101_00001_nrcalong_uncal.fits'],
                       '10678', obsid)


SCRIPT = (pathlib.Path(__file__).resolve().parents[3]
          / 'scripts' / 'reduction' / 'download_mast_observation.py')


def test_script_reads_obsid_only_once_through_observation_number():
    """Every later use goes through the padded `obs`."""
    if not SCRIPT.exists():
        pytest.skip('scripts/ not present in this install')
    tree = ast.parse(SCRIPT.read_text())
    reads = [(n, p) for p in ast.walk(tree) for n in ast.iter_child_nodes(p)
             if isinstance(n, ast.Attribute) and n.attr == 'obsid'
             and isinstance(n.value, ast.Name) and n.value.id == 'args']
    assert len(reads) == 1, (
        f'args.obsid used at lines {[n.lineno for n, _ in reads]}; use the '
        f'padded obs')
    node, parent = reads[0]
    assert (isinstance(parent, ast.Call)
            and isinstance(parent.func, ast.Name)
            and parent.func.id == 'observation_number'), (
        f'args.obsid at line {node.lineno} is not passed to observation_number')


# ---- the script's main(), against a stubbed MAST -----------------------------

OBS_TABLE_IDS = ['jw10678-o008_t001_nircam_f212n', 'jw10678-o080_t001_nircam_f212n',
                 'jw10678-o088_t001_nircam_f212n']


class _StubObservations:
    """Just enough of astroquery.mast.Observations for main().

    `download_products` writes the requested files into `download_dir`, so the
    script's own on-disk report sees them.
    """
    TIMEOUT = None
    cache_location = None
    downloaded = []
    with_image3 = True

    @classmethod
    def clear_cache(cls):
        pass

    @classmethod
    def query_criteria(cls, **kwargs):
        return Table({'obs_id': np.array(OBS_TABLE_IDS),
                      'filters': np.array(['F212N'] * 3)})

    @classmethod
    def get_product_list(cls, obs_table):
        rows = []
        for oid in ('008', '080', '088', '009'):
            for exp in ('001', '002'):
                rows.append((f'mast:JWST/product/jw10678{oid}{exp}_02101_00001_'
                             f'{{}}_uncal.fits', 'SCIENCE'))
        uris = [u.format(det) for u, _ in rows
                for det in ('nrcalong', 'mirimage')]
        types_ = ['SCIENCE'] * len(uris)
        for oid in ('008', '080'):
            kinds = ['image3', 'image2'] if cls.with_image3 else ['image2']
            for kind in kinds:
                uris.append(f'mast:JWST/product/jw10678-o{oid}_'
                            f'20260912t000000_{kind}_00001_asn.json')
                types_.append('INFO')
        return Table({'dataURI': np.array(uris), 'productType': np.array(types_)})

    @classmethod
    def filter_products(cls, products, extension):
        keep = np.array([str(u).endswith('.' + extension)
                         for u in products['dataURI']], dtype=bool)
        return products[keep]

    @classmethod
    def download_products(cls, table, download_dir):
        manifest = []
        for uri in table['dataURI']:
            path = os.path.join(download_dir, os.path.basename(uri))
            pathlib.Path(path).touch()
            cls.downloaded.append(os.path.basename(uri))
            manifest.append((path,))
        return Table({'Local Path': np.array([m[0] for m in manifest])})


def _load_script(monkeypatch, tmp_path, argv):
    if not SCRIPT.exists():
        pytest.skip('scripts/ not present in this install')
    astroquery = types.ModuleType('astroquery')
    mast = types.ModuleType('astroquery.mast')
    mast.Observations = _StubObservations
    astroquery.mast = mast
    monkeypatch.setitem(sys.modules, 'astroquery', astroquery)
    monkeypatch.setitem(sys.modules, 'astroquery.mast', mast)
    spec = importlib.util.spec_from_file_location('_download_mast_obs_under_test',
                                                  SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, 'basepath', lambda target: str(tmp_path))
    monkeypatch.setattr(sys, 'argv', ['download_mast_observation.py'] + argv)
    _StubObservations.downloaded = []
    _StubObservations.with_image3 = True
    return mod


def _argv(obsid, instrument='nircam'):
    return ['--target', 'gc-treasury', '--proposal', '10678', '--obsid', obsid,
            '--filters', 'F212N', '--instrument', instrument]


@pytest.mark.parametrize('obsid', ['8', '08', '008', ' 008'])
def test_main_fetches_the_requested_observation_for_every_spelling(
        monkeypatch, tmp_path, capsys, obsid):
    mod = _load_script(monkeypatch, tmp_path, _argv(obsid))
    assert mod.main() == 0
    out = capsys.readouterr().out
    uncal = sorted(f for f in _StubObservations.downloaded
                   if f.endswith('_uncal.fits'))
    assert uncal == ['jw10678008001_02101_00001_nrcalong_uncal.fits',
                     'jw10678008002_02101_00001_nrcalong_uncal.fits']
    assert '2 uncal to fetch (0 already on disk)' in out
    # the report and the messages use the padded number
    assert 'on disk now 2 uncal, 1 image3 asn for o008' in out
    assert 'NO image3 association' not in out
    assert 'o8' not in out.replace('o008', '').replace('o080', '').replace(
        'o088', '')


def test_main_honours_the_instrument_token(monkeypatch, tmp_path):
    """`--instrument miri` selects the `_mirimage` exposures, not `_nrc` ones."""
    mod = _load_script(monkeypatch, tmp_path, _argv('8', instrument='miri'))
    assert mod.main() == 0
    uncal = sorted(f for f in _StubObservations.downloaded
                   if f.endswith('_uncal.fits'))
    assert uncal == ['jw10678008001_02101_00001_mirimage_uncal.fits',
                     'jw10678008002_02101_00001_mirimage_uncal.fits']


def test_main_reports_a_missing_image3_association(monkeypatch, tmp_path, capsys):
    mod = _load_script(monkeypatch, tmp_path, _argv('8'))
    _StubObservations.with_image3 = False
    assert mod.main() == 0
    out = capsys.readouterr().out
    assert 'on disk now 2 uncal, 0 image3 asn for o008' in out
    assert 'NO image3 association for o008' in out


def test_main_skips_what_is_already_on_disk(monkeypatch, tmp_path, capsys):
    mod = _load_script(monkeypatch, tmp_path, _argv('08'))
    pipeline = tmp_path / 'F212N' / 'pipeline'
    pipeline.mkdir(parents=True)
    (pipeline / 'jw10678008001_02101_00001_nrcalong_uncal.fits').touch()
    assert mod.main() == 0
    out = capsys.readouterr().out
    assert '1 uncal to fetch (1 already on disk)' in out
    assert 'on disk now 2 uncal, 1 image3 asn for o008' in out


@pytest.mark.parametrize('obsid', ['', '   '])
def test_main_rejects_an_empty_obsid(monkeypatch, tmp_path, obsid):
    mod = _load_script(monkeypatch, tmp_path, _argv(obsid))
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code == 2
    assert _StubObservations.downloaded == []
