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
import os
import pathlib

import pytest

from jwst_gc_pipeline.reduction.mast_obs_scope import (
    observation_number, on_disk_counts, uncal_uri_mask)

URI = 'mast:JWST/product/{}_02101_00001_nrcalong_uncal.fits'
O008 = URI.format('jw10678008001')
O080 = URI.format('jw10678080001')
O088 = URI.format('jw10678088002')


@pytest.mark.parametrize('obsid', ['008', '8', '08'])
def test_uncal_selector_picks_the_requested_observation(obsid):
    obs = observation_number(obsid)
    mask = uncal_uri_mask([O008, O080, O088], '10678', obs, '_nrc')
    assert list(mask) == [True, False, False]


@pytest.mark.parametrize('obsid', ['08', '8'])
def test_uncal_selector_pads_an_unpadded_obsid_itself(obsid):
    """The helper pads too, so a caller that forgets cannot reintroduce the bug."""
    mask = uncal_uri_mask([O008, O080], '10678', obsid, '_nrc')
    assert list(mask) == [True, False]


def test_unpadded_needle_is_the_old_failure():
    """Pins the mechanism: the raw '08' needle misses o008 and hits o080."""
    needle = 'jw10678' + '08'
    assert needle not in O008
    assert needle in O080


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
    ]
    for name in names:
        (tmp_path / name).touch()
    assert on_disk_counts(os.listdir(tmp_path), '10678', obsid) == (1, 1)


def test_script_reads_obsid_only_once():
    """Every later use goes through the padded `obs`."""
    path = (pathlib.Path(__file__).resolve().parents[3]
            / 'scripts' / 'reduction' / 'download_mast_observation.py')
    if not path.exists():
        pytest.skip('scripts/ not present in this install')
    tree = ast.parse(path.read_text())
    uses = [n.lineno for n in ast.walk(tree)
            if isinstance(n, ast.Attribute) and n.attr == 'obsid'
            and isinstance(n.value, ast.Name) and n.value.id == 'args']
    assert len(uses) == 1, f'args.obsid used at lines {uses}; use the padded obs'
