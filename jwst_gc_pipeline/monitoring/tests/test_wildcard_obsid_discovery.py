"""A wildcard obsid registration is expanded from the products on disk.

``obsids: {nircam: '*'}`` records "this field owns every observation of the
proposal" and carries no observation numbers.  Every consumer that ENUMERATES
obsids to build one row per observation got the literal ``'*'`` back, so the
monitor showed ONE gc-treasury row whose stage counts pooled all 139 of 10678's
tiles: a tile that reduced and a tile that did not landed in one cell.

The numbers exist on disk the moment anything lands, so ``scan.observations``
reads them from there and keeps the wildcard row only while nothing has -- which
is gc-treasury's state today.  Issue #439, option 1.

The attribution is per-INSTRUMENT on purpose: 10678 takes NIRCam prime with MIRI
parallel, and a NIRCam row must not grow from MIRI products.
"""
import os

import pytest

from jwst_gc_pipeline import fields as _fields
from jwst_gc_pipeline.monitoring import scan

#: The one field registered with a wildcard obsid today.
WILDCARD_FIELD = 'gc-treasury'
WILDCARD_PROPOSAL = '10678'


def _tree(root, names):
    """``<root>/<FILTER>/pipeline/<name>`` for each ``(filter, name)``."""
    for filt, name in names:
        d = os.path.join(root, filt, 'pipeline')
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, name), 'w').close()
    return root


@pytest.fixture
def landed(tmp_path, monkeypatch):
    """Point the scanner at a throwaway tree and hand back a builder."""
    def build(names):
        scan.clear_cache()
        root = _tree(str(tmp_path / 'tree'), names)
        monkeypatch.setattr(scan, 'basepath',
                            lambda target, cutout_label=None: root)
        scan.clear_cache()
        return root
    return build


def test_the_registry_still_declares_the_wildcard():
    """If this fails the fixture below is testing a shape that no longer exists."""
    assert _fields.claims_every_observation(WILDCARD_FIELD, 'nircam')


def test_nothing_landed_keeps_the_wildcard_row(landed):
    """The premise the issue rests on: while no product exists there is nothing
    to enumerate, and a field that vanishes from the monitor is worse than one
    pooled row."""
    landed([])
    assert scan.observations(WILDCARD_FIELD, 'nircam') == [
        (WILDCARD_PROPOSAL, '*')]


def test_landed_products_become_one_row_each(landed):
    """The fix.  Both product spellings are read -- the per-exposure
    ``jw{PPPPP}{OOO}{VVV}_`` and the mosaic ``jw{PPPPP}-o{OOO}_``."""
    landed([
        ('F212N', 'jw10678001001_02101_00001_nrca1_destreak_o001_crf.fits'),
        ('F212N', 'jw10678001001_02101_00002_nrcb4_destreak_o001_crf.fits'),
        ('F480M', 'jw10678-o042_t001_nircam_clear-f480m-merged_i2d.fits'),
    ])
    assert scan.observations(WILDCARD_FIELD, 'nircam') == [
        (WILDCARD_PROPOSAL, '001'), (WILDCARD_PROPOSAL, '042')]


def test_a_parallel_instrument_does_not_supply_the_other_s_rows(landed):
    """10678's MIRI is a parallel and need not share the NIRCam observation
    number, so the NIRCam row must not grow from MIRI frames -- and vice versa."""
    landed([
        ('F212N', 'jw10678001001_02101_00001_nrca1_destreak_o001_crf.fits'),
        ('F770W', 'jw10678007001_02101_00001_mirimage_o007_crf.fits'),
    ])
    assert scan.observations(WILDCARD_FIELD, 'nircam') == [
        (WILDCARD_PROPOSAL, '001')]
    assert scan.observations(WILDCARD_FIELD, 'miri') == [
        (WILDCARD_PROPOSAL, '007')]


def test_another_proposal_in_the_same_tree_is_ignored(landed):
    """A foreign frame misfiled under the tree (which has happened -- 2221 o002
    crf in brick's LW directories) must not register as this proposal's
    observation."""
    landed([
        ('F212N', 'jw10678001001_02101_00001_nrca1_destreak_o001_crf.fits'),
        ('F212N', 'jw02221002001_05101_00001_nrca1_destreak_o002_crf.fits'),
    ])
    assert scan.observations(WILDCARD_FIELD, 'nircam') == [
        (WILDCARD_PROPOSAL, '001')]


def test_scan_field_reports_one_run_row_per_discovered_observation(landed):
    """The consequence in the monitor: the pooled row becomes per-tile rows."""
    landed([
        ('F212N', 'jw10678003001_02101_00001_nrca1_destreak_o003_crf.fits'),
        ('F212N', 'jw10678004001_02101_00001_nrca1_destreak_o004_crf.fits'),
    ])
    runs = scan.scan_field(WILDCARD_FIELD, with_headers=False)['runs']
    assert [(r['proposal'], r['obsid']) for r in runs] == [
        (WILDCARD_PROPOSAL, '003'), (WILDCARD_PROPOSAL, '004')]


@pytest.mark.parametrize('target,instrument,expected', [
    ('brick', 'nircam', [('1182', '004'), ('2221', '001')]),
    ('sickle', 'nircam', [('3958', '007')]),
    ('cloudef', 'miri', [('2092', '004'), ('2092', '006'), ('2092', '008')]),
])
def test_an_enumerated_field_is_untouched(target, instrument, expected):
    """No disk read and no change for the fifteen fields that list their
    observations: a registered observation with no products still shows as a row
    of pending stages."""
    scan.clear_cache()
    assert scan.observations(target, instrument) == expected
