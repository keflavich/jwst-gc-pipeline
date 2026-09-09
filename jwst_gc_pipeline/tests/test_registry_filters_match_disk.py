"""A field's registered filter list must cover what is reduced on disk.

`fields.yaml` is what the cataloging driver believes a target observed.  A
filter that exists on disk but not in the registry raises at the END of an m12
finalize, after the fan-out has done its work:

    KeyError: "filter 'f150w' not observed by target 'w51'; known target/filter
    map: {'6151': ['f140m', 'f162m', ...]}"

That is w51 job 39893953 (2026-08-22).  Its 6151 entry listed 13 filters while
17 were reduced under `w51/`, all of them from jw06151:

    f150w    128 crf      f444w     32 crf
    f560w     32 crf      f1000w    32 crf

Sweeping every field at the time found w51 was the only one that had drifted, so
this is a drift check rather than a systemic gap -- but the cost of the drift is
a whole m12.

Direction matters.  Registry-superset-of-disk is FINE: cloudef lists 6 filters
and has 4 reduced, which just means two are not reduced yet.  Disk-not-in-registry
is the failure, because that is what the driver refuses.
"""
import glob
import os

import pytest
import yaml


HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIELDS_YAML = os.path.join(HERE, 'fields.yaml')
TREE = '/orange/adamginsburg/jwst'


def _registry():
    doc = yaml.safe_load(open(FIELDS_YAML))
    return doc.get('fields', doc)


def _registered_filters(fd):
    out = set()
    for _prop, od in (fd.get('observations') or {}).items():
        out |= {f.lower() for f in (od.get('filters') or [])}
    return out


def _disk_filters(field):
    """Filters with at least one reduced `_crf` under the field's tree."""
    out = set()
    for d in glob.glob(f'{TREE}/{field}/F*/pipeline'):
        if glob.glob(f'{d}/*_crf.fits'):
            out.add(os.path.basename(os.path.dirname(d)).lower())
    return out


FIELDS = sorted(_registry())


@pytest.mark.parametrize('field', FIELDS)
def test_every_reduced_filter_is_registered(field):
    """The failing direction: on disk but not in the registry."""
    if not os.path.isdir(f'{TREE}/{field}'):
        pytest.skip(f'{field} tree not present (CI)')
    disk = _disk_filters(field)
    if not disk:
        pytest.skip(f'{field} has no reduced frames')
    missing = sorted(disk - _registered_filters(_registry()[field]))
    assert not missing, (
        f'{field} has reduced frames for {missing} but fields.yaml does not '
        f'list them, so a cataloging run including them dies at the m12 '
        f'finalize with "filter not observed by target"')


@pytest.mark.parametrize('filt', ['f560w', 'f1000w'])
def test_the_w51_MIRI_filters_that_were_missing(filt):
    """Pinned by name.  The sweep above skips entirely on a machine with no
    /orange, which is every CI runner -- so without this the regression has no
    guard where it would actually be caught.

    f150w and f444w were pinned here too and have been REMOVED, because 6151
    never observed either.  MAST reports its NIRCam configurations as::

        F140M  F150W2;F162M  F182M  F187N  F210M
        F335M  F360M  F410M   F444W;F405N  F480M

    -- two SERIAL PAIRS and no standalone F150W or F444W.  With two filters in
    serial the NARROWER one is the filter, so those pairs are F162M and F405N,
    both of which w51 already lists.

    How the wrong names got pinned is worth recording, because the sweep above
    could not have caught it: `_disk_filters` reads `F*/pipeline` DIRECTORY
    names, and the directories were themselves created from the same mistake --
    the reduce driver's substring mask matched `'F150W'` inside
    `'F150W2;F162M'` and wrote an `F150W/` tree beside the `F162M/` one holding
    the identical exposures (#828, mask fixed in #829).  So disk and registry
    agreed, and agreed on something that was never observed.

    wd1 and wd2 are the reason this is parametrised on MIRI bands only rather
    than dropped: 1905 has a genuine standalone `F150W` AND `F444W`, and 3523 a
    genuine standalone `F150W`, each ALONGSIDE their serial pairs.  The name
    alone never says which a product is -- only FILTER/PUPIL does."""
    assert filt in _registered_filters(_registry()['w51'])


def test_w51_lists_both_instruments():
    """6151 is NIRCam + MIRI in one filter list, which is why f560w/f1000w
    belong here rather than in a separate key -- f770w/f1280w/f2100w were
    already listed that way."""
    filters = _registered_filters(_registry()['w51'])
    # f150w was here and is gone: 6151 observed F150W2;F162M, not F150W.
    assert {'f140m', 'f162m', 'f210m'} <= filters, 'NIRCam bands missing'
    assert {'f560w', 'f770w', 'f1000w', 'f2100w'} <= filters, 'MIRI bands missing'


def test_w51_does_not_list_the_wide_half_of_its_serial_pairs():
    """The regression this PR fixes, stated positively.

    6151's SW pair is `F150W2;F162M` and its LW pair `F444W;F405N`; the
    narrower pupil band is the filter in each.  Naming the wide half made the
    reduce driver fetch the same observation twice."""
    filters = _registered_filters(_registry()['w51'])
    assert 'f150w' not in filters, '6151 observed F150W2;F162M, not F150W'
    assert 'f444w' not in filters, '6151 observed F444W;F405N, not F444W'
    assert {'f162m', 'f405n'} <= filters, 'the real bands of both pairs'


def test_registry_may_list_more_than_disk():
    """States the tolerated direction, so nobody 'fixes' it into a failure: a
    registered filter that is not reduced yet is normal."""
    if not os.path.isdir(f'{TREE}/cloudef'):
        pytest.skip('cloudef tree not present (CI)')
    extra = _registered_filters(_registry()['cloudef']) - _disk_filters('cloudef')
    assert isinstance(extra, set)     # asserting only that this is not an error
