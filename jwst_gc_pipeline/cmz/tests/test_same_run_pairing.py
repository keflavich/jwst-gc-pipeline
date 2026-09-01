"""SAME-RUN gate pairing: which catalog each shipped image is measured against.

The gate keys on MODULE as well as (filter, observation), because a disjoint
field (arches, quintuplet: NRCA and NRCB share no sky) would otherwise compare
NRCA's image with NRCB's catalog and report a pairing error as an astrometry
failure.

Keying alone is not enough.  A field can ship per-module IMAGES with a MERGED
catalog -- arches on 2026-08-30 had nrca/nrcb mosaics from 08-28 and a single
`basic_merged_...resbgsub_m8` table from 08-29 -- and a plain key intersection
between the two is then EMPTY.  The gate loop runs zero times, finds no
failures, and reports a pass having compared nothing.  That is the same defect
in a different place as the gate it belongs to: a green result standing for an
empty result set.

So these tests assert on the PAIRS, not on the return code: that the merged
fallback produces a comparison, that an own-module catalog still wins when one
ships, that a disjoint field never crosses the modules, and that an image with
no partner at all is reported instead of skipped.
"""
import importlib.util
import os

import pytest

# .../jwst_gc_pipeline/cmz/tests/test_same_run_pairing.py -> repo root (4 up)
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
_REL = os.path.join(_REPO, 'scripts', 'release')


@pytest.fixture(scope='module')
def sr():
    import sys
    if _REL not in sys.path:            # scripts/release siblings import each other
        sys.path.insert(0, _REL)
    path = os.path.join(_REL, 'stage_release.py')
    spec = importlib.util.spec_from_file_location('stage_release', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _img(filt, module, obs=None):
    return {'category': 'image', 'kind': 'science', 'filter': filt,
            'observation': obs,
            'src': f'/d/{filt.lower()}/pipeline/jw02045-o001_t001_nircam_'
                   f'clear-{filt.lower()}-{module}_i2d.fits'}


def _cat(filt, module, obs=None):
    return {'category': 'catalog', 'kind': 'catalog_per_filter_vetted',
            'filter': filt, 'observation': obs, 'module': module,
            'src': f'/d/catalogs/{filt.lower()}_{module}_vetted.fits'}


def _by_key(pairs):
    return {key: cat['src'] for key, _img_, cat in pairs}


def test_per_module_images_pair_with_a_merged_catalog(sr):
    """arches: nrca+nrcb mosaics, one merged table -> two comparisons, not zero.

    This is the case a key intersection drops entirely.
    """
    items = [_img('F212N', 'nrca'), _img('F212N', 'nrcb'),
             _cat('F212N', 'merged')]
    pairs, unpaired = sr.same_run_pairs(items)
    assert unpaired == []
    assert _by_key(pairs) == {
        ('F212N', None, 'nrca'): '/d/catalogs/f212n_merged_vetted.fits',
        ('F212N', None, 'nrcb'): '/d/catalogs/f212n_merged_vetted.fits',
    }


def test_intersection_of_the_keys_would_have_compared_nothing(sr):
    """The defect this fallback exists for, stated as the property it violates.

    Pinning the fallback alone would pass just as well against a build that
    compares nothing, so state what the old pairing did: images and catalogs
    share no key, and anything keyed on that intersection is empty.
    """
    items = [_img('F212N', 'nrca'), _img('F212N', 'nrcb'),
             _cat('F212N', 'merged')]
    img_keys = {('F212N', None, 'nrca'), ('F212N', None, 'nrcb')}
    cat_keys = {('F212N', None, 'merged')}
    assert img_keys & cat_keys == set()
    assert len(sr.same_run_pairs(items)[0]) == 2


def test_own_module_catalog_wins_over_the_merged_one(sr):
    """quintuplet: when both ship, each image is measured against its own module."""
    items = [_img('F212N', 'nrca'), _img('F212N', 'nrcb'),
             _cat('F212N', 'nrca'), _cat('F212N', 'nrcb'),
             _cat('F212N', 'merged')]
    pairs, unpaired = sr.same_run_pairs(items)
    assert unpaired == []
    assert _by_key(pairs) == {
        ('F212N', None, 'nrca'): '/d/catalogs/f212n_nrca_vetted.fits',
        ('F212N', None, 'nrcb'): '/d/catalogs/f212n_nrcb_vetted.fits',
    }


def test_a_disjoint_field_never_crosses_the_modules(sr):
    """NRCA's image is never measured against NRCB's catalog.

    The two cover non-overlapping sky, so the tie cannot succeed and the
    failure would be read as an astrometry defect.
    """
    items = [_img('F212N', 'nrca'), _cat('F212N', 'nrcb')]
    pairs, unpaired = sr.same_run_pairs(items)
    assert pairs == []
    assert unpaired == [('F212N', None, 'nrca')]


def test_an_image_with_no_partner_is_reported_not_skipped(sr):
    """A catalog release that ships an image it has no catalog for must say so."""
    items = [_img('F212N', 'merged'), _img('F323N', 'merged'),
             _cat('F212N', 'merged')]
    pairs, unpaired = sr.same_run_pairs(items)
    assert [k[0] for k, _, _ in pairs] == ['F212N']
    assert unpaired == [('F323N', None, 'merged')]


def test_unpaired_images_become_gate_failures(sr):
    """`unpaired` is not advisory -- the caller turns it into a refusal.

    Without this the widened pairing would still let an unmatched image ship.
    """
    items = [_img('F212N', 'merged'), _img('F323N', 'merged'),
             _cat('F212N', 'merged')]
    # Only the unpaired branch is exercised: F212N has a partner and would need
    # real FITS to measure, so drop it and keep the image the gate cannot pair.
    items = [it for it in items if it.get('filter') != 'F212N']
    assert sr.check_image_catalog_match(items) == []   # no catalogs -> nothing owed

    items.append(_cat('F480M', 'merged'))              # now a catalog release
    fails = sr.check_image_catalog_match(items)
    assert [k for k, _off in fails] == [('F323N', None, 'merged')]


def test_images_only_release_owes_no_partners(sr):
    """sgra ships mosaics and no catalogs by design; that is not an unpaired image."""
    items = [_img('F115W', 'nrca'), _img('F212N', 'nrcb')]
    pairs, unpaired = sr.same_run_pairs(items)
    assert (pairs, unpaired) == ([], [])


def test_observation_separates_otherwise_identical_keys(sr):
    """Two observations of one filter each pair with their own catalog."""
    items = [_img('F200W', 'merged', obs='o046'), _img('F200W', 'merged', obs='o050'),
             _cat('F200W', 'merged', obs='o046'), _cat('F200W', 'merged', obs='o050')]
    pairs, _ = sr.same_run_pairs(items)
    assert _by_key(pairs) == {
        ('F200W', 'o046', 'merged'): '/d/catalogs/f200w_merged_vetted.fits',
        ('F200W', 'o050', 'merged'): '/d/catalogs/f200w_merged_vetted.fits',
    }
    assert len(pairs) == 2


def test_merged_fallback_does_not_cross_observations(sr):
    """o046's image never falls back to o050's merged catalog."""
    items = [_img('F200W', 'nrca', obs='o046'), _cat('F200W', 'merged', obs='o050')]
    pairs, unpaired = sr.same_run_pairs(items)
    assert pairs == []
    assert unpaired == [('F200W', 'o046', 'nrca')]


def _miri_img(filt):
    return {'category': 'image', 'kind': 'science', 'filter': filt,
            'observation': None,
            'src': f'/d/images/jw02221-o002_t001_miri_{filt.lower()}_i2d.fits'}


def test_a_missing_miri_catalog_does_not_block_the_nircam_release(sr):
    """brick ships a MIRI F2550W mosaic and no MIRI catalog; sickle ships three.

    NIRCam and MIRI are independent observations -- different detectors,
    different exposures, usually a different program -- so the NIRCam catalogs
    do not owe the MIRI image a partner.  Charging it to them would refuse both
    fields for something the NIRCam release does not depend on.
    """
    items = [_img('F212N', 'merged'), _cat('F212N', 'merged'),
             _miri_img('F2550W')]
    pairs, unpaired = sr.same_run_pairs(items)
    assert [k[0] for k, _, _ in pairs] == ['F212N']
    assert unpaired == []


def test_a_miri_image_IS_owed_a_partner_once_miri_catalogs_ship(sr):
    """The exemption is per-instrument, not a blanket pass for MIRI.

    A release that ships one MIRI catalog and two MIRI mosaics has an
    unmeasured MIRI image, and that is the same defect as an unmeasured NIRCam
    one.
    """
    items = [_miri_img('F770W'), _miri_img('F1500W'),
             {'category': 'catalog', 'kind': 'catalog_per_filter_vetted',
              'filter': 'F770W', 'observation': None, 'module': 'merged',
              'src': '/d/catalogs/f770w_merged_vetted.fits'}]
    pairs, unpaired = sr.same_run_pairs(items)
    assert [k[0] for k, _, _ in pairs] == ['F770W']
    assert unpaired == [('F1500W', None, None)]


def test_the_exemption_runs_both_ways(sr):
    """A release with MIRI catalogs and NIRCam mosaics is NIRCam-images-only.

    The rule is symmetric, and deliberately so: cloudef ships its MIRI bands
    with catalogs while its NIRCam side is still images-only.  What the
    exemption does NOT cover is an instrument that ships SOME catalogs and is
    then short one -- `test_an_image_with_no_partner_is_reported_not_skipped`
    holds that line, which is the case a blanket per-instrument pass would open.
    """
    items = [_img('F212N', 'merged'),
             {'category': 'catalog', 'kind': 'catalog_per_filter_vetted',
              'filter': 'F770W', 'observation': None, 'module': 'merged',
              'src': '/d/catalogs/f770w_merged_vetted.fits'}]
    pairs, unpaired = sr.same_run_pairs(items)
    assert (pairs, unpaired) == ([], [])


# ---- the refusal message ---------------------------------------------------
#
# The gate's return shape and the caller that formats it drifted apart once:
# keys became module-keyed 3-tuples and an unpaired image reports `off=None`,
# while the caller still unpacked `(f, o), v` and formatted with `:.0f`.  Every
# same-run failure then raised inside the f-string instead of reaching the
# REFUSING TO STAGE line.  Nothing exercised the formatter, so the whole suite
# passed over it -- these tests drive it directly.

def test_the_refusal_names_filter_and_module(sr):
    """The operator has to be told WHICH pairing failed and by how much; that
    is the entire purpose of this gate's message."""
    assert sr._same_run_detail(('F212N', None, 'nrca'), 34.0) == 'F212N/nrca: 34 mas'


def test_a_three_tuple_key_does_not_raise(sr):
    """`too many values to unpack (expected 2)` -- the module-keyed shape."""
    for key in [('F212N', None, 'nrca'), ('F200W', 'o046', 'nrcb')]:
        out = sr._same_run_detail(key, 31.0)
        assert 'mas' in out and out.startswith(key[0])


def test_an_unpaired_image_reports_no_measurement_not_zero(sr):
    """`off` is None when no catalog covers the image.  Formatting that with
    `:.0f` raised; formatting it as 0 would report a perfect tie for a
    comparison that never happened."""
    out = sr._same_run_detail(('F323N', None, 'merged'), None)
    assert 'no catalog partner' in out
    assert '0 mas' not in out


def test_merged_is_not_spelled_out_as_a_module(sr):
    """'F212N: 34 mas' reads better than 'F212N/merged: 34 mas' when there is
    only one module in play."""
    assert sr._same_run_detail(('F212N', None, 'merged'), 34.0) == 'F212N: 34 mas'


def test_the_observation_appears_when_there_is_one(sr):
    """Two observations of one filter must be distinguishable in the refusal."""
    assert sr._same_run_detail(('F200W', 'o046', 'merged'), 44.0) == 'F200W/o046: 44 mas'


def test_the_old_two_tuple_shape_still_formats(sr):
    """A caller that has not been updated should format, not raise."""
    assert sr._same_run_detail(('F212N', None), 12.0) == 'F212N: 12 mas'
