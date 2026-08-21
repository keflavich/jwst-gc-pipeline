"""An excluded exposure must not reach imaging OR analysis.

arches 2045/001 exposure 4 had tracking errors, so it is excluded outright
rather than worked around.  Before the exclusion existed it was carried by a
26 mas correction floor -- a floor sized to one bad exposure, which also raises
the smallest real offset the checkpoint can act on for every other exposure in
the field.  That is the shape of workaround this module exists to remove.
"""
import json
import os

import pytest

from jwst_gc_pipeline.reduction import exposure_exclusions as ex


ARCHES_EXP4 = 'jw02045001001_02101_00004'


# --------------------------------------------------------------------------
# what the stem matcher does and does not claim
# --------------------------------------------------------------------------

@pytest.mark.parametrize('name', [
    f'{ARCHES_EXP4}_nrca1_destreak_o001_crf.fits',
    f'{ARCHES_EXP4}_nrcalong_cal.fits',
    f'{ARCHES_EXP4}_nrcb3_uncal.fits',
    f'{ARCHES_EXP4}_nrca4_destreak_o001_crf_m12_satstar_model.fits',
    f'/orange/adamginsburg/jwst/arches/F212N/pipeline/{ARCHES_EXP4}_nrcb1_rate.fits',
])
def test_every_product_of_the_exposure_is_excluded(name):
    """One rule covers `_uncal` through `_crf` and the satstar products, because
    they all carry the exposure's stem."""
    assert ex.is_excluded(name) is True
    assert ex.exclusion_reason(name)


@pytest.mark.parametrize('name', [
    'jw02045001001_02101_00003_nrca1_destreak_o001_crf.fits',   # exposure 3
    'jw02045001002_02101_00004_nrca1_destreak_o001_crf.fits',   # visit 002
    'jw02045001001_02102_00004_nrca1_destreak_o001_crf.fits',   # vgroup 02102
    'jw02221002001_02101_00004_nrca1_destreak_o002_crf.fits',   # cloudc
])
def test_its_neighbours_are_not(name):
    assert ex.is_excluded(name) is False
    assert ex.exclusion_reason(name) is None


@pytest.mark.parametrize('name', [
    'jw02045-o001_t001_nircam_clear-f212n-nrca_i2d.fits',
    'jw02045-o001_20260701t010704_image3_00001_asn.json',
    'gaia_virac2_refcat_epoch2023.21.fits',
])
def test_per_observation_products_yield_no_stem(name):
    """A mosaic or association is not per-exposure, so an exposure exclusion
    cannot be expressed against it and must not be guessed at."""
    assert ex.exposure_stem(name) is None
    assert ex.is_excluded(name) is False


# --------------------------------------------------------------------------
# the catalog spelling, which the stem cannot reach
# --------------------------------------------------------------------------

def test_the_catalog_naming_form_is_matched_too():
    """Per-exposure CATALOGS are named
    ``f212n_nrca1_visit001_vgroup02101_exp00004_m2_daophot_basic.fits`` --
    visit / vgroup / exposure / detector, but no proposal or observation -- so
    the frame stem cannot match them.  It has to: arches exposure 4 had catalogs
    through m7 written before it was excluded, and a merge globbing
    ``*_exp*_m*_daophot_basic.fits`` would ingest them.
    """
    assert ex.is_excluded_catalog_name(
        'f212n_nrca1_visit001_vgroup02101_exp00004_m2_daophot_basic.fits')
    assert ex.is_excluded_catalog_name(
        'f323n_nrcblong_visit001_vgroup02101_exp00004_resbgsub_m7_daophot_basic.fits')
    # neighbours
    assert not ex.is_excluded_catalog_name(
        'f212n_nrca1_visit001_vgroup02101_exp00003_m2_daophot_basic.fits')
    assert not ex.is_excluded_catalog_name(
        'f212n_nrca1_visit002_vgroup02101_exp00004_m2_daophot_basic.fits')


def test_is_excluded_any_covers_both_spellings():
    assert ex.is_excluded_any(f'{ARCHES_EXP4}_nrca1_destreak_o001_crf.fits')
    assert ex.is_excluded_any(
        'f212n_nrca1_visit001_vgroup02101_exp00004_m2_daophot_basic.fits')
    assert not ex.is_excluded_any('f212n_nrca1_visit001_vgroup02101_exp00005_m2_daophot_basic.fits')


# --------------------------------------------------------------------------
# the drop is announced
# --------------------------------------------------------------------------

def test_drop_excluded_announces_what_it_dropped(capsys):
    """Silence is the failure mode here.  This project hard-crashes on a dropped
    exposure precisely because a frame that vanishes quietly is
    indistinguishable from one that was never reduced; the one sanctioned drop
    has to say so every time."""
    kept, dropped = ex.drop_excluded(
        [f'{ARCHES_EXP4}_nrca1_destreak_o001_crf.fits',
         'jw02045001001_02101_00003_nrca1_destreak_o001_crf.fits'],
        label='test')
    assert len(kept) == 1 and len(dropped) == 1
    out = capsys.readouterr().out
    assert ARCHES_EXP4 in out
    assert 'excluded' in out


def test_dropping_nothing_says_nothing(capsys):
    kept, dropped = ex.drop_excluded(
        ['jw02045001001_02101_00003_nrca1_destreak_o001_crf.fits'])
    assert len(kept) == 1 and not dropped
    assert capsys.readouterr().out == ''


# --------------------------------------------------------------------------
# both halves of the pipeline consult it
# --------------------------------------------------------------------------

def test_cataloging_filters_at_its_single_enumeration_point():
    """`get_filenames` is where cataloging turns a (filter, module, visit) into
    frames; `frame_cache`, the per-phase `frame_args`, the fan-out shards and
    every merge read through it, so filtering once covers them all."""
    import inspect

    from jwst_gc_pipeline.photometry import crowdsource_catalogs_long as ccl
    src = inspect.getsource(ccl.get_filenames)
    assert 'drop_excluded' in src, (
        'get_filenames no longer filters excluded exposures; every stage '
        'downstream inherits its frame list from here')


def test_the_association_builder_filters_too():
    """"All imaging AND analysis": an exposure dropped from cataloging but still
    coadded into the mosaic leaves the catalog and the image disagreeing about
    what was observed."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(ex.__file__)))
    path = os.path.join(here, 'reduction', 'PipelineRerunNIRCAM-LONG.py')
    src = open(path).read()
    assert '_is_excluded_exposure' in src, (
        'the Image3 association builder no longer drops excluded exposures')


# --------------------------------------------------------------------------
# the entry itself
# --------------------------------------------------------------------------

def test_the_exclusion_carries_a_reason_naming_the_data_defect():
    """This is not a quality cut.  Every entry is a hand-made decision about the
    observation, so every entry has to say what is wrong with the DATA -- a
    stage that merely dislikes an exposure has other, recorded ways to say so."""
    assert ARCHES_EXP4 in ex.EXCLUDED_EXPOSURES
    why = ex.EXCLUDED_EXPOSURES[ARCHES_EXP4]
    assert 'tracking' in why.lower()
    assert len(why) > 200, 'a one-line reason is not a decision record'


def test_the_quarantine_receipt_matches_the_entry():
    """The on-disk products were renamed when the exposure was excluded; the
    receipt names the same exposure, so the two cannot drift apart silently."""
    base = '/orange/adamginsburg/jwst/arches'
    receipts = sorted(glob_receipts(base))
    if not receipts:
        pytest.skip('arches tree not present (CI)')
    rec = json.load(open(receipts[-1]))
    assert rec['exposure'] == ARCHES_EXP4
    assert rec['n'] > 0


def glob_receipts(base):
    import glob
    return glob.glob(os.path.join(base, '_EXCLUDED_exposure4_*.json'))
