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

def test_get_filenames_actually_returns_short(tmp_path, capsys):
    """BEHAVIOURAL, not a source grep.

    A grep for `drop_excluded` in the source passes even if the call is moved
    somewhere it cannot affect the return value.  This builds a tree holding one
    excluded exposure and three of its neighbours, and asserts what comes back.
    """
    from jwst_gc_pipeline.photometry.crowdsource_catalogs_long import get_filenames

    d = tmp_path / 'F212N' / 'pipeline'
    d.mkdir(parents=True)
    names = [f'jw02045001001_02101_{n:05d}_nrca1_destreak_o001_crf.fits'
             for n in (3, 4, 5, 6)]
    for n in names:
        (d / n).write_bytes(b'')

    got = get_filenames(str(tmp_path), 'F212N', '2045', '001',
                        each_suffix='destreak_o001_crf', module='nrca1',
                        visitid='001')
    base = sorted(os.path.basename(g) for g in got)
    assert len(base) == 3, base
    assert not any('_00004_' in b for b in base), base
    assert 'excluded' in capsys.readouterr().out


def test_the_association_filter_drops_the_member_and_does_not_rewrite_mast(tmp_path):
    """The imaging half, driven.

    Also pins that MAST's association is left ALONE: the filtered members go to
    a sibling file, so re-running is idempotent and the original still records
    what was observed.
    """
    import json


    mod = _load_pipeline_module()
    asn = tmp_path / 'jw02045-o001_20260701t010704_image3_00001_asn.json'
    members = [{'expname': f'jw02045001001_02101_{n:05d}_nrca1_cal.fits',
                'exptype': 'science'} for n in (3, 4, 5)]
    data = {'products': [{'name': 'p', 'members': members}]}
    asn.write_text(json.dumps(data))

    kept, path = mod._drop_excluded_asn_members(str(asn), data, members)
    assert len(kept) == 2
    assert not any('_00004_' in m['expname'] for m in kept)
    assert path != str(asn), 'must not hand back the MAST association'
    assert path.endswith('_exclfiltered_asn.json')
    # the original is untouched
    assert len(json.loads(asn.read_text())['products'][0]['members']) == 3
    # and the caller's dict was not mutated
    assert len(data['products'][0]['members']) == 3


def test_the_association_filter_is_a_no_op_when_nothing_is_excluded(tmp_path):
    """The common case writes nothing at all and returns the original path."""
    import json

    mod = _load_pipeline_module()
    asn = tmp_path / 'jw02045-o001_20260701t010704_image3_00002_asn.json'
    members = [{'expname': f'jw02045001001_02101_{n:05d}_nrca1_cal.fits',
                'exptype': 'science'} for n in (3, 5, 6)]
    data = {'products': [{'name': 'p', 'members': members}]}
    asn.write_text(json.dumps(data))

    kept, path = mod._drop_excluded_asn_members(str(asn), data, members)
    assert kept == members and path == str(asn)
    assert not list(tmp_path.glob('*_exclfiltered_asn.json'))


def _load_pipeline_module():
    """`PipelineRerunNIRCAM-LONG.py` has a hyphen, so it cannot be imported by
    name."""
    import importlib.util

    here = os.path.dirname(os.path.dirname(os.path.abspath(ex.__file__)))
    path = os.path.join(here, 'reduction', 'PipelineRerunNIRCAM-LONG.py')
    spec = importlib.util.spec_from_file_location('_pipeline_rerun', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod





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
