"""A gate is only worth its verdicts, so this tests the verdicts.

The check reads a field's reduce spec (target, proposal, observation, filters,
modules) and answers whether that reduce has inputs, before ~20 h of queue
answers it instead.

The failure the tests are shaped around is the FALSE PASS, in three forms:

* counting association files the reduce would not use (the first version read
  113 for sgra/F115W, of which one is the ``image3`` the reduce consumes and the
  rest are the earlier stage's, plus the pipeline's own catalog outputs -- so
  the gate confirmed itself from a previous run's products);
* asking a single-detector instrument which NIRCam module its frames belong to,
  which reported every complete MIRI and NIRISS field as missing both;
* accepting a spec loose enough to be satisfied by something other than what was
  asked for -- an empty filter list, or a wildcard observation whose two modules
  come from two different observations.
"""
import importlib.util
import json
import os
import pathlib

import pytest

REPO = pathlib.Path(__file__).parents[1]
SCRIPT = REPO / 'scripts' / 'reduction' / 'preflight_reduce_inputs.py'


def _load():
    """Load by path, the convention for `scripts/reduction/` in this repo.

    Deliberately not `sys.path.insert`: that leaks the whole directory into the
    rest of the pytest session.
    """
    spec = importlib.util.spec_from_file_location('preflight_reduce_inputs',
                                                  SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PF = _load()


def _field(tmp_path, target, filt, asns=(), cals=()):
    d = tmp_path / target / filt / 'pipeline'
    d.mkdir(parents=True, exist_ok=True)
    for name in asns:
        (d / name).write_text(json.dumps({'products': [{'members': []}]}))
    for name in cals:
        (d / name).write_text('')
    return str(tmp_path)


IMAGE3 = 'jw01939-o001_20260101t000000_image3_00001_asn.json'
CALS_AB = ['jw01939001001_02101_00001_nrca1_cal.fits',
           'jw01939001001_02101_00001_nrcb1_cal.fits']


# ---------------------------------------------------------------------------
# the association file the reduce actually consumes
# ---------------------------------------------------------------------------

def test_the_glob_is_the_one_the_reduce_itself_uses():
    """Pinned to the reduce, because a looser pattern is a false pass.

    Both reduce entry points build the same literal.  If either changes, this
    check would keep passing on files the reduce no longer reads.
    """
    literal = "jw0{proposal_id}-o{field}*_image3_*0[0-9][0-9]_asn.json"
    ours = PF.ASN_GLOB.replace('{proposal}', '{proposal_id}') \
                      .replace('{obsid}', '{field}')
    assert ours == literal, (
        f'preflight globs {ours!r}; the reduce globs {literal!r}')
    for name in ('PipelineRerunNIRCAM-LONG.py', 'PipelineMIRI.py'):
        src = (REPO / 'jwst_gc_pipeline' / 'reduction' / name).read_text()
        assert literal in src, f'{name} no longer builds {literal!r}'


def test_an_image2_association_is_not_counted_as_an_input(tmp_path):
    """The directory holds far more image2 associations than image3 ones, so
    counting everything made a field with no usable input read as OK."""
    root = _field(tmp_path, 'sgra', 'F115W',
                  asns=['jw01939-o001_20260101t000000_image2_00001_asn.json'],
                  cals=CALS_AB)
    rows = PF.check(root, 'sgra', '1939', '001', ['F115W'], ['nrca', 'nrcb'])
    assert not rows[0].ok
    assert 'image3' in rows[0].why


def test_the_pipelines_own_catalog_output_is_not_counted_as_an_input(tmp_path):
    """`..._mergedcat_model_asn.json` is a PRODUCT of a previous run.  Counting
    it makes the gate confirm itself."""
    root = _field(tmp_path, 'brick', 'F115W',
                  asns=['jw01182-o004_m3_daophot_basic_mergedcat_model_asn.json'],
                  cals=['jw01182004001_02101_00001_nrca1_cal.fits',
                        'jw01182004001_02101_00001_nrcb1_cal.fits'])
    rows = PF.check(root, 'brick', '1182', '004', ['F115W'], ['nrca', 'nrcb'])
    assert not rows[0].ok


def test_a_real_image3_association_passes(tmp_path):
    root = _field(tmp_path, 'sgra', 'F115W', asns=[IMAGE3], cals=CALS_AB)
    rows = PF.check(root, 'sgra', '1939', '001', ['F115W'], ['nrca', 'nrcb'])
    assert rows[0].ok, rows[0].why
    assert rows[0].n_asn == 1


# ---------------------------------------------------------------------------
# modules
# ---------------------------------------------------------------------------

def test_a_long_wavelength_frame_satisfies_its_module_family(tmp_path):
    """`nrcalong` IS module A's long-wavelength detector.  Matching the token
    literally reported every long-wavelength filter as missing both modules."""
    root = _field(tmp_path, 'brick', 'F405N', asns=[
        'jw01182-o004_20260101t000000_image3_00001_asn.json'],
        cals=['jw01182004001_02101_00001_nrcalong_cal.fits',
              'jw01182004001_02101_00001_nrcblong_cal.fits'])
    rows = PF.check(root, 'brick', '1182', '004', ['F405N'], ['nrca', 'nrcb'])
    assert rows[0].ok, rows[0].why


def test_the_module_SPEC_is_normalized_too(tmp_path):
    """The long-wavelength submitters spell it `nrcalong,nrcblong`.  Normalizing
    only the detector compared `nrcalong` against the family `nrca`."""
    root = _field(tmp_path, 'brick', 'F405N', asns=[
        'jw01182-o004_20260101t000000_image3_00001_asn.json'],
        cals=['jw01182004001_02101_00001_nrcalong_cal.fits',
              'jw01182004001_02101_00001_nrcblong_cal.fits'])
    rows = PF.check(root, 'brick', '1182', '004', ['F405N'],
                    ['nrcalong', 'nrcblong'])
    assert rows[0].ok, rows[0].why


def test_a_module_the_observation_does_not_have_is_reported(tmp_path):
    """gc2211 observation 050 is module B only, which is why its driver script
    overrides the module list."""
    root = _field(tmp_path, 'gc2211', 'F200W', asns=[
        'jw02211-o050_20260101t000000_image3_00001_asn.json'],
        cals=['jw02211050001_02101_00001_nrcb1_cal.fits'])
    rows = PF.check(root, 'gc2211', '2211', '050', ['F200W'], ['nrca', 'nrcb'])
    assert not rows[0].ok
    assert rows[0].missing == ['nrca']


def test_asking_for_the_combined_product_asks_for_BOTH_modules(tmp_path):
    """`merged` is built from the two module reductions.  Dropping it from the
    module list turned the module check off entirely for `--modules merged`."""
    root = _field(tmp_path, 'gc2211', 'F200W', asns=[
        'jw02211-o050_20260101t000000_image3_00001_asn.json'],
        cals=['jw02211050001_02101_00001_nrcb1_cal.fits'])
    rows = PF.check(root, 'gc2211', '2211', '050', ['F200W'], ['merged'])
    assert not rows[0].ok, 'a one-module observation cannot produce a merged product'
    assert rows[0].missing == ['nrca']


# ---------------------------------------------------------------------------
# instruments that have no modules
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('detector', ['mirimage', 'nis'])
def test_a_single_detector_instrument_is_not_asked_which_module(tmp_path,
                                                                detector):
    """MIRI has one imager and NIRISS one detector.  Requiring NIRCam module
    families of them reported every complete field as missing both."""
    root = _field(tmp_path, 'cloudc', 'F770W', asns=[
        'jw02526-o021_20260101t000000_image3_00001_asn.json'],
        cals=[f'jw02526021001_02101_00001_{detector}_cal.fits'])
    rows = PF.check(root, 'cloudc', '2526', '021', ['F770W'], ['nrca', 'nrcb'])
    assert rows[0].ok, rows[0].why
    assert rows[0].families == [detector]


# ---------------------------------------------------------------------------
# specs loose enough to be satisfied by the wrong thing
# ---------------------------------------------------------------------------

def test_no_filters_is_refused_rather_than_passing(tmp_path):
    with pytest.raises(ValueError):
        PF.check(str(tmp_path), 'sgra', '1939', '001', [], ['nrca'])


@pytest.mark.parametrize('bad', ['*', '0[0-9][0-9]', 'o001', ''])
def test_a_wildcard_observation_is_refused(bad):
    """It pools every observation, so two DIFFERENT observations can satisfy
    the two modules between them and the field reads as OK."""
    with pytest.raises(ValueError):
        PF.normalize_obsid(bad)


@pytest.mark.parametrize('given', ['1', '01', '001'])
def test_an_unpadded_observation_is_accepted_not_silently_missed(given):
    assert PF.normalize_obsid(given) == '001'


# ---------------------------------------------------------------------------
# the registry check, which needs no filesystem at all
# ---------------------------------------------------------------------------

def test_the_registry_catches_the_sgra_spec_with_no_disk_access():
    """The case this was written for: Sgr A* driven as proposal 4147 (Sgr C's)
    for the whole campaign."""
    ok, msg = PF.registry_verdict('sgra', '4147', '001')
    assert not ok
    assert '4147' in msg


def test_the_registry_accepts_the_corrected_spec():
    ok, msg = PF.registry_verdict('sgra', '1939', '001')
    assert ok, msg


def test_the_registry_catches_a_right_proposal_against_the_wrong_target():
    """The case the on-disk scan cannot see: 4147/012 is real data, but it is
    Sgr C's, and a scan of the sgra tree would just report it absent."""
    ok, msg = PF.registry_verdict('sgra', '4147', '012')
    assert not ok
    assert 'sgrc' in msg


def test_an_instrument_suffixed_target_is_matched_to_its_registry_name():
    """NIRISS data lives under `<target>/niriss/`, so the directory a caller
    names is not the registry's target name."""
    ok, msg = PF.registry_verdict('sgrc/niriss', '4147', '012',
                                  instrument='niriss')
    assert ok, msg


# ---------------------------------------------------------------------------
# exit status
# ---------------------------------------------------------------------------

def test_exit_status_is_nonzero_when_something_is_missing(tmp_path, capsys):
    _field(tmp_path, 'sgra', 'F115W', asns=[], cals=[])
    rc = PF.main(['--target', 'sgra', '--proposal', '4147', '--obsid', '001',
                  '--filters', 'F115W', '--root', str(tmp_path)])
    assert rc == 1
    assert 'MISMATCH' in capsys.readouterr().out


def test_exit_status_is_zero_when_everything_is_there(tmp_path, capsys):
    _field(tmp_path, 'sgra', 'F115W', asns=[IMAGE3], cals=CALS_AB)
    rc = PF.main(['--target', 'sgra', '--proposal', '1939', '--obsid', '001',
                  '--filters', 'F115W', '--root', str(tmp_path),
                  '--modules', 'nrca,nrcb'])
    assert rc == 0, capsys.readouterr().out


def test_a_deliberately_unregistered_field_can_skip_the_registry_check(
        tmp_path, capsys):
    _field(tmp_path, 'sgra', 'F115W', asns=[IMAGE3], cals=CALS_AB)
    rc = PF.main(['--target', 'sgra', '--proposal', '1939', '--obsid', '001',
                  '--filters', 'F115W', '--root', str(tmp_path),
                  '--modules', 'nrca,nrcb', '--skip-registry'])
    assert rc == 0
    assert 'registry' not in capsys.readouterr().out
