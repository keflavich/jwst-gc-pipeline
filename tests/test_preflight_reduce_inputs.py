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
import subprocess
import sys

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


def _members(*expnames):
    return {'products': [{'members': [{'expname': n, 'exptype': 'science'}
                                      for n in expnames]}]}


#: The default association members: one exposure per NIRCam module.
#:
#: An earlier version of this helper wrote `{'members': []}` for every fixture,
#: and a test asserted that returns OK.  That pinned the defect as correct: the
#: reduce raises on an association with no members, and eight of nine adversarial
#: trees passed this check and then failed the reduce.
NRCA_NRCB = ('jw01939001001_02101_00001_nrca1_cal.fits',
             'jw01939001001_02101_00001_nrcb1_cal.fits')


def _field(tmp_path, target, filt, asns=(), cals=(), members=NRCA_NRCB):
    d = tmp_path / target / filt / 'pipeline'
    d.mkdir(parents=True, exist_ok=True)
    for name in asns:
        (d / name).write_text(json.dumps(_members(*members)))
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
    literal = "{jw_prefix(proposal_id)}-o{field}*_image3_*0[0-9][0-9]_asn.json"
    ours = PF.ASN_GLOB.replace('{jw}', '{jw_prefix(proposal_id)}') \
                      .replace('{obsid}', '{field}')
    assert ours == literal, (
        f'preflight globs {ours!r}; the reduce globs {literal!r}')
    for name in ('PipelineRerunNIRCAM-LONG.py', 'PipelineMIRI.py'):
        src = (REPO / 'jwst_gc_pipeline' / 'reduction' / name).read_text()
        assert literal in src, f'{name} no longer builds {literal!r}'


def test_the_inlined_prefix_agrees_with_the_package_helper():
    """This script inlines the five-digit pad to stay runnable with no package
    on the path (see `jw_prefix`'s docstring).  Two copies can drift, so pin
    them equal over both digit widths and the sub-1000 case."""
    from jwst_gc_pipeline.mast_names import jw_prefix as canonical
    for proposal in ('2221', 2221, '02221', '1182', '10678', 10678, '12587',
                     '618', 1, 99999):
        assert PF.jw_prefix(proposal) == canonical(proposal), proposal
    assert PF.jw_prefix('10678') == 'jw10678'
    assert PF.jw_prefix('2221') == 'jw02221'


@pytest.mark.parametrize('bad', ['brick', '', None, -1, '-2221', 123456, 0,
                                 '2_221', ' 2221 ', '+2221'])
def test_the_inlined_prefix_refuses_what_the_helper_refuses(bad):
    with pytest.raises(ValueError):
        PF.jw_prefix(bad)


def test_the_script_runs_with_no_package_on_the_path():
    """The gate answers in ten seconds a question the reduce answers in 20 h of
    queue, so it must run wherever it is invoked from.  Every functional path,
    `--skip-registry` included, stays off `jwst_gc_pipeline`: an editable
    install pointing at a different checkout, or none at all, would otherwise
    take the gate down.  Measured on the real script, from a directory that is
    not the repo, with PYTHONPATH cleared."""
    env = {k: v for k, v in os.environ.items() if k != 'PYTHONPATH'}
    probe = (
        "import importlib.util, sys, json\n"
        f"spec = importlib.util.spec_from_file_location('pf', {str(SCRIPT)!r})\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mod)\n"
        "mod.check('/nonexistent-root', 'brick', '10678', '001', ['F212N'],\n"
        "          ['nrca'])\n"
        "print(json.dumps(sorted({m.split('.')[0] for m in sys.modules}\n"
        "                        & {'jwst_gc_pipeline', 'numpy', 'astropy',\n"
        "                           'jwst'})))\n"
    )
    out = subprocess.run([sys.executable, '-c', probe], cwd=os.sep,
                         env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout.strip().splitlines()[-1]) == [], out.stdout


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


def test_a_five_digit_proposals_products_are_found(tmp_path):
    """Issue #414 at this call site, pinned on the VALUE the template is
    filled with.

    ``test_the_glob_is_the_one_the_reduce_itself_uses`` pins ``ASN_GLOB``'s
    text alone, so it stays green when the ``{jw}`` field carries the old
    4-digit-only spelling.  MAST writes ``jw10678-o001...`` for the GC
    Treasury program; the old spelling globs ``jw010678-o001...`` and reads a
    complete field as having no inputs at all -- 20 h of queue answering a
    question this gate exists to answer in ten seconds.
    """
    members = ('jw10678001001_02101_00001_nrca1_cal.fits',
               'jw10678001001_02101_00001_nrcb1_cal.fits')
    root = _field(tmp_path, 'sgra', 'F212N',
                  asns=['jw10678-o001_20260816t000000_image3_00001_asn.json'],
                  cals=list(members), members=members)
    rows = PF.check(root, 'sgra', '10678', '001', ['F212N'], ['nrca', 'nrcb'])
    assert rows[0].ok, rows[0].why
    assert rows[0].n_asn == 1 and rows[0].n_cal == 2


def test_a_five_digit_proposal_does_not_match_the_over_padded_name(tmp_path):
    """The converse: products written under the WRONG prefix are not accepted
    as this proposal's inputs, so a run that fabricated ``jw010678`` names
    cannot make the gate green."""
    members = ('jw010678001001_02101_00001_nrca1_cal.fits',
               'jw010678001001_02101_00001_nrcb1_cal.fits')
    root = _field(tmp_path, 'sgra', 'F212N',
                  asns=['jw010678-o001_20260816t000000_image3_00001_asn.json'],
                  cals=list(members), members=members)
    rows = PF.check(root, 'sgra', '10678', '001', ['F212N'], ['nrca', 'nrcb'])
    assert not rows[0].ok
    assert 'jw10678-o001' in rows[0].why, rows[0].why


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
    """A module with no members in the association is reported missing.

    brick 1182/004 is not in ``MODULES_BY_PROPOSAL_FIELD_FILTER``, so the
    preflight expects both modules and says so when one has no members.  The
    registered case is the test below: once the reduce's own policy declares an
    observation single-module, asking it for the other module stops being a
    finding.
    """
    root = _field(tmp_path, 'brick', 'F405N', asns=[
        'jw01182-o004_20260101t000000_image3_00001_asn.json'],
        cals=['jw01182004001_02101_00001_nrcalong_cal.fits'],
        members=('jw01182004001_02101_00001_nrcalong_cal.fits',))
    rows = PF.check(root, 'brick', '1182', '004', ['F405N'],
                    ['nrcalong', 'nrcblong'])
    assert not rows[0].ok
    assert rows[0].missing == ['nrcb']


def test_a_registered_single_module_observation_is_not_a_false_alarm(tmp_path):
    """gc2211 observation 050 is module B only, and the reduce now knows it.

    Before the registry entry the campaign submitted module A for it and the
    reduce failed those tasks (#408, #436).  With ``2211/050`` declared NRCB
    in ``MODULES_BY_PROPOSAL_FIELD_FILTER``, asking for both modules narrows to
    the one the observation has, and module A is no longer reported missing --
    a false alarm rather than a finding.
    """
    root = _field(tmp_path, 'gc2211', 'F200W', asns=[
        'jw02211-o050_20260101t000000_image3_00001_asn.json'],
        cals=['jw02211050001_02101_00001_nrcb1_cal.fits'],
        members=('jw02211050001_02101_00001_nrcb1_cal.fits',))
    rows = PF.check(root, 'gc2211', '2211', '050', ['F200W'], ['nrca', 'nrcb'])
    assert rows[0].ok, rows[0].why
    assert rows[0].missing == []


def test_asking_for_the_combined_product_asks_for_BOTH_modules(tmp_path):
    """`merged` is built from the two module reductions.  Dropping it from the
    module list turned the module check off entirely for `--modules merged`."""
    root = _field(tmp_path, 'gc2211', 'F200W', asns=[
        'jw02211-o050_20260101t000000_image3_00001_asn.json'],
        cals=['jw02211050001_02101_00001_nrcb1_cal.fits'],
        members=('jw02211050001_02101_00001_nrcb1_cal.fits',))
    rows = PF.check(root, 'gc2211', '2211', '050', ['F200W'], ['merged'])
    assert not rows[0].ok, 'a one-module observation cannot produce a merged product'
    # the reduce's policy allows only nrcb here, so `merged` -- which asks for
    # both -- is refused outright rather than narrowed.  The row carries the
    # reduce's own refusal, which names what it would have raised.
    assert rows[0].missing == ['nrca', 'nrcb']
    assert 'No requested modules are allowed' in rows[0].why


# ---------------------------------------------------------------------------
# instruments that have no modules
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('detector', ['mirimage', 'nis'])
def test_a_single_detector_instrument_is_not_asked_which_module(tmp_path,
                                                                detector):
    """MIRI has one imager and NIRISS one detector.  Requiring NIRCam module
    families of them reported every complete field as missing both."""
    instrument = 'miri' if detector == 'mirimage' else 'niriss'
    frame = f'jw02526021001_02101_00001_{detector}_cal.fits'
    root = _field(tmp_path, 'cloudc', 'F770W', asns=[
        'jw02526-o021_20260101t000000_image3_00001_asn.json'],
        cals=[frame], members=(frame,))
    rows = PF.check(root, 'cloudc', '2526', '021', ['F770W'], ['nrca', 'nrcb'],
                    instrument=instrument)
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


# ---------------------------------------------------------------------------
# Trees that break the reduce.  Eight of these nine used to report OK.
# ---------------------------------------------------------------------------

ASN = 'jw01939-o001_20260101t000000_image3_00001_asn.json'


def _spec(root, **kw):
    return PF.check(root, 'sgra', '1939', '001', ['F115W'],
                    kw.pop('modules', ['nrca', 'nrcb']), **kw)


def test_an_association_with_no_members_is_not_usable_input(tmp_path):
    """The reduce raises `Did not find any NIRCam asn files`.  The fixtures in
    this file used to write exactly this shape for every test."""
    root = _field(tmp_path, 'sgra', 'F115W', asns=[ASN], cals=CALS_AB,
                  members=())
    assert not _spec(root)[0].ok


@pytest.mark.parametrize('foreign,label', [
    (('jw01939001001_02101_00001_nis_cal.fits',), 'NIRISS'),
    (('jw01939001001_02101_00001_mirimage_cal.fits',), 'MIRI'),
])
def test_another_instruments_association_is_not_this_reduces_input(
        tmp_path, foreign, label):
    """One observation produces associations for several instruments under the
    same proposal and observation number, and the reduce keeps only its own.
    sgrc/F480M holds a NIRCam one beside a NIRISS one right now."""
    root = _field(tmp_path, 'sgra', 'F115W', asns=[ASN], cals=CALS_AB,
                  members=foreign)
    assert not _spec(root)[0].ok, f'{label} association accepted as NIRCam input'


def test_a_module_on_disk_but_absent_from_the_association_is_reported(tmp_path):
    """The reduce narrows the association to one module's members and raises
    `No {module} members found`.  A directory listing cannot see this: the
    frames are there, and the association does not use them."""
    root = _field(tmp_path, 'sgra', 'F115W', asns=[ASN], cals=CALS_AB,
                  members=('jw01939001001_02101_00001_nrcb1_cal.fits',))
    row = _spec(root)[0]
    assert not row.ok
    assert row.missing == ['nrca']


def test_a_malformed_association_stops_rather_than_passing(tmp_path):
    root = _field(tmp_path, 'sgra', 'F115W', asns=[], cals=CALS_AB)
    (pathlib.Path(root) / 'sgra' / 'F115W' / 'pipeline' / ASN).write_text('{')
    row = _spec(root)[0]
    assert not row.ok
    assert 'cannot parse' in row.why


def test_an_association_with_no_products_stops_rather_than_passing(tmp_path):
    root = _field(tmp_path, 'sgra', 'F115W', asns=[], cals=CALS_AB)
    (pathlib.Path(root) / 'sgra' / 'F115W' / 'pipeline' / ASN).write_text(
        json.dumps({'asn_type': 'image3'}))
    assert not _spec(root)[0].ok


def test_an_unreadable_association_stops_rather_than_passing(tmp_path):
    """Same problem as a malformed one from this side, and the reduce's own
    handler does not catch OSError -- so this reads as OK and then crashes it."""
    root = _field(tmp_path, 'sgra', 'F115W', asns=[ASN], cals=CALS_AB)
    path = pathlib.Path(root) / 'sgra' / 'F115W' / 'pipeline' / ASN
    path.chmod(0o000)
    try:
        if os.access(path, os.R_OK):
            pytest.skip('running as a user that can read a 0o000 file')
        row = _spec(root)[0]
        assert not row.ok
        assert 'cannot parse' in row.why
    finally:
        path.chmod(0o644)


def test_no_cal_frames_is_reported_even_with_a_good_association(tmp_path):
    root = _field(tmp_path, 'sgra', 'F115W', asns=[ASN], cals=[])
    row = _spec(root)[0]
    assert not row.ok
    assert '_cal' in row.why


# ---------------------------------------------------------------------------
# The module spec itself
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('spec', ['nrcb nrca', 'nrca, nrcb', 'nrca,nrcb'])
def test_a_module_spec_is_split_on_commas_OR_spaces(tmp_path, spec):
    """`--filters` is space-separated and `--modules` was comma-only, so mixing
    them is the natural mistake.  `--modules "nrcb nrca"` parsed as ONE token,
    truncated to `nrcb`, and exited 0 on a field with no module A."""
    root = _field(tmp_path, 'sgra', 'F115W', asns=[ASN], cals=CALS_AB,
                  members=('jw01939001001_02101_00001_nrcb1_cal.fits',))
    rc = PF.main(['--target', 'sgra', '--proposal', '1939', '--obsid', '001',
                  '--filters', 'F115W', '--root', root, '--modules', spec,
                  '--skip-registry'])
    assert rc == 1, f'--modules {spec!r} did not notice module A is absent'


@pytest.mark.parametrize('bad', ['nrcc', 'nrca,nrcz', 'module-a', 'nrc'])
def test_an_unknown_module_token_is_refused_not_truncated(tmp_path, bad):
    """Truncated to four characters, a typo became a module name and was
    reported as genuinely missing from the data."""
    root = _field(tmp_path, 'sgra', 'F115W', asns=[ASN], cals=CALS_AB)
    with pytest.raises(SystemExit) as exc:
        PF.main(['--target', 'sgra', '--proposal', '1939', '--obsid', '001',
                 '--filters', 'F115W', '--root', root, '--modules', bad,
                 '--skip-registry'])
    assert exc.value.code == 2


def test_a_nircam_spec_pointed_at_another_instruments_directory_is_not_OK(
        tmp_path):
    """The single-detector escape used to key off the DATA, so any directory
    whose frames were all NIRISS satisfied a NIRCam module request."""
    frame = 'jw01939001001_02101_00001_nis_cal.fits'
    root = _field(tmp_path, 'sgra', 'F115W', asns=[ASN], cals=[frame],
                  members=(frame,))
    row = PF.check(root, 'sgra', '1939', '001', ['F115W'], ['nrca', 'nrcb'],
                   instrument='nircam')[0]
    assert not row.ok, 'a NIRCam spec was satisfied by NIRISS frames'


def test_an_observation_the_reduce_restricts_to_one_module_is_not_a_failure(
        tmp_path):
    """sickle 3958/007 is declared module-B-only in the reduce's own policy, so
    asking it for module A is a false alarm, not a finding -- and the README
    documented exactly that invocation."""
    root = _field(tmp_path, 'sickle', 'F187N', asns=[
        'jw03958-o007_20260101t000000_image3_00001_asn.json'],
        cals=['jw03958007001_02101_00001_nrcb1_cal.fits'],
        members=('jw03958007001_02101_00001_nrcb1_cal.fits',))
    row = PF.check(root, 'sickle', '3958', '007', ['F187N'],
                   ['nrca', 'nrcb', 'merged'])[0]
    assert row.ok, row.why


def test_the_policy_is_read_from_the_reduce_rather_than_restated(tmp_path):
    """Parsed, not imported: importing that module pulls in the whole JWST
    stack, and this check exists to run in seconds before a submission."""
    policy = PF.reduce_module_policy()
    assert policy.get('3958', {}).get('007'), (
        'the reduce no longer declares the sickle module policy where this '
        'reads it; re-point reduce_module_policy or drop the narrowing')
    # a field with no entry is unrestricted
    assert PF.allowed_modules('1939', '001', 'F115W', {'nrca', 'nrcb'},
                              policy=policy) == {'nrca', 'nrcb'}
    assert PF.allowed_modules('3958', '007', 'F187N', {'nrca', 'nrcb'},
                              policy=policy) == {'nrcb'}


# ---------------------------------------------------------------------------
# A module the reduce is not allowed to run is a failure, not a narrowing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('modules', ['nrca', 'merged', 'nrca,merged'])
def test_a_module_the_policy_excludes_is_a_FAILURE_not_a_narrowing(tmp_path,
                                                                   modules):
    """The reduce raises `No requested modules are allowed` before doing any
    work.  Narrowing to an empty set and reporting OK is the opposite verdict.

    This fired on the one field the narrowing exists for: sickle 3958/007 is
    module B only, and both `--modules nrca` and `--modules merged` read OK
    while making the reduce stop.  `merged` needs its own check because it is
    expanded to both families before the policy is consulted, which made the
    intersection non-empty and hid the failure.
    """
    root = _field(tmp_path, 'sickle', 'F187N', asns=[
        'jw03958-o007_20260101t000000_image3_00001_asn.json'],
        cals=['jw03958007001_02101_00001_nrcb1_cal.fits'],
        members=('jw03958007001_02101_00001_nrcb1_cal.fits',))
    rc = PF.main(['--target', 'sickle', '--proposal', '3958', '--obsid', '007',
                  '--filters', 'F187N', '--root', root, '--modules', modules,
                  '--skip-registry'])
    assert rc == 1, f'--modules {modules} on a module-B-only observation read OK'


def test_the_module_the_policy_DOES_allow_still_passes(tmp_path):
    root = _field(tmp_path, 'sickle', 'F187N', asns=[
        'jw03958-o007_20260101t000000_image3_00001_asn.json'],
        cals=['jw03958007001_02101_00001_nrcb1_cal.fits'],
        members=('jw03958007001_02101_00001_nrcb1_cal.fits',))
    rc = PF.main(['--target', 'sickle', '--proposal', '3958', '--obsid', '007',
                  '--filters', 'F187N', '--root', root, '--modules', 'nrcb',
                  '--skip-registry'])
    assert rc == 0


# ---------------------------------------------------------------------------
# The instrument token filter, which the module check was masking
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('instrument,foreign', [
    ('miri', 'jw02526021001_02101_00001_nrca1_cal.fits'),
    ('niriss', 'jw02526021001_02101_00001_nrca1_cal.fits'),
])
def test_a_single_detector_spec_rejects_another_instruments_association(
        tmp_path, instrument, foreign):
    """Deleting the token filter left 43 of 43 tests passing.

    The eight adversarial trees catch its absence only as a side effect of the
    module-coverage check -- and that check is SKIPPED for single-detector
    instruments, which is exactly where the token filter is the only thing
    standing between a MIRI spec and a NIRCam association.
    """
    own = f'jw02526021001_02101_00001_{"mirimage" if instrument == "miri" else "nis"}_cal.fits'
    root = _field(tmp_path, 'cloudc', 'F770W', asns=[
        'jw02526-o021_20260101t000000_image3_00001_asn.json'],
        cals=[own], members=(foreign,))
    row = PF.check(root, 'cloudc', '2526', '021', ['F770W'], ['nrca', 'nrcb'],
                   instrument=instrument)[0]
    assert not row.ok, (
        f'a {instrument} spec accepted an association whose only member is a '
        f'NIRCam exposure; the reduce keeps only its own instrument')
