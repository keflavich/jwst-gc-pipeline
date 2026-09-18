"""The guide stars the Treasury observations were taken on."""
import importlib.util
import json
import os

import pytest

_SCRIPTS = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', '..',
    'scripts', 'monitoring'))


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_SCRIPTS, f'{name}.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def gs():
    import sys
    sys.path.insert(0, _SCRIPTS)
    return _load('build_guidestars')


def _frame(directory, name, **over):
    """One exposure, with the guide-star keys a delivered frame carries."""
    from astropy.io import fits
    import numpy as np
    hdr = {'GDSTARID': 'S8DHZ0O2W5', 'GS_RA': 266.7955, 'GS_DEC': -28.6242,
           'GS_MAG': 12.871, 'GS_ORDER': 1, 'GS_V3_PA': 90.248,
           'GS_URA': 90.2, 'GS_UDEC': 68.3, 'GSC_VER': 'GSC32',
           'VISIT_ID': '10678114001', 'OBSERVTN': '114', 'INSTRUME': 'NIRCAM'}
    hdr.update(over)
    hdu = fits.PrimaryHDU(np.zeros((2, 2), dtype='float32'))
    for key, value in hdr.items():
        if value is not None:
            hdu.header[key] = value
    directory.mkdir(parents=True, exist_ok=True)
    fits.HDUList([hdu]).writeto(directory / name)


def test_one_star_over_many_frames_is_one_source(gs, tmp_path):
    """A visit is 48 frames on ONE star. Emitting a source per frame would put
    48 markers on one position and report the survey as having used 1,704
    guide stars."""
    d = tmp_path / 'o114' / 'F212N'
    for i in range(5):
        _frame(d, f'f{i}.fits')
    per_star, per_visit, missing = gs.scan(str(tmp_path))

    assert list(per_star) == ['S8DHZ0O2W5']
    assert per_star['S8DHZ0O2W5']['frames'] == 5
    assert per_star['S8DHZ0O2W5']['visits'] == {'10678114001'}
    assert not missing

    doc = gs.to_document(per_star, per_visit)
    assert doc['n'] == 1
    assert doc['sources'][0]['frames'] == 5
    assert doc['sources'][0]['visits'] == 1


def test_a_star_that_guided_several_visits_lists_all_of_them(gs, tmp_path):
    """It is one star on the sky. Two markers at one position read as two
    stars, and the observation list is what makes the overlay answer "which
    field was this guiding?"."""
    _frame(tmp_path / 'o114' / 'F212N', 'a.fits')
    _frame(tmp_path / 'o120' / 'F212N', 'b.fits',
           VISIT_ID='10678120001', OBSERVTN='120')
    per_star, per_visit, _ = gs.scan(str(tmp_path))
    doc = gs.to_document(per_star, per_visit)

    assert doc['n'] == 1
    assert doc['sources'][0]['chosen for'] == 'o114 o120'
    assert doc['sources'][0]['visits'] == 2
    # and the inverse lookup names it under BOTH, which is what the focal-plane
    # panel reads
    assert set(doc['by_obs']) == {'o114', 'o120'}
    assert doc['by_obs']['o120'][0]['guide_star'] == 'S8DHZ0O2W5'


def test_a_visit_that_reacquired_keeps_both_stars(gs, tmp_path):
    """FGS can drop lock and re-acquire on a different star mid-visit. Reading
    one frame per visit -- the cheap scan -- reports whichever sorted first as
    if it covered the whole visit, and the frames taken on the other star are
    then attributed to a star that was not guiding them."""
    d = tmp_path / 'o114' / 'F212N'
    _frame(d, 'a.fits')
    _frame(d, 'b.fits', GDSTARID='S8DH061295', GS_RA=266.71, GS_DEC=-28.66,
           GS_ORDER=2)
    per_star, per_visit, _ = gs.scan(str(tmp_path))
    doc = gs.to_document(per_star, per_visit)

    assert doc['n'] == 2
    assert len(doc['by_obs']['o114']) == 2
    assert {u['guide_star'] for u in doc['by_obs']['o114']} == \
        {'S8DHZ0O2W5', 'S8DH061295'}


def test_the_id_order_says_the_first_candidate_was_not_used(gs, tmp_path):
    """`GS_ORDER > 1` means the onboard sequence fell through to a later
    candidate. Dropping it makes a fallback acquisition indistinguishable from
    the planned one, which is the interesting case."""
    _frame(tmp_path / 'o114' / 'F212N', 'a.fits', GS_ORDER=3)
    per_star, per_visit, _ = gs.scan(str(tmp_path))
    doc = gs.to_document(per_star, per_visit)
    assert doc['sources'][0]['ID order'] == '3'


def test_a_frame_without_a_guide_star_is_reported_not_dropped(gs, tmp_path):
    """Silently skipping it makes a scan that read nothing look the same as one
    where every frame was guided."""
    _frame(tmp_path / 'o114' / 'F212N', 'ok.fits')
    _frame(tmp_path / 'o114' / 'F212N', 'bad.fits', GDSTARID=None, GS_RA=None,
           GS_DEC=None)
    per_star, _per_visit, missing = gs.scan(str(tmp_path))
    assert per_star['S8DHZ0O2W5']['frames'] == 1
    assert len(missing) == 1 and 'bad.fits' in missing[0]


def test_the_uncertainties_go_out_under_their_keyword_names_unconverted(
        gs, tmp_path):
    """The unit is not settled: the JWST keyword dictionary says arcsec, and
    15 of the 34 delivered values are 65 to 118, which as arcsec is a
    catalogue position no acquisition would survive. Publishing either
    conversion labels one of the two groups wrong by 1000x, so the values go
    out as the header wrote them, named for the keyword."""
    _frame(tmp_path / 'o114' / 'F212N', 'a.fits')
    per_star, per_visit, _ = gs.scan(str(tmp_path))
    source = gs.to_document(per_star, per_visit)['sources'][0]
    assert source['GS_URA'] == 90.2
    assert source['GS_UDEC'] == 68.3
    assert 'mas' not in ' '.join(source), 'no unit is claimed in a label'
    assert source['catalogue'] == 'GSC32'


def test_a_sub_unit_uncertainty_does_not_round_to_zero(gs, tmp_path):
    """19 of the 34 stars are Gaia-based, with values from 0.017 to 1.295.
    Rounded to one decimal, 8 of them published 0.0 -- a position known
    exactly, on the column that exists to say it is not."""
    _frame(tmp_path / 'o114' / 'F212N', 'a.fits', GS_URA=0.01810, GS_UDEC=0.0)
    per_star, per_visit, _ = gs.scan(str(tmp_path))
    source = gs.to_document(per_star, per_visit)['sources'][0]
    assert source['GS_URA'] == 0.0181
    # and a real zero stays zero rather than becoming a tiny number
    assert source['GS_UDEC'] == 0.0


def test_a_missing_uncertainty_is_null_rather_than_zero(gs, tmp_path):
    """`0.0` for an absent keyword claims the position is exact."""
    _frame(tmp_path / 'o114' / 'F212N', 'a.fits', GS_URA=None, GS_UDEC=None)
    per_star, per_visit, _ = gs.scan(str(tmp_path))
    source = gs.to_document(per_star, per_visit)['sources'][0]
    assert source['GS_URA'] is None and source['GS_UDEC'] is None


def test_the_guide_star_v3_pa_reaches_the_document(gs, tmp_path):
    """It was read into `per_visit` and never emitted. A field collected and
    dropped is how a later reader concludes the data is unavailable."""
    _frame(tmp_path / 'o114' / 'F212N', 'a.fits', GS_V3_PA=90.248)
    per_star, per_visit, _ = gs.scan(str(tmp_path))
    doc = gs.to_document(per_star, per_visit)
    assert doc['by_obs']['o114'][0]['pa_v3'] == pytest.approx(90.248)


def test_frames_that_disagree_about_a_star_are_reported(gs, tmp_path):
    """`star['ra'] = ...` on every frame made the published position depend on
    `os.walk` order. The first frame's value is kept and the disagreement is
    named, in a scan that reads every frame precisely because per-frame values
    can differ."""
    d = tmp_path / 'o114' / 'F212N'
    _frame(d, 'a.fits')
    _frame(d, 'b.fits', GS_RA=266.9000)          # same star id, moved 0.1 deg
    per_star, _per_visit, problems = gs.scan(str(tmp_path))
    assert per_star['S8DHZ0O2W5']['ra'] == pytest.approx(266.7955)
    assert any('ra' in p and 'b.fits' in p for p in problems), problems

    # frames that agree report nothing
    quiet = tmp_path / 'o120' / 'F212N'
    _frame(quiet, 'c.fits', VISIT_ID='10678120001', OBSERVTN='120')
    _frame(quiet, 'd.fits', VISIT_ID='10678120001', OBSERVTN='120')
    _per_star, _pv, problems = gs.scan(str(quiet))
    assert problems == []


# ---- what failed, and what was never flown --------------------------------
def test_a_fallback_acquisition_is_flagged_for_its_own_colour(gs, tmp_path):
    """`GS_ORDER > 1` means the observatory fell through to a later candidate,
    so an earlier one failed to acquire. That is the only acquisition failure
    the delivered data records, and it is the case worth seeing first."""
    _frame(tmp_path / 'o114' / 'F212N', 'a.fits', GS_ORDER=2)
    _frame(tmp_path / 'o120' / 'F212N', 'b.fits', GDSTARID='S8DH061295',
           GS_RA=266.71, GS_DEC=-28.66, VISIT_ID='10678120001',
           OBSERVTN='120', GS_ORDER=1)
    per_star, per_visit, _ = gs.scan(str(tmp_path))
    doc = gs.to_document(per_star, per_visit)
    flag = {s['guide star']: s['fallback'] for s in doc['sources']}
    assert flag == {'S8DHZ0O2W5': True, 'S8DH061295': False}
    assert doc['n_fallback'] == 1


def test_each_star_carries_the_observation_numbers_as_its_label(gs, tmp_path):
    """The viewer prints this beside the marker. 34 markers all reading `o`
    tell a reader nothing; the observation number is what connects a star to
    the field it was guiding."""
    _frame(tmp_path / 'o114' / 'F212N', 'a.fits')
    _frame(tmp_path / 'o098' / 'F212N', 'b.fits', VISIT_ID='10678098001',
           OBSERVTN='098')
    per_star, per_visit, _ = gs.scan(str(tmp_path))
    doc = gs.to_document(per_star, per_visit)
    assert doc['sources'][0]['label'] == '114 98'


def test_an_observation_that_was_never_flown_is_listed_with_no_guide_star(
        gs, tmp_path):
    """A skipped observation has no frames, so the scan cannot see it at all.
    Left out, the viewer says the survey used 34 guide stars and nothing
    failed; listed, the seven skipped pointings are visible as the places
    where there is no star to show.

    Their selected guide star is not recoverable: PPS assigns it at
    scheduling and it appears only in delivered data. MAST holds planned rows
    at `calib_level = -1` for o099 and nothing for the other six.
    """
    fp = tmp_path / 'footprints.json'
    fp.write_text(json.dumps({'planned': [
        {'number': 99, 'status': 'Skipped', 'target': 'GC_99',
         'ra': 266.589, 'dec': -28.658},
        {'number': 105, 'status': 'Scheduled', 'target': 'GC_105',
         'ra': 266.6, 'dec': -28.6},
        {'number': 136, 'status': 'Skipped', 'target': 'GC_136',
         'ra': 266.815, 'dec': -28.330},
    ]}))
    unflown = gs.unflown_observations(str(fp))
    assert [e['observation'] for e in unflown] == ['o099', 'o136']
    assert unflown[0]['status'] == 'Skipped'
    assert unflown[0]['ra'] == 266.589

    _frame(tmp_path / 'o114' / 'F212N', 'a.fits')
    per_star, per_visit, _ = gs.scan(str(tmp_path / 'o114'))
    doc = gs.to_document(per_star, per_visit, unflown=unflown)
    assert [e['observation'] for e in doc['unflown']] == ['o099', 'o136']
    # and they are NOT sources: there is no star to put on the sky
    assert len(doc['sources']) == 1


def test_a_missing_footprints_file_yields_no_unflown_list(gs, tmp_path):
    """The statuses come from a file the monitor rebuilds hourly. Absent, the
    catalogue is still correct about what DID fly rather than failing to
    build."""
    assert gs.unflown_observations(str(tmp_path / 'nothing.json')) == []


# ---- the acquisition ladder, and the visits that never finished it --------
def test_the_ladder_says_where_a_visit_stopped(gs):
    """FGS writes one exposure type per rung. `FGS_FINEGUIDE` present means the
    visit guided; its absence is the failure, and the highest rung reached says
    whether the star was never identified, never acquired, or acquired and then
    lost."""
    guided = gs.visit_outcome({'FGS_ID-IMAGE', 'FGS_ID-STACK', 'FGS_ACQ1',
                               'FGS_ACQ2', 'FGS_TRACK', 'FGS_FINEGUIDE'})
    assert guided == ('fine guide', 'guided')

    assert gs.visit_outcome({'FGS_ID-IMAGE', 'FGS_ID-STACK'}) == \
        ('identification', 'failed at identification')
    assert gs.visit_outcome({'FGS_ID-STACK', 'FGS_ACQ1', 'FGS_ACQ2'}) == \
        ('acquisition', 'failed at acquisition')
    assert gs.visit_outcome({'FGS_ID-STACK', 'FGS_ACQ1', 'FGS_TRACK'}) == \
        ('track', 'failed at track')
    assert gs.visit_outcome(set()) == (None, 'no guide-star exposures')


def test_the_rungs_are_ordered_not_merely_listed(gs):
    """The outcome is the HIGHEST rung present, so the order of `LADDER` is
    load-bearing. Sorting it alphabetically puts FGS_TRACK above
    FGS_FINEGUIDE and reports a guided visit as having failed at track."""
    assert gs.LADDER.index('FGS_ACQ1') < gs.LADDER.index('FGS_TRACK')
    assert gs.LADDER.index('FGS_TRACK') < gs.LADDER.index('FGS_FINEGUIDE')
    assert gs.LADDER[-1] == 'FGS_FINEGUIDE'
    # a set arriving in any order gives the same answer
    rungs = {'FGS_FINEGUIDE', 'FGS_ID-IMAGE', 'FGS_TRACK'}
    assert gs.visit_outcome(rungs)[1] == 'guided'


def _visit(obs='104', visit='10678104001', stars=('S8DHZ0P44I',), stages=(),
           acq=None):
    """One visit's rows. `acq` defaults to what the ladder implies -- a visit
    that reached fine guide reports a SUCCESSFUL acquisition -- so a test that
    does not care about GSACSTAT is not silently writing a contradiction."""
    if acq is None:
        acq = ({'SUCCESSFUL'} if 'FGS_FINEGUIDE' in stages
               else {'UNSUCCESSFUL'})
    return {visit: {'visit': visit, 'observation': f'o{int(obs):03d}',
                    'stars': {s: 2 for s in stars}, 'stages': set(stages),
                    'acq_status': set(acq)}}


def test_a_star_that_failed_one_visit_and_guided_another_carries_both(gs):
    """S8DM352124 was chosen for o098, which guided, and for o099, which never
    identified it. Marking the star failed outright would say the star is bad;
    marking it guided would hide the failure. It is the (star, visit) pair that
    failed."""
    per_star = {'S8DM352124': {'obs': {'o098', 'o099'}, 'visits': set(),
                               'orders': {1}, 'frames': 6, 'instruments': set(),
                               'ra': 266.5, 'dec': -28.6}}
    per_visit = {}
    per_visit.update(_visit('098', '10678098001', ('S8DM352124',),
                            ('FGS_ID-IMAGE', 'FGS_ACQ1', 'FGS_ACQ2',
                             'FGS_TRACK', 'FGS_FINEGUIDE')))
    per_visit.update(_visit('099', '10678099001', ('S8DM352124',),
                            ('FGS_ID-IMAGE', 'FGS_ID-STACK')))
    gs.annotate_outcomes(per_star, per_visit)

    star = per_star['S8DM352124']
    assert star['failed_obs'] == {'o099'}
    assert star['outcome'] == 'failed at identification'

    doc = gs.to_document(per_star, per_visit)
    source = doc['sources'][0]
    assert source['failed observations'] == 'o099'
    assert source['chosen for'] == 'o098 o099'
    assert doc['n_failed'] == 1
    assert doc['failed_visits'] == ['10678099001']
    # and the per-observation lookup says which of the two it was
    assert doc['by_obs']['o099'][0]['outcome'] == 'failed at identification'
    assert doc['by_obs']['o098'][0]['outcome'] == 'guided'


def test_a_failure_outcome_is_not_overwritten_by_a_later_success(gs):
    """`setdefault` on the success path and assignment on the failure path is
    the asymmetry that keeps this true whatever order the visits arrive in.
    Reversed, a star that failed o110 and guided o111 would read 'guided' and
    drop out of the red layer."""
    per_star = {'X': {'obs': {'o110', 'o111'}, 'visits': set(), 'orders': {1},
                      'frames': 4, 'instruments': set(),
                      'ra': 1.0, 'dec': 2.0}}
    per_visit = {}
    per_visit.update(_visit('110', 'v110', ('X',),
                            ('FGS_ID-STACK', 'FGS_ACQ1')))
    per_visit.update(_visit('111', 'v111', ('X',),
                            ('FGS_ID-STACK', 'FGS_ACQ1', 'FGS_ACQ2',
                             'FGS_TRACK', 'FGS_FINEGUIDE')))
    gs.annotate_outcomes(per_star, per_visit)
    assert per_star['X']['outcome'] == 'failed at acquisition'
    assert per_star['X']['failed_obs'] == {'o110'}


# ---- GSACSTAT: the evidence the ladder alone does not have ----------------
def test_a_failure_is_asserted_by_gsacstat_not_only_inferred_from_absence(gs):
    """The ladder reads failure from a MISSING `FGS_FINEGUIDE`, so a truncated
    query or a file absent from the archive reports a guided visit as failed
    at track. `GSACSTAT` is the positive evidence: over 10678 the seven failed
    visits carry `UNSUCCESSFUL` and nothing else, and all 38 guided ones carry
    at least one `SUCCESSFUL`.
    """
    failed = ('FGS_ID-IMAGE', 'FGS_ID-STACK', 'FGS_ACQ1')
    assert gs.visit_outcome(failed, ('UNSUCCESSFUL',)) == \
        ('acquisition', 'failed at acquisition')

    guided = failed + ('FGS_ACQ2', 'FGS_TRACK', 'FGS_FINEGUIDE')
    assert gs.visit_outcome(guided, ('SUCCESSFUL', 'UNSUCCESSFUL')) == \
        ('fine guide', 'guided')


def test_the_two_sources_disagreeing_is_said_rather_than_rounded(gs):
    """A SUCCESSFUL acquisition with no fine guide is the shape a truncated
    query takes. Filing it as "failed at track" -- what the ladder alone does
    -- turns a gap in the evidence into a claim about the telescope."""
    reached, outcome = gs.visit_outcome(
        ('FGS_ID-STACK', 'FGS_ACQ1', 'FGS_ACQ2', 'FGS_TRACK'), ('SUCCESSFUL',))
    assert reached == 'track'
    assert outcome == 'acquisition reported SUCCESSFUL, no fine guide (track)'
    assert not outcome.startswith('failed')

    # the mirror case: fine guide with nothing reporting success
    _r, other = gs.visit_outcome(
        ('FGS_ID-STACK', 'FGS_ACQ1', 'FGS_FINEGUIDE'), ('UNSUCCESSFUL',))
    assert other == 'fine guide reached, no acquisition reported SUCCESSFUL'
    assert other != 'guided'


def test_without_gsacstat_the_verdict_is_the_same_and_says_what_it_rests_on(gs):
    """`--source frames` has no GSACSTAT at all. The wording of the outcome is
    the ladder's either way -- it is read by people and by the viewer's layers
    -- so the provenance rides on the visit instead."""
    assert gs.visit_outcome(('FGS_ID-STACK', 'FGS_ACQ1')) == \
        ('acquisition', 'failed at acquisition')

    per_star = {'X': {'obs': {'o104'}, 'visits': set(), 'orders': {1},
                      'frames': 2, 'instruments': set(), 'ra': 1.0, 'dec': 2.0}}
    per_visit = _visit('104', 'v104', ('X',), ('FGS_ID-STACK', 'FGS_ACQ1'),
                       acq=())
    gs.annotate_outcomes(per_star, per_visit)
    assert per_visit['v104']['evidence'] == 'ladder only'

    with_status = _visit('104', 'v104', ('X',), ('FGS_ID-STACK', 'FGS_ACQ1'))
    gs.annotate_outcomes(per_star, with_status)
    assert with_status['v104']['evidence'] == 'ladder+gsacstat'


def test_the_document_carries_the_status_it_was_decided_from(gs):
    """An unread corroborator is worse than none when the claim cites it: the
    rows that decided the verdict reach the reader."""
    per_star = {'X': {'obs': {'o136'}, 'visits': set(), 'orders': {1},
                      'frames': 4, 'instruments': set(), 'ra': 1.0, 'dec': 2.0}}
    per_visit = _visit('136', 'v136', ('X',),
                       ('FGS_ID-STACK', 'FGS_ACQ1', 'FGS_TRACK'))
    gs.annotate_outcomes(per_star, per_visit)
    doc = gs.to_document(per_star, per_visit)

    assert doc['by_obs']['o136'][0]['acq_status'] == ['UNSUCCESSFUL']
    assert doc['by_obs']['o136'][0]['evidence'] == 'ladder+gsacstat'
    assert doc['sources'][0]['acquisition status'] == 'UNSUCCESSFUL'


def test_a_contested_visit_is_listed_apart_from_the_failures(gs):
    """It is neither guided nor failed. Counting it as either is the rounding
    the outcome string exists to avoid."""
    per_star = {'X': {'obs': {'o104'}, 'visits': set(), 'orders': {1},
                      'frames': 2, 'instruments': set(), 'ra': 1.0, 'dec': 2.0}}
    per_visit = _visit('104', 'v104', ('X',),
                       ('FGS_ID-STACK', 'FGS_ACQ1', 'FGS_TRACK'),
                       acq=('SUCCESSFUL',))
    gs.annotate_outcomes(per_star, per_visit)
    doc = gs.to_document(per_star, per_visit)
    assert doc['contested_visits'] == ['v104']
    assert doc['failed_visits'] == []
