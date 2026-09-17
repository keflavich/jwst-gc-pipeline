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
    assert doc['sources'][0]['observations'] == 'o114 o120'
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
