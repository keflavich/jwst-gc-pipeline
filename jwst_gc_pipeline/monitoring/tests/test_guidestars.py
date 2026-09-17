"""The guide stars the Treasury observations were taken on."""
import importlib.util
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


def test_the_position_uncertainties_are_carried_through(gs, tmp_path):
    """They are tens to hundreds of mas -- larger than anything a reader would
    assume of a marker drawn on a JWST mosaic, and larger than the offsets the
    release page publishes."""
    _frame(tmp_path / 'o114' / 'F212N', 'a.fits')
    per_star, per_visit, _ = gs.scan(str(tmp_path))
    source = gs.to_document(per_star, per_visit)['sources'][0]
    assert source['sigma RA (mas)'] == 90.2
    assert source['sigma Dec (mas)'] == 68.3
    assert source['catalogue'] == 'GSC32'
