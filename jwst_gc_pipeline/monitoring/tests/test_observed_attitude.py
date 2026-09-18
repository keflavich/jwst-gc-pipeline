"""The focal-plane overlay draws the attitude the telescope held, not the plan's."""
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
def fp():
    import sys
    sys.path.insert(0, _SCRIPTS)
    return _load('build_focal_plane')


def _frame(directory, name, pa=90.1499, ra=266.834, dec=-28.597,
           aper='NRCA1_FULL', obs='114'):
    from astropy.io import fits
    import numpy as np
    directory.mkdir(parents=True, exist_ok=True)
    primary = fits.PrimaryHDU(np.zeros((2, 2), dtype='float32'))
    primary.header['OBSERVTN'] = obs
    primary.header['APERNAME'] = aper
    primary.header['INSTRUME'] = 'NIRCAM'
    sci = fits.ImageHDU(np.zeros((2, 2), dtype='float32'))
    sci.header['PA_V3'] = pa
    sci.header['RA_REF'] = ra
    sci.header['DEC_REF'] = dec
    fits.HDUList([primary, sci]).writeto(directory / name)


def test_the_attitude_comes_from_the_frames_not_the_plan(fp, tmp_path):
    """10678 was planned at PA_V3 = 87.0 and flown at 90.15.  Drawn at the
    plan angle the NIRCam boxes sit 24" from where the frames actually are and
    the parallels, 8 to 15 arcmin off the anchor, are displaced by 35".
    """
    _frame(tmp_path / 'o114' / 'F212N', 'a.fits')
    got = fp.observed_attitude(str(tmp_path))
    assert set(got) == {'o114'}
    assert got['o114']['pa_v3'] == pytest.approx(90.1499)
    assert got['o114']['anchor'] == 'NRCA1_FULL'
    assert got['o114']['ra'] == pytest.approx(266.834)


def test_the_anchor_is_the_frame_s_own_aperture(fp, tmp_path):
    """`RA_REF`/`DEC_REF` is the reference point of the aperture that took the
    frame, so pairing it with a different anchor puts the whole observatory in
    the wrong place -- by the offset between the two apertures, which for
    NRCA1 against NRCALL_FULL is arcminutes."""
    _frame(tmp_path / 'o132' / 'F770W', 'm.fits', aper='MIRIM_ILLUM',
           obs='132')
    got = fp.observed_attitude(str(tmp_path))
    assert got['o132']['anchor'] == 'MIRIM_ILLUM'


def test_the_within_observation_spread_is_reported_not_averaged(fp, tmp_path):
    """One number per observation is only the right picture while the
    observation was held at one attitude.  Measured over the release: 0.0031
    deg worst case.  A spread far beyond that would mean it was not."""
    d = tmp_path / 'o114' / 'F212N'
    _frame(d, 'a.fits', pa=90.1499)
    _frame(d, 'b.fits', pa=90.1520)
    got = fp.observed_attitude(str(tmp_path))
    assert got['o114']['n_frames'] == 2
    assert got['o114']['pa_v3_spread_deg'] == pytest.approx(0.0021, abs=1e-5)


def test_an_observation_with_no_frames_keeps_the_planned_attitude(fp, tmp_path,
                                                                  monkeypatch):
    """101 of 139 pointings have not been taken.  They are still drawn -- that
    is what the overlay is for -- but at the plan's angle, and the entry says
    which of the two it is rather than presenting them alike."""
    footprints = tmp_path / 'footprints.json'
    footprints.write_text(json.dumps({
        'pa_v3': 87.0,
        'observed': [{'number': 114, 'ra': 266.834, 'dec': -28.597,
                      'target': 'GC_114', 'status': 'Archived'}],
        'planned': [{'number': 150, 'ra': 266.7, 'dec': -28.5,
                     'target': 'GC_150', 'status': 'Scheduled'}],
    }))
    _frame(tmp_path / 'exposures' / 'o114' / 'F212N', 'a.fits')

    class Args:
        footprints = None
        exposures = None
        out = None
        anchor = 'NRCALL_FULL'
        programme = '10678'
    args = Args()
    args.footprints = str(footprints)
    args.exposures = str(tmp_path / 'exposures')
    args.out = str(tmp_path / 'out.json')
    fp.build(args)

    doc = json.loads((tmp_path / 'out.json').read_text())
    by_id = {p['id']: p for p in doc['pointings']}
    assert by_id['o114']['attitude'] == 'as-flown'
    assert by_id['o114']['pa_v3'] == pytest.approx(90.1499)
    assert by_id['o150']['attitude'] == 'planned'
    assert by_id['o150']['pa_v3'] == pytest.approx(87.0)
    assert doc['n_as_flown'] == 1
