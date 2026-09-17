"""The full-focal-plane overlay: which SIAF an aperture belongs to, and where
the parallels land."""
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
def bf():
    import sys
    sys.path.insert(0, _SCRIPTS)
    return _load('build_footprints')


def test_every_aperture_resolves_to_its_own_siaf(bf):
    """`'NIRCam' if name.startswith('NRC') else 'MIRI'` sent FGS1_FULL to the
    MIRI SIAF -- a KeyError several frames later with nothing naming the
    cause. The instruments this tool draws all have to route correctly."""
    assert bf.aperture_instrument('NRCA1_FULL') == 'NIRCam'
    assert bf.aperture_instrument('NRCALL_FULL') == 'NIRCam'
    assert bf.aperture_instrument('MIRIM_ILLUM') == 'MIRI'
    assert bf.aperture_instrument('FGS1_FULL') == 'FGS'
    assert bf.aperture_instrument('NIS_CEN') == 'NIRISS'
    assert bf.aperture_instrument('NRS1_FULL') == 'NIRSpec'


def test_an_unknown_aperture_raises_rather_than_defaulting(bf):
    """Defaulting is what made the old form wrong: it had an answer for every
    name, including the ones it had never heard of."""
    with pytest.raises(ValueError, match='no SIAF known'):
        bf.aperture_instrument('WFI_FULL')


@pytest.mark.filterwarnings('ignore')
def test_the_parallels_land_off_the_prime_not_on_it(bf):
    """Every aperture goes through ONE attitude, anchored on the aperture APT
    points at. Anchoring each instrument on its own reference point instead
    would stack them all on the target, which looks plausible and is wrong.

    MIRI's imager sits ~7.5' from the NIRCam prime in the JWST focal plane;
    measured here to confirm the projection, not to pin pysiaf's numbers.
    """
    pytest.importorskip('pysiaf')
    import math

    polys = bf.aperture_polygons(
        266.5, -28.7, 87.0,
        ['NRCA1_FULL', 'MIRIM_ILLUM', 'FGS1_FULL'], 'NRCALL_FULL')
    assert set(polys) == {'NRCA1_FULL', 'MIRIM_ILLUM', 'FGS1_FULL'}
    assert all(len(v) == 4 for v in polys.values())

    def centre(poly):
        return (sum(p[0] for p in poly) / 4, sum(p[1] for p in poly) / 4)

    def sep_arcmin(a, b):
        d1, d2 = math.radians(a[1]), math.radians(b[1])
        dot = (math.sin(d1) * math.sin(d2)
               + math.cos(d1) * math.cos(d2) * math.cos(math.radians(a[0] - b[0])))
        return math.degrees(math.acos(max(-1, min(1, dot)))) * 60

    nrc, miri, fgs = (centre(polys[k]) for k in
                      ('NRCA1_FULL', 'MIRIM_ILLUM', 'FGS1_FULL'))
    assert 5 < sep_arcmin(nrc, miri) < 12, sep_arcmin(nrc, miri)
    assert 2 < sep_arcmin(nrc, fgs) < 8, sep_arcmin(nrc, fgs)


@pytest.mark.filterwarnings('ignore')
def test_the_attitude_follows_the_position_angle(bf):
    """The tool's claim is that it draws each pointing at ITS OWN PA_V3. A
    projection that ignored the angle would place every observation
    identically, which on a survey at one nominal PA looks entirely right."""
    pytest.importorskip('pysiaf')
    at87 = bf.aperture_polygons(266.5, -28.7, 87.0, ['NRCA1_FULL'],
                                'NRCALL_FULL')['NRCA1_FULL']
    at97 = bf.aperture_polygons(266.5, -28.7, 97.0, ['NRCA1_FULL'],
                                'NRCALL_FULL')['NRCA1_FULL']
    moved = max(abs(a[0] - b[0]) + abs(a[1] - b[1])
                for a, b in zip(at87, at97))
    assert moved > 1e-3, moved
