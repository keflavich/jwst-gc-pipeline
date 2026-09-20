"""Framing and transparency for the static full-field overview images.

The overviews are flat PNGs rendered from the HiPS layers, for the uses a HiPS
cannot serve: a talk, a paper draft, a message.  What has to hold is that the
frame is the survey (not the survey plus half a degree of blank sky north of
it), that the grid is a measurable l/b rectangle, and that sky nobody observed
is transparent rather than black.

These test the pure geometry and the alpha writing.  The tile sampling itself
is exercised by rendering against the real HiPS, which is a host fact and so
not asserted here.
"""
import json
import os
import sys

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
import astropy.units as u

pytest.importorskip('PIL')
pytest.importorskip('reproject')

_SCRIPTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))), 'scripts', 'release')
sys.path.insert(0, _SCRIPTS)
G = pytest.importorskip('gc_overview_images')


def _square(ra, dec, half=0.02):
    return [[ra - half, dec - half], [ra - half, dec + half],
            [ra + half, dec + half], [ra + half, dec - half]]


@pytest.fixture
def footprints(tmp_path):
    """Two survey tiles near the plane, plus a control field far north of it."""
    main = SkyCoord([0.0, 0.4] * u.deg, [0.0, 0.0] * u.deg,
                    frame='galactic').icrs
    north = SkyCoord(0.0 * u.deg, 0.58 * u.deg, frame='galactic').icrs
    observed = [
        {'number': '040', 'ra': main[0].ra.deg, 'dec': main[0].dec.deg,
         'nircam': [_square(main[0].ra.deg, main[0].dec.deg)], 'miri': []},
        {'number': '098', 'ra': main[1].ra.deg, 'dec': main[1].dec.deg,
         'nircam': [_square(main[1].ra.deg, main[1].dec.deg)], 'miri': []},
        {'number': '139', 'ra': north.ra.deg, 'dec': north.dec.deg,
         'nircam': [_square(north.ra.deg, north.dec.deg)], 'miri': []},
    ]
    path = tmp_path / 'footprints.json'
    path.write_text(json.dumps({'program': '10678', 'observed': observed,
                                'planned': []}))
    return str(path)


def test_the_control_field_is_outside_the_box(footprints):
    """o139 sits half a degree north with nothing between.

    Framing to include it spends most of the image on blank sky to show one
    tile, so it is dropped -- and dropped by NUMBER, so a survey that grows
    north does not silently lose tiles to a latitude cut.
    """
    (l_min, l_max, b_min, b_max), used = G.survey_box(footprints)
    assert used == 2
    assert b_max < 0.2, 'the control field pulled the frame north'
    assert l_min < 0.0 < 0.4 < l_max


def test_including_the_control_field_is_possible_and_says_so(footprints):
    _, used = G.survey_box(footprints, exclude=())
    assert used == 3
    (_, _, _, b_max), _ = G.survey_box(footprints, exclude=())
    assert b_max > 0.5


def test_the_box_comes_from_polygons_not_centres(footprints):
    """A box around the tile CENTRES understates the field by half a tile."""
    (l_min, l_max, _, _), _ = G.survey_box(footprints)
    assert l_max - l_min > 0.4, 'box is no wider than the centre separation'


def test_the_grid_is_linear_in_l_and_b(footprints):
    """CAR is exactly linear only on the equator, so CRVAL2 must stay at 0.

    Otherwise the image is not an l/b rectangle and a reader measuring off it
    gets a slightly curved answer.
    """
    box, _ = G.survey_box(footprints)
    wcs, shape, scale = G.target_wcs(box, width=800, margin_deg=1 / 60)
    assert wcs.wcs.crval[1] == 0.0
    assert wcs.wcs.ctype[0].endswith('CAR')
    # b maps to row linearly: equal steps in b are equal steps in y.
    ys = [float(wcs.world_to_pixel(SkyCoord(box[0] * u.deg, b * u.deg,
                                            frame='galactic'))[1])
          for b in (-0.10, -0.05, 0.0, 0.05, 0.10)]
    steps = np.diff(ys)
    assert np.allclose(steps, steps[0], rtol=1e-6)


def test_the_grid_covers_the_whole_box_with_its_margin(footprints):
    box, _ = G.survey_box(footprints)
    margin = 1 / 60
    wcs, shape, scale = G.target_wcs(box, width=800, margin_deg=margin)
    height, width = shape
    for l, b in ((box[0], box[2]), (box[1], box[3])):
        x, y = wcs.world_to_pixel(SkyCoord(l * u.deg, b * u.deg,
                                           frame='galactic'))
        assert -0.5 <= float(x) <= width - 0.5
        assert -0.5 <= float(y) <= height - 0.5
    assert scale == pytest.approx((box[1] - box[0] + 2 * margin) / 800)


def test_pixels_are_square(footprints):
    box, _ = G.survey_box(footprints)
    wcs, shape, scale = G.target_wcs(box, width=800, margin_deg=1 / 60)
    assert abs(wcs.wcs.cdelt[0]) == pytest.approx(abs(wcs.wcs.cdelt[1]))


@pytest.mark.parametrize('scale_arcsec,expected_min', [
    (1.69, 9),     # the production default
    (0.85, 10),
    (3.40, 8),
])
def test_the_sampling_order_matches_the_output_scale(scale_arcsec, expected_min):
    """Finer than the output (so the kernel has something to average), not
    maximally fine (which is wasted reads and, point-sampled, aliasing)."""
    level = G.sampling_level(scale_arcsec / 3600.0)
    assert level == expected_min
    hips_px = (58.6323 / 512) / 2 ** level * 3600
    assert hips_px < scale_arcsec, 'sampling coarser than the output'
    assert hips_px > scale_arcsec / 8, 'sampling far finer than needed'


def test_uncovered_sky_is_transparent(tmp_path):
    """The bug this exists to prevent.

    `reproject_adaptive` returns 0.0, not NaN, where no input covered a pixel,
    so a mask inferred with `isfinite` was true over the whole frame and the
    first rendering wrote alpha 255 everywhere -- opaque black asserted over
    sky that was never observed.  Coverage therefore comes from an explicit
    plane, and this pins what that plane means at the point it becomes alpha.
    """
    from PIL import Image
    rgb = np.full((4, 6, 3), 200.0)
    covered = np.zeros((4, 6), dtype=bool)
    covered[1:3, 2:5] = True
    path = tmp_path / 'ov.png'
    G.save_rgba(str(path), rgb, covered)

    got = np.flipud(np.asarray(Image.open(path)))
    assert got.shape == (4, 6, 4)
    assert (got[..., 3] > 0).sum() == covered.sum()
    assert np.array_equal(got[..., 3] > 0, covered)
    assert got[1, 2, 3] == 255 and got[0, 0, 3] == 0


def test_a_black_pixel_inside_the_survey_stays_opaque(tmp_path):
    """Darkness is not absence: a genuinely black pixel that WAS observed must
    not be turned transparent, which is what keying on the colour would do."""
    from PIL import Image
    rgb = np.zeros((3, 3, 3))
    covered = np.ones((3, 3), dtype=bool)
    path = tmp_path / 'black.png'
    G.save_rgba(str(path), rgb, covered)
    got = np.asarray(Image.open(path))
    assert (got[..., 3] == 255).all()
    assert (got[..., :3] == 0).all()


def test_publish_names_only_the_overviews(tmp_path, monkeypatch):
    """The docroot holds other people's files: never a directory sync."""
    for name in ('gc_treasury_overview_vminmax.png', 'gc_treasury_overview.wcs',
                 'somebody_elses_file.png'):
        (tmp_path / name).write_bytes(b'x')
    calls = []
    monkeypatch.setattr(G.__dict__['os'], 'listdir', os.listdir)
    import subprocess
    monkeypatch.setattr(subprocess, 'run',
                        lambda cmd, **kw: calls.append(cmd))
    G.publish(str(tmp_path), dry=False)
    assert calls, 'nothing was published'
    for cmd in calls:
        assert '--delete' not in cmd
        joined = ' '.join(cmd)
        assert 'somebody_elses_file' not in joined
        assert 'gc_treasury_overview_vminmax.png' in joined
