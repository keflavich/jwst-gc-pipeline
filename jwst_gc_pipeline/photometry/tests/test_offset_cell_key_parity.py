"""Per-cell offset records must spell the offset magnitude the same way,
whichever producer made them (issue #267).

``measure_offset_grid`` wrote its cells' offset as ``off`` while
``local_residual_map`` wrote ``off_mas`` -- and ``measure_offset_grid``'s own
``worst_off_cell`` summary, in the same returned dict as its cells, used
``off_mas`` too.  A consumer that reads one spelling gets an empty result from
the other with no error, which is how the pipeline monitor produced an empty
per-tile map for brick F115W visit 1 together with the derived statement that
"0/36 cells exceed 15 mas", while 14/36 cells were in fact over the gate.

``off_mas`` is the canonical key.  ``off`` is still written on
``measure_offset_grid`` cells so that checkpoints already on disk, and the
readers written against them, keep working.
"""
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord

from jwst_gc_pipeline.photometry.astrometry_offsets import (
    local_residual_map, measure_offset, measure_offset_grid)


def _field(n=4000, ra0=266.54, dec0=-28.70, span=0.02, seed=7):
    rng = np.random.default_rng(seed)
    ra = ra0 + (rng.random(n) - 0.5) * span
    dec = dec0 + (rng.random(n) - 0.5) * span
    return ra, dec


def _shift(ra, dec, dra_mas, ddec_mas):
    cosd = np.cos(np.radians(dec))
    return (ra + (dra_mas / 1000.0 / 3600.0) / cosd,
            dec + (ddec_mas / 1000.0 / 3600.0))


@pytest.fixture(scope="module")
def seamed_field():
    """A field whose top dec-band is rigidly displaced, so the per-cell maps
    have a non-trivial spread of offsets rather than a flat zero."""
    ra, dec = _field()
    a = SkyCoord(ra * u.deg, dec * u.deg)
    top = dec > -28.70
    rb, db = ra.copy(), dec.copy()
    rb[top], db[top] = _shift(ra[top], dec[top], 60.0, 0.0)
    return a, SkyCoord(rb * u.deg, db * u.deg)


def test_grid_cells_expose_the_canonical_off_mas_key(seamed_field):
    """Every ``measure_offset_grid`` cell must carry ``off_mas``.

    On the pre-fix code the cells carried only ``off`` and this KeyErrors.
    """
    a, b = seamed_field
    grid = measure_offset_grid(a, b, nx=4, ny=4, maxsep=1 * u.arcsec,
                               max_off_mas=50.0)
    assert grid["cells"], "the synthetic field must produce measurable cells"
    for c in grid["cells"]:
        assert "off_mas" in c, "grid cell is missing the canonical off_mas key"
        assert np.isfinite(c["off_mas"])


def test_local_residual_map_cells_expose_the_same_canonical_key(seamed_field):
    """The other producer of per-cell records uses the same key, so one reader
    can consume cells from either."""
    a, b = seamed_field
    glob = measure_offset(a, b, maxsep=1 * u.arcsec, sweep=False)
    assert glob is not None and glob["ok"]
    lrm = local_residual_map(a, b, glob, cell_arcsec=8.0,
                             match_radius=0.3 * u.arcsec, min_stars=10)
    assert lrm["cells"], "the synthetic field must produce measurable cells"
    for c in lrm["cells"]:
        assert "off_mas" in c
        assert np.isfinite(c["off_mas"])


def test_both_producers_agree_on_the_canonical_key(seamed_field):
    """The set-intersection statement: the key that identifies a cell's offset
    magnitude is the same string for both producers."""
    a, b = seamed_field
    grid = measure_offset_grid(a, b, nx=4, ny=4, maxsep=1 * u.arcsec)
    glob = measure_offset(a, b, maxsep=1 * u.arcsec, sweep=False)
    lrm = local_residual_map(a, b, glob, cell_arcsec=8.0,
                             match_radius=0.3 * u.arcsec, min_stars=10)
    assert grid["cells"] and lrm["cells"]
    common = set(grid["cells"][0]) & set(lrm["cells"][0])
    assert "off_mas" in common, (
        "the two per-cell producers still spell the offset magnitude "
        f"differently: grid keys {sorted(set(grid['cells'][0]))}, "
        f"residual-map keys {sorted(set(lrm['cells'][0]))}")


def test_grid_summary_worst_cell_uses_the_same_key_as_its_cells(seamed_field):
    """``worst_off_cell`` and ``worst_off_mas`` are the summary OF these cells,
    so they must be reproducible from the cells with the canonical key."""
    a, b = seamed_field
    grid = measure_offset_grid(a, b, nx=4, ny=4, maxsep=1 * u.arcsec,
                               max_off_mas=50.0)
    worst = max(c["off_mas"] for c in grid["cells"])
    assert grid["worst_off_cell"]["off_mas"] == pytest.approx(worst)
    assert grid["worst_off_mas"] == pytest.approx(worst)
    hit = [c for c in grid["cells"] if c["off_mas"] == grid["worst_off_cell"]["off_mas"]]
    assert hit, "worst_off_cell does not correspond to any cell"
    assert (hit[0]["ix"], hit[0]["iy"]) == (grid["worst_off_cell"]["ix"],
                                           grid["worst_off_cell"]["iy"])


def test_grid_cells_keep_the_historical_off_spelling(seamed_field):
    """``off`` is retained deliberately: checkpoints already written use it and
    are never rewritten, and ``diagnostics/astrometry_figs`` reads it.  Both
    keys must hold the same number."""
    a, b = seamed_field
    grid = measure_offset_grid(a, b, nx=4, ny=4, maxsep=1 * u.arcsec)
    for c in grid["cells"]:
        assert c["off"] == c["off_mas"]


def test_window_records_carry_both_spellings(seamed_field):
    """The per-window audit records that go into a checkpoint alongside the
    cells follow the same rule."""
    a, b = seamed_field
    res = measure_offset(a, b, maxsep=1 * u.arcsec, sweep=True)
    measured = [w for w in res["windows"] if w.get("dra") is not None]
    assert measured
    for w in measured:
        assert w["off_mas"] == w["off"]
