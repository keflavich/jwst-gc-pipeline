"""Gate-diagnostic persistence (Phase A2 of the color-continuity plan).

find_saturated_stars now returns per-component seed provenance
(dqsat / peak / subfloor), and the fitted satstar rows carry the gate
diagnostics (seed_kind, sat_severity_floor, satstar_implied_peak,
satstar_observed_peak); gate-REJECTED candidates are persisted separately so
the merge can flag strip-taper daophot rows.
"""
import numpy as np
from astropy.io import fits

from jwst_gc_pipeline.reduction.saturated_star_finding import (
    find_saturated_stars)

SATBIT = 2
FLOOR = 4000.0


def _fitsdata(data, satmask=None):
    ny, nx = data.shape
    dq = np.zeros((ny, nx), dtype=np.uint32)
    if satmask is not None:
        dq[satmask] = SATBIT
    return fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(data=data.astype('float32'), name='SCI'),
        fits.ImageHDU(data=dq, name='DQ'),
        fits.ImageHDU(data=np.ones((ny, nx), dtype='float32'),
                      name='VAR_POISSON'),
    ])


def _blob(shape, x, y, r=2):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return (xx - x) ** 2 + (yy - y) ** 2 <= r ** 2


def test_seed_kinds_classified():
    """One DQ component + one above-floor unflagged peak + one sub-floor
    component -> three seeds with the right provenance labels."""
    data = np.full((160, 160), 5.0)
    dqmask = _blob(data.shape, 30, 30, r=3)
    data[dqmask] = 2 * FLOOR
    data[_blob(data.shape, 90, 90, r=2)] = 1.5 * FLOOR      # peak seed
    data[_blob(data.shape, 130, 40, r=2)] = 0.6 * FLOOR     # sub-floor seed
    sat, src, coms, kinds = find_saturated_stars(
        _fitsdata(data, satmask=dqmask), severity_floor=FLOOR)
    assert len(coms) == 3
    assert sorted(kinds) == ['dqsat', 'peak', 'subfloor']


def test_kinds_align_with_coms():
    """kinds[i] describes coms[i]: the sub-floor seed is the one at (130,40)."""
    data = np.full((160, 160), 5.0)
    dqmask = _blob(data.shape, 30, 30, r=3)
    data[dqmask] = 2 * FLOOR
    data[_blob(data.shape, 130, 40, r=2)] = 0.6 * FLOOR
    sat, src, coms, kinds = find_saturated_stars(
        _fitsdata(data, satmask=dqmask), severity_floor=FLOOR)
    assert len(coms) == len(kinds) == 2
    by_pos = {}
    for (cy, cx), k in zip(coms, kinds):
        by_pos[(round(cy / 50), round(cx / 50))] = k
    assert by_pos[(1, 1)] == 'dqsat'       # com (y,x) ~ (30, 30)
    assert by_pos[(1, 3)] == 'subfloor'    # com (y,x) ~ (40, 130)


def test_dq_overlap_wins():
    """A component containing DQ pixels reads dqsat even if seeding also
    painted shoulder pixels around it."""
    data = np.full((100, 100), 5.0)
    dqmask = _blob(data.shape, 50, 50, r=2)
    data[_blob(data.shape, 50, 50, r=4)] = 1.5 * FLOOR   # bright around DQ
    data[dqmask] = 2 * FLOOR
    sat, src, coms, kinds = find_saturated_stars(
        _fitsdata(data, satmask=dqmask), severity_floor=FLOOR)
    assert len(coms) == 1
    assert kinds[0] == 'dqsat'


def test_no_sources_returns_empty_kinds():
    data = np.full((64, 64), 5.0)
    sat, src, coms, kinds = find_saturated_stars(_fitsdata(data),
                                                 severity_floor=FLOOR)
    assert coms == [] or len(coms) == 0
    assert kinds == [] or len(kinds) == 0


def test_seed_kind_survives_dedup():
    """merge_catalogs._dedup_satstar_catalog must preserve the new columns."""
    from astropy.table import Table
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from jwst_gc_pipeline.photometry.merge_catalogs import (
        _dedup_satstar_catalog)
    n = 4
    tab = Table({
        'flux_fit': [1e5, 1.1e5, 5e4, 6e4],
        'seed_kind': ['dqsat', 'dqsat', 'subfloor', 'subfloor'],
        'sat_severity_floor': [FLOOR] * n,
        'satstar_implied_peak': [8e3, 8.2e3, 2.5e3, 2.6e3],
        'satstar_observed_peak': [9e3, 9e3, 2.4e3, 2.4e3],
        'sat_area': [30, 30, 5, 5],
    })
    # two stars, each seen twice (dup positions within the footprint radius)
    tab['skycoord_fit'] = SkyCoord(
        [10.0, 10.0, 10.01, 10.01] * u.deg, [0.0, 0.0, 0.01, 0.01] * u.deg)
    out = _dedup_satstar_catalog(tab)
    assert len(out) == 2
    for colname in ('seed_kind', 'sat_severity_floor',
                    'satstar_implied_peak', 'satstar_observed_peak'):
        assert colname in out.colnames
    assert sorted(np.asarray(out['seed_kind'], dtype=str)) == ['dqsat', 'subfloor']


def test_partner_seed_painted_and_classified():
    """Phase A1: a partner-band position with local pixels above
    partner_frac*floor is seeded with kind 'partner'; a partner position on
    blank sky seeds nothing."""
    data = np.full((100, 100), 5.0)
    data[_blob(data.shape, 40, 60, r=2)] = 0.30 * FLOOR   # too faint for subfloor(0.35) but above partner frac (0.25)
    sat, src, coms, kinds = find_saturated_stars(
        _fitsdata(data), severity_floor=FLOOR,
        partner_xy=[(40, 60), (80, 20)])
    assert len(coms) == 1
    assert kinds == ['partner']


def test_partner_seed_none_is_noop():
    data = np.full((80, 80), 5.0)
    data[_blob(data.shape, 40, 40, r=2)] = 0.30 * FLOOR
    sat, src, coms, kinds = find_saturated_stars(
        _fitsdata(data), severity_floor=FLOOR)
    assert len(coms) == 0
