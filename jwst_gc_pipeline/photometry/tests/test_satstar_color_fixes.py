"""Regressions for the CMD saturation-continuity fixes (Brick audit 2026-07-10).

A saturated star's centroid scatters ~0.2", so the cross-band merge left the
satstar row and the partner band's independent detection as SEPARATE rows;
the m8 fill then measured the partner band on the star-subtracted residual
(~5 mag too faint), planting diagonal junk plumes on the CMD bright end.
Three coordinated guards are pinned here:

1. ``dedup_merged_catalog`` ``sat_link_radius``: satstar-involved pairs link
   out to 0.5" (unsaturated pairs keep the tight 0.10"); binary protection
   (same-band discrepant detections) still blocks, and beyond 0.10" rows
   sharing two or more bands are split by a shared band where both rows or
   neither are saturated (#1038).
2. ``forced_fill._satstar_partner_guard``: a saturated row with a real
   detection of the fill band nearby is NOT filled.
3. ``column_utils.color_reliable_mask``: interim plotting rule -- a color is
   invalid if one band is forced_filled while the other is saturated.
"""
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

from jwst_gc_pipeline.photometry.dedup_catalog import dedup_merged_catalog
from jwst_gc_pipeline.photometry.forced_fill import _satstar_partner_guard
from jwst_gc_pipeline.photometry.column_utils import color_reliable_mask

RA0, DEC0 = 266.54, -28.71


def _merged_cat(rows):
    """Minimal merged-catalog Table.  Each row dict: dra/ddec (arcsec offsets),
    then per-band mag/mask/replaced flags via keys like mag_f182m."""
    n = len(rows)
    ra = np.array([RA0 + r.get('dra', 0.0) / 3600 for r in rows])
    dec = np.array([DEC0 + r.get('ddec', 0.0) / 3600 for r in rows])
    t = Table({'skycoord_ref': SkyCoord(ra * u.deg, dec * u.deg)})
    for b in ('f182m', 'f212n'):
        t[f'mag_vega_{b}'] = np.array([r.get(f'mag_{b}', np.nan) for r in rows])
        t[f'mask_{b}'] = np.array([not np.isfinite(r.get(f'mag_{b}', np.nan))
                                   for r in rows])
        t[f'emag_ab_{b}'] = np.where(np.isfinite(t[f'mag_vega_{b}']), 0.01, np.nan)
        t[f'flux_jy_{b}'] = 10 ** (-0.4 * np.asarray(t[f'mag_vega_{b}']))
        t[f'replaced_saturated_{b}'] = np.array(
            [bool(r.get(f'sat_{b}', False)) for r in rows])
    return t


def _run_dedup(tmp_path, rows, **kw):
    inp, outp = str(tmp_path / 'in.fits'), str(tmp_path / 'out.fits')
    _merged_cat(rows).write(inp, overwrite=True)
    dedup_merged_catalog(inp, outp, verbose=False, **kw)
    return Table.read(outp)


def test_dedup_links_satstar_to_partner_detection(tmp_path):
    """Satstar row (f182m only) + partner detection (f212n only) 0.3" apart:
    complementary coverage + satstar involvement -> merged into one row."""
    out = _run_dedup(tmp_path, [
        dict(dra=0.0, mag_f182m=12.0, sat_f182m=True),
        dict(dra=0.3, mag_f212n=11.5),
    ])
    assert len(out) == 1
    assert np.isfinite(out['mag_vega_f182m'][0]) and np.isfinite(out['mag_vega_f212n'][0])


def test_dedup_unsaturated_pairs_keep_tight_radius(tmp_path):
    """The SAME complementary pair 0.3" apart WITHOUT saturation stays split:
    the wide radius applies only to satstar-involved pairs."""
    out = _run_dedup(tmp_path, [
        dict(dra=0.0, mag_f182m=12.0),
        dict(dra=0.3, mag_f212n=11.5),
    ])
    assert len(out) == 2


def test_dedup_satstar_binary_protection(tmp_path):
    """Satstar pair BOTH detected in f182m with discrepant mags 0.3" apart:
    the same-band collision blocks the merge (resolved pair preserved)."""
    out = _run_dedup(tmp_path, [
        dict(dra=0.0, mag_f182m=12.0, sat_f182m=True),
        dict(dra=0.3, mag_f182m=13.5, mag_f212n=11.5),
    ])
    assert len(out) == 2


def test_dedup_wide_satstar_pair_sharing_a_band_stays_split(tmp_path):
    """Two rows 0.3" apart, both detected in f182m and f212n and both
    saturated in f182m with similar mags: two satstar fits of two resolved
    stars (wd2, #1038)."""
    out = _run_dedup(tmp_path, [
        dict(dra=0.0, mag_f182m=12.0, sat_f182m=True, mag_f212n=11.6),
        dict(dra=0.3, mag_f182m=12.1, sat_f182m=True, mag_f212n=11.5),
    ])
    assert len(out) == 2
    assert sorted(np.round(out['mag_vega_f182m'], 2)) == [12.0, 12.1]


def test_dedup_wide_pair_unsaturated_in_a_shared_band_stays_split(tmp_path):
    """A saturated row and a neighbour 0.3" away, both detected in f182m and
    f212n, both unsaturated in f212n with similar mags: two daophot
    detections of two stars."""
    out = _run_dedup(tmp_path, [
        dict(dra=0.0, mag_f182m=12.0, sat_f182m=True, mag_f212n=11.6),
        dict(dra=0.3, mag_f182m=12.2, mag_f212n=11.5),
    ])
    assert len(out) == 2


def test_dedup_wide_one_band_satstar_fragment_merges(tmp_path):
    """A one-band saturated row 0.25" from a two-band row saturated in the
    same band: a fragment of the same star (on wd2 dolphot has one star for
    157 of 165 such pairs), merged at sat_dmag_collision."""
    out = _run_dedup(tmp_path, [
        dict(dra=0.0, mag_f182m=12.0, sat_f182m=True, mag_f212n=11.5),
        dict(dra=0.25, mag_f182m=12.2, sat_f182m=True),
    ])
    assert len(out) == 1


def test_dedup_wide_satstar_daophot_duplicate_merges(tmp_path):
    """The satstar fit and a daophot detection of the SAME star 0.13" apart,
    detected in two bands and saturated in only one row of each (cloudc: a
    coherent 0.13" offset between the two): still merged at
    sat_dmag_collision."""
    out = _run_dedup(tmp_path, [
        dict(dra=0.0, mag_f182m=12.0, sat_f182m=True, mag_f212n=11.4,
             sat_f212n=True),
        dict(ddec=0.13, mag_f182m=12.2, mag_f212n=11.5),
    ])
    assert len(out) == 1


def test_dedup_close_satstar_pair_keeps_loose_threshold(tmp_path):
    """Within link_radius a saturated pair sharing a band still merges at
    dmag < sat_dmag_collision (the same star's two rows)."""
    out = _run_dedup(tmp_path, [
        dict(dra=0.0, mag_f182m=12.0, sat_f182m=True),
        dict(dra=0.05, mag_f182m=12.3, mag_f212n=11.5),
    ])
    assert len(out) == 1


def test_dedup_sat_radius_disabled(tmp_path):
    """sat_link_radius=0 restores the old behaviour (0.3" satstar pair split)."""
    out = _run_dedup(tmp_path, [
        dict(dra=0.0, mag_f182m=12.0, sat_f182m=True),
        dict(dra=0.3, mag_f212n=11.5),
    ], sat_link_radius=0 * u.arcsec)
    assert len(out) == 2


def test_fill_guard_vetoes_satstar_with_nearby_detection():
    """Row 0: saturated (f182m), non-detection in f212n, real f212n detection
    (row 1) 0.3" away -> vetoed.  Row 2: unsaturated non-detection near the
    same detection -> NOT vetoed (ordinary fill target)."""
    t = _merged_cat([
        dict(dra=0.0, mag_f182m=12.0, sat_f182m=True),
        dict(dra=0.3, mag_f212n=11.5),
        dict(dra=10.0, mag_f182m=18.0),
    ])
    ref = SkyCoord(t['skycoord_ref'])
    targets = np.array([True, False, True])   # f212n fill targets
    veto = _satstar_partner_guard(t, 'f212n', ref, targets)
    assert list(veto) == [True, False, False]


def test_fill_guard_no_detection_nearby():
    """Saturated row with NO f212n detection within the guard radius: filling
    is a genuine limit -> not vetoed."""
    t = _merged_cat([
        dict(dra=0.0, mag_f182m=12.0, sat_f182m=True),
        dict(dra=5.0, mag_f212n=11.5),
    ])
    ref = SkyCoord(t['skycoord_ref'])
    veto = _satstar_partner_guard(t, 'f212n', ref, np.array([True, False]))
    assert not veto.any()


def test_color_reliable_mask():
    t = Table({
        'forced_filled_f182m':     [False, True,  False, True,  False],
        'forced_filled_f212n':     [False, False, True,  False, True],
        'replaced_saturated_f182m': [False, False, True,  False, False],
        'replaced_saturated_f212n': [False, True,  False, False, False],
        'is_saturated_f182m':      [False, False, False, False, False],
        'is_saturated_f212n':      [False, False, False, False, False],
    })
    m = color_reliable_mask(t, 'f182m', 'f212n')
    #     clean  fill+satOther  satOther+fill  fill-only  fill-only
    assert list(m) == [True, False, False, True, True]
    # catalog with none of the columns: everything reliable
    t2 = Table({'mag_vega_f182m': [1.0, 2.0]})
    assert list(color_reliable_mask(t2, 'f182m', 'f212n')) == [True, True]
