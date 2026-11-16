"""Cross-epoch seeds stay in the fit; the catalog lists them by production rules.

With ``--xepoch-field`` the m7 seed of one epoch carries the other epoch's
vetted merged catalog as ``keep_in_fit`` rows.  A secondary seeded only from
the other epoch, closer to a brighter primary than the seed-dedup radius
(0.5 FWHM), is removed by the production seed dedup, so the primary is fit
alone and takes the secondary's light.  Kept in the fit, the pair is fitted
jointly; the catalog still lists the secondary only where production would.
"""
import types

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.nddata import NDData
from astropy.table import Table
from astropy.wcs import WCS
from photutils.psf import GriddedPSFModel, SourceGrouper

from jwst_gc_pipeline.photometry import cataloging as C
from jwst_gc_pipeline.photometry.psf_fitting import _dedup_close_sources

FWHM = 2.5
SHAPE = (80, 80)
PRIMARY = (40.3, 40.2)
F_PRIMARY = 30000.0
F_SECONDARY = 0.5 * F_PRIMARY


def _gaussian_grid_psf():
    ov = 4
    n = 25 * ov + 1
    c = (n - 1) / 2
    yy, xx = np.mgrid[:n, :n]
    sig = FWHM / 2.3548 * ov
    p = np.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2 * sig ** 2))
    p /= p.sum() / ov ** 2
    return GriddedPSFModel(NDData(p[None], meta={'grid_xypos': [(0, 0)],
                                                 'oversampling': ov}))


def _wcs():
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crval = [266.6, -28.56]
    w.wcs.crpix = [40.0, 40.0]
    w.wcs.cdelt = [-0.031 / 3600, 0.031 / 3600]
    w.pixel_shape = SHAPE[::-1]
    return w


def _scene(sep_pix):
    """Primary + a secondary ``sep_pix`` away (half its flux), PSF-shaped."""
    secondary = (PRIMARY[0] + sep_pix, PRIMARY[1])
    yy, xx = np.mgrid[:SHAPE[0], :SHAPE[1]]
    sig = FWHM / 2.3548
    img = np.full(SHAPE, 10.0)
    for (x, y), f in ((PRIMARY, F_PRIMARY), (secondary, F_SECONDARY)):
        img += f / (2 * np.pi * sig ** 2) * np.exp(
            -((xx - x) ** 2 + (yy - y) ** 2) / (2 * sig ** 2))
    img += np.random.default_rng(3).normal(0, 1.0, SHAPE)
    return img, secondary


def _cat(ww, xy, flux):
    t = Table()
    t['skycoord'] = ww.pixel_to_world([p[0] for p in xy], [p[1] for p in xy])
    t['flux'] = np.asarray(flux, dtype=float)
    return t


def _run(sep_pix, xepoch):
    """Seed (this epoch: primary only; other epoch: primary + secondary when
    ``xepoch``) -> one manual pass.  Returns the catalog and the secondary."""
    ww = _wcs()
    img, secondary = _scene(sep_pix)
    prev = _cat(ww, [PRIMARY], [F_PRIMARY])
    other = (_cat(ww, [PRIMARY, secondary], [F_PRIMARY, F_SECONDARY])
             if xepoch else None)
    mask = np.zeros(SHAPE, bool)
    # local_snr_threshold far above any peak: no daofind detection enters
    # the seed, so the seed is exactly prev (+ the other epoch)
    seed = C._build_manual_seed(
        detection_image=img, nan_replaced_data=img, mask=mask, ww=ww,
        fwhm_pix=FWHM, satstar_table=None, prev_catalog=prev,
        local_snr_threshold=1e12, roundlo=-1, roundhi=1, sharplo=0.2,
        sharphi=1.0, dedup_min_sep_pix=0.5 * FWHM, label='t',
        xepoch_catalog=other)
    opts = types.SimpleNamespace(group=False, manual_group_fit=True,
                                 satstar_artifact_sigK=3.0,
                                 satstar_artifact_ratio=1.0)
    res, modsky, _ = C._manual_phot_pass(
        data=img, mask=mask, err=np.ones(SHAPE), bad=mask,
        dao_psf_model=_gaussian_grid_psf(), init_params=seed,
        aperture_radius_pix=2 * FWHM, localbkg_inner=6, localbkg_outer=10,
        grouper=SourceGrouper(2 * FWHM), options=opts,
        dq=np.zeros(SHAPE, np.uint32), satstar_model_subtracted=np.zeros(SHAPE),
        label='t', near_sat_dist_pix=1.0)
    return seed, res, modsky, secondary


def _nearest(res, xy):
    d = np.hypot(np.asarray(res['x_fit']) - xy[0],
                 np.asarray(res['y_fit']) - xy[1])
    return res[int(np.argmin(d))], float(np.min(d))


# ---------------------------------------------------------------------------
# the close pair through _build_manual_seed + _manual_phot_pass
# ---------------------------------------------------------------------------

def test_close_secondary_stays_in_fit_and_is_not_listed():
    # 1.1 px < the 0.5 FWHM = 1.25 px seed-dedup radius: production removes
    # the secondary's seed (fainter) and the primary absorbs its light.
    seed, res, _, secondary = _run(1.1, xepoch=True)
    assert len(seed) == 2, 'both other-epoch seeds survive the seed dedup'
    assert list(np.asarray(seed['keep_in_fit'])) == [True, True]
    # (this epoch's duplicate of the primary left the fit: it is covered)
    listed = dict(zip(np.round(seed['x_init'], 2), seed['keep_in_fit_listed']))
    assert listed[round(PRIMARY[0], 2)] and not listed[round(secondary[0], 2)]
    # listing: one row, the primary, fitted jointly with the secondary
    assert len(res) == 1
    row, d = _nearest(res, PRIMARY)
    assert d < 0.1
    assert float(row['flux_fit']) == pytest.approx(F_PRIMARY, rel=0.05)
    assert bool(row['keep_in_fit'])
    assert res.meta['KIFFIT'][0] == 2 and res.meta['KIFLIST'][0] == 1


def test_close_secondary_production_fits_the_blend():
    # the same scene without the other epoch: the primary takes the pair's light
    seed, res, _, _ = _run(1.1, xepoch=False)
    assert 'keep_in_fit' not in seed.colnames
    assert 'keep_in_fit' not in res.colnames
    assert len(res) == 1
    row, d = _nearest(res, PRIMARY)
    assert float(row['flux_fit']) > 1.3 * F_PRIMARY
    assert d > 0.2


def test_separated_secondary_is_listed():
    # 2 FWHM apart production keeps both seeds, so the listing keeps both
    seed, res, _, secondary = _run(2 * FWHM, xepoch=True)
    assert np.all(np.asarray(seed['keep_in_fit_listed']))
    assert len(res) == 2
    for xy, f in ((PRIMARY, F_PRIMARY), (secondary, F_SECONDARY)):
        row, d = _nearest(res, xy)
        assert d < 0.1
        assert float(row['flux_fit']) == pytest.approx(f, rel=0.05)


def test_unlisted_row_is_not_in_the_model():
    # the returned model is the listed catalog's: the unlisted secondary's
    # light stays in data - model at its position
    _, _, modsky, secondary = _run(1.1, xepoch=True)
    img, _ = _scene(1.1)
    ix, iy = int(round(secondary[0] + 1)), int(round(secondary[1]))
    assert img[iy, ix] - 10.0 - modsky[iy, ix] > 50.0


# ---------------------------------------------------------------------------
# _seed_dedup_keep_in_fit: the listing rule
# ---------------------------------------------------------------------------

def test_keep_in_fit_dedup_without_markers_is_production():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 20, (60, 2))
    flux = rng.uniform(1, 100, 60)
    prod, n_dis = _dedup_close_sources(xy, flux, 1.25)
    fit, listed, n_dis2, holes = C._seed_dedup_keep_in_fit(
        xy, flux, 1.25, None, np.zeros(60, bool))
    assert fit.tolist() == prod.tolist()
    assert listed.tolist() == prod.tolist()
    assert (n_dis2, holes) == (n_dis, 0)


def test_stand_in_for_an_unfitted_winner():
    # A (bright, this epoch) and P (keep_in_fit) 0.8 px apart: production
    # keeps A, the keep run fits P and not A, so P stands in for A.  Q is a
    # second keep_in_fit row of the cluster, farther from A: not listed.
    xy = np.array([[10.0, 10.0], [10.8, 10.0], [9.0, 10.0]])
    flux = np.array([100.0, 50.0, 40.0])
    prot = np.array([False, True, True])
    fit, listed, _, holes = C._seed_dedup_keep_in_fit(xy, flux, 1.25, None, prot)
    assert fit.tolist() == [False, True, True]
    assert listed.tolist() == [False, True, False]
    assert holes == 0


def test_no_stand_in_when_the_winner_is_fitted():
    # the winner is a saturated seed, never removed by the keep_in_fit rows
    xy = np.array([[10.0, 10.0], [10.8, 10.0]])
    flux = np.array([np.nan, 50.0])
    fit, listed, _, _ = C._seed_dedup_keep_in_fit(
        xy, flux, 1.25, np.array([True, False]), np.array([False, True]))
    assert fit.tolist() == [True, True]
    assert listed.tolist() == [True, False]


def test_hole_is_counted():
    # A is near keep_in_fit P (so not fitted) but P belongs to saturated B's
    # cluster (a saturated seed is never left out): A's production fit has no
    # stand-in.
    xy = np.array([[10.0, 10.0], [11.2, 10.0], [12.4, 10.0]])
    flux = np.array([200.0, 50.0, 100.0])  # B, P, A
    sat = np.array([True, False, False])
    prot = np.array([False, True, False])
    fit, listed, _, holes = C._seed_dedup_keep_in_fit(xy, flux, 1.25, sat, prot)
    assert fit.tolist() == [True, True, False]
    assert listed.tolist() == [True, False, False]
    assert holes == 1


# ---------------------------------------------------------------------------
# _manual_phot_pass rules on keep_in_fit rows
# ---------------------------------------------------------------------------

def test_nonpositive_keep_in_fit_row_stays_in_fit_but_not_listed():
    # a keep_in_fit seed on empty sky fits to flux <= 0: it is not listed,
    # and the star's row is unaffected
    yy, xx = np.mgrid[:SHAPE[0], :SHAPE[1]]
    sig = FWHM / 2.3548
    img = 10.0 + F_PRIMARY / (2 * np.pi * sig ** 2) * np.exp(
        -((xx - PRIMARY[0]) ** 2 + (yy - PRIMARY[1]) ** 2) / (2 * sig ** 2))
    img = img - 3.0 * np.exp(-((xx - 20) ** 2 + (yy - 60) ** 2) / 8.0)
    seed = Table()
    seed['x_init'] = [PRIMARY[0], 20.0]
    seed['y_init'] = [PRIMARY[1], 60.0]
    seed['flux_init'] = [F_PRIMARY, 100.0]
    seed['keep_in_fit'] = [False, True]
    seed['keep_in_fit_listed'] = [True, True]
    opts = types.SimpleNamespace(group=False, manual_group_fit=True,
                                 satstar_artifact_sigK=3.0,
                                 satstar_artifact_ratio=1.0)
    res, _, _ = C._manual_phot_pass(
        data=img, mask=np.zeros(SHAPE, bool), err=np.ones(SHAPE),
        bad=np.zeros(SHAPE, bool), dao_psf_model=_gaussian_grid_psf(),
        init_params=seed, aperture_radius_pix=2 * FWHM, localbkg_inner=6,
        localbkg_outer=10, grouper=SourceGrouper(2 * FWHM), options=opts,
        dq=np.zeros(SHAPE, np.uint32), satstar_model_subtracted=np.zeros(SHAPE),
        label='t', near_sat_dist_pix=1.0)
    assert len(res) == 1
    assert not bool(res['keep_in_fit'][0])
    assert res.meta['KIFLIST'][0] == 0


# ---------------------------------------------------------------------------
# xepoch_seed_catalog_path: off by default, names the writer's file, fails loud
# ---------------------------------------------------------------------------

def _opts(**kw):
    base = dict(desaturated=False, bgsub=False, blur=False, cutout_region='')
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_path_off_by_default():
    assert C.xepoch_seed_catalog_path(_opts(), 'F212N') is None


def test_path_needs_all_three():
    with pytest.raises(ValueError, match='xepoch_target'):
        C.xepoch_seed_catalog_path(
            _opts(xepoch_field='107', xepoch_proposal_id='10678'), 'F212N')


def test_path_is_the_m7_vetted_merged_name(tmp_path, monkeypatch):
    from jwst_gc_pipeline import fields
    monkeypatch.setattr(fields, 'basepath', lambda target: f'{tmp_path}/')
    o = _opts(xepoch_field='107', xepoch_proposal_id='10678',
              xepoch_target='gc-treasury')
    expect = (tmp_path / 'catalogs' /
              'f212n_merged_o107_indivexp_merged_resbgsub_m7_dao_basic_vetted.fits')
    with pytest.raises(FileNotFoundError, match='m7 vetted merged'):
        C.xepoch_seed_catalog_path(o, 'F212N')
    expect.parent.mkdir()
    expect.write_text('')
    assert C.xepoch_seed_catalog_path(o, 'F212N').replace('//', '/') == str(expect)
    # a single-observation proposal carries no observation token
    o2 = _opts(xepoch_field='2', xepoch_proposal_id='2221',
               xepoch_target='cloudc')
    expect2 = (tmp_path / 'catalogs' /
               'f212n_merged_indivexp_merged_resbgsub_m7_dao_basic_vetted.fits')
    expect2.write_text('')
    assert C.xepoch_seed_catalog_path(o2, 'F212N').replace('//', '/') == str(expect2)
