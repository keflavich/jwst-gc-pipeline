"""m7 seed of one filter = cross-band seed UNION that filter's m6 vetted catalog
(``manual_m7_seed_own_band``; AUTO by default: on for star-dominated NIRCam
fields, off on an extended-emission target and for MIRI filters).

The >=2-filter cross-band seed alone leaves out every source only ONE band's
m6 vetting accepted and m7 does not find again (a third of the m6 vetted
catalog in Brick F182M).  Own-band sources near a brighter seed source are not
added (companion cut).
"""
import ast
import inspect
import os
import textwrap
import types

import numpy as np
import pytest
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from jwst_gc_pipeline.photometry import cataloging
from jwst_gc_pipeline.photometry.band_option import band_mopt
from jwst_gc_pipeline.photometry.cataloging import (
    _auto_m7_seed_own_band, _build_i2d_augmented_seed, _build_m7_band_seed,
    _dedup_m7_band_seed,
    annotate_independent_detection, crossband_seed_file, m7_band_seed_path)
from jwst_gc_pipeline.photometry.naming import vetted_to_i2dseed
from jwst_gc_pipeline.photometry.manual_defaults import MANUAL_DEFAULTS

RA0, DEC0 = 266.5, -28.8


def _sc(dx_mas, dy_mas):
    dx = np.asarray(dx_mas, float) / 3.6e6 / np.cos(np.deg2rad(DEC0))
    dy = np.asarray(dy_mas, float) / 3.6e6
    return SkyCoord((RA0 + dx) * u.deg, (DEC0 + dy) * u.deg)


def _write(tmp_path):
    # cross-band seed: three positions, no flux column (as _build_crossband_seed writes it)
    xb = Table({'skycoord': _sc([0, 1000, 2000], [0, 0, 0]),
                'n_filt_confirmed': [2, 2, 3]})
    xpath = os.path.join(tmp_path, 'crossband_seed_manual.fits')
    xb.write(xpath)
    # own-band m6 vetted: a 10 mas neighbour of seed 0, a 25 mas neighbour of
    # seed 1, two single-band sources, and one non-positive flux row
    own = Table({'skycoord': _sc([10, 1025, 5000, 6000, 7000], [0, 0, 0, 0, 0]),
                 'flux': [50.0, 70.0, 3.0, 4.0, -1.0]})
    opath = os.path.join(tmp_path, 'own_m6.fits')
    own.write(opath)
    return xpath, opath


def test_union_and_flux_transfer(tmp_path):
    xpath, opath = _write(str(tmp_path))
    out = Table.read(_build_m7_band_seed(xpath, opath, 'F182M', 'merged',
                                         max_sep_mas=30.0))
    origin = np.asarray(out['seed_origin']).astype(str)
    assert list(origin) == ['crossband'] * 3 + ['own_m6'] * 2
    # cross-band positions take the matched own-band flux, unmatched -> 1.0
    assert list(out['flux'][:3]) == [50.0, 70.0, 1.0]
    # the single-band sources are added with their m6 flux; the negative-flux
    # row and the two matched rows are not duplicated
    assert sorted(out['flux'][3:]) == [3.0, 4.0]
    sc = out['skycoord']
    assert np.allclose(sc[:3].ra.deg, _sc([0, 1000, 2000], [0, 0, 0]).ra.deg)


def test_max_sep_controls_dedup(tmp_path):
    xpath, opath = _write(str(tmp_path))
    out = Table.read(_build_m7_band_seed(xpath, opath, 'F182M', 'merged',
                                         max_sep_mas=20.0, dedup_fwhm=0))
    # the 25 mas neighbour is now a separate own-band source
    assert len(out) == 6
    assert list(out['flux'][:3]) == [50.0, 1.0, 1.0]


def _seed_files(tmp_path, xdx, odx, oflux):
    """Cross-band seed at offsets ``xdx`` (mas, along RA) and own-band m6
    catalogue at ``odx`` with ``oflux``."""
    xpath = os.path.join(str(tmp_path), 'crossband_seed_manual.fits')
    Table({'skycoord': _sc(xdx, np.zeros(len(xdx))),
           'n_filt_confirmed': [2] * len(xdx)}).write(xpath, overwrite=True)
    opath = os.path.join(str(tmp_path), 'own_m6.fits')
    Table({'skycoord': _sc(odx, np.zeros(len(odx))), 'flux': oflux}).write(opath, overwrite=True)
    return xpath, opath


def _band_seed(xpath, opath, **kw):
    # companion cut off: these tests isolate the seed dedup; the dedup is off
    # by default, so the tests that exercise it ask for the intended 0.5
    kw.setdefault('dedup_fwhm', 0.5)
    return Table.read(_build_m7_band_seed(xpath, opath, 'F182M', 'merged',
                                          companion_fwhm=0, **kw))


def test_dedup_absorbs_own_band_beyond_max_sep(tmp_path):
    """45 mas is beyond the 30 mas match radius but inside 0.5 FWHM of F405N
    (68 mas): one row carrying the own-band flux, not a cross-band row at flux
    1.0 plus an own-band row (regime 1 of #1206)."""
    xpath, opath = _seed_files(tmp_path, [0], [45], [50.0])
    out = Table.read(_build_m7_band_seed(xpath, opath, 'F405N', 'merged',
                                         max_sep_mas=30.0, companion_fwhm=0,
                                         dedup_fwhm=0.5))
    assert list(np.asarray(out['seed_origin']).astype(str)) == ['crossband']
    assert list(out['flux']) == [50.0]
    # dedup_fwhm=0: the match radius is max_sep_mas alone, as before
    out0 = Table.read(_build_m7_band_seed(xpath, opath, 'F405N', 'merged',
                                          max_sep_mas=30.0, companion_fwhm=0,
                                          dedup_fwhm=0))
    assert list(np.asarray(out0['seed_origin']).astype(str)) == ['crossband', 'own_m6']
    assert list(out0['flux']) == [1.0, 50.0]
    # and so does the default: the dedup is opt-in until the production seeds
    # are rerun
    outd = Table.read(_build_m7_band_seed(xpath, opath, 'F405N', 'merged',
                                          max_sep_mas=30.0, companion_fwhm=0))
    assert list(outd['flux']) == [1.0, 50.0]
    assert outd.meta['DEDUPFWH'] == 0.0


def test_dedup_own_band_pair_keeps_the_brighter(tmp_path):
    """Two own-band rows 0.3 FWHM apart (F182M: 18.6 mas), no cross-band
    position near them: the brighter stays."""
    xpath, opath = _seed_files(tmp_path, [5000], [0, 18.6], [8.0, 20.0])
    out = _band_seed(xpath, opath, fwhm_arcsec=0.062)
    assert list(np.asarray(out['seed_origin']).astype(str)) == ['crossband', 'own_m6']
    assert list(out['flux']) == [1.0, 20.0]
    assert out.meta['NDEDUP'] == 1
    assert np.isclose(out['skycoord'][1].ra.deg, _sc([18.6], [0]).ra.deg[0])


def test_dedup_leaves_pairs_beyond_the_radius(tmp_path):
    """0.8 FWHM apart (49.6 mas in F182M) are two stars: both rows stay, and
    so does a cross-band position with its own-band flux."""
    xpath, opath = _seed_files(tmp_path, [5000], [0, 49.6], [8.0, 20.0])
    out = _band_seed(xpath, opath, fwhm_arcsec=0.062)
    assert list(out['flux']) == [1.0, 8.0, 20.0]
    assert out.meta['NDEDUP'] == 0
    # a cross-band position 0.8 FWHM from an own-band row does not absorb it
    xpath, opath = _seed_files(tmp_path, [0], [49.6], [8.0])
    out = _band_seed(xpath, opath, fwhm_arcsec=0.062, max_sep_mas=30.0)
    assert list(np.asarray(out['seed_origin']).astype(str)) == ['crossband', 'own_m6']
    assert list(out['flux']) == [1.0, 8.0]


def test_dedup_zero_reproduces_the_legacy_seed(tmp_path):
    xpath, opath = _write(str(tmp_path))
    out = Table.read(_build_m7_band_seed(xpath, opath, 'F182M', 'merged',
                                         max_sep_mas=30.0, dedup_fwhm=0))
    assert list(out['flux']) == [50.0, 70.0, 1.0, 3.0, 4.0]
    assert out.meta['NDEDUP'] == 0
    assert out.meta['DEDUPFWH'] == 0.0
    # the fixture's rows are all >= 25 mas apart or on a cross-band position:
    # the opt-in dedup (0.5 FWHM = 31 mas) leaves the same single-band rows
    outd = Table.read(_build_m7_band_seed(xpath, opath, 'F182M', 'merged',
                                          dedup_fwhm=0.5))
    assert list(outd['flux']) == [50.0, 70.0, 1.0, 3.0, 4.0]
    assert outd.meta['DEDUPFWH'] == 0.5
    # the default is the legacy matching
    outdef = Table.read(_build_m7_band_seed(xpath, opath, 'F182M', 'merged'))
    assert list(outdef['flux']) == [50.0, 70.0, 1.0, 3.0, 4.0]
    assert outdef.meta['DEDUPFWH'] == MANUAL_DEFAULTS['manual_m7_seed_dedup_fwhm'] == 0.0


def test_dedup_mutual_best_keeps_the_farther_own_band_row(tmp_path):
    """Two own-band rows within the match radius of one cross-band position:
    the nearer donates its flux, the farther is not dropped unused.  It stays
    an own-band seed unless it lies within dedup_fwhm of the cross-band
    position or the donor."""
    # max_sep_mas=60 widens the match radius past the 31 mas dedup radius
    xpath, opath = _seed_files(tmp_path, [0], [5, 50], [100.0, 7.0])
    out = _band_seed(xpath, opath, fwhm_arcsec=0.062, max_sep_mas=60.0)
    assert list(np.asarray(out['seed_origin']).astype(str)) == ['crossband', 'own_m6']
    assert list(out['flux']) == [100.0, 7.0]
    # 25 mas from the cross-band position: a duplicate of it, dropped
    xpath, opath = _seed_files(tmp_path, [0], [5, -25], [100.0, 7.0])
    out = _band_seed(xpath, opath, fwhm_arcsec=0.062, max_sep_mas=60.0)
    assert list(out['flux']) == [100.0]
    assert out.meta['NDEDUP'] == 1
    # within dedup_fwhm of the donor, beyond it from the cross-band position
    # (F182M: 31 mas): 20 mas from the donor at -20 mas, 35 mas from the position
    xpath, opath = _seed_files(tmp_path, [0], [15, 35, 400], [100.0, 7.0, 9.0])
    out = _band_seed(xpath, opath, fwhm_arcsec=0.062, max_sep_mas=60.0)
    assert list(out['flux']) == [100.0, 9.0]


def test_dedup_zero_legacy_when_two_own_rows_are_near_one_cross_band_position(tmp_path):
    """The one geometry where the new path differs from the old: two own-band
    rows inside max_sep_mas of one cross-band position.  Legacy: the nearer
    donates, both are left out.  The new path (mutual-best) would keep the
    farther one, so ``dedup_fwhm=0`` must not run it (pins ``dedup_fwhm > 0``)."""
    xpath, opath = _seed_files(tmp_path, [0], [5, -25], [100.0, 7.0])
    out = _band_seed(xpath, opath, fwhm_arcsec=0.062, max_sep_mas=30.0, dedup_fwhm=0)
    assert list(out['flux']) == [100.0]
    assert out.meta['NDEDUP'] == 0


def test_dedup_row_near_cross_band_position_but_not_its_donor(tmp_path):
    """F182M radii 31 / 31 mas.  Own rows at -24 mas (flux 100, the donor) and
    +25 mas (flux 7): the second is 25 mas from the cross-band position but
    49 mas from the donor, so only the 'near a cross-band position' rule
    drops it."""
    xpath, opath = _seed_files(tmp_path, [0], [-24, 25], [100.0, 7.0])
    out = _band_seed(xpath, opath, fwhm_arcsec=0.062)
    assert list(out['flux']) == [100.0]
    assert out.meta['NDEDUP'] == 1


def test_dedup_row_dropped_near_cross_band_does_not_suppress_its_neighbour(tmp_path):
    """A row dropped only for lying near a cross-band position is not a kept
    row: its fainter neighbour 25 mas away (74 mas from the donor, 50 mas from
    the cross-band position) stays."""
    xpath, opath = _seed_files(tmp_path, [0], [-24, 25, 50], [100.0, 7.0, 3.0])
    out = _band_seed(xpath, opath, fwhm_arcsec=0.062)
    assert list(out['flux']) == [100.0, 3.0]
    assert out.meta['NDEDUP'] == 1


def test_dedup_equal_flux_pair_keeps_the_same_position_in_either_order(tmp_path):
    """Equal fluxes are ordered by position, not by catalogue row: permuting
    the own-band catalogue keeps the same star."""
    kept = []
    for odx in ([0, 10], [10, 0]):
        xpath, opath = _seed_files(tmp_path, [5000], odx, [5.0, 5.0])
        out = _band_seed(xpath, opath, fwhm_arcsec=0.062)
        assert out.meta['NDEDUP'] == 1
        kept.append(out['skycoord'][1].ra.deg)
    assert np.isclose(kept[0], kept[1], rtol=0, atol=1e-12)
    assert np.isclose(kept[0], _sc([0], [0]).ra.deg[0], rtol=0, atol=1e-9)


def test_dedup_flux_donation_is_many_to_one(tmp_path):
    """Documented behaviour (not a recommendation): each cross-band position
    takes the flux of ITS nearest own-band source, so a faint cross-band
    neighbour 55 mas from a bright one inherits the bright source's flux when
    the radius is 0.5 FWHM of F405N (68 mas); at the legacy 30 mas it keeps
    1.0."""
    xpath, opath = _seed_files(tmp_path, [0, 55], [5], [500.0])
    out = _band_seed(xpath, opath, fwhm_arcsec=0.136, max_sep_mas=30.0)
    assert list(out['flux']) == [500.0, 500.0]
    out0 = _band_seed(xpath, opath, fwhm_arcsec=0.136, max_sep_mas=30.0, dedup_fwhm=0)
    assert list(out0['flux']) == [500.0, 1.0]


def test_dedup_faint_donor_drops_a_brighter_row_near_it(tmp_path):
    """Documented behaviour: donors are kept before the brightest-first pass,
    so a faint donor (flux 2, 5 mas from the cross-band position) drops a
    brighter non-donor row 30 mas from it (35 mas from the cross-band
    position, outside the 31 mas radius) and that row's flux is not used."""
    xpath, opath = _seed_files(tmp_path, [0], [-5, -35], [2.0, 100.0])
    out = _band_seed(xpath, opath, fwhm_arcsec=0.062, max_sep_mas=30.0)
    assert list(out['flux']) == [2.0]
    assert out.meta['NDEDUP'] == 1


def test_dedup_log_names_the_radius_in_use(tmp_path, capsys):
    """With the dedup on, the exclusion radius in the log line is the match
    radius (68 mas in F405N), not max_sep_mas; with it off there is no dedup
    clause at all (no '0 within 0 FWHM of a seed position')."""
    xpath, opath = _seed_files(tmp_path, [0], [45], [50.0])
    _build_m7_band_seed(xpath, opath, 'F405N', 'merged', max_sep_mas=30.0,
                        companion_fwhm=0, dedup_fwhm=0.5, fwhm_arcsec=0.136)
    msg = capsys.readouterr().out
    assert 'not within 68 mas of a cross-band position' in msg
    assert '0 within 0.5 FWHM (68 mas)' in msg
    _build_m7_band_seed(xpath, opath, 'F405N', 'merged', max_sep_mas=30.0,
                        companion_fwhm=0, dedup_fwhm=0)
    msg = capsys.readouterr().out
    assert 'not within 30 mas of a cross-band position' in msg
    assert 'of a seed position' not in msg


def test_dedup_helper_rejects_an_empty_own_band_array():
    empty = _sc([], [])
    with pytest.raises(ValueError, match='at least one own-band row'):
        _dedup_m7_band_seed(_sc([0], [0]), empty, np.zeros(0),
                            match_mas=30.0, dedup_mas=30.0)


def test_dedup_unknown_filter_names_the_filter_and_the_table(tmp_path):
    """A filter missing from the FWHM table fails with a message that names it
    (it was an IndexError), unless fwhm_arcsec is given."""
    xpath, opath = _seed_files(tmp_path, [0], [45], [50.0])
    with pytest.raises(ValueError, match='F999W is not in the FWHM table'):
        _build_m7_band_seed(xpath, opath, 'F999W', 'merged', dedup_fwhm=0.5)
    out = Table.read(_build_m7_band_seed(xpath, opath, 'F999W', 'merged',
                                         dedup_fwhm=0.5, fwhm_arcsec=0.1,
                                         companion_fwhm=0))
    assert len(out) == 1
    # dedup and the companion cut off: the table is never read
    _build_m7_band_seed(xpath, opath, 'F999W', 'merged', dedup_fwhm=0,
                        companion_fwhm=0)


def test_dedup_empty_own_catalog_unchanged(tmp_path):
    xpath, _ = _write(str(tmp_path))
    epath = os.path.join(str(tmp_path), 'empty.fits')
    Table({'skycoord': _sc([], []), 'flux': np.zeros(0)}).write(epath)
    out = Table.read(_build_m7_band_seed(xpath, epath, 'F182M', 'merged', dedup_fwhm=0.5))
    assert list(out['flux']) == [1.0, 1.0, 1.0]
    assert out.meta['NDEDUP'] == 0


def test_dedup_negative_rejected(tmp_path):
    xpath, opath = _write(str(tmp_path))
    with pytest.raises(ValueError):
        _build_m7_band_seed(xpath, opath, 'F182M', 'merged', dedup_fwhm=-0.1)


def test_output_path_per_module_and_filter(tmp_path):
    xpath, opath = _write(str(tmp_path))
    pa = _build_m7_band_seed(xpath, opath, 'F182M', 'nrca')
    pb = _build_m7_band_seed(xpath, opath, 'F182M', 'nrcb')
    pc = _build_m7_band_seed(xpath, opath, 'F212N', 'nrca')
    assert len({pa, pb, pc}) == 3
    assert pa.endswith('crossband_seed_manual_nrca_f182m_vetted.fits')
    assert os.path.exists(xpath)            # the shared seed is not rewritten


def test_empty_own_catalog_still_writes_band_file(tmp_path):
    """The band seed is never the shared cross-band seed itself: the m7 caller
    adds the residual detections to it, and _build_i2d_augmented_seed writes
    next to a ``_vetted`` input (over a suffix-less one)."""
    xpath, _ = _write(str(tmp_path))
    before = Table.read(xpath)
    epath = os.path.join(str(tmp_path), 'empty.fits')
    Table({'skycoord': _sc([], []), 'flux': np.zeros(0)}).write(epath)
    out = _build_m7_band_seed(xpath, epath, 'F182M', 'merged')
    assert out != xpath
    assert out == m7_band_seed_path(xpath, 'merged', 'F182M')
    assert vetted_to_i2dseed(out) != out
    t = Table.read(out)
    assert list(np.asarray(t['seed_origin']).astype(str)) == ['crossband'] * 3
    assert list(t['flux']) == [1.0, 1.0, 1.0]
    after = Table.read(xpath)
    assert after.colnames == before.colnames and len(after) == len(before)


def test_i2d_augmented_seed_refuses_to_overwrite_its_input(tmp_path):
    xpath, _ = _write(str(tmp_path))
    with pytest.raises(ValueError, match='overwrite'):
        _build_i2d_augmented_seed('unused_i2d.fits', xpath, 'F182M')


def _write_i2d(path, stars, shape=(80, 80), sigma=0.9, noise=1.0, seed=0):
    """SCI/ERR/WHT co-add at 0.063"/px with Gaussian stars ``(x, y, peak)``."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    sci = rng.normal(0.0, noise, shape)
    for x, y, peak in stars:
        sci += peak * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crval = [RA0, DEC0]
    w.wcs.crpix = [shape[1] / 2, shape[0] / 2]
    w.wcs.cdelt = [-0.063 / 3600, 0.063 / 3600]
    hdr = w.to_header()
    fits.HDUList([fits.PrimaryHDU(),
                  fits.ImageHDU(sci, hdr, name='SCI'),
                  fits.ImageHDU(np.full(shape, noise), hdr, name='ERR'),
                  fits.ImageHDU(np.ones(shape), hdr, name='WHT')]).writeto(path)
    return w


def test_i2d_augmented_seed_marks_residual_detections(tmp_path):
    """Seed rows keep their origin; this round's residual detections are 'i2d',
    each at least DEDUPMAS from every earlier seed."""
    i2d = os.path.join(str(tmp_path), 'resid_i2d.fits')
    w = _write_i2d(i2d, [(20, 20, 200.0), (55, 30, 150.0), (40, 60, 120.0)])
    sky = w.pixel_to_world([20.0], [20.0])
    prev = Table({'skycoord': sky, 'flux': [500.0], 'seed_origin': ['own_m6']})
    ppath = os.path.join(str(tmp_path), 'band_seed_vetted.fits')
    prev.write(ppath)
    out = Table.read(_build_i2d_augmented_seed(i2d, ppath, 'F182M'))
    origin = list(np.asarray(out['seed_origin']).astype(str))
    assert origin == ['own_m6', 'i2d', 'i2d']
    assert out['flux'][0] == 500.0
    dedup = out.meta['DEDUPMAS']
    assert 60 < dedup < 200
    sc = out['skycoord']
    assert sc[1:].separation(sc[0]).to_value(u.mas).min() > dedup

    # a vetted catalog without seed_origin (m3..m6) reads as 'prev'
    del prev['seed_origin']
    prev.write(ppath, overwrite=True)
    out = Table.read(_build_i2d_augmented_seed(i2d, ppath, 'F182M'))
    assert list(np.asarray(out['seed_origin']).astype(str)) == ['prev', 'i2d', 'i2d']


def _annot_opts(**kw):
    o = types.SimpleNamespace(desaturated=False, bgsub=False, blur=False,
                              proposal_id='4147', field='012', modules='merged')
    for k, v in kw.items():
        setattr(o, k, v)
    return o


def test_annotate_counts_m7_residual_detections(tmp_path):
    """A source only the m7 residual daofind of one band found is an
    independent detection in that band (review of #1015: it was flagged False
    in the one band that detected it)."""
    cut_bp = str(tmp_path)
    os.makedirs(f'{cut_bp}/catalogs')
    opts = _annot_opts(manual_m7_seed_own_band=True)
    # m6 vetted F182M: source A only
    Table({'skycoord': _sc([0], [0]), 'flux': [100.0]}).write(
        f'{cut_bp}/catalogs/f182m_merged_indivexp_merged_resbgsub_m6_dao_basic_vetted.fits')
    # m7 F182M seed: A (own_m6), C (cross-band only), B (this band's residual)
    xb = crossband_seed_file(cut_bp, opts)
    seed = Table({'skycoord': _sc([0, 2000, 1000], [0, 0, 0]),
                  'flux': [100.0, 1.0, 5.0],
                  'seed_origin': ['own_m6', 'crossband', 'i2d']})
    seed.meta['DEDUPMAS'] = 63.0
    seed.write(vetted_to_i2dseed(m7_band_seed_path(xb, 'merged', 'F182M')))
    # merged catalog: A; B whose fit moved 45 mas from its residual centroid
    # (beyond the 30 mas m6 radius, inside DEDUPMAS); C
    mp = f'{cut_bp}/catalogs/merged_test.fits'
    Table({'skycoord_ref': _sc([5, 1045, 2000], [0, 0, 0])}).write(mp)

    annotate_independent_detection(mp, cut_bp, ['F182M'], opts)
    assert list(Table.read(mp)['independently_detected_f182m']) == [True, True, False]

    # own-band seed off: a band seed left by an earlier run is not read
    annotate_independent_detection(mp, cut_bp, ['F182M'],
                                   _annot_opts(manual_m7_seed_own_band=False))
    assert list(Table.read(mp)['independently_detected_f182m']) == [True, False, False]

    # AUTO (None): read on a star field, not on an extended-emission run
    annotate_independent_detection(mp, cut_bp, ['F182M'],
                                   _annot_opts(manual_m7_seed_own_band=None))
    assert list(Table.read(mp)['independently_detected_f182m']) == [True, True, False]
    annotate_independent_detection(mp, cut_bp, ['F182M'],
                                   _annot_opts(manual_m7_seed_own_band=None,
                                               extended_emission=True))
    assert list(Table.read(mp)['independently_detected_f182m']) == [True, False, False]


def test_companion_of_brighter_seed_not_added(tmp_path):
    """Own-band sources within companion_fwhm FWHM of a brighter seed source
    are m6 fits in its PSF-mismatch ring (confirmed by an independent visit at
    the chance rate in Brick F182M; docs/evidence/faint_m7_seed_union) and are
    not added.  A fainter neighbour does not exclude a brighter source."""
    xpath = os.path.join(str(tmp_path), 'crossband_seed_manual.fits')
    Table({'skycoord': _sc([0], [0]), 'n_filt_confirmed': [2]}).write(xpath)
    opath = os.path.join(str(tmp_path), 'own_m6.fits')
    # the cross-band star itself (5 mas), faint sources 100 and 300 mas from
    # it, an own-band-only star at 3000 mas and a faint source 120 mas from it
    Table({'skycoord': _sc([5, 100, 300, 3000, 3120], [0] * 5),
           'flux': [100.0, 5.0, 6.0, 80.0, 4.0]}).write(opath)
    # 2.5 x 0.062" = 155 mas: the 100 mas and 3120 mas sources are left out
    out = Table.read(_build_m7_band_seed(xpath, opath, 'F182M', 'merged',
                                         companion_fwhm=2.5, fwhm_arcsec=0.062))
    assert list(out['flux']) == [100.0, 6.0, 80.0]
    assert out.meta['NCOMPAN'] == 2
    # the FWHM defaults to the filter's FWHM-table entry (F182M: 0.062")
    out_t = Table.read(_build_m7_band_seed(xpath, opath, 'F182M', 'merged',
                                           companion_fwhm=2.5))
    assert list(out_t['flux']) == [100.0, 6.0, 80.0]
    # companion_fwhm=0 disables the cut
    out0 = Table.read(_build_m7_band_seed(xpath, opath, 'F182M', 'merged',
                                          companion_fwhm=0))
    assert list(out0['flux']) == [100.0, 5.0, 6.0, 80.0, 4.0]
    assert out0.meta['NCOMPAN'] == 0


def test_companion_cut_spares_bright_companions(tmp_path):
    """The cut leaves out only companions below companion_max_ratio x their
    neighbour's flux: brighter close companions are real stars about as often
    as the sources m7 keeps (docs/evidence/m7_companion_ratio), and a star
    left out after m6 subtracted it stays whole in the final residual."""
    xpath = os.path.join(str(tmp_path), 'crossband_seed_manual.fits')
    Table({'skycoord': _sc([0], [0]), 'n_filt_confirmed': [2]}).write(xpath)
    opath = os.path.join(str(tmp_path), 'own_m6.fits')
    # the cross-band star (5 mas), companions at 0.05x, 0.3x and 0.09x of it
    # (the last one also 0.18x of the 0.3x companion, 60 mas away); an
    # own-band-only pair 3000 / 3120 mas at 0.5x
    Table({'skycoord': _sc([5, 100, -100, 0, 3000, 3120], [0, 0, 0, 120, 0, 0]),
           'flux': [100.0, 5.0, 30.0, 9.0, 80.0, 40.0]}).write(opath)
    run = dict(companion_fwhm=2.5, fwhm_arcsec=0.062)
    out = Table.read(_build_m7_band_seed(xpath, opath, 'F182M', 'merged',
                                         companion_max_ratio=0.1, **run))
    assert sorted(out['flux']) == [30.0, 40.0, 80.0, 100.0]
    assert out.meta['NCOMPAN'] == 2
    assert out.meta['COMPRAT'] == 0.1
    # a lower ratio keeps the 0.09x companion; ratio 1 is the cut without it
    out2 = Table.read(_build_m7_band_seed(xpath, opath, 'F182M', 'merged',
                                          companion_max_ratio=0.08, **run))
    assert sorted(out2['flux']) == [9.0, 30.0, 40.0, 80.0, 100.0]
    out1 = Table.read(_build_m7_band_seed(xpath, opath, 'F182M', 'merged',
                                          companion_max_ratio=1.0, **run))
    assert sorted(out1['flux']) == [80.0, 100.0]
    assert out1.meta['NCOMPAN'] == 4
    # above 1 a source would be its own "brighter" neighbour
    with pytest.raises(ValueError):
        _build_m7_band_seed(xpath, opath, 'F182M', 'merged', companion_max_ratio=1.5, **run)


def _keep_fixture(tmp_path, *, donor_saturated=False, drop_column=None):
    """Cross-band star at the origin and own-band m6 rows around it (F182M,
    FWHM 62 mas: the cut radius is 155 mas, 1.5 FWHM = 93 mas).  Rows are
    (name, x mas, y mas, flux, flux_err, is_saturated, replaced_saturated)."""
    rows = [
        ('donor', 5, 0, 100.0, 1.0, donor_saturated, False),
        ('A1_far_unsat_snr60', 130, 0, 6.0, 0.1, False, False),
        ('A2_far_unsat_snr20', -130, 0, 6.0, 0.3, False, False),
        ('A3_near_snr60', 0, 90, 6.0, 0.1, False, False),
        ('B_sat', 3000, 0, 100.0, 1.0, True, False),
        ('B1_far_satnbr', 3130, 0, 6.0, 0.1, False, False),
        ('C_replaced', 4000, 0, 100.0, 1.0, False, True),
        ('C1_far_replnbr', 4130, 0, 6.0, 0.1, False, False),
        ('D_star1', 6000, 0, 100.0, 1.0, False, False),
        ('D_star2', 6180, 0, 100.0, 1.0, False, False),
        ('D1_one_near_nbr', 6130, 0, 6.0, 0.1, False, False),
        ('E_star1', 8000, 0, 100.0, 1.0, False, False),
        ('E_star2', 8240, 0, 100.0, 1.0, False, False),
        ('E1_two_far_nbrs', 8120, 0, 6.0, 0.1, False, False),
        ('G_isolated', 10000, 0, 2.0, 0.1, False, False),
    ]
    own = Table({'skycoord': _sc([r[1] for r in rows], [r[2] for r in rows]),
                 'flux': [r[3] for r in rows], 'flux_err': [r[4] for r in rows],
                 'is_saturated': [r[5] for r in rows],
                 'replaced_saturated': [r[6] for r in rows]})
    if drop_column:
        own.remove_column(drop_column)
    xpath = os.path.join(str(tmp_path), 'crossband_seed_manual.fits')
    Table({'skycoord': _sc([0], [0]), 'n_filt_confirmed': [2]}).write(xpath, overwrite=True)
    opath = os.path.join(str(tmp_path), 'own_m6.fits')
    own.write(opath, overwrite=True)
    return xpath, opath


def test_far_companion_exemption(tmp_path):
    """With companion_keep_snr > 0 a candidate the cut would leave out is
    added when its S/N reaches the floor and every brighter seed source that
    triggers the cut is at least companion_keep_fwhm away and unsaturated."""
    xpath, opath = _keep_fixture(tmp_path)
    run = dict(companion_fwhm=2.5, fwhm_arcsec=0.062)
    off = Table.read(_build_m7_band_seed(xpath, opath, 'F182M', 'merged', **run))
    assert off.meta['NCOMPAN'] == 7 and off.meta['NCOMPKEP'] == 0
    assert off.meta['COMPKSNR'] == 0.0
    base_fluxes = sorted(off['flux'])
    assert base_fluxes == sorted([100.0] * 7 + [2.0])    # 1 cross-band + 7 own-band
    on = Table.read(_build_m7_band_seed(xpath, opath, 'F182M', 'merged',
                                        companion_keep_snr=50.0, **run))
    # A1 (one far unsaturated neighbour) and E1 (two) are added; A2 is below the
    # floor, A3 and D1 have a neighbour inside 1.5 FWHM, B1 and C1 have a
    # saturated or replaced-saturated neighbour
    assert on.meta['NCOMPAN'] == 5 and on.meta['NCOMPKEP'] == 2
    assert on.meta['COMPKSNR'] == 50.0 and on.meta['COMPKFWH'] == 1.5
    assert sorted(on['flux']) == sorted(base_fluxes + [6.0, 6.0])
    sc = on['skycoord']
    new = [i for i in range(len(on)) if on['flux'][i] == 6.0]
    xs = sorted(round(float(sc[i].separation(_sc([0], [0])[0]).to_value(u.mas)), 0) for i in new)
    assert xs == [130.0, 8120.0]
    # a wider separation floor drops E1 (120 mas), then A1 (130 mas)
    on2 = Table.read(_build_m7_band_seed(xpath, opath, 'F182M', 'merged',
                                         companion_keep_snr=50.0, companion_keep_fwhm=2.0, **run))
    assert on2.meta['NCOMPKEP'] == 1 and on2.meta['NCOMPAN'] == 6
    on3 = Table.read(_build_m7_band_seed(xpath, opath, 'F182M', 'merged',
                                         companion_keep_snr=50.0, companion_keep_fwhm=2.2, **run))
    assert on3.meta['NCOMPKEP'] == 0 and sorted(on3['flux']) == base_fluxes
    # a higher S/N floor drops both exempt rows (S/N 60)
    on4 = Table.read(_build_m7_band_seed(xpath, opath, 'F182M', 'merged',
                                         companion_keep_snr=61.0, **run))
    assert on4.meta['NCOMPKEP'] == 0 and sorted(on4['flux']) == base_fluxes


def test_far_companion_exemption_default_radius(tmp_path):
    """The default keep radius is 1.5 FWHM: a candidate 100 mas (1.6 FWHM at
    62 mas) from the brighter source it is cut for is exempt at the default
    and stays cut at 1.75 FWHM, the default before 2026-11-28."""
    own = Table({'skycoord': _sc([5, 105], [0, 0]), 'flux': [100.0, 6.0],
                 'flux_err': [1.0, 0.1], 'is_saturated': [False, False],
                 'replaced_saturated': [False, False]})
    xpath = os.path.join(str(tmp_path), 'crossband_seed_manual.fits')
    Table({'skycoord': _sc([0], [0]), 'n_filt_confirmed': [2]}).write(xpath, overwrite=True)
    opath = os.path.join(str(tmp_path), 'own_m6.fits')
    own.write(opath, overwrite=True)
    run = dict(companion_fwhm=2.5, fwhm_arcsec=0.062, companion_keep_snr=50.0)
    default = Table.read(_build_m7_band_seed(xpath, opath, 'F182M', 'merged', **run))
    assert default.meta['COMPKFWH'] == 1.5
    assert default.meta['NCOMPKEP'] == 1 and default.meta['NCOMPAN'] == 0
    wide = Table.read(_build_m7_band_seed(xpath, opath, 'F182M', 'merged',
                                          companion_keep_fwhm=1.75, **run))
    assert wide.meta['NCOMPKEP'] == 0 and wide.meta['NCOMPAN'] == 1


def test_far_companion_exemption_saturated_donor(tmp_path):
    """A cross-band position takes the saturation flag of the own-band row it
    took its flux from: a saturated donor blocks the exemption of A1."""
    xpath, opath = _keep_fixture(tmp_path, donor_saturated=True)
    out = Table.read(_build_m7_band_seed(xpath, opath, 'F182M', 'merged',
                                         companion_fwhm=2.5, fwhm_arcsec=0.062,
                                         companion_keep_snr=50.0))
    assert out.meta['NCOMPKEP'] == 1      # E1 only
    assert out.meta['NCOMPAN'] == 6


def test_far_companion_exemption_needs_its_columns(tmp_path):
    """Asking for the exemption without flux_err or without any saturation
    column raises; with the cut off the columns are not read."""
    for col in ('flux_err',):
        xpath, opath = _keep_fixture(tmp_path, drop_column=col)
        with pytest.raises(ValueError, match='flux_err'):
            _build_m7_band_seed(xpath, opath, 'F182M', 'merged', companion_fwhm=2.5,
                                fwhm_arcsec=0.062, companion_keep_snr=50.0)
        # the cut off: nothing to exempt, columns not needed
        out = Table.read(_build_m7_band_seed(xpath, opath, 'F182M', 'merged',
                                             companion_fwhm=0, fwhm_arcsec=0.062,
                                             companion_keep_snr=50.0))
        assert out.meta['NCOMPKEP'] == 0
    xpath, opath = _keep_fixture(tmp_path)
    own = Table.read(opath)
    own.remove_columns(['is_saturated', 'replaced_saturated'])
    own.write(opath, overwrite=True)
    with pytest.raises(ValueError, match='saturated'):
        _build_m7_band_seed(xpath, opath, 'F182M', 'merged', companion_fwhm=2.5,
                            fwhm_arcsec=0.062, companion_keep_snr=50.0)
    # an unsaturated catalog with only one of the two columns is accepted
    own['is_saturated'] = np.zeros(len(own), dtype=bool)
    own.write(opath, overwrite=True)
    out = Table.read(_build_m7_band_seed(xpath, opath, 'F182M', 'merged', companion_fwhm=2.5,
                                         fwhm_arcsec=0.062, companion_keep_snr=50.0))
    assert out.meta['NCOMPKEP'] == 4          # A1, E1, and B1, C1 no longer have a saturated neighbour
    for kw in ({'companion_keep_snr': -1.0}, {'companion_keep_fwhm': -0.5}):
        with pytest.raises(ValueError):
            _build_m7_band_seed(xpath, opath, 'F182M', 'merged', fwhm_arcsec=0.062, **kw)


def test_far_companion_exemption_floor_per_band(tmp_path):
    """``--manual-m7-seed-own-band-companion-keep-snr F212N=50,F480M=0`` sets the
    exemption on in F212N and off in F480M; a band that the option does not name keeps
    the default (off), and a plain number keeps its meaning for every band."""
    xpath, opath = _keep_fixture(tmp_path)
    run = dict(companion_fwhm=2.5, fwhm_arcsec=0.062)

    def seed(band, raw):
        opts = types.SimpleNamespace(manual_m7_seed_own_band_companion_keep_snr=raw)
        floor = band_mopt(opts, 'manual_m7_seed_own_band_companion_keep_snr', band)
        out = Table.read(_build_m7_band_seed(xpath, opath, band, 'merged',
                                             companion_keep_snr=floor, **run))
        return out.meta['COMPKSNR'], out.meta['NCOMPKEP'], sorted(out['flux'])

    off = seed('F480M', '0')
    assert off[:2] == (0.0, 0)
    plain = seed('F212N', '50')
    assert plain[:2] == (50.0, 2)
    assert plain[2] == sorted(off[2] + [6.0, 6.0])
    # F212N gets the plain-number result, F480M and a band the option does not name get none
    assert seed('F212N', 'F212N=50,F480M=0') == plain
    assert seed('f212n', 'f212n=50') == plain
    assert seed('F480M', 'F212N=50,F480M=0') == off
    assert seed('F182M', 'F212N=50,F480M=0') == off
    assert seed('F182M', 'F212N=50') == off
    # a plain number applies to every band
    assert seed('F480M', '50') == plain
    assert seed('F182M', 50.0) == plain
    # a floor that no source reaches (the two candidates have S/N 60) leaves the cut as it
    # is, and the table records the floor of its band
    assert seed('F480M', 'F212N=50,F480M=61') == (61.0,) + off[1:]


def test_flag_array_reads_the_stored_forms():
    flag = cataloging._flag_array
    assert list(flag(np.array([True, False]))) == [True, False]
    assert list(flag(np.array([1, 0, 2]))) == [True, False, True]
    assert list(flag(np.array(['True', 'false', 'TRUE', 'False']))) == [True, False, True, False]
    masked = Table({'f': np.ma.MaskedArray([True, True, False], mask=[False, True, False])})
    assert list(flag(masked['f'])) == [True, False, False]


def test_pipeline_defaults():
    # AUTO by default (docs/evidence/faint_defaults_on); see
    # docs/evidence/faint_m7_seed_union for the full-field realness
    assert MANUAL_DEFAULTS['manual_m7_seed_own_band'] is None
    assert MANUAL_DEFAULTS['manual_m7_seed_own_band_companion_fwhm'] == 2.5
    assert MANUAL_DEFAULTS['manual_m7_seed_own_band_companion_max_ratio'] == 0.1
    # the far-companion exemption is opt-in (docs/evidence/m7_companion_keep)
    assert MANUAL_DEFAULTS['manual_m7_seed_own_band_companion_keep_snr'] == 0.0
    assert MANUAL_DEFAULTS['manual_m7_seed_own_band_companion_keep_fwhm'] == 1.5
    # opt-in until the production seeds are rerun (decision 2026-11-14)
    assert MANUAL_DEFAULTS['manual_m7_seed_dedup_fwhm'] == 0.0


def test_auto_own_band_off_on_extended_emission_and_miri():
    auto = MANUAL_DEFAULTS['manual_m7_seed_own_band']
    for target in ('w51', 'sickle', 'wd2', 'ngc6334'):
        assert _auto_m7_seed_own_band(auto, types.SimpleNamespace(target=target)) is False
    for target in ('brick', 'sgrb2', 'sgra', 'cloudc'):
        assert _auto_m7_seed_own_band(auto, types.SimpleNamespace(target=target)) is True
        assert _auto_m7_seed_own_band(
            auto, types.SimpleNamespace(target=target), miri=True) is False
    # --extended-emission / --no-extended-emission override the target list
    assert _auto_m7_seed_own_band(
        auto, types.SimpleNamespace(target='brick', extended_emission=True)) is False
    assert _auto_m7_seed_own_band(
        auto, types.SimpleNamespace(target='w51', extended_emission=False)) is True
    # the flag and the --no- flag are used verbatim, on MIRI too
    assert _auto_m7_seed_own_band(False, types.SimpleNamespace(target='brick')) is False
    assert _auto_m7_seed_own_band(True, types.SimpleNamespace(target='w51')) is True
    assert _auto_m7_seed_own_band(
        True, types.SimpleNamespace(target='brick'), miri=True) is True


def test_pipeline_passes_the_companion_options():
    """run_manual_pipeline hands the companion-cut options, including the
    far-companion exemption, to the builder."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(cataloging.run_manual_pipeline)))
    (call,) = [c for c in ast.walk(tree) if isinstance(c, ast.Call)
               and getattr(c.func, 'id', None) == '_build_m7_band_seed']
    kw = {k.arg: ast.unparse(k.value) for k in call.keywords}
    for arg in ('companion_fwhm', 'companion_max_ratio', 'companion_keep_fwhm'):
        assert kw[arg] == f"float(mopt(opts_phase, 'manual_m7_seed_own_band_{arg}'))"
    # the exemption floor is the value of the band the seed is built for
    assert kw['companion_keep_snr'] == (
        "band_mopt(opts_phase, 'manual_m7_seed_own_band_companion_keep_snr', filt, "
        "label=f'{phase}:{filt}')")
    assert kw['dedup_fwhm'] == "float(mopt(opts_phase, 'manual_m7_seed_dedup_fwhm'))"
