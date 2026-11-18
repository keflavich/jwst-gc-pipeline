"""Tests for the optional per-row position-correction tables (poscorr.py) and the builder hook that reads them
(build_treasury_pm --src-poscorr / --ref-poscorr): the table is checked against the catalogue it is applied to,
the sky direction and size of a correction are what the table says, and the default build is unchanged.
"""
import json

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u

from jwst_gc_pipeline.astrometry import poscorr as PC
from jwst_gc_pipeline.astrometry.tests.test_build_treasury_pm import (
    CENTER, _synthetic_two_epoch_field, _write_ref_fits)


def _write_table(path, ra, dec, dxi, deta, **meta):
    """A table as the producer writes it: row-parallel to a catalogue with positions (ra, dec) [deg]."""
    rachk, decchk = PC.position_checksums(ra, dec)
    t = Table()
    t['row'] = np.arange(len(ra), dtype=np.int64)
    t['dxi_mas'] = np.asarray(dxi, float)
    t['deta_mas'] = np.asarray(deta, float)
    t.meta.update(NROWS=len(ra), RACHK=rachk, DECCHK=decchk, CATALOG='test.fits', **meta)
    t.write(path, overwrite=True)


def _positions(n=50, seed=3):
    rng = np.random.default_rng(seed)
    ra = CENTER.ra.deg + rng.uniform(-30, 30, n) / 3600 / np.cos(CENTER.dec.rad)
    dec = CENTER.dec.deg + rng.uniform(-30, 30, n) / 3600
    return ra, dec


def test_checksums_follow_the_positions():
    ra, dec = _positions()
    assert PC.position_checksums(ra, dec) == PC.position_checksums(ra.copy(), dec.copy())
    ra2 = ra.copy()
    ra2[7] = np.nextafter(ra2[7], 400.0)            # one ulp in one row
    assert PC.position_checksums(ra2, dec)[0] != PC.position_checksums(ra, dec)[0]
    assert PC.position_checksums(ra2, dec)[1] == PC.position_checksums(ra, dec)[1]


def test_load_roundtrip_returns_the_corrections_and_the_header(tmp_path):
    ra, dec = _positions()
    dxi, deta = np.linspace(-5, 5, len(ra)), np.linspace(2, -3, len(ra))
    p = tmp_path / 'pc.fits'
    _write_table(p, ra, dec, dxi, deta, SOLUTION='map', MAPKIND='dqfull')
    gx, gy, meta = PC.load_poscorr(str(p), ra, dec)
    np.testing.assert_array_equal(gx, dxi)
    np.testing.assert_array_equal(gy, deta)
    assert meta['SOLUTION'] == 'map' and meta['MAPKIND'] == 'dqfull'


def test_load_refuses_a_table_made_for_another_catalogue(tmp_path):
    ra, dec = _positions()
    p = tmp_path / 'pc.fits'
    _write_table(p, ra, dec, np.zeros(len(ra)), np.zeros(len(ra)))
    with pytest.raises(PC.PosCorrMismatchError, match='rows'):         # row count
        PC.load_poscorr(str(p), ra[:-1], dec[:-1])
    with pytest.raises(PC.PosCorrMismatchError, match='checksum'):     # same length, positions re-measured
        PC.load_poscorr(str(p), ra + 1e-9, dec)
    swapped = np.arange(len(ra))[::-1]
    with pytest.raises(PC.PosCorrMismatchError, match='checksum'):     # same rows, other order
        PC.load_poscorr(str(p), ra[swapped], dec[swapped])


def test_load_refuses_non_finite_corrections_and_a_scrambled_row_column(tmp_path):
    ra, dec = _positions()
    dxi = np.zeros(len(ra))
    dxi[4] = np.nan
    p = tmp_path / 'nan.fits'
    _write_table(p, ra, dec, dxi, np.zeros(len(ra)))
    with pytest.raises(PC.PosCorrMismatchError, match='non-finite'):
        PC.load_poscorr(str(p), ra, dec)
    t = Table.read(p)
    t['row'][3], t['row'][4] = t['row'][4], t['row'][3]
    t['dxi_mas'][4] = 0.0
    t.write(tmp_path / 'rows.fits', overwrite=True)
    with pytest.raises(PC.PosCorrMismatchError, match='row column'):
        PC.load_poscorr(str(tmp_path / 'rows.fits'), ra, dec)


def test_apply_zero_correction_returns_the_positions_unchanged():
    ra, dec = _positions()
    sc = SkyCoord(ra * u.deg, dec * u.deg)
    out = PC.apply_poscorr(sc, np.zeros(len(ra)), np.zeros(len(ra)))
    np.testing.assert_array_equal(out.ra.deg, ra)
    np.testing.assert_array_equal(out.dec.deg, dec)


def test_apply_moves_east_and_north_by_the_stated_mas():
    """+dxi is east on the sky (ra grows by dxi/cos dec), +deta is north; offsets come out in mas."""
    sc = SkyCoord([CENTER.ra.deg] * 2 * u.deg, [CENTER.dec.deg] * 2 * u.deg)
    out = PC.apply_poscorr(sc, np.array([12.0, 0.0]), np.array([0.0, -7.0]))
    dra, ddec = sc[0].spherical_offsets_to(out[0])
    assert dra.to(u.mas).value == pytest.approx(12.0, abs=1e-3) and ddec.to(u.mas).value == pytest.approx(0, abs=1e-3)
    dra, ddec = sc[1].spherical_offsets_to(out[1])
    assert dra.to(u.mas).value == pytest.approx(0.0, abs=1e-3) and ddec.to(u.mas).value == pytest.approx(-7.0, abs=1e-3)
    assert out[0].ra.deg > sc[0].ra.deg and out[1].dec.deg < sc[1].dec.deg


# ---------------------------------------------------------------------------
# builder hook
# ---------------------------------------------------------------------------

def _zero_table_for(path, out_path, **meta):
    t = Table.read(path)
    sc = SkyCoord(t['skycoord'])
    z = np.zeros(len(t))
    _write_table(out_path, sc.ra.deg, sc.dec.deg, z, z, **meta)
    return sc


def test_load_one_applies_the_table_to_the_usable_rows_and_keeps_the_catalogue_row(tmp_path):
    from jwst_gc_pipeline.astrometry.build_treasury_pm import _load_one
    rng = np.random.default_rng(5)
    n = 60
    ra, dec = _positions(n, 6)
    flux = 10 ** rng.uniform(-6, -4, n)
    flux[::7] = 1e-9                                 # S/N < 5: dropped, so row != usable index
    cat = tmp_path / 'cat.fits'
    _write_ref_fits(cat, SkyCoord(ra * u.deg, dec * u.deg), flux)
    dxi, deta = rng.uniform(-8, 8, n), rng.uniform(-8, 8, n)
    pc = tmp_path / 'pc.fits'
    _write_table(pc, ra, dec, dxi, deta, SOLUTION='unit')
    plain = _load_one(str(cat), 'f212n', 2024.7, 5.0, 0)
    hooked = _load_one(str(cat), 'f212n', 2024.7, 5.0, 0, poscorr=str(pc))
    assert 'sc_cat' not in plain and 'poscorr_meta' not in plain          # default: nothing new
    ok = np.nonzero(flux / (flux / 20.0) > 5.0)[0]
    np.testing.assert_array_equal(plain['row'], ok)
    np.testing.assert_array_equal(hooked['row'], ok)
    np.testing.assert_array_equal(hooked['sc_cat'].ra.deg, plain['sc'].ra.deg)
    np.testing.assert_array_equal(hooked['sc_cat'].dec.deg, plain['sc'].dec.deg)
    expect = PC.apply_poscorr(plain['sc'], dxi[ok], deta[ok])
    np.testing.assert_allclose(hooked['sc'].ra.deg, expect.ra.deg, rtol=0, atol=1e-14)
    np.testing.assert_allclose(hooked['sc'].dec.deg, expect.dec.deg, rtol=0, atol=1e-14)
    sep = plain['sc'].separation(hooked['sc']).to(u.mas).value
    np.testing.assert_allclose(sep, np.hypot(dxi[ok], deta[ok]), atol=1e-3)
    assert hooked['poscorr_meta']['SOLUTION'] == 'unit'
    # the arrays the PM builder reads are the same rows, with and without the table
    for k in ('ex', 'ey', 'flux', 'mag', 'sat', 'flux_noise'):
        np.testing.assert_array_equal(hooked[k], plain[k])


def test_load_filter_catalog_keeps_row_and_catalogue_positions_through_the_dedup(tmp_path):
    from jwst_gc_pipeline.astrometry.build_treasury_pm import load_filter_catalog
    ra, dec = _positions(40, 8)
    flux = np.full(40, 1e-5)
    pa, pb = tmp_path / 'a.fits', tmp_path / 'b.fits'
    _write_ref_fits(pa, SkyCoord(ra * u.deg, dec * u.deg), flux)
    # b repeats the first 10 stars (duplicates, removed by the dedup) plus 30 new ones
    ra_b = np.concatenate([ra[:10], ra[10:] + 20 / 3600])
    dec_b = np.concatenate([dec[:10], dec[10:] + 20 / 3600])
    _write_ref_fits(pb, SkyCoord(ra_b * u.deg, dec_b * u.deg), np.full(40, 1e-5))
    za = tmp_path / 'za.fits'
    zb = tmp_path / 'zb.fits'
    _write_table(za, ra, dec, np.full(40, 3.0), np.zeros(40))
    _write_table(zb, ra_b, dec_b, np.full(40, -4.0), np.zeros(40))
    cat = load_filter_catalog([str(pa), str(pb)], 'f212n', 2024.7, poscorr_paths=[str(za), str(zb)])
    assert cat['n'] == 70 and len(cat['sc_cat']) == 70 and len(cat['row']) == 70
    assert len(cat['obs_index']) == 70 and len(cat['poscorr_meta']) == 2
    from_a = cat['obs_index'] == 0
    from_b = cat['obs_index'] == 1
    assert from_a.sum() == 40 and from_b.sum() == 30                          # duplicates dropped from b
    np.testing.assert_array_equal(cat['row'][from_a], np.arange(40))
    np.testing.assert_array_equal(cat['row'][from_b], np.arange(10, 40))
    # each kept row carries ITS table's correction, applied to its own catalogue position
    off = cat['sc_cat'].spherical_offsets_to(cat['sc'])
    np.testing.assert_allclose(off[0].to(u.mas).value[from_a], 3.0, atol=1e-3)
    np.testing.assert_allclose(off[0].to(u.mas).value[from_b], -4.0, atol=1e-3)
    with pytest.raises(ValueError, match='exactly one per catalog'):
        load_filter_catalog([str(pa), str(pb)], 'f212n', 2024.7, poscorr_paths=[str(za)])


def test_load_one_refuses_a_table_for_a_recatalogued_file(tmp_path):
    """Same row count, moved positions (a re-run of the catalogue): refused, not silently applied."""
    from jwst_gc_pipeline.astrometry.build_treasury_pm import _load_one
    ra, dec = _positions(30, 9)
    flux = np.full(30, 1e-5)
    cat, pc = tmp_path / 'cat.fits', tmp_path / 'pc.fits'
    _write_ref_fits(cat, SkyCoord(ra * u.deg, dec * u.deg), flux)
    _write_table(pc, ra, dec, np.zeros(30), np.zeros(30))
    _write_ref_fits(cat, SkyCoord((ra + 3e-9) * u.deg, dec * u.deg), flux)    # ~0.01 mas move
    with pytest.raises(PC.PosCorrMismatchError):
        _load_one(str(cat), 'f212n', 2024.7, 5.0, 0, poscorr=str(pc))


@pytest.fixture
def two_epoch(tmp_path):
    pytest.importorskip('flystar')
    src, ref, n_all, _ = _synthetic_two_epoch_field(tmp_path, np.random.default_rng(41))
    return tmp_path, src, ref, n_all


def test_build_without_tables_has_the_same_columns_as_before(two_epoch):
    from jwst_gc_pipeline.astrometry import build_treasury_pm as btp
    tmp_path, src, ref, _ = two_epoch
    pm = btp.build([src], [ref], 'f212n', 2024.7, 2026.7, str(tmp_path / 'pm.fits'), verbose=False)
    new = {'src_row', 'src_obs_index', 'ref_row', 'ra_src_cat', 'dec_src_cat', 'ra_ref_corr', 'dec_ref_corr'}
    assert not (new & set(pm.colnames))
    assert not any(k.startswith(('src_poscorr', 'ref_poscorr')) for k in pm.meta)


def test_build_with_zero_tables_reproduces_the_default_table(two_epoch):
    """Plumbing check: all-zero corrections give identical PM numbers; the row columns and provenance are the
    only additions, and ra_ref/dec_ref keep their default meaning (untied catalogue-frame position)."""
    from jwst_gc_pipeline.astrometry import build_treasury_pm as btp
    tmp_path, src, ref, _ = two_epoch
    base = btp.build([src], [ref], 'f212n', 2024.7, 2026.7, str(tmp_path / 'base.fits'), verbose=False)
    zs, zr = tmp_path / 'zs.fits', tmp_path / 'zr.fits'
    _zero_table_for(src, zs, SOLUTION='zero')
    _zero_table_for(ref, zr, SOLUTION='zero')
    hook = btp.build([src], [ref], 'f212n', 2024.7, 2026.7, str(tmp_path / 'hook.fits'), verbose=False,
                     src_poscorr=[str(zs)], ref_poscorr=[str(zr)])
    assert len(hook) == len(base)
    for c in base.colnames:
        np.testing.assert_array_equal(np.asarray(hook[c]), np.asarray(base[c]), err_msg=c)
    assert {'src_row', 'src_obs_index', 'ref_row'} <= set(hook.colnames)
    for which in ('src', 'ref'):
        prov = json.loads(hook.meta[f'{which}_poscorr_json'])
        assert [q['solution'] for q in prov] == ['zero'] and prov[0]['table'] == f'z{which[0]}.fits'
    # rows point back at the catalogue files
    ts, tr = Table.read(src), Table.read(ref)
    sc_s, sc_r = SkyCoord(ts['skycoord']), SkyCoord(tr['skycoord'])
    np.testing.assert_array_equal(hook['ra_src_cat'], sc_s.ra.deg[hook['src_row']])
    np.testing.assert_array_equal(hook['ra_ref'], sc_r.ra.deg[hook['ref_row']])
    np.testing.assert_array_equal(hook['ra_ref_corr'], hook['ra_ref'])


def test_build_applies_a_src_correction_with_the_right_sign_and_size(two_epoch):
    """One src star moved 20 mas east in the catalogue frame: its PM falls by 20/dt mas/yr in RA (the star
    appears to have moved less east between the epochs) and the other stars' PMs are unchanged."""
    from jwst_gc_pipeline.astrometry import build_treasury_pm as btp
    tmp_path, src, ref, _ = two_epoch
    base = btp.build([src], [ref], 'f212n', 2024.7, 2026.7, str(tmp_path / 'base.fits'), verbose=False,
                     emit_rows=True)
    ts = Table.read(src)
    sc = SkyCoord(ts['skycoord'])
    trust = np.nonzero(np.asarray(base['trustworthy']))[0]
    pick = int(base['src_row'][trust[np.argmax(np.asarray(base['mag_src'])[trust])]])   # faintest: not a tie star
    dxi = np.zeros(len(ts))
    deta = np.zeros(len(ts))
    dxi[pick] = 20.0
    zs = tmp_path / 'zs.fits'
    _write_table(zs, sc.ra.deg, sc.dec.deg, dxi, deta, SOLUTION='one-star')
    hook = btp.build([src], [ref], 'f212n', 2024.7, 2026.7, str(tmp_path / 'hook.fits'), verbose=False,
                     src_poscorr=[str(zs)])
    assert len(hook) == len(base)
    np.testing.assert_array_equal(hook['src_row'], base['src_row'])
    moved = np.asarray(hook['src_row']) == pick
    assert moved.sum() == 1
    d_pmra = np.asarray(hook['pm_ra'] - base['pm_ra'])
    d_pmdec = np.asarray(hook['pm_dec'] - base['pm_dec'])
    dt = float(base.meta['baseline_yr'])
    assert d_pmra[moved][0] == pytest.approx(-20.0 / dt, abs=0.15)
    assert abs(d_pmdec[moved][0]) < 0.15
    # a faint star is not in the tie sample (brightest 20%), so the tie and every other star are unchanged
    assert np.nanmax(np.abs(d_pmra[~moved])) < 0.01 and np.nanmax(np.abs(d_pmdec[~moved])) < 0.01
    # the untied catalogue-frame ref positions are the catalogue's own; the src ones are reported uncorrected
    np.testing.assert_array_equal(hook['ra_ref'], base['ra_ref'])
    sc_pick = sc[pick]
    assert float(hook['ra_src_cat'][moved][0]) == pytest.approx(sc_pick.ra.deg, abs=1e-12)
    assert float(hook['ra0'][moved][0]) > float(base['ra0'][moved][0])


def test_build_applies_a_ref_correction_with_the_opposite_sign(two_epoch):
    from jwst_gc_pipeline.astrometry import build_treasury_pm as btp
    tmp_path, src, ref, _ = two_epoch
    base = btp.build([src], [ref], 'f212n', 2024.7, 2026.7, str(tmp_path / 'base.fits'), verbose=False,
                     emit_rows=True)
    tr = Table.read(ref)
    sc = SkyCoord(tr['skycoord'])
    trust = np.nonzero(np.asarray(base['trustworthy']))[0]
    pick = int(base['ref_row'][trust[np.argmax(np.asarray(base['mag_src'])[trust])]])
    dxi = np.zeros(len(tr))
    deta = np.zeros(len(tr))
    deta[pick] = 15.0
    zr = tmp_path / 'zr.fits'
    _write_table(zr, sc.ra.deg, sc.dec.deg, dxi, deta, SOLUTION='one-star')
    hook = btp.build([src], [ref], 'f212n', 2024.7, 2026.7, str(tmp_path / 'hook.fits'), verbose=False,
                     ref_poscorr=[str(zr)])
    moved = np.asarray(hook['ref_row']) == pick
    assert moved.sum() == 1
    dt = float(base.meta['baseline_yr'])
    d_pmdec = np.asarray(hook['pm_dec'] - base['pm_dec'])
    assert d_pmdec[moved][0] == pytest.approx(+15.0 / dt, abs=0.15)
    assert float(hook['dec_ref_corr'][moved][0]) > float(hook['dec_ref'][moved][0])


def test_cli_passes_the_poscorr_options_and_defaults_to_none(monkeypatch):
    import sys
    from jwst_gc_pipeline.astrometry import build_treasury_pm as btp
    seen = {}
    monkeypatch.setattr(btp, 'build', lambda *a, **k: seen.update(k) or None)
    base = ['x', '--src', 's1.fits', 's2.fits', '--src-epoch', '2024.7', '--ref', 'r1.fits', 'r2.fits',
            '--ref-epoch', '2026.7', '--filter', 'f212n', '--out', 'o.fits']
    monkeypatch.setattr(sys, 'argv', base)
    btp.main()
    assert seen['src_poscorr'] is None and seen['ref_poscorr'] is None and seen['emit_rows'] is False
    monkeypatch.setattr(sys, 'argv', base + ['--src-poscorr', 'a.fits', 'b.fits', '--ref-poscorr', 'c.fits',
                                             'd.fits', '--emit-rows'])
    btp.main()
    assert seen['src_poscorr'] == ['a.fits', 'b.fits'] and seen['ref_poscorr'] == ['c.fits', 'd.fits']
    assert seen['emit_rows'] is True
