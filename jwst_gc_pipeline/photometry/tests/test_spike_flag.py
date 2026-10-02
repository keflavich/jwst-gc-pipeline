import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
import astropy.units as u

from jwst_gc_pipeline.photometry import spike_flag as sf

DEC0, RA0 = -57.76, 155.95
PA_V3 = 140.8


def _offset(center, r_arcsec, pa_deg):
    return center.directional_offset_by(pa_deg * u.deg, r_arcsec * u.arcsec)


def _table(rows):
    """rows: list of (SkyCoord, mags dict band->mag or nan)."""
    bands = ['f115w', 'f200w', 'f212n']
    t = Table()
    t['skycoord_ref'] = SkyCoord([r[0] for r in rows])
    for b in bands:
        t[f'mag_vega_{b}'] = [r[1].get(b, np.nan) for r in rows]
        t[f'mask_{b}'] = np.zeros(len(rows), bool)
        t[f'forced_filled_{b}'] = np.zeros(len(rows), bool)
    return t


def _synthetic(off=0.0, r=np.arange(5, 20, 1.0)):
    parent = SkyCoord(RA0 * u.deg, DEC0 * u.deg)
    spike0 = sf.spike_position_angles([PA_V3])[0]
    rows = [(parent, dict(f115w=8.0, f200w=7.5, f212n=7.6))]
    for ri in r:
        rows.append((_offset(parent, ri, spike0 + off), dict(f212n=19.0)))
    return rows, parent, spike0


def test_spike_pas():
    pa = sf.spike_position_angles([140.8])
    assert len(pa) == 6
    assert np.allclose(pa, np.sort((140.8 + 60 * np.arange(6)) % 360))
    assert len(sf.spike_position_angles([140.8, 140.8], struts=True)) == 8
    assert len(sf.spike_position_angles([])) == 0


def test_collect_pa_v3(tmp_path):
    d = tmp_path / 'F200W' / 'pipeline'
    d.mkdir(parents=True)
    for i, v in enumerate([140.82, 140.84, 10.0]):
        h = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(np.zeros((2, 2)), name='SCI')])
        h['SCI'].header['PA_V3'] = v
        h.writeto(d / f'x{i}_cal.fits')
    assert np.allclose(sf.collect_pa_v3(str(tmp_path), ['f200w']), [10.0, 140.8])
    assert sf.collect_pa_v3(str(tmp_path), ['f444w']).size == 0


def test_n_real_bands():
    rows, _, _ = _synthetic(r=[5.0])
    t = _table(rows)
    t['forced_filled_f200w'][0] = True
    t['mask_f115w'][0] = True
    n, mm = sf.n_real_bands(t)
    assert list(n) == [1, 1]
    assert np.isclose(mm[0], 7.5)


def test_wedge_on_off_spike():
    rows, parent, s0 = _synthetic()
    res = sf.flag_spike_artifacts(_table(rows), [PA_V3])
    assert res['spike_wedge'][1:].all()
    assert not res['spike_wedge'][0]
    rows, _, _ = _synthetic(off=30.0)
    res = sf.flag_spike_artifacts(_table(rows), [PA_V3])
    assert not res['spike_wedge'].any()


def test_multiband_and_far_not_flagged():
    rows, parent, s0 = _synthetic(r=[8.0, 9.0, 10.0, 11.0])
    rows.append((_offset(parent, 12.0, s0), dict(f115w=18, f200w=18, f212n=18)))
    rows.append((_offset(parent, 39.0, s0), dict(f212n=19.0)))  # beyond L
    res = sf.flag_spike_artifacts(_table(rows), [PA_V3])
    assert res['spike_wedge'][1:5].all()
    assert not res['spike_wedge'][5]
    assert not res['spike_wedge'][6]


def test_density_flag():
    c = SkyCoord(RA0 * u.deg, DEC0 * u.deg)
    rng = np.random.default_rng(1)
    rows = [(_offset(c, 0.5 * rng.random(), 360 * rng.random()), dict(f212n=19.0))
            for _ in range(20)]
    rows.append((_offset(c, 100.0, 0.0), dict(f212n=19.0)))
    rows.append((_offset(c, 3.0, 90.0), dict(f115w=8.0, f200w=7.5)))   # bright parent
    res = sf.flag_spike_artifacts(_table(rows), [])
    assert res['single_band_crowd'][:20].all()
    assert not res['single_band_crowd'][20:].any()
    assert not res['spike_wedge'].any()
    assert (res['spike_artifact'] == res['single_band_crowd']).all()


def test_empty_pa_message(capsys):
    rows, _, _ = _synthetic()
    res = sf.flag_spike_artifacts(_table(rows), np.array([]))
    assert not res['spike_wedge'].any()
    assert 'no PA_V3' in capsys.readouterr().out


def test_flag_m8_file(tmp_path):
    rows, _, _ = _synthetic()
    t = _table(rows)
    d = tmp_path / 'F212N' / 'pipeline'
    d.mkdir(parents=True)
    h = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(np.zeros((2, 2)), name='SCI')])
    h['SCI'].header['PA_V3'] = PA_V3
    h.writeto(d / 'a_cal.fits')
    (tmp_path / 'catalogs').mkdir()
    p = str(tmp_path / 'catalogs' / 'm8_dedup.fits')
    t.write(p)
    flags = sf.flag_m8_spike_artifacts(p, str(tmp_path), bands=['f212n'])
    t2 = Table.read(p)
    for c in ('spike_wedge', 'single_band_crowd', 'spike_artifact', 'n_real_bands'):
        assert c in t2.colnames
    assert t2['spike_wedge'].sum() == flags['spike_wedge'].sum() > 0


def _random_field(rng, n=3000, half=60.0):
    c = SkyCoord(RA0 * u.deg, DEC0 * u.deg)
    r = half * np.sqrt(rng.random(n))
    pa = 360 * rng.random(n)
    return [(_offset(c, ri, pai), dict(f212n=19.0)) for ri, pai in zip(r, pa)], c


def test_chain_over_background_flagged():
    rng = np.random.default_rng(3)
    rows, parent = _random_field(rng, n=150)
    rows[0] = (parent, dict(f115w=8.0, f200w=7.5, f212n=7.6))
    s0 = sf.spike_position_angles([PA_V3])[0]
    chain = [(_offset(parent, ri, s0), dict(f212n=19.0)) for ri in np.arange(4, 20, 0.6)]
    res = sf.flag_spike_artifacts(_table(rows + chain), [PA_V3])
    nb = len(rows)
    assert res['spike_wedge'][nb:].mean() >= 0.8
    # background rows are only flagged if they happen to sit on a spike
    assert res['spike_wedge'][:nb].sum() <= 0.05 * nb


def test_uniform_field_flags_almost_nothing():
    rng = np.random.default_rng(4)
    rows, parent = _random_field(rng, n=3000)
    rows[0] = (parent, dict(f115w=8.0, f200w=7.5, f212n=7.6))
    res = sf.flag_spike_artifacts(_table(rows), [PA_V3])
    assert res['spike_wedge'].sum() <= 0.01 * len(rows)
    assert res['single_band_crowd'].sum() <= 0.01 * len(rows)
    # the non-adaptive wedge marks every row on a spike line
    res0 = sf.flag_spike_artifacts(_table(rows), [PA_V3], adaptive=False)
    assert res0['spike_wedge'].sum() > res['spike_wedge'].sum()


def test_crowd_contrast():
    rng = np.random.default_rng(5)
    rows, c = _random_field(rng, n=6000, half=60.0)   # dense uniform field, ~0.5 /arcsec^2
    clump = [(_offset(c, 0.4 * rng.random() + 0.0, 360 * rng.random()), dict(f212n=19.0))
             for _ in range(40)]
    shifted = [(_offset(r[0], 30.0, 0.0), r[1]) for r in clump]  # clump away from centre
    shifted.append((_offset(c, 33.0, 0.0), dict(f115w=8.0, f200w=7.5)))  # bright parent
    res = sf.flag_spike_artifacts(_table(rows + shifted), [])
    nb = len(rows)
    assert res['single_band_crowd'][nb:-1].mean() > 0.9
    assert res['single_band_crowd'][:nb].mean() < 0.02
    # absolute-count version flags the whole dense field core far more
    res0 = sf.flag_spike_artifacts(_table(rows + shifted), [], adaptive=False)
    assert res0['single_band_crowd'][:nb].sum() >= res['single_band_crowd'][:nb].sum()


def test_crowd_needs_bright_parent():
    c = SkyCoord(RA0 * u.deg, DEC0 * u.deg)
    rng = np.random.default_rng(1)
    clump = [(_offset(c, 0.5 * rng.random(), 360 * rng.random()), dict(f212n=19.0))
             for _ in range(20)]
    res = sf.flag_spike_artifacts(_table(clump), [])
    assert not res['single_band_crowd'].any()
    res0 = sf.flag_spike_artifacts(_table(clump), [], crowd_near_parent=False)
    assert res0['single_band_crowd'].all()


def test_crowd_ignores_footprint_edge():
    """A dense uniform single-band field ending at a straight edge: next to the
    edge the 8-20" annulus is about half off the catalog.  Dividing by the full
    annulus area halves the expected density there and flags the edge strip;
    the covered-area density does not."""
    rng = np.random.default_rng(6)
    c = SkyCoord(RA0 * u.deg, DEC0 * u.deg)
    nrow = 18000
    xs = 60.0 * rng.random(nrow)                     # x in [0, 60]: edge at x=0
    ys = 100.0 * rng.random(nrow) - 50.0
    r = np.hypot(xs, ys); pa = np.degrees(np.arctan2(xs, ys))
    rows = [(_offset(c, ri, pai), dict(f212n=19.0)) for ri, pai in zip(r, pa)]
    rows.append((c, dict(f115w=8.0, f200w=7.5)))     # bright parent on the edge
    kw = dict(crowd_near_parent=False)
    raw = sf.flag_spike_artifacts(_table(rows), [], ann_probe_r=0.0, **kw)
    cov = sf.flag_spike_artifacts(_table(rows), [], **kw)
    assert raw['single_band_crowd'].sum() > 20
    assert cov['single_band_crowd'].sum() <= 0.1 * raw['single_band_crowd'].sum()


def test_collect_pa_v3_multi_program(tmp_path):
    d = tmp_path / 'F212N' / 'pipeline'
    d.mkdir(parents=True)
    for i, (pre, v) in enumerate([('jw01182004001', 91.0)] * 60 + [('jw02221001001', 89.0)] * 3
                                 + [('jw02221002001', 275.5)] * 2):
        h = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(np.zeros((2, 2)), name='SCI')])
        h['SCI'].header['PA_V3'] = v
        h.writeto(d / f'{pre}_{i:05d}_nrca1_cal.fits')
    out = sf.collect_pa_v3(str(tmp_path), ['f212n'])
    assert np.allclose(out, [89.0, 91.0, 275.5])
    assert len(sf._cluster_pa([280.0, 280.1, 359.9, 0.1])) == 2
