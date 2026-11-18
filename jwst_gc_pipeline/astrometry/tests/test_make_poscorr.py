"""Tests for the hook-table producer (make_poscorr.py) on synthetic fields: no network, CRDS or survey data.

A field is a set of catalogue rows seen in several frames, each frame a rotated (optionally mirrored) linear map
from the sky to the detector, so the Jacobian the producer fits is known and the expected correction can be written
down: for a detector-axis correction c (mas) in a frame whose sky -> pixel map is J = R / pixscale, the sky vector
is inv(R) @ c, and a row's correction is the mean of those over its kept detections (or, for a row with none and
for a saturated row, over every frame whose footprint covers it).
"""
import os
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.table import Table

from jwst_gc_pipeline.astrometry import make_poscorr as MP
from jwst_gc_pipeline.astrometry import poscorr as PC

RA0, DEC0 = 266.5, -28.5
PIXSCALE = 0.031          # arcsec / pixel
C = (0.6, -0.25)          # a detector-axis correction (mas)


def _to_sky(xy):
    """Tangent-plane offsets (east*cos(dec), north; arcsec) -> ra, dec (deg)."""
    return RA0 + xy[:, 0] / 3600.0 / np.cos(np.radians(DEC0)), DEC0 + xy[:, 1] / 3600.0


def _rot(theta_deg, mirror=False):
    t = np.radians(theta_deg)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return R @ np.diag([-1.0, 1.0]) if mirror else R


ROTS = [_rot(0.0), _rot(35.0), _rot(100.0, mirror=True), _rot(200.0)]


class _Const:
    """A solution that applies the same detector-axis correction everywhere."""
    name = 'const'

    def __init__(self, cx, cy):
        self.c = (cx, cy)

    def __call__(self, det, x, y):
        return np.full(np.shape(x), self.c[0]), np.full(np.shape(y), self.c[1])


def _expected(c, frames, rots=ROTS):
    """Mean over ``frames`` (indices into rots) of the sky vector inv(R) @ c, mas."""
    return np.mean([np.linalg.inv(rots[k]) @ np.asarray(c) for k in frames], axis=0)


def _random_xy(seed=7, half=20.0, step=2.2, jitter=0.4, avoid=((1.0, 1.0), (-3.0, 4.0), (5.0, -6.0))):
    """Rows on a jittered grid in a 2*half arcsec square: every pair is at least step - 2*jitter = 1.4 arcsec apart,
    so association is not what most tests probe, and none is within 1 arcsec of the ``avoid`` points, where the
    tests place rows of their own."""
    rng = np.random.default_rng(seed)
    g = np.arange(-half, half, step)
    xy = np.array([(x, y) for x in g for y in g]) + rng.uniform(-jitter, jitter, (len(g) ** 2, 2))
    far = np.all([np.hypot(*(xy - np.asarray(p)).T) > 1.0 for p in avoid], axis=0)
    return xy[far]


def _write_catalog(path, xy, sat, nmatch, nmatch_good, suffix=''):
    ra, dec = _to_sky(xy)
    t = Table()
    t[f'skycoord{suffix}.ra'] = ra
    t[f'skycoord{suffix}.dec'] = dec
    t[f'nmatch{suffix}'] = np.asarray(nmatch, np.int16)
    t[f'nmatch_good{suffix}'] = np.asarray(nmatch_good, np.int16)
    t['is_saturated'] = np.asarray(sat, bool)
    t.write(path, overwrite=True)


def _write_frame(path, xy, R, flux=None, detector='NRCA1', centre=(1024.0, 1024.0)):
    ra, dec = _to_sky(xy)
    pix = np.asarray(centre) + (R @ xy.T).T / PIXSCALE
    t = Table()
    t['x_fit'] = pix[:, 0]
    t['y_fit'] = pix[:, 1]
    t['skycoord_centroid.ra'] = ra
    t['skycoord_centroid.dec'] = dec
    t['flux_fit'] = np.full(len(xy), 100.0, np.float32) if flux is None else np.asarray(flux, np.float32)
    t.meta['PIXSCALE'] = PIXSCALE
    t.meta['DETECTOR'] = detector
    t.write(path, overwrite=True)


def _make_field(tmp, rots=ROTS, xy=None, missing=None, no_det=(), shifts=None, flux=None, sat=(), good=None,
                suffix='', tag=''):
    """Write a catalogue and one frame catalogue per rotation.

    missing: {frame: rows not detected in that frame}; no_det: rows detected in no frame; shifts: {(frame, row):
    (east, north) displacement of that detection in mas}; flux: {(frame, row): flux_fit}; sat: saturated rows;
    good: {row: nmatch_good the catalogue carries} (default: every detection is good)."""
    xy = _random_xy() if xy is None else xy
    missing, shifts, flux, good = missing or {}, shifts or {}, flux or {}, good or {}
    n = len(xy)
    nmatch = np.zeros(n, int)
    frames = []
    for k, R in enumerate(rots):
        det = np.array([i for i in range(n) if i not in no_det and i not in missing.get(k, ())])
        pos = xy[det].copy()
        fl = np.full(len(det), 100.0)
        for j, i in enumerate(det):
            if (k, i) in shifts:
                pos[j] += np.asarray(shifts[(k, i)]) / 1000.0
            if (k, i) in flux:
                fl[j] = flux[(k, i)]
        nmatch[det] += 1
        path = tmp / f'f212n_nrca1_{tag}exp{k:05d}.fits'
        _write_frame(path, pos, R, fl)
        frames.append(str(path))
    ngood = nmatch.copy()
    for i, g in good.items():
        ngood[i] = g
    sat_mask = np.zeros(n, bool)
    sat_mask[list(sat)] = True
    cat = tmp / f'cat{tag}.fits'
    _write_catalog(cat, xy, sat_mask, nmatch, ngood, suffix)
    return SimpleNamespace(cat=str(cat), frames=frames, xy=xy, n=n)


def _run(f, solution=None, out=None, **kw):
    out = out or os.path.join(os.path.dirname(f.cat), 'out', 'poscorr.fits')
    return MP.run(f.cat, f.frames, solution or _Const(*C), 0.10, str(out), verbose=False, **kw)


def test_a_rows_correction_is_the_mean_of_its_frames_sky_vectors(tmp_path):
    f = _make_field(tmp_path)
    t = _run(f)
    exp = _expected(C, range(len(ROTS)))
    assert not np.allclose(exp, C, atol=0.05)           # the frame rotations matter: not the detector-axis vector
    np.testing.assert_allclose(t['dxi_mas'], exp[0], atol=1e-3)
    np.testing.assert_allclose(t['deta_mas'], exp[1], atol=1e-3)
    assert (t['estimate'] == 1).all()
    assert (t['n_assoc'] == len(ROTS)).all() and (t['n_keep'] == len(ROTS)).all()
    assert (t['nmatch_cat'] == len(ROTS)).all() and (t['set_consistent'] == 1).all()
    assert np.abs(t['recon_dxi_mas']).max() < 1e-3 and np.abs(t['recon_deta_mas']).max() < 1e-3


def test_a_mirrored_frame_flips_the_sky_direction_of_the_correction(tmp_path):
    f = _make_field(tmp_path, rots=[_rot(0.0, mirror=True)])
    t = _run(f, _Const(1.0, 0.0))
    np.testing.assert_allclose(t['dxi_mas'], -1.0, atol=1e-3)     # +x of a mirrored frame is west
    np.testing.assert_allclose(t['deta_mas'], 0.0, atol=1e-3)


def test_header_guards_and_the_builders_reader(tmp_path):
    f = _make_field(tmp_path)
    t = _run(f)
    ra, dec = MP.catalogue_positions(f.cat)
    assert (t.meta['RACHK'], t.meta['DECCHK']) == PC.position_checksums(ra, dec)
    assert t.meta['NROWS'] == f.n and t.meta['NFRAMES'] == len(ROTS) and t.meta['NDET'] == len(ROTS) * f.n
    assert t.meta['SOLUTION'] == 'const' and t.meta['RADIUS'] == 0.10 and t.meta['MODE'] == 'clip'
    assert t.meta['F_ASSOC'] == 1.0 and t.meta['F_SETOK'] == 1.0
    dxi, deta, meta = PC.load_poscorr(str(tmp_path / 'out' / 'poscorr.fits'), ra, dec)
    np.testing.assert_array_equal(dxi, t['dxi_mas'])
    np.testing.assert_array_equal(deta, t['deta_mas'])
    assert meta['CATALOG'] == f.cat
    assert [str(c.dtype) for c in (t['row'], t['dxi_mas'], t['n_assoc'], t['estimate'])] == [
        'int32', 'float32', 'int16', 'int16']
    with pytest.raises(PC.PosCorrMismatchError):                      # another catalogue's positions are refused
        PC.load_poscorr(str(tmp_path / 'out' / 'poscorr.fits'), ra + 1e-6, dec)


def test_a_displaced_detection_leaves_the_clipped_mean_but_stays_in_the_all_column(tmp_path):
    rots = ROTS + [_rot(70.0), _rot(300.0)]
    rows = range(10)
    f = _make_field(tmp_path, rots=rots, shifts={(5, i): (60.0, 0.0) for i in rows}, good={i: 5 for i in rows})
    t = _run(f)
    kept, every = _expected(C, range(5), rots), _expected(C, range(6), rots)
    assert not np.allclose(kept, every, atol=0.02)
    sel = np.arange(10)
    assert (t['n_assoc'][sel] == 6).all() and (t['n_keep'][sel] == 5).all()
    np.testing.assert_allclose(t['dxi_mas'][sel], kept[0], atol=1e-3)
    np.testing.assert_allclose(t['deta_mas'][sel], kept[1], atol=1e-3)
    np.testing.assert_allclose(t['dxi_all_mas'][sel], every[0], atol=1e-3)
    np.testing.assert_allclose(t['deta_all_mas'][sel], every[1], atol=1e-3)
    assert np.abs(t['recon_dxi_mas'][sel]).max() < 1e-3               # kept positions reproduce the catalogue
    assert (t['set_consistent'] == 1).all()
    assert (t['n_keep'][10:] == 6).all()
    ta = _run(f, mode='all')
    np.testing.assert_array_equal(ta['dxi_mas'], t['dxi_all_mas'])
    assert ta.meta['MODE'] == 'all'


def test_a_row_whose_every_detection_is_clipped_falls_back_to_all_of_them(tmp_path):
    # row 0: frame 0 has an outlying flux, frame 1 an outlying east position, frame 2 an outlying north position,
    # so each clip removes a different detection and none is left.
    f = _make_field(tmp_path, rots=ROTS[:3], shifts={(1, 0): (50.0, 0.0), (2, 0): (0.0, 50.0)},
                    flux={(0, 0): 1000.0})
    t = _run(f)
    assert t['n_assoc'][0] == 3 and t['n_keep'][0] == 3
    assert (t['n_keep'][1:] == 3).all()


def _field_with_unseen_rows(tmp_path, **kw):
    xy = np.vstack([_random_xy(), [[1.0, 1.0], [700.0, 0.0], [-3.0, 4.0], [5.0, -6.0]]])
    n = len(xy) - 4
    a, b, c, d = n, n + 1, n + 2, n + 3
    # a: detected nowhere, inside every footprint;  b: detected nowhere, outside every footprint;
    # c: saturated, detected in frames 0-1 only;  d: not saturated, detected in frames 2-3 only
    f = _make_field(tmp_path, xy=xy, no_det=(a, b), missing={2: [c], 3: [c], 0: [d], 1: [d]}, sat=(c,), **kw)
    return f, (a, b, c, d)


def test_rows_with_no_detection_and_saturated_rows_get_the_footprint_estimate(tmp_path):
    f, (a, b, c, d) = _field_with_unseen_rows(tmp_path)
    t = _run(f)
    every = _expected(C, range(4))
    assert t['estimate'][a] == 2 and t['n_assoc'][a] == 0 and t['n_foot'][a] == 4
    np.testing.assert_allclose([t['dxi_mas'][a], t['deta_mas'][a]], every, atol=1e-3)
    assert t['estimate'][b] == 0 and t['n_foot'][b] == 0
    assert t['dxi_mas'][b] == 0 and t['deta_mas'][b] == 0 and t['foot_dxi_mas'][b] == 0
    assert t['estimate'][c] == 2 and t['n_assoc'][c] == 2          # saturated: footprint, although detected twice
    np.testing.assert_allclose([t['dxi_mas'][c], t['deta_mas'][c]], every, atol=1e-3)
    np.testing.assert_allclose([t['dxi_all_mas'][c], t['deta_all_mas'][c]], every, atol=1e-3)
    seen = _expected(C, (2, 3))
    assert not np.allclose(seen, every, atol=0.02)
    assert t['estimate'][d] == 1 and t['n_assoc'][d] == 2          # not saturated: the detection mean ...
    np.testing.assert_allclose([t['dxi_mas'][d], t['deta_mas'][d]], seen, atol=1e-3)
    np.testing.assert_allclose([t['foot_dxi_mas'][d], t['foot_deta_mas'][d]], every, atol=1e-3)   # ... footprint kept
    assert t['n_foot'][d] == 4
    assert (t['estimate'][:f.n - 4] == 1).all()
    assert t.meta['F_ASSOC'] == pytest.approx(1.0 - 2 / f.n)


def test_extra_frames_only_widen_the_footprint(tmp_path):
    f, (a, b, c, d) = _field_with_unseen_rows(tmp_path)
    xyb = _random_xy(seed=11)
    rots_b = [_rot(10.0), _rot(190.0, mirror=True)]
    fb = _make_field(tmp_path, rots=rots_b, xy=xyb, tag='b')
    t0 = _run(f)
    t1 = _run(f, extra=[(fb.cat, fb.frames)])
    assert t1['n_foot'][a] == 6 and t0['n_foot'][a] == 4
    both = np.mean([np.linalg.inv(R) @ np.asarray(C) for R in ROTS + rots_b], axis=0)
    np.testing.assert_allclose([t1['dxi_mas'][a], t1['deta_mas'][a]], both, atol=1e-3)
    assert t1['n_foot'][b] == 0
    for col in ('n_assoc', 'n_keep', 'estimate'):
        np.testing.assert_array_equal(t1[col], t0[col])
    detected = np.arange(f.n - 4)
    np.testing.assert_array_equal(t1['dxi_mas'][detected], t0['dxi_mas'][detected])


def test_a_frame_with_too_few_associated_detections_is_skipped(tmp_path):
    f = _make_field(tmp_path)
    sparse = tmp_path / 'f212n_nrca1_exp99999.fits'
    _write_frame(sparse, f.xy[:MP.MIN_ASSOCIATED - 1], ROTS[0])
    t = _run(f, out=str(tmp_path / 'out' / 'p.fits'))
    t2 = MP.run(f.cat, f.frames + [str(sparse)], _Const(*C), 0.10, str(tmp_path / 'out' / 'p2.fits'), verbose=False)
    assert t2.meta['NDET'] == t.meta['NDET']                          # the skipped frame contributes no detection
    assert t2['n_assoc'].max() == len(ROTS)
    np.testing.assert_array_equal(t2['dxi_mas'], t['dxi_mas'])


def test_set_consistent_needs_both_the_matched_and_the_kept_count_of_the_merge(tmp_path):
    rots = ROTS + [_rot(70.0), _rot(300.0)]
    f = _make_field(tmp_path, rots=rots, shifts={(5, i): (60.0, 0.0) for i in range(10)})   # clipped here: n_keep 5
    c = Table.read(f.cat)
    c['nmatch'][20:25] += 1                       # the merge matched a detection that the association does not
    c.write(f.cat, overwrite=True)
    t = _run(f)
    expected = np.ones(f.n, int)
    expected[:10] = 0                             # nmatch_good 6 in the catalogue (nothing clipped), n_keep 5 here
    expected[20:25] = 0                           # nmatch 7 in the catalogue, n_assoc 6 here
    np.testing.assert_array_equal(t['set_consistent'], expected)
    assert t.meta['F_SETOK'] == pytest.approx(1 - 15 / f.n)


def test_a_catalogue_without_the_merges_counts_gets_no_count_columns_and_no_f_setok_card(tmp_path):
    f = _make_field(tmp_path)
    c = Table.read(f.cat)
    c.remove_columns(['nmatch', 'nmatch_good'])
    c.write(f.cat, overwrite=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        t = _run(f)
    assert not [w for w in caught if 'F_SETOK' in str(w.message)]      # astropy skips (and warns about) a NaN card
    assert not {'nmatch_cat', 'nmatch_good_cat', 'set_consistent'} & set(t.colnames)
    assert 'F_SETOK' not in t.meta and t.meta['F_ASSOC'] == 1.0
    ra, dec = MP.catalogue_positions(f.cat)
    dxi, deta, meta = PC.load_poscorr(str(tmp_path / 'out' / 'poscorr.fits'), ra, dec)
    np.testing.assert_array_equal(dxi, t['dxi_mas'])
    assert 'F_SETOK' not in meta and meta['POSCONV'].startswith('add dxi_mas')


def test_suffixed_catalogue_columns_are_found_by_filter(tmp_path):
    f = _make_field(tmp_path, suffix='_f212n')
    t = _run(f)
    assert (t['n_assoc'] == len(ROTS)).all() and (t['nmatch_cat'] == len(ROTS)).all()
    with pytest.raises(KeyError, match='skycoord_f323n'):
        MP.catalogue_positions(f.cat, 'f323n')


def _brute_force_mutual(fr_xy, cat_xy, radius):
    """Mutual nearest neighbours within radius from the full distance matrix (flat sky; positions in arcsec)."""
    d = np.hypot(fr_xy[:, None, 0] - cat_xy[None, :, 0], fr_xy[:, None, 1] - cat_xy[None, :, 1])
    nearest_row, nearest_det = d.argmin(axis=1), d.argmin(axis=0)
    ok = (nearest_det[nearest_row] == np.arange(len(fr_xy))) & (d[np.arange(len(fr_xy)), nearest_row] < radius)
    return set(zip(np.nonzero(ok)[0].tolist(), nearest_row[ok].tolist()))


def test_association_is_mutual_nearest_within_the_radius():
    cat = np.array([[0.0, 0.0], [0.0, 1.0]])
    det = np.array([[0.010, 0.0],       # nearest row 0 (10 mas), and row 0's nearest detection
                    [0.020, 0.0],       # also nearest to row 0, but not row 0's nearest detection: contested
                    [0.0, 1.150]])      # nearest row 1 but 150 mas away: outside the radius
    ia, ib = MP.associate(*_to_sky(det), *_to_sky(cat), 0.10)
    assert list(zip(ia.tolist(), ib.tolist())) == [(0, 0)]


def test_association_matches_a_brute_force_mutual_match_in_a_dense_field():
    rng = np.random.default_rng(5)
    cat = rng.uniform(0, 3.0, (400, 2))                          # ~0.15" between rows: many contested detections
    det = cat[rng.choice(400, 300, replace=False)] + rng.normal(0, 0.05, (300, 2))
    ia, ib = MP.associate(*_to_sky(det), *_to_sky(cat), 0.10)
    got = set(zip(ia.tolist(), ib.tolist()))
    # SkyCoord separations and the flat-sky distances differ by < 1e-6 arcsec here: no pair sits that close to 0.1"
    assert got == _brute_force_mutual(det, cat, 0.10)
    assert 100 < len(got) < 300
    assert len(set(ia.tolist())) == len(ia) and len(set(ib.tolist())) == len(ib)


def _write_maps(path, dx=2.0, dy=-1.0, dets=('NRCA1',)):
    arrays = {}
    for kind, s in (('loofull', 1.0), ('dqfull', 3.0)):
        for det in dets:
            arrays[f'{kind}_{det}_dx'] = np.full((16, 16), dx * s)
            arrays[f'{kind}_{det}_dy'] = np.full((16, 16), dy * s)
    np.savez(path, **arrays)
    return str(path)


def test_the_map_solution_is_minus_the_map_of_the_requested_kind(tmp_path):
    npz = _write_maps(tmp_path / 'maps.npz')
    sol = MP.Solution('map', npz, 'loofull')
    cx, cy = sol('nrca1', np.array([10.0, 1500.0]), np.array([300.0, 2000.0]))
    np.testing.assert_allclose(cx, -2.0)
    np.testing.assert_allclose(cy, 1.0)
    cx, cy = MP.Solution('map', npz, 'dqfull')('NRCA1', [100.0], [100.0])
    np.testing.assert_allclose([cx[0], cy[0]], [-6.0, 3.0])
    assert (sol.name, sol.map_file, sol.map_kind) == ('map', npz, 'loofull')
    z = MP.Solution('zero')
    assert z.name == 'zero' and z.map_file == '' and z.map_kind == ''
    assert not np.any(z('NRCB2', [5.0], [6.0]))


def test_the_map_solution_refuses_what_it_cannot_evaluate(tmp_path):
    npz = _write_maps(tmp_path / 'maps.npz')
    with pytest.raises(ValueError, match='NRCB2'):                  # a detector the npz has no map for
        MP.Solution('map', npz, 'loofull')('NRCB2', [1.0], [1.0])
    with pytest.raises(ValueError, match='dqfull'):                 # no maps of that kind: the kinds present are named
        MP.Solution('map', npz, 'nosuchkind')
    with pytest.raises(ValueError, match='gdc_v1'):
        MP.Solution('gdc_v1')
    with pytest.raises(ValueError, match='map_kind'):
        MP.Solution('map', npz)


def test_map_is_bilinear_between_cell_centres_and_extrapolates_linearly_beyond_them():
    g = np.tile(np.arange(16.0)[:, None], (1, 16))                   # value = ix
    bm = MP.BinnedMapField(g, -g)
    c = (np.arange(16) + 0.5) * 128.0
    cx, cy = bm(c[3], 1000.0)
    assert (cx, cy) == (pytest.approx(3.0), pytest.approx(-3.0))
    assert bm(0.5 * (c[3] + c[4]), 1000.0)[0] == pytest.approx(3.5)
    assert bm(0.0, 1000.0)[0] == pytest.approx(-0.5)                 # outer half cell: extrapolated, not held at 0
    assert bm(2047.0, 1000.0)[0] == pytest.approx(15.0 + (2047.0 - c[15]) / 128.0)


@pytest.mark.filterwarnings('ignore:Mean of empty slice')            # nanmean of an all-NaN map: the case tested
def test_map_nan_cells_are_filled_from_their_neighbours_then_zero():
    g = np.full((16, 16), 2.0)
    g[5, 5] = np.nan
    assert MP.BinnedMapField(g, g)(5.5 * 128.0, 5.5 * 128.0)[0] == pytest.approx(2.0)
    assert MP.BinnedMapField(np.full((16, 16), np.nan), g)(100.0, 100.0)[0] == 0.0


def test_main_writes_a_table_with_the_zero_and_map_solutions(tmp_path):
    f = _make_field(tmp_path)
    frame_args = ['--catalog', f.cat, '--frame-dir', str(tmp_path), '--frame-regex', r'^f212n_nrca1_exp\d+\.fits$']
    out = tmp_path / 'hooks' / 'poscorr.fits'
    ra, dec = MP.catalogue_positions(f.cat)

    MP.main(frame_args + ['--solution', 'zero', '--out', str(out)])
    dxi, deta, meta = PC.load_poscorr(str(out), ra, dec)
    assert not dxi.any() and not deta.any() and meta['SOLUTION'] == 'zero'

    npz = _write_maps(tmp_path / 'maps.npz')
    MP.main(frame_args + ['--solution', 'map', '--map-npz', npz, '--map-kind', 'loofull', '--out', str(out)])
    dxi, deta, meta = PC.load_poscorr(str(out), ra, dec)
    exp = _expected((-2.0, 1.0), range(len(ROTS)))                   # c = -M
    np.testing.assert_allclose(dxi, exp[0], atol=1e-3)
    np.testing.assert_allclose(deta, exp[1], atol=1e-3)
    assert (meta['SOLUTION'], meta['MAPFILE'], meta['MAPKIND']) == ('map', npz, 'loofull')
    assert os.listdir(out.parent) == ['poscorr.fits']                # the temporary of the atomic write is gone


def test_main_rejects_bad_invocations(tmp_path):
    f = _make_field(tmp_path)
    args = ['--catalog', f.cat, '--frame-dir', str(tmp_path), '--out', str(tmp_path / 'x.fits')]
    with pytest.raises(SystemExit) as e:                             # argparse error: map without its npz
        MP.main(args + ['--frame-regex', r'^f212n_nrca1_exp\d+\.fits$', '--solution', 'map'])
    assert e.value.code == 2
    with pytest.raises(SystemExit, match='no frames match'):
        MP.main(args + ['--frame-regex', r'^nothing$', '--solution', 'zero'])
    with pytest.raises(SystemExit, match='--extra'):
        MP.main(args + ['--frame-regex', r'^f212n_nrca1_exp\d+\.fits$', '--solution', 'zero',
                        '--extra', f'{f.cat}::^nothing$'])
    assert not (tmp_path / 'x.fits').exists()
