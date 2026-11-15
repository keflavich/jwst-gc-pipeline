"""Tests for the stringent m7 cross-band seed + independent-detection provenance.

Failure mode being guarded (2026-06-30): the legacy seed UNIONED every filter's
m6 detections and force-fit every filter at every position, so a single-band (or
structure) detection became a fake multi-band source.  The stringent seed only
keeps positions confirmed in >= min_filters; annotate_independent_detection flags
which per-band measurements are real m6 detections vs cross-band-seeded.
"""
import json
import os
import types
import numpy as np
import pytest
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

from jwst_gc_pipeline.photometry import cataloging as C


def _opts():
    return types.SimpleNamespace(
        desaturated=False, bgsub=False, blur=False,
        # 4147 (sgrc), NOT a brick proposal -- `_write_m6` below writes
        # UNTOKENED names and 1182/2221 are per-obs-merged now (#590).
        proposal_id='4147', field='012', modules='merged')


def _write_m6(cut_bp, filt, ras, decs, snr=20.0, qfit=0.05, module='merged'):
    os.makedirs(f'{cut_bp}/catalogs', exist_ok=True)
    n = len(ras)
    t = Table()
    t['skycoord'] = SkyCoord(np.array(ras) * u.deg, np.array(decs) * u.deg)
    flux = np.full(n, 1000.0)
    t['flux'] = flux
    t['flux_err'] = flux / snr
    t['qfit'] = np.full(n, qfit)
    p = f'{cut_bp}/catalogs/{filt}_{module}_indivexp_merged_resbgsub_m6_dao_basic_vetted.fits'
    t.write(p, overwrite=True)
    return p


def test_stringent_seed_drops_single_band(tmp_path):
    cut_bp = str(tmp_path)
    # shared multi-band source at (10.0000, 20.0000); single-band only in f405n
    multi = (10.00000, 20.00000)
    single = (10.00100, 20.00100)   # ~3.6" away, f405n only
    _write_m6(cut_bp, 'f405n', [multi[0], single[0]], [multi[1], single[1]])
    _write_m6(cut_bp, 'f410m', [multi[0] + 5e-6], [multi[1]])      # ~16 mas off -> same cluster
    _write_m6(cut_bp, 'f187n', [multi[0]], [multi[1] + 5e-6])

    out = C._build_crossband_seed(cut_bp, ['merged'], ['f405n', 'f410m', 'f187n'], _opts())
    seed = Table.read(out)
    # only the multi-band position survives (>=2 filters); single-band dropped
    assert len(seed) == 1
    assert seed['n_filt_confirmed'][0] >= 2
    sc = SkyCoord(seed['skycoord'])
    assert sc.separation(SkyCoord(multi[0] * u.deg, multi[1] * u.deg)).arcsec[0] < 0.05


def test_stringent_seed_rejects_low_snr_and_bad_qfit(tmp_path):
    cut_bp = str(tmp_path)
    pos = (10.0, 20.0)
    # confirmed in f405n (good) but f410m is low-SNR and f187n is bad-qfit -> only 1 good filter
    _write_m6(cut_bp, 'f405n', [pos[0]], [pos[1]], snr=20, qfit=0.05)
    _write_m6(cut_bp, 'f410m', [pos[0]], [pos[1]], snr=2.0, qfit=0.05)   # SNR<5 -> not confirmed
    _write_m6(cut_bp, 'f187n', [pos[0]], [pos[1]], snr=20, qfit=0.9)     # qfit>0.2 -> not confirmed
    with pytest.raises(ValueError):
        C._build_crossband_seed(cut_bp, ['merged'], ['f405n', 'f410m', 'f187n'], _opts())


def test_min_filters_one_keeps_single_band(tmp_path):
    cut_bp = str(tmp_path)
    _write_m6(cut_bp, 'f405n', [10.0], [20.0])
    o = _opts(); o.manual_crossband_seed_min_filters = 1
    out = C._build_crossband_seed(cut_bp, ['merged'], ['f405n', 'f410m', 'f187n'], o)
    assert len(Table.read(out)) == 1   # legacy-like behavior recoverable


def test_annotate_independent_detection(tmp_path):
    cut_bp = str(tmp_path)
    # merged catalog: row0 at a real f405n m6 position; row1 at a seeded-only position
    real = (10.0, 20.0)
    seeded = (10.05, 20.05)
    m = Table()
    m['skycoord_ref'] = SkyCoord([real[0], seeded[0]] * u.deg, [real[1], seeded[1]] * u.deg)
    mp = f'{cut_bp}/catalogs/merged_test.fits'
    os.makedirs(f'{cut_bp}/catalogs', exist_ok=True)
    m.write(mp, overwrite=True)
    _write_m6(cut_bp, 'f405n', [real[0]], [real[1]])    # only the real one has an m6 detection
    _write_m6(cut_bp, 'f410m', [real[0]], [real[1]])
    C.annotate_independent_detection(mp, cut_bp, ['f405n', 'f410m'], _opts())
    out = Table.read(mp)
    assert bool(out['independently_detected_f405n'][0]) is True
    assert bool(out['independently_detected_f405n'][1]) is False
    assert out['n_filt_independent'][0] == 2
    assert out['n_filt_independent'][1] == 0


# ---------------------------------------------------------------------------
# The m7 fan-out resume needs to know WHICH files the seed is built from, so a
# marker written against a superseded seed can be refit (#570 review).  The seed
# FILE itself is no use: every m7 shard rewrites it at the same path, so its
# mtime is always "now".  `crossband_seed_inputs` is that list, and the builder
# consumes it -- one spelling, or the resume watches files the seed never read.
# ---------------------------------------------------------------------------

def test_the_seed_inputs_are_the_files_the_builder_reads(tmp_path):
    cut_bp = str(tmp_path)
    written = [_write_m6(cut_bp, f, [10.0], [20.0])
               for f in ('f405n', 'f410m', 'f187n')]
    got = C.crossband_seed_inputs(cut_bp, ['merged'],
                                  ['f405n', 'f410m', 'f187n'], _opts())
    assert [p for _m, _f, p in got] == written


def test_a_filter_with_no_m6_catalog_is_not_listed(tmp_path):
    """Only existing inputs: a path that was never written cannot make a marker
    stale, and the builder skips it too."""
    cut_bp = str(tmp_path)
    _write_m6(cut_bp, 'f405n', [10.0], [20.0])
    got = C.crossband_seed_inputs(cut_bp, ['merged'],
                                  ['f405n', 'f480m'], _opts())
    assert [os.path.basename(p) for _m, _f, p in got] == [
        'f405n_merged_indivexp_merged_resbgsub_m6_dao_basic_vetted.fits']


def test_the_builder_reads_the_seed_input_list(tmp_path):
    """Behavioural: hide one filter's catalog from the list and the seed loses
    that filter's confirmation, so a 2-filter position drops out."""
    cut_bp = str(tmp_path)
    _write_m6(cut_bp, 'f405n', [10.0], [20.0])
    _write_m6(cut_bp, 'f410m', [10.0], [20.0])
    out = C._build_crossband_seed(cut_bp, ['merged'], ['f405n', 'f410m'],
                                  _opts())
    assert len(Table.read(out)) == 1
    os.remove(C.crossband_seed_inputs(cut_bp, ['merged'], ['f410m'],
                                      _opts())[0][2])
    with pytest.raises(ValueError):
        C._build_crossband_seed(cut_bp, ['merged'], ['f405n', 'f410m'], _opts())


# -- seed health (issue #981) ------------------------------------------------

@pytest.fixture(autouse=True)
def _fresh_collapse_warning_guard():
    """The once-per-process warning guard is module state: clear it around each
    test so a test never inherits (or leaves behind) a suppressed warning."""
    C._CROSSBAND_SEED_COLLAPSE_WARNED.clear()
    yield
    C._CROSSBAND_SEED_COLLAPSE_WARNED.clear()


def _health_path(out):
    return out.replace('.fits', '.health.json')


def _load_health(out):
    with open(_health_path(out)) as fh:
        return json.load(fh)


def _stars(n, seed=7):
    rng = np.random.default_rng(seed)
    return (266.0 + rng.uniform(0, 0.02, n), -28.9 + rng.uniform(0, 0.02, n))


def _shifted(ra, dec, shift_mas):
    return ra + shift_mas / 3.6e6 / np.cos(np.radians(dec)), dec


def _two_band_field(cut_bp, n=2000, shift_mas=0.0, seed=7):
    """n well-separated stars in F212N; F480M is the same stars shifted in RA."""
    ra, dec = _stars(n, seed)
    _write_m6(cut_bp, 'f212n', ra, dec)
    _write_m6(cut_bp, 'f480m', *_shifted(ra, dec, shift_mas))
    return ['f212n', 'f480m']


def _sparse_pair(cut_bp, n_a, n_b, n_aligned):
    """Two bands of isolated stars on a 10 arcsec grid: the first ``n_aligned``
    of each band coincide, the rest sit 5 arcsec from anything in the other
    band.  Band fractions are exactly n_aligned / n_a and n_aligned / n_b."""
    step = 10.0 / 3600.0
    ra_a = 266.0 + step * np.arange(n_a)
    ra_b = np.concatenate([ra_a[:n_aligned],
                           266.0 + step * (np.arange(n_a, n_a + n_b - n_aligned) + 0.5)])
    dec = np.full(max(n_a, n_b), -28.9)
    _write_m6(cut_bp, 'f212n', ra_a, dec[:n_a])
    _write_m6(cut_bp, 'f480m', ra_b, dec[:n_b])
    return ['f212n', 'f480m']


def test_seed_health_collapse_warns_and_records_offset(tmp_path, capsys):
    cut_bp = str(tmp_path)
    filts = _two_band_field(cut_bp, shift_mas=80.0)
    out = C._build_crossband_seed(cut_bp, ['merged'], filts, _opts())
    text = capsys.readouterr().out
    assert 'WARNING [m7] crossband seed collapsed' in text
    health = _load_health(out)
    assert health['collapsed'] is True
    assert health['frac'] < 0.3
    assert health['n_good'] == {'f212n': 2000, 'f480m': 2000}
    assert health['offset_filters'] == ['f212n', 'f480m']
    assert health['offset_ok'] is True
    assert 0.0 <= health['offset_window_edge_fraction'] <= 1.0
    assert health['offset_window_edge_fraction'] < 0.5
    assert abs(health['offset_dra_mas'] - 80.0) < 12.0
    assert abs(health['offset_ddec_mas']) < 12.0
    # the printed line carries the recorded number, not just a prefix
    assert f"dRA*cos(dec)={health['offset_dra_mas']:.1f} mas" in text
    assert 'f480m-f212n offset' in text
    # the meta mirrors the JSON
    meta = Table.read(out).meta
    assert meta['SEEDCOLL'] is True
    assert meta['SEEDFRAC'] == pytest.approx(health['frac'])
    assert meta['SEEDNGD'] == 2000
    assert meta['SEEDOFFX'] == pytest.approx(health['offset_dra_mas'])
    assert meta['SEEDOFFY'] == pytest.approx(health['offset_ddec_mas'])


def test_offset_beyond_the_window_is_not_printed_as_a_measurement(tmp_path, capsys):
    """A shift larger than the 0.3 arcsec window gives the highest noise bin
    (ok False); a log line with that number is what someone types into an
    offsets table.  Nothing is printed or stored as an offset."""
    cut_bp = str(tmp_path)
    filts = _two_band_field(cut_bp, n=3000, shift_mas=1000.0)
    out = C._build_crossband_seed(cut_bp, ['merged'], filts, _opts())
    text = capsys.readouterr().out
    assert 'WARNING [m7] crossband seed collapsed' in text
    assert 'dRA*cos(dec)' not in text
    assert 'No coherent f480m-f212n offset peak' in text
    health = _load_health(out)
    assert health['offset_ok'] is False
    assert health['offset_dra_mas'] is None and health['offset_ddec_mas'] is None
    assert health['offset_contrast'] is not None
    meta = Table.read(out).meta
    assert 'SEEDOFFX' not in meta and 'SEEDOFFY' not in meta


def test_seed_health_collapse_action_raise(tmp_path):
    cut_bp = str(tmp_path)
    filts = _two_band_field(cut_bp, shift_mas=80.0)
    o = _opts(); o.manual_crossband_seed_collapse_action = 'raise'
    with pytest.raises(C.CrossbandSeedCollapsedError):
        C._build_crossband_seed(cut_bp, ['merged'], filts, o)
    # the seed and its health record are on disk for inspection
    assert os.path.exists(C.crossband_seed_file(cut_bp, o))
    assert os.path.exists(_health_path(C.crossband_seed_file(cut_bp, o)))


def test_seed_health_aligned_bands_are_quiet(tmp_path, capsys):
    cut_bp = str(tmp_path)
    filts = _two_band_field(cut_bp, shift_mas=0.0)
    o = _opts(); o.manual_crossband_seed_collapse_action = 'raise'
    out = C._build_crossband_seed(cut_bp, ['merged'], filts, o)
    assert 'collapsed' not in capsys.readouterr().out
    health = _load_health(out)
    assert health['collapsed'] is False
    assert health['frac'] > 0.95
    assert abs(health['offset_dra_mas']) < 12.0


def test_seed_health_few_pairs_skips_offset(tmp_path):
    cut_bp = str(tmp_path)
    filts = _two_band_field(cut_bp, n=50, shift_mas=0.0)
    out = C._build_crossband_seed(cut_bp, ['merged'], filts, _opts())
    health = _load_health(out)
    assert health['offset_dra_mas'] is None and health['offset_ok'] is None
    assert health['frac'] > 0.95


def test_seed_health_single_filter_has_no_fraction(tmp_path, capsys):
    cut_bp = str(tmp_path)
    _write_m6(cut_bp, 'f405n', [10.0, 10.001], [20.0, 20.001])
    o = _opts(); o.manual_crossband_seed_min_filters = 1
    out = C._build_crossband_seed(cut_bp, ['merged'], ['f405n'], o)
    health = _load_health(out)
    assert health['frac'] is None and health['collapsed'] is False
    assert 'collapsed' not in capsys.readouterr().out


def test_seed_health_json_is_not_a_seed_input(tmp_path):
    """Build a seed so the JSON exists, then check every reader of the catalog
    directory that matters for the seed: the input list and the seed globs."""
    import glob
    cut_bp = str(tmp_path)
    filts = _two_band_field(cut_bp, n=50)
    out = C._build_crossband_seed(cut_bp, ['merged'], filts, _opts())
    assert os.path.exists(_health_path(out))
    inputs = [p for _m, _f, p in C.crossband_seed_inputs(
        cut_bp, ['merged'], filts, _opts())]
    assert len(inputs) == 2
    assert not any(p.endswith('.json') for p in inputs)
    assert glob.glob(f'{cut_bp}/catalogs/crossband_seed_manual*.fits') == [out]
    # rebuilding with the JSON present reads the same inputs and gives the same seed
    first = Table.read(out)
    C._build_crossband_seed(cut_bp, ['merged'], filts, _opts())
    again = Table.read(out)
    assert np.array_equal(first['skycoord'].ra.deg, again['skycoord'].ra.deg)


def test_seed_health_rejects_unknown_action(tmp_path):
    cut_bp = str(tmp_path)
    filts = _two_band_field(cut_bp, n=50)
    o = _opts(); o.manual_crossband_seed_collapse_action = 'ignore'
    with pytest.raises(ValueError, match='collapse_action'):
        C._build_crossband_seed(cut_bp, ['merged'], filts, o)


def test_seed_health_defaults():
    from jwst_gc_pipeline.photometry.manual_defaults import MANUAL_DEFAULTS
    assert MANUAL_DEFAULTS['manual_crossband_seed_min_frac'] == 0.3
    assert MANUAL_DEFAULTS['manual_crossband_seed_collapse_action'] == 'warn'


def test_default_floor_separates_fraction_02_from_04(tmp_path, capsys):
    """With default options (no health attributes on the options object) a
    band confirmed at 0.2 is collapsed and one at 0.4 is not, so the default
    floor sits between them."""
    low = tmp_path / 'low'; high = tmp_path / 'high'
    low.mkdir(); high.mkdir()
    out = C._build_crossband_seed(str(low), ['merged'],
                                  _sparse_pair(str(low), 10, 10, 2), _opts())
    assert _load_health(out)['collapsed'] is True
    C._CROSSBAND_SEED_COLLAPSE_WARNED.clear()
    out = C._build_crossband_seed(str(high), ['merged'],
                                  _sparse_pair(str(high), 10, 10, 4), _opts())
    assert _load_health(out)['collapsed'] is False


def test_fraction_equal_to_the_floor_is_not_collapsed(tmp_path):
    cut_bp = str(tmp_path)
    filts = _sparse_pair(cut_bp, 10, 10, 3)
    o = _opts()
    out = C._build_crossband_seed(cut_bp, ['merged'], filts, o)
    h = _load_health(out)
    assert h['frac'] == pytest.approx(0.3) and h['collapsed'] is False
    o.manual_crossband_seed_min_frac = 0.31
    out = C._build_crossband_seed(cut_bp, ['merged'], filts, o)
    assert _load_health(out)['collapsed'] is True


def test_unequal_depth_bands_share_the_smallest_denominator(tmp_path):
    """Band A has 10 good detections, band B 20, and 4 of each are confirmed: the
    fraction is 4 / min(n_good) = 0.4 for both, not 4/20 = 0.2 for the deeper
    band (its extra detections can never be confirmed by the shallower one)."""
    cut_bp = str(tmp_path)
    filts = _sparse_pair(cut_bp, 10, 20, 4)
    out = C._build_crossband_seed(cut_bp, ['merged'], filts, _opts())
    h = _load_health(out)
    assert h['band_frac'] == {'f212n': pytest.approx(0.4), 'f480m': pytest.approx(0.4)}
    assert h['frac'] == pytest.approx(0.4) and h['collapsed'] is False
    assert h['n_good'] == {'f212n': 10, 'f480m': 20}
    assert Table.read(out).meta['SEEDNGD'] == 10


def test_unequal_depth_collapse_is_still_flagged(tmp_path):
    """Same depths, 2 confirmed: 2 / min(n_good) = 0.2 < 0.3 is a collapse."""
    cut_bp = str(tmp_path)
    filts = _sparse_pair(cut_bp, 10, 20, 2)
    h = _load_health(C._build_crossband_seed(cut_bp, ['merged'], filts, _opts()))
    assert h['frac'] == pytest.approx(0.2) and h['collapsed'] is True


def test_deep_band_does_not_trip_a_healthy_pair(tmp_path):
    """gc-treasury shape (#981 review): the bands differ 2.4x in
    good detections, 70% of the smaller band confirmed.  The pooled form gave 0.70; dividing the deep band by
    its own n_good gave 0.29 and a false collapse warning."""
    cut_bp = str(tmp_path)
    filts = _sparse_pair(cut_bp, 100, 240, 70)
    h = _load_health(C._build_crossband_seed(cut_bp, ['merged'], filts, _opts()))
    assert h['band_frac']['f480m'] == pytest.approx(0.70)
    assert h['frac'] == pytest.approx(0.70) and h['collapsed'] is False


def test_fraction_is_capped_when_a_cluster_holds_several_detections(tmp_path):
    """Every star of the 10-star band has two detections ~7 mas apart in the
    other band: that band has 20 detections in kept clusters against a smallest
    band of 10, and its fraction is capped at 1, not 2."""
    cut_bp = str(tmp_path)
    step = 10.0 / 3600.0
    ra_a = 266.0 + step * np.arange(10)
    ra_b = np.concatenate([ra_a, ra_a + 2e-6])
    _write_m6(cut_bp, 'f212n', ra_a, np.full(10, -28.9))
    _write_m6(cut_bp, 'f480m', ra_b, np.full(20, -28.9))
    out = C._build_crossband_seed(cut_bp, ['merged'], ['f212n', 'f480m'], _opts())
    h = _load_health(out)
    assert h['n_in_seed'] == {'f212n': 10, 'f480m': 20}
    assert h['band_frac'] == {'f212n': pytest.approx(1.0), 'f480m': pytest.approx(1.0)}
    assert h['collapsed'] is False


def test_good_detections_are_summed_over_modules(tmp_path):
    cut_bp = str(tmp_path)
    ra, dec = _stars(40)
    for mod, sl in (('nrca', slice(0, 20)), ('nrcb', slice(20, 40))):
        _write_m6(cut_bp, 'f212n', ra[sl], dec[sl], module=mod)
        _write_m6(cut_bp, 'f480m', ra[sl], dec[sl], module=mod)
    out = C._build_crossband_seed(cut_bp, ['nrca', 'nrcb'], ['f212n', 'f480m'], _opts())
    h = _load_health(out)
    assert h['n_good'] == {'f212n': 40, 'f480m': 40}
    assert h['n_in_seed'] == {'f212n': 40, 'f480m': 40}
    assert h['frac'] == pytest.approx(1.0)


def test_fraction_never_exceeds_one_with_overlapping_modules(tmp_path):
    """Three bands over two modules that both cover the same stars: clusters hold
    more detections than there are stars, but each band's fraction stays <= 1."""
    cut_bp = str(tmp_path)
    ra, dec = _stars(60)
    for mod in ('nrca', 'nrcb'):
        for filt in ('f115w', 'f200w', 'f444w'):
            _write_m6(cut_bp, filt, ra, dec, module=mod)
    out = C._build_crossband_seed(cut_bp, ['nrca', 'nrcb'],
                                  ['f115w', 'f200w', 'f444w'], _opts())
    h = _load_health(out)
    # every detection sits in a kept cluster: exactly 1, and never above it
    assert h['band_frac'] == {'f115w': 1.0, 'f200w': 1.0, 'f444w': 1.0}
    assert h['n_good'] == {'f115w': 120, 'f200w': 120, 'f444w': 120}
    assert h['collapsed'] is False


def _three_band_field(cut_bp, shift_f444w_mas, n=2000):
    """f115w and f200w agree; f444w is shifted.  Median S/N: f200w > f444w > f115w."""
    ra, dec = _stars(n)
    _write_m6(cut_bp, 'f115w', ra, dec, snr=10.0)
    _write_m6(cut_bp, 'f200w', ra, dec, snr=50.0)
    _write_m6(cut_bp, 'f444w', *_shifted(ra, dec, shift_f444w_mas), snr=30.0)
    return ['f115w', 'f200w', 'f444w']


def test_one_misregistered_band_of_three_is_flagged(tmp_path, capsys):
    cut_bp = str(tmp_path)
    filts = _three_band_field(cut_bp, 80.0)
    out = C._build_crossband_seed(cut_bp, ['merged'], filts, _opts())
    text = capsys.readouterr().out
    assert 'WARNING [m7] crossband seed collapsed' in text
    h = _load_health(out)
    assert h['collapsed'] is True and h['worst_band'] == 'f444w'
    assert h['band_frac']['f115w'] > 0.95 and h['band_frac']['f200w'] > 0.95
    assert h['frac'] < 0.3
    # the offset is the worst band against the highest-S/N other band (f200w,
    # not the faint f115w), ordered by S/N: f444w - f200w
    assert h['offset_filters'] == ['f200w', 'f444w']
    assert abs(h['offset_dra_mas'] - 80.0) < 12.0


def test_three_aligned_bands_are_quiet(tmp_path, capsys):
    cut_bp = str(tmp_path)
    filts = _three_band_field(cut_bp, 0.0)
    out = C._build_crossband_seed(cut_bp, ['merged'], filts, _opts())
    assert 'collapsed' not in capsys.readouterr().out
    h = _load_health(out)
    assert h['collapsed'] is False and h['frac'] > 0.95
    assert h['worst_band'] in h['offset_filters']


def test_collapse_warning_prints_once_per_seed_and_again_after_reset(tmp_path, capsys):
    cut_bp = str(tmp_path)
    filts = _two_band_field(cut_bp, n=300, shift_mas=80.0)   # confirms nothing

    def build():
        with pytest.raises(ValueError, match='0 positions confirmed'):
            C._build_crossband_seed(cut_bp, ['merged'], filts, _opts())
        return capsys.readouterr().out.count('WARNING [m7] crossband seed collapsed')

    assert build() == 1
    assert build() == 0
    C._CROSSBAND_SEED_COLLAPSE_WARNED.clear()
    assert build() == 1


def test_empty_seed_still_leaves_the_health_record(tmp_path, capsys):
    """Two bands that confirm nothing: the build raises ValueError, but the
    JSON was written first and the warning printed."""
    cut_bp = str(tmp_path)
    filts = _sparse_pair(cut_bp, 5, 5, 0)
    with pytest.raises(ValueError, match='0 positions confirmed'):
        C._build_crossband_seed(cut_bp, ['merged'], filts, _opts())
    out = C.crossband_seed_file(cut_bp, _opts())
    h = _load_health(out)
    assert h['n_seed'] == 0 and h['frac'] == 0.0 and h['collapsed'] is True
    assert 'WARNING [m7] crossband seed collapsed' in capsys.readouterr().out
