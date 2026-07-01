"""Tests for the stringent m7 cross-band seed + independent-detection provenance.

Failure mode being guarded (2026-06-30): the legacy seed UNIONED every filter's
m6 detections and force-fit every filter at every position, so a single-band (or
structure) detection became a fake multi-band source.  The stringent seed only
keeps positions confirmed in >= min_filters; annotate_independent_detection flags
which per-band measurements are real m6 detections vs cross-band-seeded.
"""
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
        proposal_id='2221', field='001', modules='merged')


def _write_m6(cut_bp, filt, ras, decs, snr=20.0, qfit=0.05):
    os.makedirs(f'{cut_bp}/catalogs', exist_ok=True)
    n = len(ras)
    t = Table()
    t['skycoord'] = SkyCoord(np.array(ras) * u.deg, np.array(decs) * u.deg)
    flux = np.full(n, 1000.0)
    t['flux'] = flux
    t['flux_err'] = flux / snr
    t['qfit'] = np.full(n, qfit)
    p = f'{cut_bp}/catalogs/{filt}_merged_indivexp_merged_resbgsub_m6_dao_basic_vetted.fits'
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
