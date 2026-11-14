"""The cross-band ECSV twin is written from the annotated FITS (#1228).

``merge_daophot`` used to write ``{table}.ecsv`` before
``annotate_independent_detection`` added ``independently_detected_<filt>`` and
``n_filt_independent`` to the FITS, so the ECSV had fewer columns (sgrb2 m7:
576 against 588).  ``run_manual_pipeline`` now skips the ECSV in the merge and
calls ``write_crossband_ecsv`` after the annotation and propagation steps.
"""
import inspect

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table

from jwst_gc_pipeline.photometry import cataloging, merge_catalogs


def _annotated_fits(tmp_path, n=20, seed=3):
    rng = np.random.default_rng(seed)
    t = Table()
    t['skycoord_ref'] = SkyCoord((266.5 + rng.uniform(0, 0.01, n)) * u.deg,
                                 (-28.9 + rng.uniform(0, 0.01, n)) * u.deg)
    t['flux_f212n'] = rng.uniform(1, 10, n)
    t['independently_detected_f212n'] = rng.uniform(size=n) > 0.5
    t['independently_detected_f480m'] = rng.uniform(size=n) > 0.5
    t['n_filt_independent'] = (t['independently_detected_f212n'].astype('i4')
                               + t['independently_detected_f480m'].astype('i4'))
    t.meta['EXTNAME'] = 'm7'
    stem = str(tmp_path / 'basic_merged_indivexp_photometry_tables_merged_resbgsub_m7')
    t.write(f'{stem}.fits')
    return stem, t


def test_ecsv_has_the_fits_columns_and_values(tmp_path):
    stem, _ = _annotated_fits(tmp_path)
    out = cataloging.write_crossband_ecsv(f'{stem}.fits')
    assert out == f'{stem}.ecsv'
    fits_t = Table.read(f'{stem}.fits')
    ecsv_t = Table.read(out)
    assert ecsv_t.colnames == fits_t.colnames
    assert {'independently_detected_f212n', 'independently_detected_f480m',
            'n_filt_independent'} <= set(ecsv_t.colnames)
    for c in ('independently_detected_f212n', 'independently_detected_f480m',
              'n_filt_independent', 'flux_f212n'):
        np.testing.assert_array_equal(ecsv_t[c], fits_t[c])
        assert ecsv_t[c].dtype.newbyteorder('=') == fits_t[c].dtype.newbyteorder('=')
    np.testing.assert_allclose(ecsv_t['skycoord_ref'].ra.deg,
                               fits_t['skycoord_ref'].ra.deg, rtol=0, atol=1e-12)


def test_an_undumpable_meta_value_is_stringified(tmp_path):
    stem, t = _annotated_fits(tmp_path)
    with fits.open(f'{stem}.fits', mode='update') as hdul:
        hdul[1].header['BADCARD'] = fits.card.Undefined()
    out = cataloging.write_crossband_ecsv(f'{stem}.fits')
    assert Table.read(out).colnames == t.colnames


def test_a_missing_fits_writes_nothing(tmp_path):
    assert cataloging.write_crossband_ecsv(str(tmp_path / 'absent.fits')) is None
    assert list(tmp_path.iterdir()) == []


def test_a_stale_ecsv_is_replaced(tmp_path):
    stem, _ = _annotated_fits(tmp_path)
    (tmp_path / f'{stem.rsplit("/", 1)[1]}.ecsv').write_text('an earlier run')
    cataloging.write_crossband_ecsv(f'{stem}.fits')
    assert 'n_filt_independent' in Table.read(f'{stem}.ecsv').colnames


def test_run_manual_pipeline_writes_the_ecsv_last():
    lines = inspect.getsource(cataloging.run_manual_pipeline).splitlines()
    run = ' '.join(' '.join(ln for ln in lines
                            if not ln.lstrip().startswith('#')).split())
    assert "vetted=True, write_ecsv=False)" in run
    i_ann = run.index("annotate_independent_detection(_xb, cut_bp, filternames, options)")
    i_exc = run.index("except Exception as _aex:")
    i_prop = run.index("propagate_independent_detection(_xb)")
    i_ecsv = run.index("write_crossband_ecsv(_xb)")
    assert i_ann < i_exc < i_prop < i_ecsv
    # the ECSV call follows the annotation's except-block at the same level, so
    # a failed annotation still gets an ECSV (without the flags)
    assert "if module in _xb_ecsv_modules: try: write_crossband_ecsv(_xb)" in run


def test_the_merge_still_strips_bad_meta_via_the_shared_helper():
    src = ' '.join(inspect.getsource(merge_catalogs.merge_catalogs).split())
    assert "_sanitise_meta_for_yaml(basetable) _write_ecsv_twin(basetable, tablename, write_ecsv)" in src
