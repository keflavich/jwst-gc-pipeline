"""Satstar fits use the same hybrid PSF as the unsaturated fits (#1208).

With ``PSF_EPSF_CORE_DIR`` set, ``get_psf_model`` hands the unsaturated fits an
ePSF core inside 10 px; ``get_psf`` (the satstar fits) returned plain STPSF, so
a step appeared at the saturation boundary for the small cores.  Synthetic
grids and cores only, as in ``photometry/tests/test_epsf_hybrid.py``: no stpsf
data, no network.
"""
import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from jwst_gc_pipeline.photometry import epsf_hybrid as eh
from jwst_gc_pipeline.photometry import satstar_cache
from jwst_gc_pipeline.photometry.tests.test_epsf_hybrid import _core, _stpsf_grid
from jwst_gc_pipeline.reduction import saturated_star_finding as ssf


def _header(instrume='NIRCAM', detector='NRCBLONG', filt='F480M', program='10678',
            module='B'):
    hdr = fits.Header()
    hdr['INSTRUME'] = instrume
    hdr['DETECTOR'] = detector
    hdr['FILTER'] = filt
    hdr['PUPIL'] = 'CLEAR'
    hdr['MODULE'] = module
    hdr['DATE-OBS'] = '2023-02-01'
    if program is not None:
        hdr['PROGRAM'] = program
    return hdr


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A get_psf that runs without stpsf data, MAST or a token: the grid file
    exists (so the cache branch is taken) and ``load_stpsf_grid`` returns the
    synthetic STPSF grid."""
    st = _stpsf_grid()
    fake = type('FakePSFGen', (), {})
    monkeypatch.setattr(ssf.stpsf, 'NIRCam', fake, raising=False)
    monkeypatch.setattr(ssf.stpsf, 'MIRI', fake, raising=False)
    monkeypatch.setattr(ssf, 'get_fwhm', lambda header, instrument_replacement=None: (0.1, 2.0))
    monkeypatch.setattr(ssf, 'load_stpsf_grid', lambda fn: st)
    monkeypatch.setenv('HOME', str(tmp_path))
    (tmp_path / '.mast_api_token').write_text('token')
    psfs = tmp_path / 'psfs'
    psfs.mkdir()
    for name in ('nircam_nrcb5_f480m', 'nircam_nrcb5_f212n', 'nircam_nrcb1_f480m',
                 'miri_mirim_f770w'):
        (psfs / f'{name}_fovp512_samp2_npsf16.fits').write_text('fake')
        (psfs / f'{name}_fovp1024_samp2_npsf16.fits').write_text('fake')
    cores = tmp_path / 'cores'
    cores.mkdir()
    eh.write_epsf_core(cores / eh.epsf_core_filename('NRCB5', 'F480M'), _core(), 'NRCB5', 'F480M')
    monkeypatch.delenv(eh.EPSF_CORE_DIR_ENV, raising=False)
    monkeypatch.delenv(eh.ALLOW_OTHER_PROGRAM_ENV, raising=False)
    monkeypatch.setattr(eh, '_APPLIED', {})
    monkeypatch.setattr(eh, '_HYBRID_CACHE', {})
    return dict(st=st, psfs=str(psfs), cores=str(cores))


def _get(env, header, fov_pixels=1024):
    return ssf.get_psf(header, path_prefix=env['psfs'], fov_pixels=fov_pixels)


def test_unset_returns_the_stpsf_grid_object(env):
    assert _get(env, _header()) is env['st']


def test_set_returns_the_hybrid(env, monkeypatch):
    monkeypatch.setenv(eh.EPSF_CORE_DIR_ENV, env['cores'])
    grid = _get(env, _header())
    assert grid is not env['st']
    assert grid.meta['epsf_core'].endswith('epsf_core_nrcb5_f480m.fits')
    # the wing is STPSF's
    assert np.array_equal(grid.data[:, 0, :], np.stack(
        [eh._stpsf_data_at(env['st'], x, y) for (x, y) in grid.grid_xypos])[:, 0, :])


def test_miri_header_is_untouched(env, monkeypatch):
    monkeypatch.setenv(eh.EPSF_CORE_DIR_ENV, env['cores'])
    grid = _get(env, _header(instrume='MIRI', detector='MIRIMAGE', filt='F770W',
                             module='MIRIMAGE'))
    assert grid is env['st']


def test_filter_without_a_core_keeps_stpsf(env, monkeypatch):
    monkeypatch.setenv(eh.EPSF_CORE_DIR_ENV, env['cores'])
    assert _get(env, _header(filt='F212N')) is env['st']


def test_detector_is_passed_when_the_grid_meta_lacks_it(env, monkeypatch):
    """With detector=None and no ``detector`` in the grid meta the core is
    skipped with a warning; the satstar path names the detector itself."""
    monkeypatch.setenv(eh.EPSF_CORE_DIR_ENV, env['cores'])
    meta = {k: v for k, v in dict(env['st'].meta).items() if k != 'detector'}
    from astropy.nddata import NDData
    from photutils.psf import GriddedPSFModel
    nodetector = GriddedPSFModel(NDData(np.asarray(env['st'].data), meta=meta))
    assert eh.grid_detector(nodetector) is None
    monkeypatch.setattr(ssf, 'load_stpsf_grid', lambda fn: nodetector)
    grid = _get(env, _header())
    assert grid is not nodetector and grid.meta['epsf_core']


def test_header_without_program_does_not_raise(env, monkeypatch):
    monkeypatch.setenv(eh.EPSF_CORE_DIR_ENV, env['cores'])
    grid = _get(env, _header(program=None))
    assert grid.meta['epsf_core']


def test_leading_zero_program_matches_the_core(env, monkeypatch):
    monkeypatch.setenv(eh.EPSF_CORE_DIR_ENV, env['cores'])
    assert _get(env, _header(program='010678')).meta['epsf_core']


def test_other_program_raises_unless_allowed(env, monkeypatch):
    monkeypatch.setenv(eh.EPSF_CORE_DIR_ENV, env['cores'])
    with pytest.raises(ValueError, match='built from program'):
        _get(env, _header(program='01182'))
    monkeypatch.setenv(eh.ALLOW_OTHER_PROGRAM_ENV, '1')
    with pytest.warns(UserWarning, match='using it anyway'):
        assert _get(env, _header(program='01182')).meta['epsf_core']


# -- provenance --------------------------------------------------------------

def test_provenance_is_limited_to_the_frames_detector(env, monkeypatch):
    monkeypatch.setenv(eh.EPSF_CORE_DIR_ENV, env['cores'])
    _get(env, _header())
    hdr = _header()
    meta = eh.psf_provenance_meta('F480M', detector=ssf._psf_provenance_detector(hdr))
    assert meta['PSFMODEL'] == 'STPSF+EPSFCORE'
    assert meta['EPSFSHA'] == eh.expected_psf_signature('F480M', 'NRCBLONG')
    # another detector's core is not recorded against this frame
    assert eh.psf_provenance_meta('F480M', detector='NRCA5')['PSFMODEL'] == 'STPSF'
    # no detector given: every detector applied in the process (unchanged behaviour)
    assert eh.psf_provenance_meta('F480M')['PSFMODEL'] == 'STPSF+EPSFCORE'


def test_provenance_detector_for_merged_module_is_none():
    assert ssf._psf_provenance_detector(_header(module='merged')) is None
    assert ssf._psf_provenance_detector(_header(detector='NRCALONG')) == 'NRCA5'
    assert ssf._psf_provenance_detector(_header(detector='NRCB1')) == 'NRCB1'


# -- expected_psf_signature ----------------------------------------------------

def test_expected_signature_empty_without_a_core(env, monkeypatch):
    assert eh.expected_psf_signature('F480M', 'NRCBLONG') == ''
    monkeypatch.setenv(eh.EPSF_CORE_DIR_ENV, env['cores'])
    assert eh.expected_psf_signature('F212N', 'NRCBLONG') == ''
    assert eh.expected_psf_signature('F480M', 'NRCA5') == ''


def test_expected_signature_is_the_core_sha(env, monkeypatch):
    monkeypatch.setenv(eh.EPSF_CORE_DIR_ENV, env['cores'])
    path = f"{env['cores']}/epsf_core_nrcb5_f480m.fits"
    sig = eh.expected_psf_signature('F480M', 'NRCBLONG')
    assert sig == eh._sha256(path)
    assert eh.expected_psf_signature('f480m', 'nrcb5') == sig
    eh.write_epsf_core(path, _core(dx=0.05), 'NRCB5', 'F480M')
    assert eh.expected_psf_signature('F480M', 'NRCBLONG') != sig


# -- cache check ---------------------------------------------------------------

def _frame(tmp_path, detector='NRCBLONG', filt='F480M'):
    fn = tmp_path / 'frame.fits'
    fits.PrimaryHDU(header=_header(detector=detector, filt=filt)).writeto(fn, overwrite=True)
    return str(fn)


def _cached(tmp_path, epsfsha=None, recov='off'):
    t = Table({'flux': [1.0, 2.0]})
    t.meta['SATRECOV'] = recov
    if epsfsha is not None:
        t.meta['EPSFSHA'] = epsfsha
    t.write(tmp_path / 'frame_satstar_catalog.fits', overwrite=True)


@pytest.fixture
def cache_env(tmp_path, monkeypatch, env):
    import jwst_gc_pipeline.photometry.crowdsource_catalogs_long as CL
    monkeypatch.setenv('SATSTAR_SEED_CORE_DQ', '0')
    calls = {'n': 0}

    def _stub(filename, overwrite=True, recovery_signature=None, **kw):
        calls['n'] += 1
        _cached(tmp_path, epsfsha=eh.expected_psf_signature('F480M', 'NRCBLONG') or None,
                recov=str(recovery_signature))
    monkeypatch.setattr(CL, 'remove_saturated_stars', _stub)
    return CL, calls, _frame(tmp_path)


@pytest.mark.parametrize('recov', ['off', None])
def test_stpsf_cache_is_kept_by_an_stpsf_run(tmp_path, cache_env, recov):
    CL, calls, fn = cache_env
    _cached(tmp_path)
    CL.load_or_make_satstar_catalog(fn, path_prefix=str(tmp_path), recovery_signature=recov)
    assert calls['n'] == 0


@pytest.mark.parametrize('recov', ['off', None])
def test_stpsf_cache_is_refitted_by_a_hybrid_run(tmp_path, cache_env, monkeypatch, env, recov):
    CL, calls, fn = cache_env
    monkeypatch.setenv(eh.EPSF_CORE_DIR_ENV, env['cores'])
    _cached(tmp_path)
    out = CL.load_or_make_satstar_catalog(fn, path_prefix=str(tmp_path),
                                          recovery_signature=recov)
    assert calls['n'] == 1
    assert out.meta['EPSFSHA'] == eh.expected_psf_signature('F480M', 'NRCBLONG')


def test_hybrid_cache_is_kept_by_a_hybrid_run_and_refitted_by_an_stpsf_run(
        tmp_path, cache_env, monkeypatch, env):
    CL, calls, fn = cache_env
    monkeypatch.setenv(eh.EPSF_CORE_DIR_ENV, env['cores'])
    _cached(tmp_path, epsfsha=eh.expected_psf_signature('F480M', 'NRCBLONG'))
    CL.load_or_make_satstar_catalog(fn, path_prefix=str(tmp_path), recovery_signature='off')
    assert calls['n'] == 0
    monkeypatch.delenv(eh.EPSF_CORE_DIR_ENV)
    CL.load_or_make_satstar_catalog(fn, path_prefix=str(tmp_path), recovery_signature='off')
    assert calls['n'] == 1


def test_changed_core_file_invalidates_the_cache(tmp_path, cache_env, monkeypatch, env):
    CL, calls, fn = cache_env
    monkeypatch.setenv(eh.EPSF_CORE_DIR_ENV, env['cores'])
    _cached(tmp_path, epsfsha=eh.expected_psf_signature('F480M', 'NRCBLONG'))
    eh.write_epsf_core(f"{env['cores']}/epsf_core_nrcb5_f480m.fits", _core(dx=0.05),
                       'NRCB5', 'F480M')
    CL.load_or_make_satstar_catalog(fn, path_prefix=str(tmp_path), recovery_signature='off')
    assert calls['n'] == 1


def test_filter_without_a_core_keeps_its_cache_in_a_hybrid_run(
        tmp_path, cache_env, monkeypatch, env):
    CL, calls, _ = cache_env
    monkeypatch.setenv(eh.EPSF_CORE_DIR_ENV, env['cores'])
    fn = _frame(tmp_path, filt='F212N')
    _cached(tmp_path)
    CL.load_or_make_satstar_catalog(fn, path_prefix=str(tmp_path), recovery_signature='off')
    assert calls['n'] == 0


# -- cross-phase content key -----------------------------------------------------

def _key(tmp_path, env_dict):
    fn = tmp_path / 'key_frame.fits'
    hdul = fits.HDUList([fits.PrimaryHDU(header=_header()),
                         fits.ImageHDU(np.ones((4, 4), dtype=np.float32), name='SCI'),
                         fits.ImageHDU(np.zeros((4, 4), dtype=np.uint32), name='DQ')])
    hdul.writeto(fn, overwrite=True)
    return satstar_cache.satstar_content_key(
        str(fn), path_prefix=str(tmp_path / 'nopsfs'), env=env_dict)


def test_content_key_sees_the_core(tmp_path, env):
    base = _key(tmp_path, {})
    # an unset variable, or one with no core for this frame, leaves the key alone
    assert _key(tmp_path, {eh.EPSF_CORE_DIR_ENV: str(tmp_path / 'empty')}) == base
    hybrid = _key(tmp_path, {eh.EPSF_CORE_DIR_ENV: env['cores']})
    assert hybrid != base
    eh.write_epsf_core(f"{env['cores']}/epsf_core_nrcb5_f480m.fits", _core(dx=0.05),
                       'NRCB5', 'F480M')
    assert _key(tmp_path, {eh.EPSF_CORE_DIR_ENV: env['cores']}) != hybrid


def test_content_key_unchanged_when_no_core_applies(tmp_path):
    """The psf_core term is fed only when a core applies, so an STPSF key is
    exactly what it was before."""
    h = _header()
    assert satstar_cache._psf_core_signature(h, {}) == ''
    assert satstar_cache._psf_core_signature(h, {eh.EPSF_CORE_DIR_ENV: str(tmp_path)}) == ''
    assert satstar_cache._psf_core_signature(_header(instrume='MIRI'),
                                             {eh.EPSF_CORE_DIR_ENV: str(tmp_path)}) == ''


# -- the writer stamps what the cache check expects ---------------------------

def test_remove_saturated_stars_stamps_the_signature_the_cache_expects(
        tmp_path, env, monkeypatch):
    """The real writer, with the fitter stubbed to apply the core: the
    catalog's EPSFSHA is the one ``load_or_make_satstar_catalog`` expects for
    the frame, whatever other detectors the process has fitted; an STPSF run
    stamps no EPSFSHA, which reads back as the expected ''."""
    import builtins
    import jwst_gc_pipeline.photometry.crowdsource_catalogs_long as CL
    fn = _frame(tmp_path)
    with fits.open(fn, mode='update') as hl:
        hl.append(fits.ImageHDU(np.zeros((8, 8), 'float32'), name='SCI'))
        hl.append(fits.ImageHDU(np.zeros((8, 8), 'int32'), name='DQ'))
    for attr in ('satstar_wingcal_measurements', 'satstar_rejected',
                 'satstar_model', 'satstar_resid', 'satstar_flagimg'):
        monkeypatch.delattr(builtins, attr, raising=False)

    def fit(fh, **kw):
        eh.maybe_apply_epsf_core(env['st'], 'NRCB5', 'F480M', program='10678')
        eh._APPLIED[('NRCA5', 'F480M')] = dict(path='other.fits', sha256='x' * 64,
                                               program='10678', r0=10.0, r1=12.0,
                                               shift_max_mpix=0.0)
        return Table({'flux_fit': [1.0]})
    monkeypatch.setattr(ssf, 'get_saturated_stars', fit)
    cat = fn.replace('.fits', '_satstar_catalog.fits')
    monkeypatch.setenv(eh.EPSF_CORE_DIR_ENV, env['cores'])
    ssf.remove_saturated_stars(fn, recovery_signature='off')
    meta = Table.read(cat).meta
    assert meta['PSFMODEL'] == 'STPSF+EPSFCORE'
    assert meta['EPSFSHA'] == CL._satstar_psf_signature(fn) != ''
    monkeypatch.delenv(eh.EPSF_CORE_DIR_ENV)
    monkeypatch.setattr(eh, '_APPLIED', {})
    monkeypatch.setattr(ssf, 'get_saturated_stars',
                        lambda fh, **kw: Table({'flux_fit': [1.0]}))
    ssf.remove_saturated_stars(fn, recovery_signature='off')
    meta = Table.read(cat).meta
    assert meta['PSFMODEL'] == 'STPSF' and 'EPSFSHA' not in meta
    assert CL._satstar_psf_signature(fn) == ''
