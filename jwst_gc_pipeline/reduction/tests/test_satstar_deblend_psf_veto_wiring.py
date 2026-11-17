"""``get_saturated_stars`` hands the PSF veto to the ZEROFRAME deblend (#1195).

The veto itself is tested in ``test_satstar_deblend_psf_veto.py``; here
``get_psf``, ``deblend_psf_stamp`` and ``build_deblended_source_records`` are
replaced, so the tests pin only the wiring: which keyword arguments reach the
deblend, and that the in-FOV PSF grid is built once (the veto reads it before
the deblend, the fit reuses it afterwards).  The deblend stub returns no
records, so the call returns after "No saturated stars found" without a fit.
"""
import numpy as np
import pytest
from astropy.io import fits

from jwst_gc_pipeline.reduction import satstar_deblend as sd
from jwst_gc_pipeline.reduction import saturated_star_finding as ssf

NY = NX = 128


def _hdul(detector):
    data = np.full((NY, NX), 5.0, dtype='float32')
    dq = np.zeros((NY, NX), dtype='uint32')
    yy, xx = np.mgrid[0:NY, 0:NX]
    disk = (xx - 64) ** 2 + (yy - 64) ** 2 <= 9
    dq[disk] = 2
    data[disk] = 1e5
    hdr = fits.Header({'INSTRUME': 'NIRCAM', 'DETECTOR': detector,
                       'FILTER': 'F182M', 'TELESCOP': 'JWST'})
    one = np.ones((NY, NX), dtype='float32')
    return fits.HDUList([
        fits.PrimaryHDU(header=hdr),
        fits.ImageHDU(data, name='SCI', header=fits.Header({'PIXAR_SR': 1e-13})),
        fits.ImageHDU(dq, name='DQ'),
        fits.ImageHDU(one, name='VAR_POISSON'),
        fits.ImageHDU(one, name='ERR')])


@pytest.fixture
def spies(monkeypatch):
    calls = {'get_psf': [], 'stamp': [], 'build': []}

    def _get_psf(header, **kw):
        calls['get_psf'].append(kw)
        return 'GRID'

    def _stamp(grid):
        calls['stamp'].append(grid)
        return 'STAMP'

    def _build(*a, **kw):
        calls['build'].append(kw)
        return []

    monkeypatch.setattr(ssf, 'get_psf', _get_psf)
    # get_saturated_stars builds an stpsf.NIRCam() generator up front, which
    # reads the STPSF data directory; CI has none and these tests need no PSF.
    monkeypatch.setattr(ssf.stpsf, 'NIRCam', lambda *a, **kw: object())
    monkeypatch.setattr(ssf, 'deblend_psf_stamp', _stamp)
    monkeypatch.setattr(sd, 'build_deblended_source_records', _build)
    return calls


def _run(detector='NRCA2'):
    zf = np.full((NY, NX), 100.0)
    return ssf.get_saturated_stars(_hdul(detector), zeroframe=zf,
                                   zeroframe_deblend=True, plot=False)


@pytest.mark.parametrize('detector, fov', [('NRCA2', 512), ('NRCALONG', 1024)])
def test_margin_on_passes_the_stamp_and_builds_the_grid_once(
        spies, monkeypatch, detector, fov):
    monkeypatch.setenv('SATSTAR_DEBLEND_PSF_MARGIN', '0.3')
    _run(detector)
    assert len(spies['build']) == 1
    kw = spies['build'][0]
    assert kw['psf_stamp'] == 'STAMP'
    assert kw['psf_margin'] == 0.3
    assert spies['stamp'] == ['GRID']
    assert [c['fov_pixels'] for c in spies['get_psf']] == [fov]


def test_margin_off_leaves_the_deblend_call_alone(spies, monkeypatch):
    monkeypatch.delenv('SATSTAR_DEBLEND_PSF_MARGIN', raising=False)
    _run()
    assert len(spies['build']) == 1
    assert 'psf_stamp' not in spies['build'][0]
    assert 'psf_margin' not in spies['build'][0]
    assert spies['stamp'] == []
    assert [c['fov_pixels'] for c in spies['get_psf']] == [512]


def test_a_non_gridded_psf_runs_without_the_veto(spies, monkeypatch, capsys):
    monkeypatch.setenv('SATSTAR_DEBLEND_PSF_MARGIN', '0.25')
    monkeypatch.setattr(ssf, 'deblend_psf_stamp', lambda grid: None)
    _run()
    assert 'psf_stamp' not in spies['build'][0]
    assert 'running without it' in capsys.readouterr().out
    assert len(spies['get_psf']) == 1
