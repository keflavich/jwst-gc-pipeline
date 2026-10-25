"""The vetting force-keep of saturated/replaced rows skips satstar rows with no
counterpart in the data i2d.

A single-frame satstar fit on a detector artifact (SATURATED-flagged reference
pixels at a detector edge, a hot/dead cluster, a cosmic-ray footprint) enters
the merged catalog as a ``replaced_saturated`` row with a large flux at a
position where the data i2d shows blank sky.  Force-keeping it puts that flux
in the vetted catalog, and the merged-catalog render subtracts it from every
frame covering the position.
"""
import numpy as np
from astropy.table import Table
from astropy.wcs import WCS

from jwst_gc_pipeline.photometry.cataloging import (_filter_extended_emission,
                                                    _peak_excess,
                                                    _satstar_phantom_rows)

NY = NX = 200
CAL = [(25, 25), (60, 25), (95, 25), (130, 25), (165, 25),
       (25, 60), (165, 60), (25, 95), (165, 95)]
AMP = 60.0
FLUX = 2 * np.pi * AMP           # sigma = 1 px Gaussian of peak AMP
# satstar rows (replaced_saturated); positions at least 25 px apart
REAL = (60, 140)                 # star at the catalog position
PHANTOM = (100, 140)             # blank sky, low qfit (passes the qfit gate)
PHANTOM_HQ = (140, 140)          # blank sky, high qfit (only the force-keep)
NANCORE = (60, 175)              # NaN core (saturated in every exposure)
OFFSET = (100, 175)              # star 2 px from the catalog position
FAINT = (140, 175)               # blank sky, flux below the expected-S/N floor
SATFLAG = (100, 100)             # blank sky, is_saturated instead of replaced


def _wcs():
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crpix = [NX / 2, NY / 2]
    w.wcs.crval = [266.5, -28.7]
    w.wcs.cdelt = [-0.031 / 3600, 0.031 / 3600]
    return w


def _gauss(data, x, y, amp, sig=1.0):
    yy, xx = np.mgrid[0:NY, 0:NX]
    data += amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sig ** 2))


def _scene():
    rng = np.random.default_rng(11)
    data = rng.normal(0.0, 1.0, (NY, NX))
    for x, y in CAL:
        _gauss(data, x, y, AMP)
    _gauss(data, *REAL, AMP)
    _gauss(data, OFFSET[0] + 2.0, OFFSET[1], AMP)
    x0, y0 = NANCORE
    data[y0 - 1:y0 + 2, x0 - 1:x0 + 2] = np.nan
    names = ['cal'] * len(CAL) + ['real', 'phantom', 'phantom_hq', 'nancore',
                                  'offset', 'faint', 'satflag']
    xy = np.array(CAL + [REAL, PHANTOM, PHANTOM_HQ, NANCORE, OFFSET, FAINT,
                         SATFLAG], float)
    n = len(xy)
    flux = np.full(n, FLUX)
    flux[names.index('faint')] = 10.0
    qfit = np.full(n, 0.05)
    qfit[names.index('phantom_hq')] = 0.9
    qfit[names.index('nancore')] = 0.9
    qfit[names.index('offset')] = 0.9
    qfit[names.index('faint')] = 0.9
    qfit[names.index('satflag')] = 0.9
    rs = np.zeros(n, bool)
    rs[len(CAL):len(CAL) + 6] = True
    sat = np.zeros(n, bool)
    sat[names.index('satflag')] = True
    w = _wcs()
    cat = Table({'id': np.arange(n),
                 'skycoord': w.pixel_to_world(xy[:, 0], xy[:, 1]),
                 'qfit': qfit, 'flags': np.zeros(n), 'local_bkg': np.zeros(n),
                 'flux': flux, 'flux_err': flux / 50.0,
                 'group_size': np.ones(n), 'replaced_saturated': rs,
                 'is_saturated': sat, 'satstar_nframes': np.ones(n, int)})
    return data, w, cat, names


def _kept(**kw):
    data, w, cat, names = _scene()
    out = _filter_extended_emission(cat, data_i2d_image=data, ww_i2d=w,
                                    sky_clean_keep=False, label='test', **kw)
    ids = set(np.asarray(out['id']).tolist())
    return {nm for i, nm in enumerate(names) if i in ids and nm != 'cal'}, \
        sum(1 for i, nm in enumerate(names) if i in ids and nm == 'cal')


def test_peak_excess_of_a_psf_and_blank_sky():
    data = np.zeros((NY, NX))
    _gauss(data, 80.0, 80.0, 50.0)
    pk = _peak_excess(data, np.array([80.0, 82.0, 120.0, 3.0]),
                      np.array([80.0, 80.0, 120.0, 80.0]))
    assert abs(pk[0] - 50.0) < 1e-6          # centred
    assert abs(pk[1] - 50.0) < 1e-6          # star 2 px away, inside r_peak
    assert abs(pk[2]) < 1e-6                 # blank
    assert np.isnan(pk[3])                   # within 10 px of the edge


def test_phantom_rows_dropped_real_and_ambiguous_kept(monkeypatch):
    monkeypatch.delenv('VET_SATSTAR_PHANTOM', raising=False)
    kept, ncal = _kept()
    assert ncal == len(CAL)
    assert {'real', 'nancore', 'offset', 'faint'} <= kept
    assert not ({'phantom', 'phantom_hq', 'satflag'} & kept)


def test_veto_off_restores_force_keep(monkeypatch):
    monkeypatch.setenv('VET_SATSTAR_PHANTOM', '0')
    kept, ncal = _kept()
    assert ncal == len(CAL)
    assert kept == {'real', 'phantom', 'phantom_hq', 'nancore', 'offset',
                    'faint', 'satflag'}


def test_thresholds_are_parameters(monkeypatch):
    monkeypatch.delenv('VET_SATSTAR_PHANTOM', raising=False)
    # an expected-S/N floor above every row's S/N disables the veto
    kept, _ = _kept(satstar_phantom_snr_min=1e9)
    assert {'phantom', 'phantom_hq', 'satflag'} <= kept
    # the call-site switch (MIRI passes False)
    kept, _ = _kept(satstar_phantom_veto=False)
    assert {'phantom', 'phantom_hq', 'satflag'} <= kept


def test_phantom_rows_measurements_and_calibration_floor():
    data, w, cat, names = _scene()
    xx, yy = w.world_to_pixel(cat['skycoord'])
    cand = np.asarray(cat['replaced_saturated']) | np.asarray(cat['is_saturated'])
    calib = ~cand
    ph, core_rel, peak_rel, snr, info = _satstar_phantom_rows(
        data, xx, yy, np.asarray(cat['flux']), cand, calib)
    assert info['n_calib'] == len(CAL)
    i = {nm: names.index(nm) for nm in names if nm != 'cal'}
    assert 0.8 < core_rel[i['real']] < 1.2 and 0.8 < peak_rel[i['real']] < 1.2
    assert abs(core_rel[i['phantom']]) < 0.1 and peak_rel[i['phantom']] < 0.2
    assert snr[i['phantom']] > 20 and snr[i['faint']] < 20
    assert np.isnan(core_rel[i['nancore']])
    assert peak_rel[i['offset']] > 0.8
    assert np.isnan(core_rel[:len(CAL)]).all()       # not candidates
    assert set(np.flatnonzero(ph)) == {i['phantom'], i['phantom_hq'], i['satflag']}
    # fewer calibration rows than min_calib turns the test off
    ph2, *_ = _satstar_phantom_rows(data, xx, yy, np.asarray(cat['flux']),
                                    cand, calib, min_calib=len(CAL) + 1)
    assert not ph2.any()
