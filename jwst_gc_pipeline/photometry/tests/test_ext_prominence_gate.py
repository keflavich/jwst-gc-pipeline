"""The NIRCam extended-emission prominence gate (``ext_prom_min``) must reject a
source that sits on a flat emission plateau (core barely rises above the local
annulus -> low prominence) EVEN when the star_like OR-branches would keep it
(qfit<=qfit_max, or flags==1 from landing in a satstar wing), while keeping a
real star (sharp core -> high prominence).

This is the W51 dark-filament F480M leak: emission bumps fit with tight qfit or
inherit a saturated-star wing flag and bypass every star test; only the deep-i2d
annulus-MAD prominence separates them (real median ~5, false ~0.9).
"""
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import astropy.units as u

from jwst_gc_pipeline.photometry.cataloging import _filter_extended_emission


def _simple_wcs(nx, ny):
    w = WCS(naxis=2)
    w.wcs.crpix = [nx / 2.0, ny / 2.0]
    w.wcs.cdelt = [-0.063 / 3600.0, 0.063 / 3600.0]
    w.wcs.crval = [266.5, -28.8]
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    return w


def _synthetic_i2d(nx=61, ny=61):
    """Flat background 100; a SHARP real star at (15,15); a broad FLAT emission
    plateau centred at (45,45) whose core barely exceeds its own annulus."""
    img = np.full((ny, nx), 100.0)
    yy, xx = np.mgrid[0:ny, 0:nx]
    # real star: narrow gaussian, tall core -> high prominence
    img += 5000.0 * np.exp(-(((xx - 15) ** 2 + (yy - 15) ** 2) / (2 * 1.2 ** 2)))
    # emission plateau: wide, low-contrast (sigma ~ 12 px so the 4-10px annulus
    # is nearly as bright as the r<1.5 core -> prominence ~0)
    img += 400.0 * np.exp(-(((xx - 45) ** 2 + (yy - 45) ** 2) / (2 * 12.0 ** 2)))
    return img


def _catalog(w, positions, qfit, flags):
    sc = SkyCoord.from_pixel(np.array([p[0] for p in positions]),
                             np.array([p[1] for p in positions]), w)
    n = len(positions)
    return Table({
        'skycoord': sc,
        'qfit': np.asarray(qfit, float),
        'flags': np.asarray(flags, float),
        'flux': np.full(n, 1000.0),
        'flux_err': np.full(n, 50.0),
        'local_bkg': np.full(n, 100.0),
        'group_size': np.ones(n, int),
    })


def test_prom_gate_drops_emission_keeps_star():
    w = _simple_wcs(61, 61)
    img = _synthetic_i2d(61, 61)
    # both pass star_like: the real via qfit, the emission via flags==1 (wing)
    t = _catalog(w, [(15, 15), (45, 45)], qfit=[0.05, 0.05], flags=[0, 1])

    # gate OFF -> both kept (flags==1 emission bump leaks, the bug)
    off = _filter_extended_emission(t.copy(), data_i2d_image=img, ww_i2d=w,
                                    ext_prom_min=0.0, label='off')
    assert len(off) == 2

    # gate ON -> emission plateau dropped, real star kept
    on = _filter_extended_emission(t.copy(), data_i2d_image=img, ww_i2d=w,
                                   ext_prom_min=3.0, label='on')
    kept = SkyCoord(on['skycoord'])
    xk, yk = kept.to_pixel(w)
    assert len(on) == 1
    assert abs(xk[0] - 15) < 1.5 and abs(yk[0] - 15) < 1.5


def test_prom_gate_satstar_forcekeep_overrides():
    """A replaced_saturated source with low prominence (broad subtracted core)
    must survive the gate via the model==catalog force-keep."""
    w = _simple_wcs(61, 61)
    img = _synthetic_i2d(61, 61)
    t = _catalog(w, [(45, 45)], qfit=[0.05], flags=[1])
    t['replaced_saturated'] = np.array([True])
    on = _filter_extended_emission(t.copy(), data_i2d_image=img, ww_i2d=w,
                                   ext_prom_min=3.0, label='force')
    assert len(on) == 1  # force-kept despite low prominence


def test_prom_gate_off_is_noop_default():
    """ext_prom_min<=0 must be byte-identical to no gate (star-dominated fields)."""
    w = _simple_wcs(61, 61)
    img = _synthetic_i2d(61, 61)
    t = _catalog(w, [(15, 15), (45, 45)], qfit=[0.05, 0.05], flags=[0, 1])
    a = _filter_extended_emission(t.copy(), data_i2d_image=img, ww_i2d=w,
                                  ext_prom_min=0.0, label='a')
    b = _filter_extended_emission(t.copy(), data_i2d_image=img, ww_i2d=w,
                                  label='b')  # ext_prom_min defaults to 0.0
    assert len(a) == len(b) == 2
