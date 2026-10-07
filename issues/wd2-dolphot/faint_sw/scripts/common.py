"""Shared helpers for the faint-SW investigation (read-only on tree_*)."""
import sys, glob, os, re
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
import analyze as an

Q = an.Q
FS = f'{Q}/faint_sw'
BANDS4 = ['150W', '182M', '187N', '162M']
FWHM_PX = {'150W': 1.66, '182M': 2.0, '187N': 2.06, '162M': 1.79}   # ~1.03 lambda/D / 0.031"
RX = re.compile(r'_(nrc[ab][0-9a-z]+)_visit\d+_vgroup\d+_exp0*(\d+)_')


def sky_cols(tab, prefix):
    if prefix in tab.colnames:
        return tab[prefix]
    return SkyCoord(np.asarray(tab[prefix + '.ra'], float) * u.deg, np.asarray(tab[prefix + '.dec'], float) * u.deg)


def frame_files(tree, band):
    """dict (det, exp) -> m7 daophot basic path"""
    out = {}
    for p in glob.glob(f'{tree}/F{band.upper()}/f{band.lower()}_*_resbgsub_m7_daophot_basic.fits'):
        det, exp = RX.search(os.path.basename(p)).groups()
        out[(det, int(exp))] = p
    return out


def pipe_stem(tree, band, det, exp):
    g = glob.glob(f'{tree}/F{band.upper()}/pipeline/jw*_0000{exp}_{det}_*crf_resbgsub_m7_satstar_catalog.fits')
    return g[0].replace('_catalog.fits', '') if g else None
