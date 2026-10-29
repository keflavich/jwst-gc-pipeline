"""Shared fixtures for the transdimensional (PCAT) tests."""
import os
from pathlib import Path

import numpy as np
import pytest

OUT = Path('/blue/adamginsburg/adamginsburg/jwst/brick/astrometry_paper/outputs/pcat_cutout')
EPSF_PATH = OUT / 'psf' / 'epsf_f182m_nrcb3_e16_os5.fits'
PCATDE_DIR = Path('/blue/adamginsburg/adamginsburg/repos/pcat/pcat-de/pcat-de')


def lib_dir():
    """Kernel directory: ``$PCAT_JWST_LIBDIR``, else the package default."""
    from ..kernel import default_lib_dir
    return default_lib_dir()


def synthetic_epsf(oversample=5, size=205, fwhm=1.95):
    """Pixel-integrated Gaussian ePSF in the common-ePSF convention (centre at
    index size // 2, values = flux fraction per detector pixel)."""
    from scipy.special import erf
    s = fwhm / 2.3548
    u = (np.arange(size) - size // 2) / oversample

    def integ(u):
        return 0.5 * (erf((u + 0.5) / (np.sqrt(2) * s)) - erf((u - 0.5) / (np.sqrt(2) * s)))
    return np.outer(integ(u), integ(u))


def get_epsf(require_real=False):
    """``(epsf, oversample, is_real)``: the common F182M ePSF when present,
    else a synthetic Gaussian ePSF (or skip when ``require_real``)."""
    if EPSF_PATH.exists():
        from ..psf_poly import load_epsf
        e, os_, _ = load_epsf(EPSF_PATH)
        return e, os_, True
    if require_real:
        pytest.skip(f'{EPSF_PATH} not available')
    return synthetic_epsf(), 5, False


def have_compiler():
    import shutil
    return shutil.which(os.environ.get('CC', 'gcc')) is not None
