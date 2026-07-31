"""Where to find the PSF-FWHM reference table.

``fwhm_table.ecsv`` lists the PSF FWHM of every NIRCam and MIRI filter. Those
are instrument constants, so one table serves every target, and a copy ships
in this package. A target tree may still carry its own
``reduction/fwhm_table.ecsv``; when it does, that copy wins.

NIRISS has a separate table because its pixels are a different size.
"""
import os
from pathlib import Path

PACKAGED = Path(__file__).resolve().parent / 'fwhm_table.ecsv'
PACKAGED_NIRISS = Path(__file__).resolve().parent / 'fwhm_table_niriss.ecsv'


def fwhm_table_path(basepath=None, instrument=None):
    """Return the FWHM table to read for ``instrument`` under ``basepath``.

    Parameters
    ----------
    basepath : str or None
        A target's data directory. Its ``reduction/fwhm_table.ecsv`` is used
        if it exists. ``None`` selects the packaged table.
    instrument : str or None
        ``'NIRISS'`` selects the NIRISS table, which is only ever the packaged
        one.
    """
    if str(instrument).upper() == 'NIRISS':
        return str(PACKAGED_NIRISS)
    if basepath:
        local = os.path.join(str(basepath), 'reduction', 'fwhm_table.ecsv')
        if os.path.exists(local):
            return local
    return str(PACKAGED)
