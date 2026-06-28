"""Small shared astrometry helpers (masked->NaN coercion + PM propagation).

Factored out of ``reduction/build_gaia_virac2_refcat.py``,
``reduction/build_gaia_virac2_refcat_byquery.py``, and
``photometry/generate_offsets_table.py``, which each carried a byte-identical
copy of ``farr``/``prop`` (drift risk: any change to the PM-propagation formula
had to be mirrored by hand in three places).
"""
import numpy as np


def farr(x):
    """Coerce ``x`` to a plain float array, turning masked / non-finite to NaN."""
    return np.asarray(np.ma.filled(np.ma.masked_invalid(np.asarray(x, float)), np.nan), float)


def prop(ra, dec, pmra, pmde, dt):
    """Proper-motion propagate ``(ra, dec)`` [deg] by ``dt`` years.

    ``pmra``/``pmde`` are in mas/yr (pmra is the on-sky rate, i.e. already
    includes the cos(dec) factor, so it is divided back out here).  Non-finite
    proper motions are treated as zero.
    """
    pmra = np.where(np.isfinite(pmra), pmra, 0.)
    pmde = np.where(np.isfinite(pmde), pmde, 0.)
    return ra + (pmra * dt / 3.6e6) / np.cos(np.radians(dec)), dec + (pmde * dt / 3.6e6)
