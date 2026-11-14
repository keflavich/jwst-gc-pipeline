"""Column conventions of the per-frame satstar catalogs (#949).

The satstar fitter works in a pad-centred stamp around each source, so the
raw ``x_fit``/``y_fit``/``x_init``/``y_init`` it records are STAMP coordinates
(about 0-85 px) and the per-stamp photutils ``id`` is 1 on every row.  The
full-detector pixel position is ``xcentroid``/``ycentroid`` (stamp position
plus the stamp origin), which is what the in-tree readers use.  A reader that
takes ``x_fit`` at face value gets a stamp coordinate, the trap behind #1055.

:func:`finalize_satstar_columns` rewrites a table into the detector-frame
vintage:

* ``x_fit``/``y_fit`` hold the detector pixel position (``xcentroid``/
  ``ycentroid``), and ``x_init``/``y_init`` the initial guess in the same frame;
* the stamp values survive as ``x_fit_cutout``/``y_fit_cutout``/
  ``x_init_cutout``/``y_init_cutout``;
* the per-stamp ``id`` becomes ``cutout_id`` and ``id`` is a running index;
* ``meta['XYFRAME'] = 'detector'`` marks the table.  A table without the key is
  the older stamp-local vintage.

The helper is idempotent through that key, so it is safe to call on every read
path: a catalog written before this change is normalised in memory, one written
after it passes through untouched.
"""
import logging

import numpy as np
from astropy.table import Column

__all__ = ['XYFRAME_KEY', 'XYFRAME_DETECTOR', 'finalize_satstar_columns',
           'is_detector_frame']

log = logging.getLogger(__name__)

#: Header key marking a satstar table whose ``x_fit``/``y_fit``/``x_init``/
#: ``y_init`` are detector pixels.  Absent on the older stamp-local vintage.
XYFRAME_KEY = 'XYFRAME'
XYFRAME_DETECTOR = 'detector'

_STAMP_COLUMNS = ('x_fit', 'y_fit', 'x_init', 'y_init')


def is_detector_frame(tbl):
    """True when ``tbl`` carries the detector-frame vintage stamp."""
    return str(tbl.meta.get(XYFRAME_KEY, '')).lower() == XYFRAME_DETECTOR


def _as_float(col):
    return np.ma.filled(np.ma.asarray(col).astype(float), np.nan)


def finalize_satstar_columns(tbl, x0_col='xcentroid', y0_col='ycentroid'):
    """Put a satstar table in the detector-frame column vintage, in place.

    Parameters
    ----------
    tbl : `~astropy.table.Table`
        A satstar catalog, either vintage.
    x0_col, y0_col : str
        The detector-pixel columns.  The stamp origin of each row is
        ``x0_col - x_fit`` (the relation the fitter used to build them).

    Returns
    -------
    `~astropy.table.Table`
        ``tbl`` itself.  A table already carrying ``XYFRAME`` is returned
        unchanged; one without the detector columns is returned unchanged with a
        warning, because its ``x_fit`` cannot be converted.
    """
    if is_detector_frame(tbl):
        return tbl
    if x0_col not in tbl.colnames or y0_col not in tbl.colnames:
        log.warning("satstar table has no %s/%s columns and no %s stamp: its "
                    "x_fit/y_fit are stamp-local and are left as they are",
                    x0_col, y0_col, XYFRAME_KEY)
        return tbl

    x0 = _as_float(tbl[x0_col])
    y0 = _as_float(tbl[y0_col])
    if 'x_fit' in tbl.colnames and 'y_fit' in tbl.colnames:
        x_origin = x0 - _as_float(tbl['x_fit'])
        y_origin = y0 - _as_float(tbl['y_fit'])
    else:
        x_origin = y_origin = None

    for name in _STAMP_COLUMNS:
        if name not in tbl.colnames or f'{name}_cutout' in tbl.colnames:
            continue
        tbl[f'{name}_cutout'] = tbl[name]
    if 'id' in tbl.colnames and 'cutout_id' not in tbl.colnames:
        tbl.rename_column('id', 'cutout_id')

    if 'x_fit' in tbl.colnames:
        tbl['x_fit'] = x0
    if 'y_fit' in tbl.colnames:
        tbl['y_fit'] = y0
    if x_origin is not None:
        if 'x_init' in tbl.colnames:
            tbl['x_init'] = _as_float(tbl['x_init_cutout']) + x_origin
        if 'y_init' in tbl.colnames:
            tbl['y_init'] = _as_float(tbl['y_init_cutout']) + y_origin

    _id = np.arange(1, len(tbl) + 1, dtype=np.int32)
    if 'id' in tbl.colnames:
        tbl['id'] = _id
    else:
        tbl.add_column(Column(_id, name='id'), index=0)
    tbl.meta[XYFRAME_KEY] = XYFRAME_DETECTOR
    return tbl
