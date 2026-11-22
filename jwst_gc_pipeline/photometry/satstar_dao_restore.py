"""Opt-in restore of the daophot row that a satstar overwrote and the in-field
dedup would drop.

``replace_saturated`` overwrites the flux and position of the daophot row it
pairs with a satstar.  The in-field dedup of ``cataloging._clean_offfov_dups_and_offfield``
(``SATSTAR_INFIELD_DEDUP`` ``cofit`` and ``cofit_residual``) then drops the
``replaced_saturated`` row of a satstar that no per-exposure run fit beside a
better supported neighbour within 1".  When that row is a real star that has its
own daophot detections, the drop removes the star from the catalog: nothing
renders it, and its light stays in ``*_mergedcat_residual_i2d``.  gc-treasury
10678 o105 F212N: a replay of the dedup on the 10354 in-field rows of the
consolidated satstar catalog drops 57.  55 of the 57 have no row in the production
m7 merged catalog (the other two were not examined), and 24 of the 55 have a
daophot cluster (1-5 frames) within 80 mas.  Five of the 24 are 63-79 mas from
the satstar at 0.002-0.031 of its flux (the ``dqsat`` seeds of single-exposure
transients, which the dedup rightly drops).  The other 19 sit within 21 mas of it
at 0.80-1.06 of its flux, 18 of them within 5 mas.

With ``SATSTAR_DAO_RESTORE_FRAC_MIN`` above 0 the two steps cooperate:

1. ``replace_saturated`` (the per-frame merge only) records the flux, flux error
   and position of every daophot row it overwrites in the ``satstar_dao_*``
   columns.
2. The in-field dedup, for each row it is about to drop, checks that record.  A
   row whose daophot flux is within ``[FRAC_MIN, FRAC_MAX)`` of the satstar flux,
   lies within ``SEP_MAX_ARCSEC`` of the satstar and has no surviving satstar row
   within ``NEIGHBOUR_MIN_ARCSEC`` is kept as the daophot row it was: flux,
   flux error and position from the record, ``replaced_saturated`` False,
   ``is_saturated`` unchanged (the vetting force-keep reads it), the satstar
   ensemble columns blank and ``satstar_dao_restored`` True.

Every other row is dropped or kept exactly as without the switch.  A restored
row carries the daophot photometry that the satstar replaced, not the satstar's:
the replays in ``SATURATED_PIXEL_HANDLING.md`` found a restored satstar fit
over-subtracts by 25-40 % or sits about 0.2" off.

Only tables whose pixel and magnitude columns are absent are restored (the
per-frame merge's ``minimal_table`` has none); a table with ``x``/``y`` or
magnitude columns would need those columns recorded as well.
"""
import os
from collections import namedtuple

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky

#: restore on when above 0: lower end of the daophot / satstar flux ratio
ENV_FRAC_MIN = 'SATSTAR_DAO_RESTORE_FRAC_MIN'

#: upper end of the daophot / satstar flux ratio (exclusive)
ENV_FRAC_MAX = 'SATSTAR_DAO_RESTORE_FRAC_MAX'

#: largest separation in arcsec between the daophot row and the satstar
ENV_SEP_MAX = 'SATSTAR_DAO_RESTORE_SEP_MAX_ARCSEC'

#: smallest separation in arcsec between the daophot position and any
#: surviving satstar row (0 turns the check off)
ENV_NEIGHBOUR_MIN = 'SATSTAR_DAO_RESTORE_NEIGHBOUR_MIN_ARCSEC'

DEFAULT_FRAC_MAX = 1.25
DEFAULT_SEP_MAX_ARCSEC = 0.03
DEFAULT_NEIGHBOUR_MIN_ARCSEC = 0.1

PRIOR_FLUX = 'satstar_dao_flux'
PRIOR_FLUX_ERR = 'satstar_dao_flux_err'
PRIOR_RA = 'satstar_dao_ra'
PRIOR_DEC = 'satstar_dao_dec'

#: what ``replace_saturated`` recorded of the daophot row it overwrote
PRIOR_COLUMNS = (PRIOR_FLUX, PRIOR_FLUX_ERR, PRIOR_RA, PRIOR_DEC)

RESTORED_COLUMN = 'satstar_dao_restored'

#: columns ``replace_saturated`` also overwrites on a table that has them; the
#: record does not cover these, so such a table is not restored
UNTRACKED_COLUMNS = ('x', 'y', 'dx', 'dy', 'x_fit', 'y_fit', 'x_err', 'y_err',
                     'mag_ab', 'mag_vega', 'emag_ab')

#: satstar-only columns blanked on a restored row (the value on a row that was
#: never a satstar)
_ENSEMBLE_BLANKS = (('satstar_nframes', 0), ('satstar_nmeas', 0),
                    ('satstar_std_ra', np.nan), ('satstar_std_dec', np.nan))

Settings = namedtuple('Settings', 'frac_min frac_max sep_max_arcsec '
                                  'neighbour_min_arcsec')


def _env_number(name, default):
    raw = os.environ.get(name, '')
    if not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        value = np.nan
    if not (np.isfinite(value) and value >= 0.0):
        raise ValueError(f"{name}={raw!r}: expected a finite number >= 0")
    return value


def settings():
    """``None`` when the restore is off, else the ``Settings`` it runs with.

    ``SATSTAR_DAO_RESTORE_FRAC_MIN`` unset, blank or 0 is off, the default; the
    other three variables are then not read.  A value that is not a finite
    number >= 0, or a ``FRAC_MAX`` that does not exceed ``FRAC_MIN``, raises
    ``ValueError`` naming the variable.
    """
    frac_min = _env_number(ENV_FRAC_MIN, 0.0)
    if frac_min == 0.0:
        return None
    frac_max = _env_number(ENV_FRAC_MAX, DEFAULT_FRAC_MAX)
    if frac_max <= frac_min:
        raise ValueError(f"{ENV_FRAC_MAX}={frac_max:g} must exceed "
                         f"{ENV_FRAC_MIN}={frac_min:g}")
    return Settings(frac_min, frac_max,
                    _env_number(ENV_SEP_MAX, DEFAULT_SEP_MAX_ARCSEC),
                    _env_number(ENV_NEIGHBOUR_MIN, DEFAULT_NEIGHBOUR_MIN_ARCSEC))


def enabled():
    """Whether the per-frame merge records the daophot rows it overwrites."""
    return settings() is not None


def flux_columns(table):
    """``(flux column, flux error column)`` of a merged table, as
    ``replace_saturated`` picks them: ``flux`` (error ``dflux``, ``flux_err`` or
    ``eflux``, whichever comes first) or the daophot ``flux_fit`` (error
    ``flux_err``).  ``(None, None)`` for a table with neither flux column; the
    error is ``None`` when the table has no error column.
    """
    if 'flux' in table.colnames:
        for name in ('dflux', 'flux_err', 'eflux'):
            if name in table.colnames:
                return 'flux', name
        return 'flux', None
    if 'flux_fit' in table.colnames:
        return 'flux_fit', 'flux_err' if 'flux_err' in table.colnames else None
    return None, None


def _floats(column):
    return np.ma.filled(np.ma.asarray(column).astype(float), np.nan)


def record_prior(table, rows, flux_col, err_col):
    """Record the flux, flux error and position of ``table[rows]`` in the
    ``satstar_dao_*`` columns, for the rows ``replace_saturated`` is about to
    overwrite.  Every other row gets NaN; a column already present is reset.
    Call it before the overwrite.  Returns the number of rows recorded.
    """
    n = len(table)
    for name in PRIOR_COLUMNS:
        table[name] = np.full(n, np.nan)
    rows = np.asarray(rows, dtype=int)
    if not len(rows):
        return 0
    table[PRIOR_FLUX][rows] = _floats(table[flux_col])[rows]
    if err_col is not None:
        table[PRIOR_FLUX_ERR][rows] = _floats(table[err_col])[rows]
    sky = table['skycoord'][rows]
    table[PRIOR_RA][rows] = sky.ra.deg
    table[PRIOR_DEC][rows] = sky.dec.deg
    return int(len(rows))


def restorable(table, rows, survivors, flux_col, cfg):
    """Boolean over ``rows`` (indices into ``table``): which would-be-dropped
    rows go back to the daophot row recorded for them.

    A row qualifies when the recorded flux is finite and positive, its ratio to
    the row's own (satstar) flux lies in ``[frac_min, frac_max)``, the recorded
    position is within ``sep_max_arcsec`` of the row's position and the nearest
    row of ``survivors`` (indices of the satstar rows that stay) is at least
    ``neighbour_min_arcsec`` from the recorded position.  The last test stops a
    daophot row that is the star a surviving satstar row already carries from
    entering the catalog twice.  A table without the record, or with columns
    the record does not cover, restores nothing.
    """
    rows = np.asarray(rows, dtype=int)
    ok = np.zeros(len(rows), dtype=bool)
    if (not len(rows) or any(c not in table.colnames for c in PRIOR_COLUMNS)
            or untracked_columns(table)):
        return ok
    sat = _floats(table[flux_col])[rows]
    prior = _floats(table[PRIOR_FLUX])[rows]
    ra = _floats(table[PRIOR_RA])[rows]
    dec = _floats(table[PRIOR_DEC])[rows]
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = prior / sat
        ok = (np.isfinite(ratio) & (sat > 0) & (prior > 0)
              & (ratio >= cfg.frac_min) & (ratio < cfg.frac_max)
              & np.isfinite(ra) & np.isfinite(dec))
    cand = np.flatnonzero(ok)
    if not len(cand):
        return ok
    prior_sky = SkyCoord(ra[cand] * u.deg, dec[cand] * u.deg)
    near = (prior_sky.separation(table['skycoord'][rows[cand]]).arcsec
            <= cfg.sep_max_arcsec)
    ok[cand[~near]] = False
    cand, prior_sky = cand[near], prior_sky[near]
    survivors = np.asarray(survivors, dtype=int)
    if len(cand) and len(survivors) and cfg.neighbour_min_arcsec > 0:
        nearest = match_coordinates_sky(
            prior_sky, table['skycoord'][survivors])[1].arcsec
        ok[cand[nearest < cfg.neighbour_min_arcsec]] = False
    return ok


def untracked_columns(table):
    """The columns of ``table`` that ``replace_saturated`` overwrites and the
    record does not cover (``UNTRACKED_COLUMNS``)."""
    return [c for c in UNTRACKED_COLUMNS if c in table.colnames]


def select(table, rows, dropped, flux_col, cfg):
    """The rows of ``table`` to keep as daophot rows instead of dropping.

    ``rows`` are the candidate (in-field ``replaced_saturated``) rows of the
    in-field dedup and ``dropped`` is the boolean over ``rows`` of those the
    dedup drops; the rows of ``rows`` that stay are the survivors.  Returns the
    indices into ``table`` of the dropped rows that ``restorable`` accepts, and
    prints how many, or why none could be (no ``satstar_dao_*`` record, columns
    the record does not cover).
    """
    rows = np.asarray(rows, dtype=int)
    dropped = np.asarray(dropped, dtype=bool)
    missing = [c for c in PRIOR_COLUMNS if c not in table.colnames]
    untracked = untracked_columns(table)
    if missing:
        print(f"  WARNING: {ENV_FRAC_MIN} is set but the merged table has no "
              f"{', '.join(missing)} column(s) (written by a merge that ran "
              f"with the switch on); restoring none of the {int(dropped.sum())} "
              f"row(s) the in-field dedup drops", flush=True)
        return np.zeros(0, dtype=int)
    if untracked:
        print(f"  WARNING: {ENV_FRAC_MIN} is set but the merged table has "
              f"{', '.join(untracked)} column(s) that the satstar_dao_* record "
              f"does not cover; restoring none of the {int(dropped.sum())} "
              f"row(s) the in-field dedup drops", flush=True)
        return np.zeros(0, dtype=int)
    keep = restorable(table, rows[dropped], rows[~dropped], flux_col, cfg)
    chosen = rows[dropped][keep]
    print(f"  satstar in-field dedup: kept {len(chosen)} of the "
          f"{int(dropped.sum())} dropped row(s) as daophot rows "
          f"({ENV_FRAC_MIN}: daophot flux {cfg.frac_min:g}-{cfg.frac_max:g} of "
          f"the satstar flux, within {cfg.sep_max_arcsec:g}\" of the satstar, no "
          f"surviving satstar row within {cfg.neighbour_min_arcsec:g}\")",
          flush=True)
    return chosen


def restore(table, rows, flux_col, err_col):
    """Turn ``table[rows]`` back into the daophot rows recorded for them.

    Flux, flux error and position come from the ``satstar_dao_*`` record,
    ``replaced_saturated`` becomes False, the satstar ensemble columns are
    blanked and ``satstar_dao_restored`` (created False for every row when
    absent) becomes True.  ``is_saturated`` and every other column stay.
    Mutates ``table``.
    """
    rows = np.asarray(rows, dtype=int)
    if RESTORED_COLUMN not in table.colnames:
        table[RESTORED_COLUMN] = np.zeros(len(table), dtype=bool)
    if not len(rows):
        return
    table[flux_col][rows] = table[PRIOR_FLUX][rows]
    if err_col is not None:
        table[err_col][rows] = table[PRIOR_FLUX_ERR][rows]
    table['skycoord'][rows] = SkyCoord(_floats(table[PRIOR_RA])[rows] * u.deg,
                                       _floats(table[PRIOR_DEC])[rows] * u.deg)
    table['replaced_saturated'][rows] = False
    for name, blank in _ENSEMBLE_BLANKS:
        if name in table.colnames:
            table[name][rows] = blank
    table[RESTORED_COLUMN][rows] = True
