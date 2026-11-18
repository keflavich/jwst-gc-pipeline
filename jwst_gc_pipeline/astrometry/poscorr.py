"""
Optional per-row position-correction tables for the two-epoch PM builders.

A table is row-parallel to ONE merged catalogue (the FITS file the builder reads) and carries, for every row,
the sky-plane correction to ADD to the catalogue position: ``dxi_mas`` along east (x cos dec) and ``deta_mas``
along north.  The producer is a distortion-correction hook: the mean, over the row's per-frame detections, of a
detector-axis correction c(x_fit, y_fit) rotated onto the sky with each frame's Jacobian (the merged position is
the uniform mean of the kept frames' positions, so the corrected position is position + mean(c_sky) to first
order).  This module only reads and applies such a table; it does not know which correction produced it.

The builder passes a table's catalogue positions through ``load_poscorr`` first.  ``load_poscorr`` refuses a table
whose catalogue row count or position checksum differs from the catalogue being loaded, so a hook table made for
another catalogue version (e.g. before 5365 was re-catalogued with new PSF cores) cannot be applied silently.
"""
import zlib

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table

MAS_TO_DEG = 1.0 / 3.6e6


class PosCorrMismatchError(ValueError):
    """The position-correction table does not belong to this catalogue."""


def position_checksums(ra_deg, dec_deg):
    """CRC32 of the little-endian float64 bytes of ra and of dec (both in degrees)."""
    return (zlib.crc32(np.ascontiguousarray(ra_deg, dtype='<f8').tobytes()),
            zlib.crc32(np.ascontiguousarray(dec_deg, dtype='<f8').tobytes()))


def load_poscorr(path, ra_deg, dec_deg):
    """Read a position-correction table and check it against the catalogue it will be applied to.

    ``ra_deg``/``dec_deg`` are the catalogue's positions for ALL its rows (before any usability cut).
    Returns ``(dxi_mas, deta_mas, meta)`` as float64 arrays of the catalogue's length; ``meta`` is the table's
    header (solution name, map file, association radius, ...), kept for the PM table's provenance.
    Raises PosCorrMismatchError when the row count or the position checksums differ.
    """
    t = Table.read(path)
    n = len(ra_deg)
    if len(t) != n or int(t.meta.get('NROWS', len(t))) != n:
        raise PosCorrMismatchError(
            f'{path}: {len(t)} rows (NROWS={t.meta.get("NROWS")}), catalogue has {n}')
    if not np.array_equal(np.asarray(t['row']), np.arange(n)):
        raise PosCorrMismatchError(f'{path}: row column is not 0..{n - 1}')
    rachk, decchk = position_checksums(ra_deg, dec_deg)
    if int(t.meta.get('RACHK', -1)) != rachk or int(t.meta.get('DECCHK', -1)) != decchk:
        raise PosCorrMismatchError(
            f'{path}: position checksum differs from the catalogue (table made for '
            f'{t.meta.get("CATALOG")}); re-run the hook on this catalogue')
    dxi = np.asarray(t['dxi_mas'], float)
    deta = np.asarray(t['deta_mas'], float)
    if not (np.all(np.isfinite(dxi)) and np.all(np.isfinite(deta))):
        raise PosCorrMismatchError(f'{path}: non-finite corrections')
    return dxi, deta, dict(t.meta)


def apply_poscorr(sc, dxi_mas, deta_mas):
    """Corrected positions: ra += dxi / cos(dec), dec += deta (mas -> deg).  Returns a new SkyCoord in
    ``sc``'s frame; with all-zero corrections the positions are returned unchanged."""
    ra = sc.ra.deg + np.asarray(dxi_mas, float) * MAS_TO_DEG / np.cos(sc.dec.rad)
    dec = sc.dec.deg + np.asarray(deta_mas, float) * MAS_TO_DEG
    return SkyCoord(ra=ra, dec=dec, unit='deg', frame=sc.frame.replicate_without_data())
