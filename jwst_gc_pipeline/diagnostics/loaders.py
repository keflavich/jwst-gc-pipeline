"""Catalogue and mosaic loading for the diagnostic write-ups.

The write-ups read products that are already on disk and never modify them.
Two things make that less trivial than it sounds:

* the cross-band tables are wide (a 300k-row m8 merge carries ~250 columns),
  so the loaders take an explicit column list and pull only those;
* the per-filter tables carry raw ``flux`` in image units with no photometric
  calibration, while the cross-band table carries ``flux_jy`` and
  ``mag_vega``.  :func:`photometric_zeropoints` recovers the per-filter
  conversion from the cross-band table so that per-filter figures can be
  drawn on a physical magnitude axis whenever a cross-band merge exists, and
  falls back to an instrumental magnitude (clearly labelled) when it does not.
"""

import os
import warnings

import numpy as np
from astropy.io import fits
from astropy.table import Table

# mag_ab = -2.5 log10(flux_jy) + 8.90
ABMAG_OFFSET = 8.90


class MissingColumnsWarning(UserWarning):
    """A requested column is absent from a product; the panel degrades."""


def _present(path, wanted):
    """Subset of *wanted* actually present in the FITS table at *path*."""
    with fits.open(path, memmap=True) as hdul:
        hdu = next(h for h in hdul if getattr(h, 'columns', None) is not None)
        have = set(hdu.columns.names)
    return [c for c in wanted if c in have], [c for c in wanted if c not in have]


def read_columns(path, columns, label=''):
    """Read only *columns* from the FITS table at *path*.

    Columns are named as they appear in the FITS file, where a ``SkyCoord`` is
    stored as the pair ``skycoord.ra`` / ``skycoord.dec``.  ``Table.read``
    reassembles that pair into a single mixin column ``skycoord``, so a
    request for either half is resolved to the mixin.

    Missing columns are warned about and omitted rather than raising: a
    catalogue written before a given column existed should degrade one panel,
    not abort the whole document.
    """
    keep, missing = _present(path, list(columns))
    if missing:
        warnings.warn(
            f'{label or os.path.basename(path)}: no column(s) '
            f'{", ".join(missing)}; the panels that need them are omitted.',
            MissingColumnsWarning)
    tbl = Table.read(path, memmap=True)
    resolved = []
    for name in keep:
        if name in tbl.colnames:
            resolved.append(name)
        elif '.' in name and name.rsplit('.', 1)[0] in tbl.colnames:
            resolved.append(name.rsplit('.', 1)[0])
    # dict.fromkeys: a mixin requested twice (via .ra and .dec) appears once.
    return tbl[list(dict.fromkeys(resolved))]


def skycoord_of(tbl, column='skycoord'):
    """The SkyCoord in *column*, whether stored as a mixin or as ra/dec pairs."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    if column in tbl.colnames:
        col = tbl[column]
        if isinstance(col, SkyCoord):
            return col
        return SkyCoord(col)
    ra, dec = f'{column}.ra', f'{column}.dec'
    if ra in tbl.colnames and dec in tbl.colnames:
        return SkyCoord(np.asarray(tbl[ra]) * u.deg, np.asarray(tbl[dec]) * u.deg)
    raise KeyError(f'no sky coordinate column {column!r} in table')


def column(tbl, name, fill=np.nan):
    """*name* as a plain float array with masked entries filled, or all-*fill*.

    A ``skycoord.ra``-style name is resolved against the reassembled mixin.
    """
    if name not in tbl.colnames:
        if '.' in name:
            prefix, attr = name.rsplit('.', 1)
            if prefix in tbl.colnames and hasattr(tbl[prefix], attr):
                return np.asarray(getattr(tbl[prefix], attr).deg, dtype=float)
        return np.full(len(tbl), fill, dtype=float)
    col = tbl[name]
    values = col.filled(fill) if hasattr(col, 'filled') else col
    return np.asarray(values, dtype=float)


def photometric_zeropoints(crossband_path, filters):
    """``{filter: (conv_jy, vega_zp_jy)}`` recovered from a cross-band table.

    ``flux_jy = flux * conv_jy`` and ``mag_vega = -2.5 log10(flux_jy/vega_zp)``.
    Both constants are recovered as medians of per-source ratios, which is
    exact up to the floating-point noise of the original conversion and does
    not require knowing which zero-point file the merge used.

    Returns an empty dict when *crossband_path* is None.
    """
    if not crossband_path:
        return {}
    wanted = []
    for filt in filters:
        wanted += [f'flux_{filt}', f'flux_jy_{filt}', f'mag_vega_{filt}']
    keep, _missing = _present(crossband_path, wanted)
    if not keep:
        return {}
    tbl = Table.read(crossband_path, memmap=True)
    out = {}
    for filt in filters:
        fcol, jcol, vcol = f'flux_{filt}', f'flux_jy_{filt}', f'mag_vega_{filt}'
        if fcol not in tbl.colnames or jcol not in tbl.colnames:
            continue
        flux = column(tbl, fcol)
        fjy = column(tbl, jcol)
        good = np.isfinite(flux) & np.isfinite(fjy) & (flux > 0) & (fjy > 0)
        if good.sum() < 10:
            continue
        conv = float(np.median(fjy[good] / flux[good]))
        vega = np.nan
        if vcol in tbl.colnames:
            mag = column(tbl, vcol)
            ok = good & np.isfinite(mag)
            if ok.sum() >= 10:
                vega = float(np.median(fjy[ok] / 10 ** (-0.4 * mag[ok])))
        out[filt] = (conv, vega)
    return out


def magnitudes(flux, zeropoint):
    """Vega (or AB, or instrumental) magnitudes from raw *flux*.

    *zeropoint* is the ``(conv_jy, vega_zp)`` pair from
    :func:`photometric_zeropoints`, or None.  Returns ``(mag, label)`` so the
    caller can axis-label honestly: with no calibration available the returned
    magnitude is instrumental and the label says so.
    """
    flux = np.asarray(flux, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        if zeropoint is None:
            return -2.5 * np.log10(np.where(flux > 0, flux, np.nan)), \
                r'instrumental $-2.5\log_{10}(\mathrm{flux})$'
        conv, vega = zeropoint
        fjy = np.where(flux > 0, flux * conv, np.nan)
        if np.isfinite(vega) and vega > 0:
            return -2.5 * np.log10(fjy / vega), 'Vega mag'
        return -2.5 * np.log10(fjy) + ABMAG_OFFSET, 'AB mag'


def load_mosaic(path, downsample=4):
    """``(image, wcs, header)`` for an ``i2d`` mosaic, block-averaged.

    ``i2d`` products are rectified plain ``RA---TAN`` grids with no SIP, so
    ``astropy.wcs.WCS(header)`` is exact for them (this is the documented
    exemption from the GWCS rule -- see ``CLAUDE.md``).  They are also large;
    *downsample* block-averages by that factor and rescales the WCS to match,
    which is ample for the surface-brightness comparisons drawn here.
    """
    from astropy.wcs import WCS
    with fits.open(path, memmap=True) as hdul:
        sci = hdul['SCI'] if 'SCI' in hdul else hdul[1]
        data = np.asarray(sci.data, dtype=np.float32)
        header = sci.header.copy()
    wcs = WCS(header)
    if downsample and downsample > 1:
        ny, nx = data.shape
        ny -= ny % downsample
        nx -= nx % downsample
        block = data[:ny, :nx].reshape(ny // downsample, downsample,
                                       nx // downsample, downsample)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)  # all-NaN blocks
            data = np.nanmean(block, axis=(1, 3))
        wcs = wcs[::downsample, ::downsample]
    return data, wcs, header


def sample_mosaic(path, coords, downsample=4):
    """Mosaic surface brightness at each of *coords* (NaN off the footprint)."""
    from astropy.wcs import NoConvergence
    data, wcs, _header = load_mosaic(path, downsample=downsample)
    try:
        x, y = wcs.world_to_pixel(coords)
    except NoConvergence:
        # The inverse WCS did not converge for these coordinates (the m8
        # all_world2pix failure mode, issue #187).  Treat every source as off
        # the footprint so the mosaic panel blanks rather than taking the whole
        # figure down with it.
        return np.full(len(coords), np.nan, dtype=float)
    xi = np.round(x).astype(int)
    yi = np.round(y).astype(int)
    inside = ((xi >= 0) & (xi < data.shape[1]) &
              (yi >= 0) & (yi < data.shape[0]) &
              np.isfinite(x) & np.isfinite(y))
    out = np.full(len(coords), np.nan, dtype=float)
    out[inside] = data[yi[inside], xi[inside]]
    return out
