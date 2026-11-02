"""Load PSF grids saved by STPSF / WebbPSF ``psf_grid(save=True)``.

``stpsf.utils.to_griddedpsfmodel`` pairs data plane ``i`` with header key
``DET_YX{i}``.  The writer (``CreatePSFLibrary.create_grid``) does not keep that
pairing: since webbpsf 1.3.0 (commit f629e65d, 2023-12) it saves
``model.data``, the planes after photutils sorted them by y and then x, while
the ``DET_YX{i}`` keys stay in the order the PSFs were computed
(``itertools.product(loc_list, loc_list)``, x slowest).  For the square grids
``psf_grid`` builds, the round trip puts the PSF computed at detector (x, y) at
(y, x).  Every 4x4 grid in the shared PSF store (stpsf 2.2.0) is affected.

The transposition matters for photometry because the distorted STPSF PSF
conserves surface brightness: its sum scales as 1/AREA(x, y), which is what
makes a PSF fit to the MJy/sr ``crf`` return the true flux without a
pixel-area map.  With the planes transposed the sum follows 1/AREA(y, x), and
a star reads faint by ``2.5 log10(AREA(x, y) / AREA(y, x))``: over the wd2
stars, rms 0.012 mag (SW) and 0.022 mag (LW), 1-99 % range +-0.03 mag (SW) and
+-0.055 mag (LW).

:func:`load_stpsf_grid` assigns each plane to the position it was computed at:

- ``DET_YX{i}`` holds (y, x); the keys give the set of grid positions.
- Files from webbpsf >= 1.3.0 and stpsf (2.x) store the planes sorted by
  (y, x), so plane ``j`` belongs to the j-th position in that order.
- Files from webbpsf <= 1.2.1 store the planes in ``DET_YX`` key order.
- When the keys are already in (y, x) order both rules agree and any version
  is accepted.  Otherwise a version outside the checked ranges raises.
"""
import re
from collections import OrderedDict

import numpy as np
from astropy.io import fits
from astropy.nddata import NDData
from packaging.version import InvalidVersion, Version
from photutils.psf import GriddedPSFModel

#: Last webbpsf release whose psf_grid wrote the planes in DET_YX key order.
LAST_KEY_ORDER_VERSION = Version('1.2.1')
#: First webbpsf release whose psf_grid writes the planes sorted by (y, x).
FIRST_SORTED_VERSION = Version('1.3.0')
#: Newest (major, minor) checked to write sorted planes with unsorted keys.
#: A newer writer may have changed either side; check it before extending.
LAST_CHECKED_SORTED = (2, 2)

_DET_YX = re.compile(r'^DET_YX(\d+)$')


class PSFGridOrderError(ValueError):
    """The plane order of a saved PSF grid cannot be determined."""


def det_yx_positions(header):
    """The (x, y) position of each ``DET_YX{i}`` key, in key-index order."""
    keys = sorted(((int(m.group(1)), k) for k in header
                   if (m := _DET_YX.match(k))), key=lambda t: t[0])
    if not keys:
        raise KeyError("PSF grid header has no 'DET_YX{i}' keys")
    xy = []
    for _, k in keys:
        y, x = (float(v) for v in str(header[k]).strip().strip('()').split(','))
        xy.append((x, y))
    return np.array(xy)


def _yx_sort_order(xy):
    # photutils' GriddedPSFModel order: y primary, x secondary
    return np.lexsort((xy[:, 0], xy[:, 1]))


def planes_sorted_by_yx(header):
    """True when the writer stored the planes sorted by (y, x), False when it
    stored them in ``DET_YX`` key order.

    Raises `PSFGridOrderError` for a ``VERSION`` this module has not checked.
    """
    raw = header.get('VERSION')
    if raw is None:
        raise PSFGridOrderError('PSF grid header has no VERSION key')
    try:
        version = Version(str(raw))
    except InvalidVersion as ex:
        raise PSFGridOrderError(f'unparseable PSF grid VERSION {raw!r}') from ex
    if version <= LAST_KEY_ORDER_VERSION:
        return False
    if version >= FIRST_SORTED_VERSION and version.release[:2] <= LAST_CHECKED_SORTED:
        return True
    raise PSFGridOrderError(
        f'PSF grid written by STPSF/WebbPSF {raw}: the plane order of this '
        f'version has not been checked (sorted by (y, x) for '
        f'{FIRST_SORTED_VERSION} to {".".join(map(str, LAST_CHECKED_SORTED))}.x, '
        f'DET_YX key order up to {LAST_KEY_ORDER_VERSION}).  Check how its '
        f'CreatePSFLibrary.create_grid saves the data and extend '
        f'jwst_gc_pipeline.photometry.psf_grid_io.')


def grid_plane_positions(header, nplanes):
    """The (x, y) detector position each data plane was computed at."""
    xy = det_yx_positions(header)
    if len(xy) != nplanes:
        raise PSFGridOrderError(
            f'{len(xy)} DET_YX keys for {nplanes} PSF planes')
    order = _yx_sort_order(xy)
    if np.array_equal(order, np.arange(len(xy))):
        return xy
    if planes_sorted_by_yx(header):
        return xy[order]
    return xy


def load_stpsf_grid(hdulist_or_filename, ext_data=0, ext_header=0):
    """A `~photutils.psf.GriddedPSFModel` from a saved STPSF/WebbPSF grid.

    Drop-in replacement for ``stpsf.utils.to_griddedpsfmodel`` (same
    arguments, same lower-cased ``(value, comment)`` meta) that pairs each
    plane with the position it was computed at; see the module docstring.
    """
    if isinstance(hdulist_or_filename, fits.HDUList):
        return _from_hdulist(hdulist_or_filename, ext_data, ext_header)
    with fits.open(hdulist_or_filename) as hdul:
        return _from_hdulist(hdul, ext_data, ext_header)


def _from_hdulist(hdul, ext_data, ext_header):
    data = np.array(hdul[ext_data].data)
    header = hdul[ext_header].header
    if 'OVERSAMP' not in header:
        raise KeyError("PSF grid header has no 'OVERSAMP' key")
    if data.ndim == 2:
        data = data[np.newaxis]
    xypos = grid_plane_positions(header, data.shape[0])

    header = header.copy(strip=True)
    for key in ('COMMENT', 'HISTORY', ''):
        header.remove(key, ignore_missing=True, remove_all=True)
    meta = OrderedDict((card.keyword.lower(), (card.value, card.comment))
                       for card in header.cards)
    meta['grid_xypos'] = [tuple(p) for p in xypos]
    meta.setdefault('oversampling', header['OVERSAMP'])
    return GriddedPSFModel(NDData(data, meta=meta))
