"""Hybrid NIRCam PSF: empirical ePSF core, STPSF wing.

STPSF is too sharp in the core for the 10678 Galactic Center frames: it lacks
the jitter / charge-diffusion blur measured in issue #1002, so every fitted star
leaves a negative core and a positive 2-4 px ring (about -0.35% of the flux per
pixel at F212N).  An empirical ePSF built from the field stars removes most of
that, but beyond ~12 px it is built from too little light per star in a
confusion-limited field, and its wing absorbs neighbour light (issue #1007:
around saturated cores the pure ePSF is slightly WORSE than STPSF in F480M).

The hybrid keeps each half where it is right: the ePSF inside ``R0`` = 10 px,
the STPSF grid outside ``R1`` = 12 px, and a linear blend in between.  On the
held-out o081 frame (issue #1007) it is the best of the three models on every
frame-level number in both F212N and F480M.

The STPSF grid stays the reference for the two conventions downstream code
relies on:

* **normalisation**: the core is scaled so that its flux inside ``R0`` equals
  the STPSF grid's flux inside ``R0`` at the same position, so total
  normalisation (and hence the zero point / aperture corrections calibrated on
  STPSF) is unchanged;
* **centre**: the core is shifted so that its centroid inside 1.5 px equals the
  STPSF grid's, so fitted positions keep STPSF's astrometric convention.  The
  per-node shifts that took are recorded in the returned grid's ``meta``.

Opt-in.  Nothing changes unless ``PSF_EPSF_CORE_DIR`` names a directory of core
files (``epsf_core_<detector>_<filter>.fits``, written by
``scripts/analysis/epsf_map/export_epsf_core.py``).  The 10678 cores live in
their own repository, ``JWST-GC/epsfs`` (kept out of this one so binary PSF
data never enter its history): clone it and point ``PSF_EPSF_CORE_DIR`` at the
checkout.  A set directory that has
no file for the requested detector/filter keeps the STPSF grid and says so.

Units.  photutils evaluates ``flux * interp(data)`` and stpsf's ``psf_grid``
stores the pixel-integrated PSF times ``oversampling**2``, so a grid's data are
already "fraction of the flux in a native pixel whose centre is at this
offset" -- the same quantity as the ePSF core (normalised so that
``sum(P[r <= r_norm]) / O**2 == 1``).  The hybrid is therefore built directly
in data space.
"""
import os
import warnings

import numpy as np
from astropy.io import fits
from astropy.nddata import NDData
from photutils.psf import GriddedPSFModel
from scipy import ndimage

EPSF_CORE_DIR_ENV = 'PSF_EPSF_CORE_DIR'
R0 = 10.0       # ePSF inside this radius [native px]
R1 = 12.0       # STPSF outside this radius [native px]
R_CENTROID = 1.5
FORMAT_VERSION = 1


def _bilinear_weights(x, y, node_x, node_y):
    """Bilinear (node index, weight) pairs at (x, y) on a regular lattice, index
    ``j * len(node_x) + i``; clamped to the outermost nodes, as photutils does
    for a GriddedPSFModel."""
    def axis(v, nodes):
        if len(nodes) == 1:
            return [(0, 1.0)]
        f = float(np.clip(np.interp(v, nodes, np.arange(len(nodes))), 0, len(nodes) - 1))
        i0 = min(int(np.floor(f)), len(nodes) - 2)
        t = f - i0
        return [(i0, 1 - t), (i0 + 1, t)]
    nx = len(node_x)
    return [(j * nx + i, wx * wy) for j, wy in axis(y, node_y)
            for i, wx in axis(x, node_x) if wx * wy > 0]


class EPSFCore:
    """A spatially varying ePSF core on a regular lattice of detector positions.

    ``data[j * len(node_x) + i]`` is the core at ``(node_x[i], node_y[j])``,
    sampled at ``oversampling`` nodes per native pixel out to ``radius`` px
    (shape ``(2 * O * R + 1,) * 2``, the star at the central node).
    """

    def __init__(self, data, oversampling, radius, node_x, node_y, meta=None):
        self.data = np.asarray(data, dtype=np.float64)
        self.oversampling = int(oversampling)
        self.radius = int(radius)
        self.node_x = np.asarray(node_x, dtype=np.float64)
        self.node_y = np.asarray(node_y, dtype=np.float64)
        self.meta = dict(meta or {})
        n = 2 * self.oversampling * self.radius + 1
        if self.data.shape != (len(self.node_x) * len(self.node_y), n, n):
            raise ValueError(f'ePSF core data shape {self.data.shape} does not match '
                             f'{len(self.node_x)}x{len(self.node_y)} nodes of {n}x{n}')
        if self.radius < R1:
            raise ValueError(f'ePSF core radius {self.radius} px < blend radius R1={R1}')
        self._coef = [ndimage.spline_filter(P, order=3, mode='constant') for P in self.data]

    def weights(self, x, y):
        return _bilinear_weights(x, y, self.node_x, self.node_y)

    def sample(self, x, y, ux, uy):
        """Core at detector position (x, y), evaluated at offsets (ux, uy) [native px]
        from the star; zero beyond ``radius``."""
        O, R = self.oversampling, self.radius
        coords = [np.ravel(uy) * O + O * R, np.ravel(ux) * O + O * R]
        out = np.zeros(np.size(ux))
        for k, w in self.weights(x, y):
            out += w * ndimage.map_coordinates(self._coef[k], coords, order=3,
                                               prefilter=False, mode='constant', cval=0.0)
        return out.reshape(np.shape(ux))


def epsf_core_filename(detector, filtername):
    return f'epsf_core_{detector.lower()}_{filtername.lower()}.fits'


def write_epsf_core(path, core, detector, filtername):
    hdr = fits.Header()
    for k, v in core.meta.items():          # provenance first: the keys below always win
        if k.upper() not in ('FILENAME',):
            hdr[k[:8].upper()] = v
    hdr['EPSFVER'] = (FORMAT_VERSION, 'epsf_hybrid core file format')
    hdr['DETECTOR'] = (detector.upper(), 'stpsf detector name (NRCB5 = NRCBLONG)')
    hdr['FILTER'] = filtername.upper()
    hdr['OVERSAMP'] = (core.oversampling, 'nodes per native pixel')
    hdr['RADIUS'] = (core.radius, 'core half-size [native px]')
    hdus = [fits.PrimaryHDU(core.data.astype(np.float32), header=hdr),
            fits.ImageHDU(core.node_x, name='NODE_X'),
            fits.ImageHDU(core.node_y, name='NODE_Y')]
    fits.HDUList(hdus).writeto(path, overwrite=True)


def read_epsf_core(path):
    with fits.open(path) as hl:
        hdr = hl[0].header
        if hdr.get('EPSFVER') != FORMAT_VERSION:
            raise ValueError(f'{path}: EPSFVER={hdr.get("EPSFVER")!r}, expected {FORMAT_VERSION}')
        meta = {k: hdr[k] for k in hdr if k not in ('SIMPLE', 'BITPIX', 'EXTEND', 'COMMENT', 'HISTORY')
                and not k.startswith('NAXIS')}
        return EPSFCore(hl[0].data, hdr['OVERSAMP'], hdr['RADIUS'],
                        hl['NODE_X'].data, hl['NODE_Y'].data, meta=meta)


def _centroid(img, ux, uy, rmax):
    w = np.where(np.hypot(ux, uy) <= rmax, np.clip(img, 0, None), 0.0)
    s = w.sum()
    if not s > 0:
        raise ValueError('PSF has no positive flux inside the centroid radius')
    return (w * ux).sum() / s, (w * uy).sum() / s


def hybrid_stamp(stpsf_data, origin, oversampling, core, x, y, r0=R0, r1=R1,
                 tol=1e-5, maxiter=100):
    """One hybrid ePSF on the STPSF stamp's own node grid.

    Returns ``(data, (dx, dy))``: the hybrid array (same shape as ``stpsf_data``)
    and the shift applied to the core to put its centroid on STPSF's [native px].
    """
    D = np.asarray(stpsf_data, dtype=np.float64)
    ny, nx = D.shape
    oy, ox = np.broadcast_to(np.asarray(oversampling, dtype=float), (2,))   # photutils order (y, x)
    ux = (np.arange(nx) - origin[0]) / ox
    uy = (np.arange(ny) - origin[1]) / oy
    UX, UY = np.meshgrid(ux, uy)
    r = np.hypot(UX, UY)
    cx_s, cy_s = _centroid(D, UX, UY, R_CENTROID)
    # Fixed-point iteration on the shift.  The hard-edged centroid aperture on a
    # 2x-sampled grid responds to a shift with a gain below 1, so a fixed number
    # of steps leaves mpix residuals: iterate to tolerance instead.
    dx = dy = 0.0
    for _ in range(maxiter):
        C = core.sample(x, y, UX - dx, UY - dy)
        cx, cy = _centroid(C, UX, UY, R_CENTROID)
        ex, ey = cx_s - cx, cy_s - cy
        if np.hypot(ex, ey) < tol:
            break
        dx, dy = dx + ex, dy + ey
    else:
        raise RuntimeError(f'ePSF core centroid did not converge onto STPSF at ({x:.0f}, {y:.0f}): '
                           f'residual {np.hypot(ex, ey):.2e} px')
    inner = r <= r0
    k = D[inner].sum() / C[inner].sum()
    t = np.clip((r - r0) / (r1 - r0), 0.0, 1.0)
    return (1.0 - t) * k * C + t * D, (dx, dy)


def _stpsf_data_at(grid, x, y):
    """STPSF grid data at (x, y), bilinear in the grid's own nodes (photutils' rule)."""
    xy = np.asarray(grid.grid_xypos, dtype=float)
    gx, gy = np.unique(xy[:, 0]), np.unique(xy[:, 1])
    lookup = {(float(a), float(b)): i for i, (a, b) in enumerate(xy)}
    out = np.zeros(grid.data.shape[1:])
    for k, w in _bilinear_weights(x, y, gx, gy):
        i, j = k % len(gx), k // len(gx)
        out += w * grid.data[lookup[(float(gx[i]), float(gy[j]))]]
    return out


def make_hybrid_grid(grid, core, r0=R0, r1=R1):
    """A new GriddedPSFModel: the ePSF core inside r0, ``grid`` (STPSF) outside r1.

    The hybrid is built at the CORE's nodes (the lattice the ePSF was measured
    on), with the STPSF data interpolated to each node -- the construction
    validated in issue #1007.
    """
    origin = getattr(grid, 'origin', None)
    if origin is None:
        origin = ((grid.data.shape[2] - 1) / 2.0, (grid.data.shape[1] - 1) / 2.0)
    data, xypos, shifts = [], [], []
    for y in core.node_y:
        for x in core.node_x:
            D = _stpsf_data_at(grid, x, y)
            H, sh = hybrid_stamp(D, origin, grid.oversampling, core, x, y, r0, r1)
            data.append(H); xypos.append((x, y)); shifts.append(sh)
    shifts = np.asarray(shifts)
    meta = {k: v for k, v in dict(grid.meta).items() if k not in ('grid_xypos', 'oversampling')}
    meta.update(grid_xypos=xypos, oversampling=grid.oversampling,
                epsf_core=core.meta.get('FILENAME', ''), epsf_r0=r0, epsf_r1=r1,
                epsf_centroid_shift_max_px=float(np.hypot(*shifts.T).max()),
                epsf_centroid_shifts_px=shifts.tolist())
    return GriddedPSFModel(NDData(np.asarray(data), meta=meta))


def grid_detector(grid):
    """The detector a stpsf grid was built for (its ``meta['detector']``), or None."""
    v = dict(getattr(grid, 'meta', {}) or {}).get('detector')
    if isinstance(v, (tuple, list)):
        v = v[0]
    return str(v).upper() if v else None


def maybe_apply_epsf_core(grid, detector, filtername, environ=None):
    """Return the hybrid grid if ``PSF_EPSF_CORE_DIR`` provides a core for this
    detector/filter; otherwise ``grid`` unchanged (the same object).

    ``detector`` is the stpsf detector name (``NRCB5``, ``NRCA1``, ...); None
    takes it from the grid's metadata.  ``grid`` may be a list (the merged-module
    path): each element is handled on its own, by its own metadata.
    """
    env = os.environ if environ is None else environ
    core_dir = env.get(EPSF_CORE_DIR_ENV, '')
    if not core_dir:
        return grid
    if isinstance(grid, list):
        return [maybe_apply_epsf_core(g, None, filtername, env) for g in grid]
    detector = detector or grid_detector(grid)
    if detector is None:
        warnings.warn(f'{EPSF_CORE_DIR_ENV} is set but no detector is known for this '
                      f'{filtername} grid; keeping STPSF')
        return grid
    path = os.path.join(core_dir, epsf_core_filename(detector, filtername))
    if not os.path.exists(path):
        print(f'{EPSF_CORE_DIR_ENV}={core_dir}: no ePSF core for {detector}/{filtername} '
              f'({os.path.basename(path)}); using the plain STPSF grid', flush=True)
        return grid
    core = read_epsf_core(path)
    if str(core.meta.get('DETECTOR', detector)).upper() != detector.upper() or \
            str(core.meta.get('FILTER', filtername)).upper() != filtername.upper():
        raise ValueError(f'{path} holds {core.meta.get("DETECTOR")}/{core.meta.get("FILTER")}, '
                         f'not {detector}/{filtername}')
    core.meta['FILENAME'] = path
    hyb = make_hybrid_grid(grid, core)
    print(f'Hybrid PSF for {detector}/{filtername}: ePSF core r<{R0:g} px from {path}, '
          f'STPSF wing r>{R1:g} px; core re-centred onto STPSF by <= '
          f'{hyb.meta["epsf_centroid_shift_max_px"]*1e3:.1f} mpix', flush=True)
    return hyb
