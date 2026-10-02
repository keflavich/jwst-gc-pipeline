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
  STPSF) is unchanged.  The match is made separately for each sub-pixel phase
  of the grid (``_phase_scale``): the empirical cores' native-pixel flux varies
  with phase by ~1-2%, STPSF's by 0.15-0.4%, and the hybrid keeps STPSF's;
* **centre**: the core is shifted so that its centroid inside 1.5 px equals the
  STPSF grid's, so fitted positions keep STPSF's astrometric convention.  The
  per-node shifts that took are recorded in the returned grid's ``meta``.

Opt-in.  Nothing changes unless ``PSF_EPSF_CORE_DIR`` names a directory of core
files (``epsf_core_<detector>_<filter>.fits``, written by
``scripts/analysis/epsf_map/export_epsf_core.py``).  The 10678 cores live in
their own repository, https://github.com/JWST-GC/epsfs (kept out of this one
so binary PSF data never enter its history; the ten cores of PR #1009 are its
commit 7a09209, exported here at af5c2db8): clone it and point
``PSF_EPSF_CORE_DIR`` at the checkout.  A set directory that has
no file for the requested detector/filter keeps the STPSF grid and says so.

Provenance.  A hybrid catalog must never be mistaken for, or overwrite, an
STPSF one (it is ~0.4% / ~1.7-2.4% brighter at F212N / F480M; PR #1009).  So
with ``PSF_EPSF_CORE_DIR`` set, every catalog / residual filename carries the
``_hybpsf`` token (:func:`hybrid_psf_token`, appended after ``_epsf`` at every
site that builds or globs those names, so writers and readers agree), and the
catalog meta records which model was actually used (:func:`psf_provenance_meta`:
``PSFMODEL``, the core file and its sha256, R0/R1, the largest centroid shift).
The token follows the configuration; ``PSFMODEL`` follows what was applied --
a filter with no core file is ``_hybpsf`` but ``PSFMODEL = STPSF``.

The cores measure program 10678's observing conditions.  A frame from another
program is refused unless ``PSF_EPSF_ALLOW_OTHER_PROGRAM=1`` (that is an
extrapolation that needs its own check; see the PR #1009 description).

Units.  photutils evaluates ``flux * interp(data)`` and stpsf's ``psf_grid``
stores the pixel-integrated PSF times ``oversampling**2``, so a grid's data are
already "fraction of the flux in a native pixel whose centre is at this
offset" -- the same quantity as the ePSF core (normalised so that
``sum(P[r <= r_norm]) / O**2 == 1``).  The hybrid is therefore built directly
in data space.
"""
import hashlib
import os
import warnings

import numpy as np
from astropy.io import fits
from astropy.nddata import NDData
from photutils.psf import GriddedPSFModel
from scipy import ndimage

EPSF_CORE_DIR_ENV = 'PSF_EPSF_CORE_DIR'
ALLOW_OTHER_PROGRAM_ENV = 'PSF_EPSF_ALLOW_OTHER_PROGRAM'
HYBRID_TOKEN = '_hybpsf'   # must not contain '_epsf': residual globs match tokens by substring
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
    k = _phase_scale(D, C, r <= r0, oy, ox)
    t = np.clip((r - r0) / (r1 - r0), 0.0, 1.0)
    return (1.0 - t) * k * C + t * D, (dx, dy)


def _phase_scale(D, C, inner, oy, ox):
    """Scale of the core onto STPSF inside r0, one value per sub-pixel phase.

    The samples ``[a::oy, b::ox]`` of a pixel-integrated, oversampled PSF are
    the native-pixel image of a star at one sub-pixel phase, so their sum
    inside r0 is that star's flux inside r0.  The empirical core's sum varies
    with phase by ~1-2% (finite phase coverage of the ePSF stars; PR #1009
    review), STPSF's by 0.15-0.4%.  Matching each phase separately gives the
    hybrid STPSF's phase dependence -- and STPSF's flux inside r0 at every
    phase, not just on average.  A non-integer oversampling has no phase
    classes and falls back to one scale.
    """
    if not (float(oy).is_integer() and float(ox).is_integer()):
        return D[inner].sum() / C[inner].sum()
    oy, ox = int(oy), int(ox)
    k = np.empty_like(D)
    for a in range(oy):
        for b in range(ox):
            sel = np.zeros(D.shape, bool)
            sel[a::oy, b::ox] = True
            sel &= inner
            k[a::oy, b::ox] = D[sel].sum() / C[sel].sum()
    return k


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


# Hybrid grids built in this process, keyed on (core sha256, STPSF grid
# fingerprint): get_psf_model reloads the STPSF grid on every call, and a
# samp4 hybrid takes ~5 s to build.
_HYBRID_CACHE = {}
# What maybe_apply_epsf_core applied in this process, by (DETECTOR, FILTER):
# the catalog writer stamps it into the catalog meta (psf_provenance_meta).
_APPLIED = {}


def hybrid_psf_token(environ=None):
    """``'_hybpsf'`` when ``PSF_EPSF_CORE_DIR`` is set, else ``''``.

    Appended to the ``_epsf`` filename token everywhere, so a hybrid run never
    writes over (or is globbed as) an STPSF catalog.
    """
    env = os.environ if environ is None else environ
    return HYBRID_TOKEN if env.get(EPSF_CORE_DIR_ENV, '') else ''


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _grid_fingerprint(grid):
    h = hashlib.sha1(np.ascontiguousarray(grid.data).tobytes())
    h.update(np.asarray(grid.grid_xypos, dtype=float).tobytes())
    h.update(np.asarray(grid.oversampling, dtype=float).tobytes())
    return h.hexdigest()


def _check_program(core, path, program, env):
    core_prog = str(core.meta.get('PROGRAM', '') or '')
    if not (program and core_prog):
        return
    progs = {p.strip().lstrip('0') for p in core_prog.split(',')}
    if str(program).strip().lstrip('0') in progs:
        return
    msg = (f'{path} was built from program {core_prog}, not {program}: its core measures '
           f'{core_prog}\'s jitter / charge diffusion / wavefront epoch')
    if env.get(ALLOW_OTHER_PROGRAM_ENV, '') == '1':
        warnings.warn(msg + f' ({ALLOW_OTHER_PROGRAM_ENV}=1: using it anyway)')
        return
    raise ValueError(msg + f'; set {ALLOW_OTHER_PROGRAM_ENV}=1 to use it anyway')


def maybe_apply_epsf_core(grid, detector, filtername, environ=None, program=None):
    """Return the hybrid grid if ``PSF_EPSF_CORE_DIR`` provides a core for this
    detector/filter; otherwise ``grid`` unchanged (the same object).

    ``detector`` is the stpsf detector name (``NRCB5``, ``NRCA1``, ...); None
    takes it from the grid's metadata.  ``grid`` may be a list (the merged-module
    path): each element is handled on its own, by its own metadata.
    ``program`` (the frame's proposal id), when given, must be one the core was
    built from (see the module docstring).
    """
    env = os.environ if environ is None else environ
    core_dir = env.get(EPSF_CORE_DIR_ENV, '')
    if not core_dir:
        return grid
    if isinstance(grid, list):
        return [maybe_apply_epsf_core(g, None, filtername, env, program) for g in grid]
    detector = detector or grid_detector(grid)
    if detector is None:
        warnings.warn(f'{EPSF_CORE_DIR_ENV} is set but no detector is known for this '
                      f'{filtername} grid; keeping STPSF')
        return grid
    key = (detector.upper(), filtername.upper())
    path = os.path.join(core_dir, epsf_core_filename(detector, filtername))
    if not os.path.exists(path):
        print(f'{EPSF_CORE_DIR_ENV}={core_dir}: no ePSF core for {detector}/{filtername} '
              f'({os.path.basename(path)}); using the plain STPSF grid', flush=True)
        _APPLIED.setdefault(key, None)
        return grid
    core = read_epsf_core(path)
    if str(core.meta.get('DETECTOR', detector)).upper() != detector.upper() or \
            str(core.meta.get('FILTER', filtername)).upper() != filtername.upper():
        raise ValueError(f'{path} holds {core.meta.get("DETECTOR")}/{core.meta.get("FILTER")}, '
                         f'not {detector}/{filtername}')
    _check_program(core, path, program, env)
    core.meta['FILENAME'] = path
    sha = _sha256(path)
    ckey = (sha, _grid_fingerprint(grid))
    hyb = _HYBRID_CACHE.get(ckey)
    if hyb is None:
        hyb = make_hybrid_grid(grid, core)
        hyb.meta['epsf_core_sha256'] = sha
        _HYBRID_CACHE[ckey] = hyb
    _APPLIED[key] = dict(path=path, sha256=sha, program=str(core.meta.get('PROGRAM', '')),
                         r0=hyb.meta['epsf_r0'], r1=hyb.meta['epsf_r1'],
                         shift_max_mpix=1e3 * hyb.meta['epsf_centroid_shift_max_px'])
    print(f'Hybrid PSF for {detector}/{filtername}: ePSF core r<{R0:g} px from {path}, '
          f'STPSF wing r>{R1:g} px; core re-centred onto STPSF by <= '
          f'{hyb.meta["epsf_centroid_shift_max_px"]*1e3:.1f} mpix', flush=True)
    return hyb


def psf_provenance_meta(filtername, environ=None):
    """Catalog meta (FITS-length keys) recording the PSF model used for ``filtername``.

    ``PSFMODEL`` is ``'STPSF+EPSFCORE'`` when this process applied a core for
    the filter, else ``'STPSF'``.  With a core: ``EPSFCORE`` (file name),
    ``EPSFSHA`` (sha256), ``EPSFPROG``, ``EPSFR0`` / ``EPSFR1`` [px] and
    ``EPSFSHFT`` (largest core re-centring [mpix]).  ``EPSFDIR`` records a set
    ``PSF_EPSF_CORE_DIR`` either way.
    """
    env = os.environ if environ is None else environ
    core_dir = env.get(EPSF_CORE_DIR_ENV, '')
    used = [v for (det, filt), v in sorted(_APPLIED.items())
            if filt == filtername.upper() and v is not None]
    meta = {'PSFMODEL': 'STPSF+EPSFCORE' if used else 'STPSF'}
    if core_dir:
        meta['EPSFDIR'] = core_dir
    if used:
        meta.update(EPSFCORE=','.join(os.path.basename(v['path']) for v in used),
                    EPSFSHA=','.join(v['sha256'] for v in used),
                    EPSFPROG=','.join(sorted({v['program'] for v in used})),
                    EPSFR0=float(used[0]['r0']), EPSFR1=float(used[0]['r1']),
                    EPSFSHFT=float(max(v['shift_max_mpix'] for v in used)))
    return meta
