"""Build and load the PCAT C kernel (ctypes) and render PSF models.

The shared library combines

* ``_vendor/pcat_lion_kernel.c``: the PCAT-DE / Lion model + likelihood kernel
  (MIT, patched as listed in its header), and
* ``pcat_regions.c``: region-label chi^2 bookkeeping used by
  :mod:`.pcat_sampler`.

Three BLAS back ends are supported: ``openblas`` (the conda env's
``libcblas`` -> OpenBLAS), ``mkl`` (``$MKLROOT``) and ``none`` (plain loops).
``auto`` loads ``openblas`` and falls back to ``none`` (with a warning) when
the OpenBLAS kernel cannot be built, e.g. outside a conda env.

Rendering convention (identical to PCAT ``image_model_eval``)
--------------------------------------------------------------
A source at 0-based pixel position ``(x, y)`` (pixel centres at integers) is
drawn with stamp-centre index ``nc // 2`` on pixel ``(ceil(y), ceil(x))`` and
sub-pixel coefficients ``dx = ceil(x) - x``, ``dy = ceil(y) - y`` in ``[0, 1)``.
Arrays are ``(ny, nx)`` row-major; the C row length is ``nx``.
"""
import ctypes
import os
import shutil
import subprocess
import warnings
from pathlib import Path

import numpy as np
from numpy.ctypeslib import ndpointer

__all__ = ['build_kernel', 'load_kernel', 'basis_terms', 'design_rows',
           'PSFRenderer', 'default_lib_dir']

_HERE = Path(__file__).resolve().parent
_SOURCES = [_HERE / '_vendor' / 'pcat_lion_kernel.c', _HERE / 'pcat_regions.c']
CONDA_PREFIX = Path(os.environ.get('CONDA_PREFIX', '/blue/adamginsburg/adamginsburg/miniconda3/envs/python313'))

DEFAULT_MKLROOT = '/apps/compilers/intel/2025/1.0.666/mkl/latest'

_f4 = ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')
_f8 = ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
_i4 = ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
_int = ctypes.c_int

_LOADED = {}


def default_lib_dir():
    """Directory holding the compiled kernels (env ``PCAT_JWST_LIBDIR`` or ``_lib``)."""
    return Path(os.environ.get('PCAT_JWST_LIBDIR', _HERE / '_lib'))


def _lib_name(blas):
    return f'libpcatjwst_{blas}.so'


def build_kernel(lib_dir=None, blas='openblas', march='x86-64-v3', cc='gcc', verbose=False):
    """Compile the kernel into ``lib_dir/libpcatjwst_<blas>.so`` and return its path.

    ``march=x86-64-v3`` (AVX2/FMA) runs on every HiPerGator AMD EPYC node.
    """
    lib_dir = Path(lib_dir) if lib_dir is not None else default_lib_dir()
    lib_dir.mkdir(parents=True, exist_ok=True)
    out = lib_dir / _lib_name(blas)
    cmd = [cc, '-O3', f'-march={march}', '-fPIC', '-shared', '-std=gnu11',
           '-o', str(out)] + [str(s) for s in _SOURCES]
    if blas == 'openblas':
        # the env's include/cblas.h is the ILP64 (openblas64_) header while
        # lib/libcblas.so is LP64 OpenBLAS: declare the LP64 prototype locally
        cmd += ['-DPCAT_DECLARE_CBLAS', f'-L{CONDA_PREFIX}/lib', '-lcblas',
                f'-Wl,-rpath,{CONDA_PREFIX}/lib']
    elif blas == 'mkl':
        mklroot = os.environ.get('MKLROOT', DEFAULT_MKLROOT)
        if not Path(mklroot, 'include', 'mkl_cblas.h').exists():
            raise RuntimeError(f'no mkl_cblas.h under {mklroot}; module load mkl')
        # sequential MKL (libmkl_rt defaults to the Intel OpenMP layer, which
        # needs libiomp5 and aborts with "undefined symbol: omp_get_num_procs")
        cmd += ['-DPCAT_USE_MKL', f'-I{mklroot}/include', f'-L{mklroot}/lib/intel64',
                '-Wl,--no-as-needed', '-lmkl_intel_lp64', '-lmkl_sequential', '-lmkl_core',
                '-lpthread', '-lm', '-ldl', f'-Wl,-rpath,{mklroot}/lib/intel64']
    elif blas == 'none':
        cmd += ['-DPCAT_NO_CBLAS']
    else:
        raise ValueError(f'unknown blas {blas!r}')
    if shutil.which(cc) is None:
        raise RuntimeError(f'compiler {cc} not found')
    res = subprocess.run(cmd, capture_output=True, text=True)
    if verbose or res.returncode != 0:
        print(' '.join(cmd))
        print(res.stdout, res.stderr)
    if res.returncode != 0:
        raise RuntimeError(f'kernel build failed ({res.returncode}): {res.stderr}')
    return out


def load_kernel(blas='openblas', lib_dir=None, build=True, nthreads=1):
    """Load (building if missing) the kernel; returns a ctypes ``CDLL``.

    ``lib.blas`` is the back end loaded (``blas='auto'``: ``openblas``, else
    ``none``).

    ``nthreads`` sets the OpenBLAS thread count (1 is fastest for the small
    sgemm calls of the sampler).
    """
    if blas == 'auto':
        key = f'auto:{lib_dir}'
        if key not in _LOADED:
            try:
                _LOADED[key] = load_kernel('openblas', lib_dir=lib_dir, build=build, nthreads=nthreads)
            except (RuntimeError, OSError) as err:
                warnings.warn(f'OpenBLAS kernel unavailable ({err}); using the plain-loop kernel')
                _LOADED[key] = load_kernel('none', lib_dir=lib_dir, build=build, nthreads=nthreads)
        return _LOADED[key]
    lib_dir = Path(lib_dir) if lib_dir is not None else default_lib_dir()
    path = lib_dir / _lib_name(blas)
    key = str(path)
    if key in _LOADED:
        return _LOADED[key]
    if not path.exists():
        if not build:
            raise FileNotFoundError(path)
        build_kernel(lib_dir, blas=blas)
    lib = ctypes.CDLL(str(path))
    lib.blas = blas
    lib.pcat_model_eval.restype = None
    lib.pcat_model_eval.argtypes = [_int, _int, _int, _int, _int, _f4, _f4, _f4, _i4, _i4,
                                    _f4, _f4, _f4, _f8, _int, _int, _int, _int]
    lib.pcat_like_eval.restype = None
    lib.pcat_like_eval.argtypes = [_int, _int, _f4, _f4, _f4, _f8, _int, _int, _int, _int]
    lib.pcat_imag_acpt.restype = None
    lib.pcat_imag_acpt.argtypes = [_int, _int, _f4, _f4, _i4, _int, _int, _int, _int]
    lib.pcat_dchi2_regions.restype = None
    lib.pcat_dchi2_regions.argtypes = [_int, _f4, _f4, _f4, _i4, _int, _f8]
    lib.pcat_apply_accept.restype = None
    lib.pcat_apply_accept.argtypes = [_int, _f4, _i4, _i4, _f4, _f4]
    lib.pcat_dchi2_global.restype = ctypes.c_double
    lib.pcat_dchi2_global.argtypes = [_int, _f4, _f4, _f4]
    # raw-pointer handles for the sampler's hot loop (ndpointer checks cost
    # ~10 us per argument); callers pass ``arr.ctypes.data`` of C-contiguous
    # arrays of the right dtype
    vp = ctypes.c_void_p
    fast = lib['pcat_model_eval']
    fast.restype = None
    fast.argtypes = [_int] * 5 + [vp] * 9 + [_int] * 4
    lib.fast_model_eval = fast
    fast = lib['pcat_dchi2_regions']
    fast.restype = None
    fast.argtypes = [_int, vp, vp, vp, vp, _int, vp]
    lib.fast_dchi2_regions = fast
    fast = lib['pcat_apply_accept']
    fast.restype = None
    fast.argtypes = [_int, vp, vp, vp, vp, vp]
    lib.fast_apply_accept = fast
    if blas == 'openblas' and nthreads is not None:
        try:
            setter = ctypes.CDLL(str(CONDA_PREFIX / 'lib' / 'libcblas.so')).openblas_set_num_threads
        except (OSError, AttributeError):
            setter = None
        if setter is not None:
            setter.argtypes = [_int]
            setter(int(nthreads))
    _LOADED[key] = lib
    return lib


# ----------------------------------------------------------------------------
# polynomial bases
# ----------------------------------------------------------------------------
def basis_terms(basis):
    """Exponent pairs ``(a, b)`` of the sub-pixel basis ``dx**a * dy**b``.

    ``'cubic10'`` is PCAT's 10-term total-degree-3 basis in PCAT's column order
    (1, dx, dy, dx^2, dx dy, dy^2, dx^3, dx^2 dy, dx dy^2, dy^3); ``'bicubic16'``
    adds the six terms of degree 4-6 with each exponent <= 3; ``'quartic15'`` is
    total degree <= 4.
    """
    pcat10 = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (1, 2), (0, 3)]
    if basis == 'cubic10':
        return pcat10
    if basis == 'bicubic16':
        return pcat10 + [(3, 1), (2, 2), (1, 3), (3, 2), (2, 3), (3, 3)]
    if basis == 'quartic15':
        return pcat10 + [(4, 0), (3, 1), (2, 2), (1, 3), (0, 4)]
    raise ValueError(f'unknown basis {basis!r}')


def design_rows(dx, dy, terms):
    """``(n, k)`` float32 matrix of basis terms evaluated at ``(dx, dy)``."""
    dx = np.asarray(dx, dtype=np.float64).ravel()
    dy = np.asarray(dy, dtype=np.float64).ravel()
    deg = max(max(a, b) for a, b in terms)
    px = [np.ones_like(dx)]
    py = [np.ones_like(dy)]
    for _ in range(deg):
        px.append(px[-1] * dx)
        py.append(py[-1] * dy)
    out = np.empty((dx.size, len(terms)), dtype=np.float32)
    for j, (a, b) in enumerate(terms):
        out[:, j] = px[a] * py[b]
    return out


class PSFRenderer:
    """Render point sources on a ``(ny, nx)`` float32 array with the C kernel.

    Parameters
    ----------
    cf : (k, nc*nc) float32 coefficient table (see :mod:`.psf_poly`).
    nc : stamp size (odd).
    terms : basis exponent list matching ``cf`` rows.
    lib : ctypes kernel or None (pure numpy fallback).
    """

    def __init__(self, cf, nc, terms, lib=None):
        self.cf = np.ascontiguousarray(cf, dtype=np.float32)
        self.nc = int(nc)
        self.rad = self.nc // 2
        self.terms = list(terms)
        self.k = len(self.terms)
        if self.cf.shape != (self.k, self.nc * self.nc):
            raise ValueError(f'cf shape {self.cf.shape} != ({self.k}, {self.nc**2})')
        self.lib = lib
        self._dummy = {}

    def _dummies(self, shape):
        if shape not in self._dummy:
            self._dummy[shape] = (np.zeros(shape, np.float32), np.zeros(shape, np.float32),
                                  np.zeros(4, np.float64))
        return self._dummy[shape]

    def render(self, x, y, f, image):
        """Add sources (positions in array pixel coordinates) to ``image`` in place.

        Sources must satisfy ``rad <= ceil(x) <= nx-1-rad`` (and likewise in y);
        the sampler pads its arrays so that this always holds.
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        f = np.asarray(f, dtype=np.float64)
        n = x.size
        if n == 0:
            return image
        ix = np.ceil(x).astype(np.int32)
        iy = np.ceil(y).astype(np.int32)
        ny, nx = image.shape
        if (ix.min() < self.rad or iy.min() < self.rad or ix.max() > nx - 1 - self.rad
                or iy.max() > ny - 1 - self.rad):
            raise ValueError('source stamp outside the (padded) array')
        dd = design_rows(ix - x, iy - y, self.terms) * f.astype(np.float32)[:, None]
        if self.lib is None:
            recon = (dd @ self.cf).reshape(n, self.nc, self.nc)
            r = self.rad
            for i in range(n):
                image[iy[i] - r:iy[i] + r + 1, ix[i] - r:ix[i] + r + 1] += recon[i]
            return image
        if image.dtype != np.float32 or not image.flags.c_contiguous:
            raise TypeError('image must be a C-contiguous float32 array')
        dd = np.ascontiguousarray(dd, dtype=np.float32)
        recon = np.empty((n, self.nc * self.nc), dtype=np.float32)
        ref, wt, diff2 = self._dummies(image.shape)
        # the kernel merges phonions on the same pixel in place (ix, iy are
        # fresh arrays here)
        ix = np.ascontiguousarray(ix, dtype=np.int32)
        iy = np.ascontiguousarray(iy, dtype=np.int32)
        self.lib.fast_model_eval(nx, ny, n, self.nc, self.k, dd.ctypes.data, self.cf.ctypes.data,
                                 recon.ctypes.data, ix.ctypes.data, iy.ctypes.data,
                                 image.ctypes.data, ref.ctypes.data, wt.ctypes.data,
                                 diff2.ctypes.data, max(nx, ny), 0, 0, 0)
        return image
