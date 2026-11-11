"""Flag NIRCam pixels whose ramp-fit rate came from a single group.

The problem
-----------
When every group after the first is flagged SATURATED, ``ramp_fit`` (with
``suppress_one_group=False``, as ``PipelineRerunNIRCAM-LONG.py`` runs it)
returns ``rate = G1 / tgroup`` for that pixel.  G1 is the average of the
NFRAMES frames of group 1, so its effective exposure time is not ``tgroup``:
for BRIGHT2 (NFRAMES=2, TGROUP=21.47 s) it is about 0.75 tgroup, so the rate
is about 0.75x the true value.  The cal/crf PIXELDQ stays 0 there (SATURATED
reaches the rate DQ only when ALL groups saturate), so PSF and satstar fits
use those pixels as good data.

Measured on brick 2221 o001 F182M jw02221001001_07101_00001_nrcb4, star at
crf pixel (427, 1794): the 16 pixels with group 2 flagged come out at exactly
G1/tgroup, the growth-ring pixels (flagged only by n_pix_grow_sat=1) at
0.55-0.66x their (G2-G1)/tgroup, and the star at 0.87 mJy against an expected
1.05-1.12 mJy.  The F182M/F187N ratio of stars with F200W < ~16.4 AB drops
from 1.00 to 0.78.

The fix
-------
Read GROUPDQ from the sibling Detector1 ``_ramp.fits`` and, for pixels with
exactly one usable group (no SATURATED, no DO_NOT_USE) in every integration
that has any, OR ``SATURATED | DO_NOT_USE`` into the frame DQ and set
SCI/ERR/VAR_* to NaN, which is how the crf carries pixels with zero good
groups.  The satstar finder then seeds them (it keys on SATURATED), the
daophot path masks them (``truly_lost_saturated_mask`` = SATURATED &
DO_NOT_USE), and the satstar first-read recovery rebuilds them from group 0
with its empirically calibrated scale.

Wired in two places:
  * ``write_perexposure_crf`` in ``PipelineRerunNIRCAM-LONG.py`` applies it to
    each per-exposure crf it writes (persisted, header ``G1ONLYDQ``);
  * cataloging's ``load_data`` and satstar's ``remove_saturated_stars`` apply
    it in memory, so crf written before this change are corrected at
    cataloging time without a reduction rerun.
Applying it twice is a no-op (OR of the same bits, NaN of the same pixels).

``G1ONLY_SAT_DQ=0`` disables it everywhere.  A NIRCam per-exposure frame with
no sibling ``_ramp.fits`` (or one whose GROUPDQ does not match the frame
shape) is reported with a WARNING and counted in ``COUNTS``; it is never
skipped silently.  Mosaics and non-NIRCam frames are out of scope and skipped
without a warning.
"""
import os
import re
import sys
import warnings
from collections import Counter

import numpy as np
from astropy.io import fits

SATURATED = 2
DO_NOT_USE = 1
FLAG_BITS = SATURATED | DO_NOT_USE
ENV = 'G1ONLY_SAT_DQ'
HEADER_KEY = 'G1ONLYDQ'
NAN_EXTS = ('SCI', 'ERR', 'VAR_POISSON', 'VAR_RNOISE', 'VAR_FLAT')

# Per-exposure NIRCam product: jw<prop><obs><visit>_<grp><seq><act>_<exp>_nrc<det>_...
_PEREXP_RE = re.compile(r'^jw\d{11}_\d{5}_\d{5}_nrc[ab](?:long|[1-4])_')

#: Running tally for this process: 'applied', 'flagged_px', 'no_single_group',
#: 'missing_ramp', 'shape_mismatch', 'disabled'.  Missing ramps are the number
#: to watch: those frames keep the biased single-group rate.
COUNTS = Counter()
MISSING = []

_MASK_CACHE = {}


class G1OnlyRampMissingWarning(UserWarning):
    """A NIRCam per-exposure frame has no usable sibling ``_ramp.fits``."""


def enabled(env=None):
    env = os.environ if env is None else env
    return str(env.get(ENV, '1')).strip().lower() not in ('0', 'false', 'no', 'off', '')


def single_group_mask(groupdq):
    """Pixels whose rate rests on exactly one usable group.

    ``groupdq`` is ``(nints, ngroups, ny, nx)`` or ``(ngroups, ny, nx)``.  A
    group is usable when it carries neither SATURATED nor DO_NOT_USE.  A pixel
    is flagged when its best integration has exactly one usable group, i.e. no
    integration gives the ramp fit two groups to difference.  Pixels with zero
    usable groups everywhere are already SATURATED | DO_NOT_USE in the rate and
    are not returned.  NGROUPS=1 ramps return an all-False mask.
    """
    gdq = np.asarray(groupdq)
    if gdq.ndim == 3:
        gdq = gdq[None]
    if gdq.ndim != 4:
        raise ValueError(f"GROUPDQ must be 3-D or 4-D, got shape {gdq.shape}")
    if gdq.shape[1] < 2:
        # NGROUPS=1: every pixel is single-group by design, uniformly; that is
        # a calibration question, not a saturation flag.
        return np.zeros(gdq.shape[2:], dtype=bool)
    usable = (gdq & FLAG_BITS) == 0
    n_usable = usable.sum(axis=1, dtype=np.int16)      # (nints, ny, nx)
    return n_usable.max(axis=0) == 1


def is_perexposure_nircam(filename, header):
    return (bool(_PEREXP_RE.match(os.path.basename(str(filename))))
            and 'NIRCAM' in str(header.get('INSTRUME', '')).upper())


def find_ramp(filename):
    # The same lookup the satstar ZEROFRAME loader uses, so both read one ramp.
    from jwst_gc_pipeline.reduction.saturated_star_finding import _find_ramp_for
    return _find_ramp_for(filename)


def ramp_single_group_mask(ramp_path):
    """``single_group_mask`` of a ``_ramp.fits`` GROUPDQ, cached per path+mtime."""
    key = (os.path.realpath(ramp_path), os.path.getmtime(ramp_path))
    if key not in _MASK_CACHE:
        # memmap=False: some ramps carry BZERO on GROUPDQ, which astropy
        # refuses to memory-map.
        with fits.open(ramp_path, memmap=False) as rh:
            if 'GROUPDQ' not in rh:
                raise KeyError(f"{ramp_path} has no GROUPDQ extension")
            mask = single_group_mask(rh['GROUPDQ'].data)
        _MASK_CACHE.clear()          # one frame at a time; keep memory flat
        _MASK_CACHE[key] = mask
    return _MASK_CACHE[key]


def _warn_missing(filename, why):
    COUNTS['missing_ramp' if 'ramp' in why else 'shape_mismatch'] += 1
    MISSING.append((str(filename), why))
    msg = (f"G1-only DQ: {os.path.basename(str(filename))}: {why}; single-group "
           f"(G1/tgroup, ~0.75x for NFRAMES=2) pixels stay UNFLAGGED in this "
           f"frame [{COUNTS['missing_ramp']} missing ramp, "
           f"{COUNTS['shape_mismatch']} shape mismatch so far]")
    print(f"WARNING: {msg}", flush=True)
    warnings.warn(msg, G1OnlyRampMissingWarning, stacklevel=3)


def frame_mask(filename, header, dq_shape):
    """Mask to flag for this frame, or None when not applicable / unavailable.

    None is returned silently for a non-NIRCam or non-per-exposure product and
    when disabled; loudly (warning + ``COUNTS``) for a NIRCam per-exposure
    frame with no ramp or a mismatched GROUPDQ.
    """
    if not enabled():
        COUNTS['disabled'] += 1
        return None
    if not is_perexposure_nircam(filename, header):
        return None
    ramp = find_ramp(filename)
    if ramp is None:
        _warn_missing(filename, "no sibling _ramp.fits")
        return None
    try:
        mask = ramp_single_group_mask(ramp)
    except (OSError, KeyError, ValueError) as ex:
        _warn_missing(filename, f"ramp unreadable ({type(ex).__name__}: {ex})")
        return None
    if mask.shape != tuple(dq_shape):
        _warn_missing(filename, f"GROUPDQ shape {mask.shape} != frame DQ {tuple(dq_shape)}")
        return None
    return mask


def signature(filename):
    """Cache-key part: ``'g1'`` when this frame gets pixels flagged, else ``''``."""
    try:
        header = fits.getheader(filename)
    except OSError:
        return ''
    if not enabled() or not is_perexposure_nircam(filename, header):
        return ''
    ramp = find_ramp(filename)
    if ramp is None:
        return ''
    try:
        return 'g1' if ramp_single_group_mask(ramp).any() else ''
    except (OSError, KeyError, ValueError):
        return ''


def apply_to_hdul(hdul, filename, label=''):
    """Flag single-group pixels in an open HDUList, in memory.

    Replaces the DQ (and SCI/ERR/VAR_*) arrays of ``hdul`` with corrected
    copies and stamps ``G1ONLYDQ`` (number of pixels flagged) in the primary
    header.  Returns that number, or None when nothing was applied.
    """
    if 'DQ' not in hdul or hdul['DQ'].data is None:
        return None
    dq = hdul['DQ'].data
    mask = frame_mask(filename, hdul[0].header, dq.shape)
    if mask is None:
        return None
    n = int(mask.sum())
    if n == 0:
        COUNTS['no_single_group'] += 1
        hdul[0].header[HEADER_KEY] = (0, 'px flagged SAT|DNU: one usable ramp group')
        return 0
    new_dq = np.array(dq, copy=True)
    already = int(((new_dq[mask] & FLAG_BITS) == FLAG_BITS).sum())
    new_dq[mask] |= np.array(FLAG_BITS, dtype=new_dq.dtype)
    hdul['DQ'].data = new_dq
    for ext in NAN_EXTS:
        if ext in hdul and hdul[ext].data is not None \
                and np.issubdtype(hdul[ext].data.dtype, np.floating) \
                and hdul[ext].data.shape == mask.shape:
            arr = np.array(hdul[ext].data, copy=True)
            arr[mask] = np.nan
            hdul[ext].data = arr
    hdul[0].header[HEADER_KEY] = (n, 'px flagged SAT|DNU: one usable ramp group')
    COUNTS['applied'] += 1
    COUNTS['flagged_px'] += n
    print(f"G1-only DQ{(' (' + label + ')') if label else ''}: "
          f"{os.path.basename(str(filename))}: flagged {n} single-group px "
          f"SATURATED|DO_NOT_USE and NaN ({already} already flagged)", flush=True)
    return n


def apply_to_file(path, label=''):
    """Apply in place to a FITS file on disk.  Returns as ``apply_to_hdul``."""
    with fits.open(path, mode='update') as hdul:
        n = apply_to_hdul(hdul, path, label=label)
        if n is None:
            return None
    return n


def summary():
    return dict(COUNTS)


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(
        description="Flag single-group (G1-only) NIRCam pixels SATURATED|DO_NOT_USE "
                    "in per-exposure cal/crf files from their sibling _ramp.fits.")
    p.add_argument('files', nargs='+')
    p.add_argument('--dry-run', action='store_true',
                   help="report the count per file without writing")
    args = p.parse_args(argv)
    for fn in args.files:
        if args.dry_run:
            with fits.open(fn) as hdul:
                mask = frame_mask(fn, hdul[0].header, hdul['DQ'].shape)
            print(f"{fn}: {'n/a' if mask is None else int(mask.sum())}")
        else:
            n = apply_to_file(fn, label='cli')
            print(f"{fn}: {'n/a' if n is None else n}")
    print(f"summary: {summary()}")
    return 1 if (COUNTS['missing_ramp'] or COUNTS['shape_mismatch']) else 0


if __name__ == '__main__':
    sys.exit(main())
