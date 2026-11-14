"""Flag NIRCam pixels whose ramp-fit rate came from a single group, when biased.

The problem
-----------
When every group after the first is flagged SATURATED, ``ramp_fit`` (with
``suppress_one_group=False``, as ``PipelineRerunNIRCAM-LONG.py`` runs it)
returns a rate from group 1 alone.  G1 is the average of the NFRAMES frames of
group 1, so its effective exposure time is ``(NFRAMES+1) * TFRAME / 2``, not
``TGROUP``.  A ramp fit that divides by ``TGROUP`` gives 1/r of the true rate,
``r = TGROUP / ((NFRAMES+1) * TFRAME / 2)``: 1/1.333 = 0.75x for BRIGHT2.
The cal/crf PIXELDQ stays 0 there (SATURATED reaches the rate DQ only when ALL
groups saturate), so PSF and satstar fits use those pixels as good data.

Measured on brick 2221 o001 F182M jw02221001001_07101_00001_nrcb4, star at
crf pixel (427, 1794): the 16 pixels with group 2 flagged come out at exactly
G1/tgroup, the growth-ring pixels (flagged only by n_pix_grow_sat=1) at
0.55-0.66x their (G2-G1)/tgroup, and the star at 0.87 mJy against an expected
1.05-1.12 mJy.  The F182M/F187N ratio of stars with F200W < ~16.4 AB drops
from 1.00 to 0.78.

Not every frame is biased.  stcal 1.7.3 (jwst 1.14.1) divides group 1 by
group_time only in its NGROUPS=2 path (``fit_2_group``); its NGROUPS>2 path
(``fit_single_read``) and the stcal 1.18 C fitter (jwst 1.21) divide by
``(NFRAMES+1) * TFRAME / 2``.  Measured m (definition below): brick 2221 o001
and cloudc 2221 o002 (BRIGHT2, NGROUPS=2, jwst 1.14.1 / 1.12.1) 0.99-1.00,
biased; gc-treasury 10678 (BRIGHT2, NGROUPS=4, jwst 1.21) 1.32-1.36, brick
1182 o004 and sgrb2 5365 o001 (SHALLOW2, NGROUPS 7 / 3) 3.29 / 3.32 against
r = 3.33, all unbiased.  Flagging the unbiased gc-treasury F480M frames gave
about 100 fewer bright (F >= 1e4) stars per m7 frame catalog.

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

The gate (``G1ONLY_SAT_DQ``)
----------------------------
``auto`` (default, also when unset): decide per frame from the data.  At the
single-group pixels whose usable group is group 1,

    m = (SCI - B) / ((G1 / TGROUP) * C)

where C (frame units per DN/s) and B (additive offset, e.g. destreak/bgsub)
are fitted on clean pixels, ``SCI = C * (G2 - G1) / TGROUP + B``.  A biased
ramp fit gives m = 1; an unbiased one gives m = r.  The frame is flagged when
the median m is below ``(1 + r) / 2`` and left alone otherwise.  Fallbacks:

  * fewer than ``N_MIN_PX`` measurable pixels (or no ramp SCI, no readout
    keywords, too few clean reference pixels, a non-finite m): flag -- the
    pre-gate behaviour;
    the frame has few single-group pixels, so either choice moves few pixels;
  * ``r < R_MIN_SEPARABLE`` (NFRAMES=1 readouts: G1/TGROUP is the right rate
    under either ramp-fit timing): skip;
  * the frame header already carries a decision (``G1ONLYDC``): reuse it --
    a flagged crf has NaN where m would be measured;
  * ``G1ONLYDQ > 0`` with no decision key (flagged before the gate existed):
    flag.

``1`` / ``force`` flags every frame (the pre-gate behaviour); ``0`` / ``off``
disables it everywhere; any other value raises ``ValueError``.

The decision and m are stamped next to ``G1ONLYDQ`` (``G1ONLYDC`` flag/skip,
``G1ONLYWY`` reason, ``G1ONLYM`` median m, ``G1ONLYN`` pixels used,
``G1ONLYR`` r) and printed once per application.  Decisions are cached per
(frame, ramp) path+mtime within a process.

Wired in two places, all through ``apply_to_hdul`` / ``signature``:
  * ``write_perexposure_crf`` in ``PipelineRerunNIRCAM-LONG.py`` applies it to
    each per-exposure crf it writes (persisted, header ``G1ONLYDQ`` + gate keys);
  * cataloging's ``load_data`` and satstar's ``remove_saturated_stars`` apply
    it in memory, so crf written before this change are corrected at
    cataloging time without a reduction rerun; the satstar cache keys carry
    ``signature`` ('g1' only for frames that get flagged).
Applying it twice is a no-op (OR of the same bits, NaN of the same pixels).

A NIRCam per-exposure frame with no sibling ``_ramp.fits`` (or one whose
GROUPDQ does not match the frame shape) is reported with a WARNING and counted
in ``COUNTS``; it is never skipped silently.  Mosaics and non-NIRCam frames
are out of scope and skipped without a warning.
"""
import os
import re
import sys
import warnings
from collections import Counter, namedtuple

import numpy as np
from astropy.io import fits

SATURATED = 2
DO_NOT_USE = 1
FLAG_BITS = SATURATED | DO_NOT_USE
ENV = 'G1ONLY_SAT_DQ'
HEADER_KEY = 'G1ONLYDQ'
DECISION_KEY = 'G1ONLYDC'
REASON_KEY = 'G1ONLYWY'
M_KEY = 'G1ONLYM'
N_KEY = 'G1ONLYN'
R_KEY = 'G1ONLYR'
NAN_EXTS = ('SCI', 'ERR', 'VAR_POISSON', 'VAR_RNOISE', 'VAR_FLAT')

MODE_AUTO = 'auto'
MODE_FORCE = 'force'
MODE_OFF = 'off'
_AUTO_VALUES = ('auto',)
_FORCE_VALUES = ('1', 'true', 'yes', 'on', 'force')
# '' kept as "off": it disabled the flag before the gate existed.
_OFF_VALUES = ('0', 'false', 'no', 'off', '')

#: Minimum single-group pixels with a measurable m for an ``auto`` decision.
N_MIN_PX = 20
#: Minimum clean reference pixels (all groups GROUPDQ == 0, crf DQ clean,
#: G2 - G1 >= REF_MIN_DN) to fit C.
N_MIN_REF = 100
#: Reference pixels need this much signal per group difference (DN).
REF_MIN_DN = 200.0
#: Single-group pixels need G1 at least this high (DN) to enter m.
G1_MIN_DN = 200.0
#: Below this r the biased and unbiased rates differ by < 5%: no decision.
R_MIN_SEPARABLE = 1.05

# Per-exposure NIRCam product: jw<prop><obs><visit>_<grp><seq><act>_<exp>_nrc<det>_...
_PEREXP_RE = re.compile(r'^jw\d{11}_\d{5}_\d{5}_nrc[ab](?:long|[1-4])_')

#: Running tally for this process: 'applied', 'flagged_px', 'no_single_group',
#: 'skipped_unbiased', 'fallback_flag', 'missing_ramp', 'shape_mismatch',
#: 'disabled'.  Missing ramps are the number to watch: those frames keep the
#: single-group rate whether or not it is biased.
COUNTS = Counter()
MISSING = []

_MASK_CACHE = {}
_RAMP_CACHE = {}
_DECISION_CACHE = {}

#: flag: bool; source: 'measured' | 'fallback' | 'readout' | 'header' |
#: 'forced'; reason: str; m: median m or None; n: pixels used; r: expected
#: unbiased m or None; pct: (5, 25, 75, 95) percentiles of m or None.
Decision = namedtuple('Decision', 'flag source reason m n r pct')


class G1OnlyRampMissingWarning(UserWarning):
    """A NIRCam per-exposure frame has no usable sibling ``_ramp.fits``."""


class G1OnlyAmbiguousWarning(UserWarning):
    """``auto`` measured an m that matches neither 1 (biased) nor r (unbiased)."""


def mode(env=None):
    """``'auto'``, ``'force'`` or ``'off'`` from ``G1ONLY_SAT_DQ``.

    Unset or ``auto`` -> auto; ``1/true/yes/on/force`` -> force;
    ``0/false/no/off`` or empty -> off; anything else raises ``ValueError``.
    """
    env = os.environ if env is None else env
    raw = env.get(ENV)
    if raw is None:
        return MODE_AUTO
    val = str(raw).strip().lower()
    if val in _AUTO_VALUES:
        return MODE_AUTO
    if val in _FORCE_VALUES:
        return MODE_FORCE
    if val in _OFF_VALUES:
        return MODE_OFF
    raise ValueError(f"{ENV}={raw!r}: expected 'auto' (default), '1'/'force' "
                     f"(flag every frame) or '0'/'off' (disable)")


def enabled(env=None):
    return mode(env) != MODE_OFF


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


def _file_key(path):
    try:
        return (os.path.realpath(path), os.path.getmtime(path))
    except OSError:
        return None


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


def ramp_measurement_inputs(ramp_path):
    """Arrays ``auto`` needs from a ``_ramp.fits``, cached per path+mtime.

    Returns ``(g1, diff, ref, header)``: ``g1`` is group 1 (DN) averaged over
    the integrations where it is the only usable group (NaN elsewhere),
    ``diff`` is G2 - G1 (DN) averaged over integrations, ``ref`` marks pixels
    with GROUPDQ == 0 in every group of every integration, and ``header`` is
    the ramp primary header (readout keywords).  Only groups 1-2 of SCI are
    read.
    """
    key = (os.path.realpath(ramp_path), os.path.getmtime(ramp_path))
    if key not in _RAMP_CACHE:
        with fits.open(ramp_path, memmap=False) as rh:
            for ext in ('GROUPDQ', 'SCI'):
                if ext not in rh:
                    raise KeyError(f"{ramp_path} has no {ext} extension")
            gdq = np.asarray(rh['GROUPDQ'].data)
            if gdq.ndim == 3:
                gdq = gdq[None]
            if gdq.ndim != 4 or gdq.shape[1] < 2:
                raise ValueError(f"{ramp_path}: GROUPDQ shape {gdq.shape} has < 2 groups")
            sci = np.asarray(rh['SCI'].section[..., 0:2, :, :], dtype=np.float32)
            if sci.ndim == 3:
                sci = sci[None]
            header = rh[0].header.copy()
        usable = (gdq & FLAG_BITS) == 0
        n_usable = usable.sum(axis=1, dtype=np.int16)
        one_g1 = (n_usable == 1) & usable[:, 0]               # (nints, ny, nx)
        n_one = one_g1.sum(axis=0)
        with np.errstate(invalid='ignore', divide='ignore'):
            g1 = np.where(one_g1, sci[:, 0], 0).sum(axis=0, dtype=np.float64) / n_one
        g1 = np.where(n_one > 0, g1, np.nan).astype(np.float32)
        diff = (sci[:, 1] - sci[:, 0]).mean(axis=0, dtype=np.float64).astype(np.float32)
        ref = (gdq == 0).all(axis=(0, 1))
        _RAMP_CACHE.clear()          # one frame at a time; keep memory flat
        _RAMP_CACHE[key] = (g1, diff, ref, header)
    return _RAMP_CACHE[key]


def expected_ratio(*headers):
    """``(r, tgroup)`` from the first header carrying NFRAMES and TFRAME.

    ``r = TGROUP / ((NFRAMES + 1) * TFRAME / 2)``: the m an unbiased ramp fit
    gives (1.333 for BRIGHT2, 3.33 for SHALLOW2).  TGROUP falls back to
    ``(NFRAMES + GROUPGAP) * TFRAME``.  ``(None, None)`` when no header has
    the keywords.
    """
    for hdr in headers:
        if hdr is None or 'NFRAMES' not in hdr or 'TFRAME' not in hdr:
            continue
        nframes = float(hdr['NFRAMES'])
        tframe = float(hdr['TFRAME'])
        tgroup = hdr.get('TGROUP')
        tgroup = (float(tgroup) if tgroup is not None
                  else (nframes + float(hdr.get('GROUPGAP', 0))) * tframe)
        return tgroup / ((nframes + 1) * tframe / 2), tgroup
    return None, None


def measure_ratio(mask, sci, dq, g1, diff, ref, tgroup):
    """Median m at the single-group pixels.

    Fits ``SCI = C * diff / tgroup + B`` on clean reference pixels through two
    robust points: the medians of the faint half and of the bright pixels
    (``diff >= REF_MIN_DN`` and in the top 10% of clean ``diff``).  When the
    bright point is less than twice the faint one in rate (no lever arm), B
    is set to 0 and C is the median SCI/rate of the bright pixels.  Then
    ``m = (SCI - B) / ((g1 / tgroup) * C)`` at ``mask`` pixels with finite
    SCI and ``g1 >= G1_MIN_DN``.  Returns ``(m, n, pct, nref)``: ``n`` such
    pixels, ``nref`` bright reference pixels; ``m`` and ``pct`` are None when
    ``n < N_MIN_PX`` or ``nref < N_MIN_REF``.
    """
    sci = np.asarray(sci, dtype=np.float64)
    finite = np.isfinite(sci)
    sel = mask & finite & np.isfinite(g1) & (g1 >= G1_MIN_DN)
    n = int(sel.sum())
    clean = ref & finite & np.isfinite(diff)
    if dq is not None:
        clean &= (np.asarray(dq) & FLAG_BITS) == 0
    nclean = int(clean.sum())
    bright_cut = max(REF_MIN_DN, float(np.percentile(diff[clean], 90))) if nclean else REF_MIN_DN
    bright = clean & (diff >= bright_cut)
    nref = int(bright.sum())
    if n < N_MIN_PX or nref < N_MIN_REF:
        return None, n, None, nref
    rate = diff.astype(np.float64) / tgroup
    faint = clean & (diff <= np.median(diff[clean]))
    rate_f, sci_f = np.median(rate[faint]), np.median(sci[faint])
    rate_b, sci_b = np.median(rate[bright]), np.median(sci[bright])
    if rate_f > 0 and rate_b >= 2 * rate_f:
        conv = (sci_b - sci_f) / (rate_b - rate_f)
        offset = sci_f - conv * rate_f
    else:
        conv = np.median(sci[bright] / rate[bright])
        offset = 0.0
    with np.errstate(invalid='ignore', divide='ignore'):     # C = 0 -> non-finite m
        m = (sci[sel] - offset) / ((g1[sel].astype(np.float64) / tgroup) * conv)
    pct = tuple(float(v) for v in np.percentile(m, (5, 25, 75, 95)))
    return float(np.median(m)), n, pct, nref


def _header_decision(header):
    """A decision already recorded in ``header``, or None."""
    if header is None:
        return None
    prior = str(header.get(DECISION_KEY, '')).strip().lower()
    if prior in ('flag', 'skip'):
        return Decision(prior == 'flag', 'header',
                        str(header.get(REASON_KEY, f"{DECISION_KEY}={prior}")),
                        header.get(M_KEY), int(header.get(N_KEY, 0) or 0),
                        header.get(R_KEY), None)
    prior_n = header.get(HEADER_KEY)
    if isinstance(prior_n, (int, np.integer)) and prior_n > 0:
        # Flagged before the gate existed: SCI is NaN where m would be measured.
        return Decision(True, 'header', f"pre-gate {HEADER_KEY}={prior_n}",
                        None, 0, None, None)
    return None


def _auto_decision(mask, header, ramp, sci, dq):
    """Measure m and decide; every unmeasurable case falls back to flag."""
    hdr_decision = _header_decision(header)
    if hdr_decision is not None:
        return hdr_decision
    try:
        g1, diff, ref, ramp_header = ramp_measurement_inputs(ramp)
    except (OSError, KeyError, ValueError, TypeError) as ex:
        return Decision(True, 'fallback', f"ramp SCI unreadable ({type(ex).__name__})",
                        None, 0, None, None)
    r, tgroup = expected_ratio(ramp_header, header)
    if r is None:
        return Decision(True, 'fallback', "no NFRAMES/TFRAME in ramp or frame header",
                        None, 0, None, None)
    if r < R_MIN_SEPARABLE:
        return Decision(False, 'readout', f"r={r:.3f}<{R_MIN_SEPARABLE}: unbiased either way",
                        None, 0, r, None)
    if sci is None or g1.shape != mask.shape or np.shape(sci) != mask.shape:
        return Decision(True, 'fallback', "frame SCI unavailable", None, 0, r, None)
    m, n, pct, nref = measure_ratio(mask, sci, dq, g1, diff, ref, tgroup)
    if m is None:
        why = (f"{n} measurable px (<{N_MIN_PX})" if n < N_MIN_PX
               else f"{nref} reference px (<{N_MIN_REF})")
        return Decision(True, 'fallback', why, None, n, r, None)
    if not np.isfinite(m):
        # e.g. a constant SCI (C = 0): no measurement, so the pre-gate default.
        return Decision(True, 'fallback', f"m not finite ({m})", None, n, r, None)
    threshold = (1 + r) / 2
    if m < threshold:
        return Decision(True, 'measured', f"m<{threshold:.3f}: G1/TGROUP (biased)",
                        m, n, r, pct)
    return Decision(False, 'measured', f"m>={threshold:.3f}: unbiased", m, n, r, pct)


def _decision_key(filename, ramp):
    """Cache key: frame and ramp path+mtime, or None when either is not on disk."""
    fkey, rkey = _file_key(filename), _file_key(ramp)
    return None if fkey is None or rkey is None else fkey + rkey


def decide(filename, header, mask, ramp, sci, dq, env=None):
    """Gate decision for one frame (``Decision``), cached per frame+ramp."""
    md = mode(env)
    if md == MODE_FORCE:
        return Decision(True, 'forced', f"{ENV} forces flagging", None, 0, None, None)
    key = _decision_key(filename, ramp)
    if key is not None and key in _DECISION_CACHE:
        return _DECISION_CACHE[key]
    dec = _auto_decision(mask, header, ramp, sci, dq)
    if dec.m is not None and dec.r is not None and dec.r >= R_MIN_SEPARABLE:
        quarter = 0.25 * (dec.r - 1)
        if abs(dec.m - 1) > quarter and abs(dec.m - dec.r) > quarter:
            msg = (f"G1-only DQ: {os.path.basename(str(filename))}: m={dec.m:.3f} "
                   f"matches neither 1 (biased) nor r={dec.r:.3f} (unbiased); "
                   f"decided {'flag' if dec.flag else 'skip'} at the midpoint")
            print(f"WARNING: {msg}", flush=True)
            warnings.warn(msg, G1OnlyAmbiguousWarning, stacklevel=3)
    if key is not None:
        _DECISION_CACHE[key] = dec
    return dec


def _warn_missing(filename, why):
    COUNTS['missing_ramp' if 'ramp' in why else 'shape_mismatch'] += 1
    MISSING.append((str(filename), why))
    msg = (f"G1-only DQ: {os.path.basename(str(filename))}: {why}; single-group "
           f"pixels stay UNFLAGGED in this frame (rate G1/TGROUP, 0.75x for "
           f"BRIGHT2, if its ramp fit was the biased one) "
           f"[{COUNTS['missing_ramp']} missing ramp, "
           f"{COUNTS['shape_mismatch']} shape mismatch so far]")
    print(f"WARNING: {msg}", flush=True)
    warnings.warn(msg, G1OnlyRampMissingWarning, stacklevel=3)


def frame_mask(filename, header, dq_shape):
    """Single-group mask for this frame, or None when not applicable / unavailable.

    None is returned silently for a non-NIRCam or non-per-exposure product and
    when disabled; loudly (warning + ``COUNTS``) for a NIRCam per-exposure
    frame with no ramp or a mismatched GROUPDQ.  This is the mask before the
    ``auto`` gate; ``frame_plan`` adds the decision.
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


def frame_plan(filename, header, sci, dq):
    """``(mask, decision)`` for one frame.

    ``mask`` is None when not applicable / unavailable (see ``frame_mask``);
    ``decision`` is None when the mask is empty (nothing to decide).
    """
    mask = frame_mask(filename, header, np.shape(dq))
    if mask is None or not mask.any():
        return mask, None
    return mask, decide(filename, header, mask, find_ramp(filename), sci, dq)


def signature(filename):
    """Cache-key part: ``'g1'`` when this frame gets pixels flagged, else ``''``."""
    try:
        with fits.open(filename) as fh:
            header = fh[0].header
            if not enabled() or not is_perexposure_nircam(filename, header):
                return ''
            ramp = find_ramp(filename)
            if ramp is None:
                return ''
            try:
                mask = ramp_single_group_mask(ramp)
            except (OSError, KeyError, ValueError):
                return ''
            if not mask.any():
                return ''
            if mode() == MODE_AUTO:
                cached = _DECISION_CACHE.get(_decision_key(filename, ramp))
                if cached is not None:          # skip reading SCI/DQ again
                    return 'g1' if cached.flag else ''
            if 'SCI' not in fh or 'DQ' not in fh or fh['DQ'].data is None \
                    or fh['DQ'].data.shape != mask.shape:
                return ''
            dec = decide(filename, header, mask, ramp, fh['SCI'].data, fh['DQ'].data)
    except OSError:
        return ''
    return 'g1' if dec.flag else ''


def _stamp(header, n, dec):
    header[HEADER_KEY] = (n, 'px flagged SAT|DNU: one usable ramp group')
    if dec.source == 'header':
        return                       # keep the recorded decision as written
    header[DECISION_KEY] = ('flag' if dec.flag else 'skip', 'G1-only gate decision')
    header[REASON_KEY] = f"{dec.source}: {dec.reason}"[:68]
    if dec.m is not None:
        header[M_KEY] = (round(float(dec.m), 4), 'median SCI/(G1/TGROUP) at G1-only px')
    if dec.n:
        header[N_KEY] = (int(dec.n), 'G1-only px used for G1ONLYM')
    if dec.r is not None:
        header[R_KEY] = (round(float(dec.r), 4), 'G1ONLYM if unbiased: TGROUP/((NF+1)TF/2)')


def _describe(dec):
    parts = []
    if dec.m is not None:
        pct = '' if dec.pct is None else (f", 5-95% {dec.pct[0]:.3f}-{dec.pct[3]:.3f}")
        parts.append(f"m={dec.m:.3f} (n={dec.n}{pct})")
    if dec.r is not None:
        parts.append(f"r={dec.r:.3f}")
    parts.append(f"[{dec.source}: {dec.reason}]")
    return ' '.join(parts)


def apply_to_hdul(hdul, filename, label=''):
    """Flag single-group pixels in an open HDUList, in memory, when the gate says so.

    Replaces the DQ (and SCI/ERR/VAR_*) arrays of ``hdul`` with corrected
    copies and stamps ``G1ONLYDQ`` (number of pixels flagged) and the gate
    keys in the primary header.  Returns that number (0 when the frame has no
    single-group pixels or the gate skipped it), or None when nothing was
    applied (disabled, out of scope, no ramp).
    """
    if 'DQ' not in hdul or hdul['DQ'].data is None:
        return None
    dq = hdul['DQ'].data
    sci = hdul['SCI'].data if 'SCI' in hdul else None
    mask, dec = frame_plan(filename, hdul[0].header, sci, dq)
    if mask is None:
        return None
    tag = f"G1-only DQ{(' (' + label + ')') if label else ''}: {os.path.basename(str(filename))}"
    n = int(mask.sum())
    if n == 0:
        COUNTS['no_single_group'] += 1
        hdul[0].header[HEADER_KEY] = (0, 'px flagged SAT|DNU: one usable ramp group')
        return 0
    if not dec.flag:
        COUNTS['skipped_unbiased'] += 1
        _stamp(hdul[0].header, 0, dec)
        print(f"{tag}: skip {n} single-group px: {_describe(dec)}", flush=True)
        return 0
    if dec.source == 'fallback':
        COUNTS['fallback_flag'] += 1
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
    _stamp(hdul[0].header, n, dec)
    COUNTS['applied'] += 1
    COUNTS['flagged_px'] += n
    print(f"{tag}: flagged {n} single-group px SATURATED|DO_NOT_USE and NaN "
          f"({already} already flagged): {_describe(dec)}", flush=True)
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
                    "in per-exposure cal/crf files from their sibling _ramp.fits "
                    f"(gate: {ENV}=auto|1|0).")
    p.add_argument('files', nargs='+')
    p.add_argument('--dry-run', action='store_true',
                   help="report the count and gate decision per file without writing")
    args = p.parse_args(argv)
    for fn in args.files:
        if args.dry_run:
            with fits.open(fn) as hdul:
                mask, dec = frame_plan(fn, hdul[0].header,
                                       hdul['SCI'].data if 'SCI' in hdul else None,
                                       hdul['DQ'].data)
            if mask is None:
                print(f"{fn}: n/a")
            else:
                verdict = '' if dec is None else (
                    f" {'flag' if dec.flag else 'skip'}: {_describe(dec)}")
                print(f"{fn}: {int(mask.sum())} single-group px{verdict}")
        else:
            n = apply_to_file(fn, label='cli')
            print(f"{fn}: {'n/a' if n is None else n}")
    print(f"summary: {summary()}")
    return 1 if (COUNTS['missing_ramp'] or COUNTS['shape_mismatch']) else 0


if __name__ == '__main__':
    sys.exit(main())
