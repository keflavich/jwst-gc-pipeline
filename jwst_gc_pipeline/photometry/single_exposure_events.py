"""Single-exposure transient events ("snowballs") in the per-frame NIRCam images.

A cosmic-ray shower that saturates a few pixels of ONE exposure leaves a compact
bright core in that frame and nothing in the frames that cover the same sky.  The
jump step flags the surrounding disc (JUMP_DET) and the core (SATURATED), but the
core pixels keep a finite, very high rate -- the slope of the few groups read
before the ramp saturated -- and carry no DO_NOT_USE.  ``ResampleStep`` drops
only DO_NOT_USE pixels, so the event enters the coadd at weight 1/N.  The data
i2d then shows a star of 1/N of the event's peak, and the merged-catalog
residual i2d keeps it, because no catalog model subtracts a star that is in one
frame only.

``outlier_detection`` would remove them, but it is skipped on the crowded
Galactic Center fields (``image3_steps_for``: it flags real PSF spikes), so this
module does the one test that tells an event from a bright star and nothing
else: a star is bright in every frame that covers it, an event in one.

For every compact SATURATED component of every frame (``find_candidates``) the
other covering frames are asked for the local excess over their own background
in the same 5x5 box (the second pass of ``find_events``).  The candidate is an event when at least
``min_other_frames`` frames cover it with finite, unsaturated data and every one
of them stays below ``ratio`` of the candidate's excess.  A frame whose box
holds a SATURATED pixel counts as bright (the star saturates there too), and a
frame with too few finite pixels is not counted at all.  Positions covered by
one frame only cannot be tested and are left alone.

``find_ramp_events`` (env ``SINGLE_EXPOSURE_EVENT_RAMP``) asks the exposure
itself instead of its neighbours, so it also tests positions that one frame
covers and events that saturate no pixel.  An exposure reads every pixel in two
integrations (the sibling Detector1 ``_ramp.fits``, ``SCI[integration, group,
y, x]``): a star gives the same counts in each of them, a cosmic-ray hit adds
counts to one.  Per frame the rate of each integration (the least-squares slope
of its groups, in DN per group, which a constant offset of the integration such
as the pedestal does not enter) is summed over ``box`` x ``box`` pixels; a pixel
disagrees between the integrations when the strongest exceeds the weakest (the
lower median, for more than two integrations) by ``nsigma`` times the frame's own
scatter of that difference and the weakest stays below ``1 - min_diff_frac`` of
the strongest.  The mosaic is built from the crf, not from the ramp, and the jump
step has already removed most hits from it, so a disagreeing pixel is an event
pixel only where the crf itself holds the extra counts: its value exceeds
``calibration * rate`` of the weakest integration (``calibration`` = the frame's
median crf per DN per group on bright pixels that agree between the integrations)
by ``vis_min`` and by ``vis_rel`` of that prediction.  The box sum leaks the
disagreement onto the neighbours of a hit, and a real star next to it passes the
box test but not this one.  A connected set of event pixels is an event when its
pixel of largest excess over the prediction reaches ``min_excess``; its footprint
is the set, dilated.  A frame without a usable ramp (no sibling file, one integration, a
cutout) or without pixels to calibrate on is not tested; the mosaic builders
print how many frames were.

The footprint of a saturated-core event (``footprint``) is its SATURATED core, dilated, plus
the connected pixels above ``max(wing_sigma * sigma, wing_frac * excess)``,
limited to ``max_radius`` pixels.  The result is a boolean mask per frame;
``apply_mask_to_dq`` sets DO_NOT_USE on it, which is what the mosaic builders
do before they resample.

The numerics take the frames as objects with ``name``, ``shape``, ``wcs`` and
``load()`` (the SCI and DQ arrays), so they run without the pipeline;
``CrfFrame`` is that object for a crf file.

The same mosaic builders also drop the extreme negative pixels
(``negative_pixel_mask``, env ``NEGATIVE_PIXEL_MASK_NSIGMA``): a pixel far below
the frame's own noise is a detector defect, not sky, and it enters the data
mosaic and the merged-catalog residual as a one-pixel crater.
"""
import os
import re
from dataclasses import dataclass, field

import numpy as np
from astropy.io import fits
from scipy import ndimage

DO_NOT_USE = 1
SATURATED = 2

#: environment switch of the mosaic builders; off unless set
ENV_FLAG = 'SINGLE_EXPOSURE_EVENT_MASK'

#: environment switch of the integration-consistency (ramp) test; off unless set
ENV_RAMP = 'SINGLE_EXPOSURE_EVENT_RAMP'

#: environment switch of the extreme-negative-pixel mask; off unless set
ENV_NEGATIVE = 'NEGATIVE_PIXEL_MASK_NSIGMA'

#: detector name of a crf file, and the sibling Detector1 ramp it is made from
#: (the pattern of ``cataloging._load_ramp_group0``)
_RAMP_NAME = re.compile(r'(_nrc[ab](?:long|[1-4]))_.*\.fits$')


def _switch(name):
    """The on/off environment variable ``name``: unset or blank is off;
    ``1/true/yes/on`` and ``0/false/no/off`` are accepted in any case.  Any other
    value raises ``ValueError`` naming the variable, so a typo is not read as
    either state."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return False
    word = raw.strip().lower()
    if word in ('1', 'true', 'yes', 'on'):
        return True
    if word in ('0', 'false', 'no', 'off'):
        return False
    raise ValueError(f'{name}={raw!r}: expected 1/true/yes/on or 0/false/no/off')


def event_mask_enabled():
    """``SINGLE_EXPOSURE_EVENT_MASK``: the cross-frame test of ``find_events``.
    Values as ``_switch``."""
    return _switch(ENV_FLAG)


def ramp_event_enabled():
    """``SINGLE_EXPOSURE_EVENT_RAMP``: the integration-consistency test of
    ``find_ramp_events``.  Values as ``_switch``."""
    return _switch(ENV_RAMP)


@dataclass(frozen=True)
class EventParams:
    # a SATURATED component larger than this is the core of a real star
    max_core_area: int = 150
    # MJy/sr over the local background that the candidate's brightest pixel
    # must reach; a smaller event is below the leftover scan's floor
    min_excess: float = 300.0
    # other frames must stay below this fraction of the candidate's excess
    ratio: float = 0.1
    # half-width of the box the frames are compared in (2 = 5x5)
    cmp_half: int = 2
    # fewest finite pixels a comparison box needs to count as a frame that
    # covers the position (of (2 * cmp_half + 1) ** 2)
    min_finite: int = 12
    # annulus for the local background and its scatter, in pixels
    bkg_r_in: float = 8.0
    bkg_r_out: float = 20.0
    # other covering frames with finite, unsaturated data needed to call an event
    min_other_frames: int = 1
    # footprint
    wing_sigma: float = 5.0
    wing_frac: float = 0.002
    core_dilate: int = 2
    max_radius: float = 12.0
    # candidates this close to the array edge are not tested
    edge_margin: float = 6.0


@dataclass(frozen=True)
class RampParams:
    # the integrations are compared in the sum over box x box pixels
    box: int = 3
    # the strongest integration must exceed the weakest by this many times the
    # frame's scatter (1.4826 MAD) of the difference of the first two integrations
    nsigma: float = 10.0
    # ... and the weakest must stay below (1 - min_diff_frac) of the strongest
    min_diff_frac: float = 0.5
    # such a pixel belongs to an event only where the crf exceeds what the weakest
    # integration alone gives (calibration * rate) by vis_min MJy/sr and by
    # vis_rel of that prediction (the pixel-to-pixel scatter of the calibration)
    vis_min: float = 50.0
    vis_rel: float = 0.1
    # the calibration is the median crf per DN per group over the pixels with at
    # least calib_rate DN per group in every integration, the integrations within
    # calib_spread of the brightest, and no SATURATED or DO_NOT_USE flag; a frame
    # with fewer than calib_min_pixels of them is not tested
    calib_rate: float = 400.0
    calib_spread: float = 0.1
    calib_min_pixels: int = 20
    # MJy/sr over that prediction that the brightest pixel of the event must
    # reach; a smaller one is below the leftover scan's floor
    min_excess: float = 200.0
    # the footprint is the event pixels dilated by this many pixels
    dilate: int = 1
    # a connected set of event pixels larger than this is a detector or frame
    # defect, not a hit, and is left alone
    max_area: int = 5000


@dataclass
class Event:
    frame: str
    x: float
    y: float
    area: int
    peak: float
    excess: float
    bkg: float
    sigma: float
    others: list = field(default_factory=list)   # (frame name, excess) of the frames compared
    ys: np.ndarray = None                        # footprint pixels
    xs: np.ndarray = None
    ramp: tuple = None                           # (strongest, weakest) integration rate at the peak, DN per group


class CrfFrame:
    """A crf file as a frame of ``find_events``.  The WCS and the shape are read
    on construction, the arrays on every ``load`` (they are not held)."""

    def __init__(self, path):
        from jwst_gc_pipeline.frame_wcs import frame_wcs
        self.path = os.fspath(path)
        self.name = os.path.basename(self.path)
        self.wcs = frame_wcs(self.path)
        hdr = fits.getheader(self.path, 'SCI')
        self.shape = (int(hdr['NAXIS2']), int(hdr['NAXIS1']))

    def load(self):
        with fits.open(self.path, memmap=False) as hdul:
            return (np.asarray(hdul['SCI'].data, dtype=float),
                    np.asarray(hdul['DQ'].data).astype(np.int64))

    def ramp_path(self):
        """The sibling Detector1 ramp of this crf (``jw..._nrcb3_ramp.fits``), or
        None when the file name carries no NIRCam detector."""
        path = _RAMP_NAME.sub(r'\1_ramp.fits', self.path)
        return None if path == self.path else path

    def load_ramp(self):
        """SCI cube of the sibling ramp, (integration, group, y, x) float32, or None
        when there is no sibling file, no SCI cube, or the cube is not on the
        detector grid of this frame (a ``--cutout-region`` crop)."""
        path = self.ramp_path()
        if path is None or not os.path.exists(path):
            return None
        with fits.open(path, memmap=False) as hdul:
            try:
                data = hdul['SCI'].data
            except KeyError:
                return None
            if data is None or data.ndim != 4 or tuple(data.shape[-2:]) != self.shape:
                return None
            return np.asarray(data, dtype=np.float32)


# ---------------------------------------------------------------------------
# Pixel-level helpers
# ---------------------------------------------------------------------------

def _annulus_stats(sci, x, y, r_in, r_out):
    """Median and MAD-sigma of the finite pixels of ``sci`` between ``r_in`` and
    ``r_out`` of (x, y); NaN, NaN when fewer than 20 pixels are finite."""
    ny, nx = sci.shape
    h = int(np.ceil(r_out))
    y0, y1 = max(int(round(y)) - h, 0), min(int(round(y)) + h + 1, ny)
    x0, x1 = max(int(round(x)) - h, 0), min(int(round(x)) + h + 1, nx)
    win = sci[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    r = np.hypot(xx - x, yy - y)
    vals = win[(r >= r_in) & (r <= r_out) & np.isfinite(win)]
    if vals.size < 20:
        return np.nan, np.nan
    med = float(np.median(vals))
    return med, float(1.4826 * np.median(np.abs(vals - med)))


def _box(sci, dq, x, y, half):
    """(peak, n_finite, any_saturated) over the box of half-width ``half`` at
    the pixel nearest (x, y); peak is NaN when no pixel is finite."""
    ny, nx = sci.shape
    xi, yi = int(round(x)), int(round(y))
    y0, y1 = max(yi - half, 0), min(yi + half + 1, ny)
    x0, x1 = max(xi - half, 0), min(xi + half + 1, nx)
    b = sci[y0:y1, x0:x1]
    fin = np.isfinite(b)
    peak = float(b[fin].max()) if fin.any() else np.nan
    return peak, int(fin.sum()), bool(((dq[y0:y1, x0:x1] & SATURATED) > 0).any())


# ---------------------------------------------------------------------------
# One frame: candidates and footprints
# ---------------------------------------------------------------------------

def find_candidates(sci, dq, params):
    """Compact SATURATED components of one frame whose brightest finite pixel
    is at least ``min_excess`` over the local background.

    Returns a list of dicts ``x, y, area, peak, excess, bkg, sigma, label`` and
    the label image (so the footprint can start from the component).
    """
    sat = (dq & SATURATED) > 0
    lab, n = ndimage.label(sat)
    out = []
    if n == 0:
        return out, lab
    ny, nx = sci.shape
    areas = np.bincount(lab.ravel(), minlength=n + 1)
    boxes = ndimage.find_objects(lab)
    h = params.cmp_half
    m = params.edge_margin
    for k in range(1, n + 1):
        if areas[k] > params.max_core_area:
            continue
        sl = boxes[k - 1]
        cy, cx = ndimage.center_of_mass(lab[sl] == k)
        cy += sl[0].start
        cx += sl[1].start
        if not (m <= cx <= nx - 1 - m and m <= cy <= ny - 1 - m):
            continue
        y0, y1 = max(sl[0].start - h, 0), min(sl[0].stop + h, ny)
        x0, x1 = max(sl[1].start - h, 0), min(sl[1].stop + h, nx)
        patch = sci[y0:y1, x0:x1]
        fin = np.isfinite(patch)
        if not fin.any():
            continue                      # an all-NaN core: a real star's
        peak = float(patch[fin].max())
        bkg, sig = _annulus_stats(sci, cx, cy, params.bkg_r_in, params.bkg_r_out)
        if not np.isfinite(bkg):
            continue
        if peak - bkg < params.min_excess:
            continue
        out.append(dict(x=float(cx), y=float(cy), area=int(areas[k]), peak=peak,
                        excess=peak - bkg, bkg=bkg, sigma=sig, label=k))
    return out, lab


def footprint(sci, lab, cand, params):
    """Pixel indices (ys, xs) the event covers in its own frame."""
    ny, nx = sci.shape
    r = int(np.ceil(params.max_radius)) + params.core_dilate + 2
    xc, yc = cand['x'], cand['y']
    y0, y1 = max(int(round(yc)) - r, 0), min(int(round(yc)) + r + 1, ny)
    x0, x1 = max(int(round(xc)) - r, 0), min(int(round(xc)) + r + 1, nx)
    win = sci[y0:y1, x0:x1] - cand['bkg']
    core = lab[y0:y1, x0:x1] == cand['label']
    thr = max(params.wing_sigma * (cand['sigma'] if np.isfinite(cand['sigma']) else 0.0),
              params.wing_frac * cand['excess'])
    with np.errstate(invalid='ignore'):
        above = np.isfinite(win) & (win > thr)
    grown, _ = ndimage.label(above | core, structure=np.ones((3, 3)))
    keep = np.isin(grown, np.unique(grown[core]))
    keep |= ndimage.binary_dilation(core, iterations=params.core_dilate)
    keep = ndimage.binary_dilation(keep, structure=np.ones((3, 3)))
    yy, xx = np.mgrid[y0:y1, x0:x1]
    keep &= np.hypot(xx - xc, yy - yc) <= params.max_radius
    return yy[keep], xx[keep]


# ---------------------------------------------------------------------------
# Many frames
# ---------------------------------------------------------------------------

def _world(frame, x, y):
    ra, dec = frame.wcs.pixel_to_world_values(np.atleast_1d(x), np.atleast_1d(y))
    return np.asarray(ra, dtype=float), np.asarray(dec, dtype=float)


def find_events(frames, params=EventParams(), log=None):
    """Single-exposure events of ``frames`` (all of one filter and one tile).

    Returns a list of ``Event``.  ``log`` is an optional ``print``-like callable.
    """
    log = log or (lambda *a, **k: None)
    names = [f.name for f in frames]
    if len(set(names)) != len(names):
        raise ValueError('find_events: frame names are not unique')
    # pass 1: candidates, with their footprints (small) and sky positions
    cands = []                     # (frame index, dict, ys, xs, ra, dec)
    for i, fr in enumerate(frames):
        sci, dq = fr.load()
        found, lab = find_candidates(sci, dq, params)
        for c in found:
            ys, xs = footprint(sci, lab, c, params)
            ra, dec = _world(fr, c['x'], c['y'])
            cands.append((i, c, ys, xs, float(ra[0]), float(dec[0])))
        log(f'  candidates {fr.name}: {len(found)}')
    if not cands:
        return []
    ra = np.array([c[4] for c in cands])
    dec = np.array([c[5] for c in cands])
    owner = np.array([c[0] for c in cands])
    # pass 2: every frame is loaded once for the candidates that fall on it
    others = [[] for _ in cands]   # per candidate: (frame index, excess) of clean frames
    blocked = np.zeros(len(cands), dtype=bool)     # a frame is bright there too
    h = params.cmp_half
    for g, fr in enumerate(frames):
        gx, gy = fr.wcs.world_to_pixel_values(ra, dec)
        gx = np.asarray(gx, dtype=float)
        gy = np.asarray(gy, dtype=float)
        ny, nx = fr.shape
        m = params.edge_margin
        inside = (np.isfinite(gx) & np.isfinite(gy) & (owner != g)
                  & (gx >= m) & (gx <= nx - 1 - m) & (gy >= m) & (gy <= ny - 1 - m))
        idx = np.flatnonzero(inside)
        if idx.size == 0:
            continue
        sci, dq = fr.load()
        for j in idx:
            peak, nfin, sat = _box(sci, dq, gx[j], gy[j], h)
            if sat:
                blocked[j] = True
                continue
            if nfin < params.min_finite:
                continue
            bkg, _ = _annulus_stats(sci, gx[j], gy[j], params.bkg_r_in, params.bkg_r_out)
            if not np.isfinite(bkg):
                continue
            others[j].append((g, peak - bkg))
    events = []
    for j, (i, c, ys, xs, _ra, _dec) in enumerate(cands):
        if blocked[j] or len(others[j]) < params.min_other_frames:
            continue
        if max(e for _g, e in others[j]) >= params.ratio * c['excess']:
            continue
        events.append(Event(frame=names[i], x=c['x'], y=c['y'], area=c['area'],
                            peak=c['peak'], excess=c['excess'], bkg=c['bkg'],
                            sigma=c['sigma'],
                            others=[(names[g], e) for g, e in others[j]],
                            ys=ys, xs=xs))
    return events


# ---------------------------------------------------------------------------
# One frame, one exposure's integrations: the ramp test
# ---------------------------------------------------------------------------

def ramp_slopes(ramp):
    """Rate of each integration of ``ramp`` (integration, group, y, x): the
    least-squares slope of its groups, DN per group, (integration, y, x) float32.
    A constant offset of an integration (the pedestal of the pixel, a dark
    subtraction that is off) does not enter; the group-mean level carries it.
    Needs two or more groups."""
    ramp = np.asarray(ramp, dtype=np.float32)
    ngroup = ramp.shape[1]
    if ngroup < 2:
        raise ValueError(f'ramp_slopes needs two or more groups, got {ngroup}')
    w = np.arange(ngroup, dtype=np.float32) - np.float32((ngroup - 1) / 2.0)
    return np.tensordot(ramp, w, axes=([1], [0])) / np.float32((w * w).sum())


def box_sums(values, box):
    """``values`` (integration, y, x) summed over ``box`` x ``box`` pixels."""
    return np.stack([ndimage.uniform_filter(values[i], size=box, mode='nearest')
                     * np.float32(box * box) for i in range(values.shape[0])])


def ramp_calibration(sci, dq, rate, params):
    """crf per DN per group of one frame: the median of ``sci`` over the mean over
    integrations of ``rate`` on the pixels that are bright (``calib_rate`` in
    every integration) and agree between the integrations (the brightest within
    ``calib_spread`` of the faintest), finite and neither SATURATED nor
    DO_NOT_USE in the crf.  NaN when fewer than ``calib_min_pixels`` pixels
    qualify."""
    lo = rate.min(axis=0)
    hi = rate.max(axis=0)
    with np.errstate(invalid='ignore'):
        ok = (np.isfinite(sci) & (lo >= params.calib_rate)
              & (hi - lo <= params.calib_spread * hi)
              & ((np.asarray(dq) & (SATURATED | DO_NOT_USE)) == 0))
    if int(ok.sum()) < params.calib_min_pixels:
        return np.nan
    return float(np.median(sci[ok] / rate.mean(axis=0)[ok]))


def ramp_event_pixels(sums, params):
    """Pixels where one integration holds counts the others do not.

    ``sums`` is ``box_sums(ramp_slopes(ramp), box)`` for two or more
    integrations.  Returns ``(mask, strongest, weakest, sigma)``: ``strongest``
    is the largest value over the integrations, ``weakest`` the lower median (the
    smallest for two integrations, the middle one for three), and ``sigma`` the
    1.4826 MAD of the difference of the first two integrations over the frame.
    ``mask`` is True where ``strongest - weakest >= nsigma * sigma``,
    ``strongest`` is positive and ``weakest <= (1 - min_diff_frac) * strongest``.
    None when ``sigma`` is not a positive number (a frame that cannot tell noise
    from signal).
    """
    nint = sums.shape[0]
    if nint < 2:
        raise ValueError(f'ramp_event_pixels needs two or more integrations, got {nint}')
    d01 = sums[0] - sums[1]
    with np.errstate(invalid='ignore'):
        sigma = float(1.4826 * np.nanmedian(np.abs(d01 - np.nanmedian(d01))))
    if not (np.isfinite(sigma) and sigma > 0.0):
        return None
    strongest = sums.max(axis=0)
    weakest = np.sort(sums, axis=0)[(nint - 1) // 2]
    with np.errstate(invalid='ignore'):
        mask = ((strongest - weakest >= params.nsigma * sigma) & (strongest > 0)
                & (weakest <= (1.0 - params.min_diff_frac) * strongest))
    return mask, strongest, weakest, sigma


def find_ramp_candidates(sci, dq, rate, params, event_params=EventParams()):
    """Events of one frame from the integrations of its exposure.

    ``sci`` and ``dq`` are the crf image and its DQ and ``rate`` the
    ``ramp_slopes`` of the sibling ramp, all on the same pixel grid.
    Returns ``(found, sigma, calibration)``: ``found`` is a list of dicts ``x, y,
    area, peak, excess, bkg, sigma, strongest, weakest, ys, xs`` (the footprint;
    ``peak`` the crf at the pixel of largest excess, ``bkg`` what the weakest
    integration alone gives there, ``strongest`` and ``weakest`` the rates of
    that pixel), ``sigma`` the scatter of ``ramp_event_pixels`` and
    ``calibration`` that of ``ramp_calibration`` (each NaN when the frame cannot
    be tested).
    """
    got = ramp_event_pixels(box_sums(rate, params.box), params)
    if got is None:
        return [], np.nan, np.nan
    mask, _strongest, _weakest, sigma = got
    calibration = ramp_calibration(sci, dq, rate, params)
    if not np.isfinite(calibration):
        return [], sigma, np.nan
    strongest = rate.max(axis=0)
    weakest = np.sort(rate, axis=0)[(rate.shape[0] - 1) // 2]
    predicted = calibration * weakest.astype(float)
    over = sci - predicted                          # what the crf holds beyond the weakest integration
    with np.errstate(invalid='ignore'):
        mask &= over >= np.maximum(params.vis_min, params.vis_rel * np.abs(predicted))
        mask &= (np.asarray(dq) & DO_NOT_USE) == 0
    ny, nx = sci.shape
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    found = []
    if n == 0:
        return found, sigma, calibration
    areas = np.bincount(lab.ravel(), minlength=n + 1)
    pad = int(params.dilate)
    for k, sl in enumerate(ndimage.find_objects(lab), 1):
        if areas[k] > params.max_area:
            continue
        sub = lab[sl] == k
        yy, xx = np.nonzero(sub)
        yy = yy + sl[0].start
        xx = xx + sl[1].start
        j = int(np.argmax(over[yy, xx]))
        xc, yc = float(xx[j]), float(yy[j])
        if not over[yy[j], xx[j]] >= params.min_excess:
            continue
        sig = _annulus_stats(sci, xc, yc, event_params.bkg_r_in, event_params.bkg_r_out)[1]
        bkg = float(predicted[yy[j], xx[j]])
        peak = float(sci[yy[j], xx[j]])
        y0, y1 = max(sl[0].start - pad, 0), min(sl[0].stop + pad, ny)
        x0, x1 = max(sl[1].start - pad, 0), min(sl[1].stop + pad, nx)
        patch = np.zeros((y1 - y0, x1 - x0), dtype=bool)
        patch[sl[0].start - y0:sl[0].stop - y0, sl[1].start - x0:sl[1].stop - x0] = sub
        if pad:
            patch = ndimage.binary_dilation(patch, structure=np.ones((3, 3)), iterations=pad)
        fy, fx = np.nonzero(patch)
        found.append(dict(x=xc, y=yc, area=int(areas[k]), peak=peak, excess=peak - bkg,
                          bkg=bkg, sigma=sig,
                          strongest=float(strongest[int(yc), int(xc)]),
                          weakest=float(weakest[int(yc), int(xc)]),
                          ys=fy + y0, xs=fx + x0))
    return found, sigma, calibration


def find_ramp_events(frames, params=RampParams(), event_params=EventParams(), log=None):
    """Single-integration events of ``frames`` (one filter, one tile).

    Each frame needs ``name``, ``shape``, ``load()`` (SCI, DQ) and ``load_ramp()``
    (the ``(integration, group, y, x)`` cube of its exposure, or None).  A frame
    without a cube of two or more integrations and groups on its own pixel grid is not
    tested, nor is one with no pixels to calibrate the crf against its ramp.
    Returns ``(events, n_tested)``: a list of ``Event`` with ``others == []``,
    ``bkg`` what the weakest integration alone gives at the peak and ``ramp`` =
    (strongest, weakest) rate there, DN per group, and the number of frames that were tested.
    ``log`` is an optional ``print``-like callable.
    """
    log = log or (lambda *a, **k: None)
    names = [f.name for f in frames]
    if len(set(names)) != len(names):
        raise ValueError('find_ramp_events: frame names are not unique')
    events = []
    n_tested = 0
    for fr in frames:
        ramp = fr.load_ramp()
        if (ramp is None or ramp.ndim != 4 or ramp.shape[0] < 2 or ramp.shape[1] < 2
                or tuple(ramp.shape[-2:]) != tuple(fr.shape)):
            log(f'  ramp {fr.name}: no usable ramp')
            continue
        sci, dq = fr.load()
        found, sigma, calibration = find_ramp_candidates(
            sci, dq, ramp_slopes(ramp), params, event_params)
        del ramp
        if not np.isfinite(sigma):
            log(f'  ramp {fr.name}: no usable scatter of the integration difference, not tested')
            continue
        if not np.isfinite(calibration):
            log(f'  ramp {fr.name}: fewer than {params.calib_min_pixels} pixels to calibrate '
                f'the crf against the ramp, not tested')
            continue
        n_tested += 1
        for c in found:
            events.append(Event(frame=fr.name, x=c['x'], y=c['y'], area=c['area'],
                                peak=c['peak'], excess=c['excess'], bkg=c['bkg'],
                                sigma=c['sigma'], others=[], ys=c['ys'], xs=c['xs'],
                                ramp=(c['strongest'], c['weakest'])))
        log(f'  ramp {fr.name}: {len(found)} events (difference scatter {sigma:.3g} DN, '
            f'calibration {calibration:.4g} MJy/sr per DN/group)')
    return events, n_tested


def event_masks(events, shapes):
    """``{frame name: bool array}`` for the frames that carry an event.
    ``shapes`` maps a frame name to its (ny, nx)."""
    masks = {}
    for ev in events:
        mk = masks.get(ev.frame)
        if mk is None:
            mk = masks[ev.frame] = np.zeros(shapes[ev.frame], dtype=bool)
        mk[ev.ys, ev.xs] = True
    return masks


def apply_mask_to_dq(dq, mask):
    """``dq`` with DO_NOT_USE set where ``mask`` is True (a new array)."""
    out = np.array(dq, copy=True)
    out[np.asarray(mask, dtype=bool)] |= DO_NOT_USE
    return out


def negative_pixel_nsigma():
    """``NEGATIVE_PIXEL_MASK_NSIGMA``: ``None`` (off) when unset, blank or 0;
    otherwise the number of ERR below zero at which a pixel is masked.  A value
    that is not a finite number >= 0 raises ``ValueError`` naming the variable,
    so a typo is not read as off."""
    raw = os.environ.get(ENV_NEGATIVE)
    if raw is None or not raw.strip():
        return None
    try:
        nsigma = float(raw)
    except ValueError:
        nsigma = np.nan
    if not (np.isfinite(nsigma) and nsigma >= 0.0):
        raise ValueError(f'{ENV_NEGATIVE}={raw!r}: expected a finite number >= 0 '
                         f'(0 is off)')
    return nsigma or None


def negative_pixel_mask(sci, err, dq, nsigma):
    """Bool mask of the pixels whose SCI is finite and below ``-nsigma * ERR``
    (``err`` finite and > 0), and that carry neither SATURATED nor DO_NOT_USE.

    A saturated core is left to the satstar handling, a DO_NOT_USE pixel is
    already dropped.  The remaining pixel that sits many ERR below zero is a
    detector defect (o105 F212N nrca4 pixel (1140, 1644): -1.9e4 MJy/sr, 166 ERR
    below zero, DQ = JUMP_DET only, in all six exposures); sky and stars never
    go below zero by more than a few ERR."""
    sci = np.asarray(sci, dtype=float)
    err = np.asarray(err, dtype=float)
    dq = np.asarray(dq)
    usable = np.isfinite(sci) & np.isfinite(err) & (err > 0)
    unflagged = (dq & (DO_NOT_USE | SATURATED)) == 0
    with np.errstate(invalid='ignore'):
        return usable & unflagged & (sci < -float(nsigma) * err)
