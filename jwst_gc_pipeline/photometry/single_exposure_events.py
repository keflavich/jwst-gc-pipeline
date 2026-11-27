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

The footprint of an event (``footprint``) is its SATURATED core, dilated, plus
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

Their positive twin is the isolated one-pixel spike (``find_positive_spikes``,
env ``POSITIVE_SPIKE_MASK_MIN_EXCESS``): one pixel hundreds of MJy/sr over its
surroundings whose eight neighbours stay at the background and whose DQ carries
neither SATURATED nor DO_NOT_USE, so it is neither a snowball core nor a pixel
the pipeline already drops (F480M o130 nrcalong exp00001 pixel (1961, 1452):
2228 MJy/sr against 28 MJy/sr at the same sky position in the next exposure).
It enters a six-exposure mosaic at 1/6 of its excess and the merged-catalog
residual keeps it as a star-sized blob, because no catalog model subtracts it.
The test needs no other frame: the PSF of a star puts at least 0.3 of its peak
into the brightest neighbour, a one-pixel spike puts none.  A pixel in the
noisy halo of a saturated star can also stand over calm neighbours; the excess
must therefore be at least ``SPIKE_MIN_SIGMA`` times the scatter of the annulus
the background comes from (halo pixels: 0.2-1.1 sigma, the defects of one
exposure: 10.8 sigma or more).

A spike whose 3x3 box holds a dropped pixel is not judged by default, so a defect
that sits beside a NaN or DO_NOT_USE pixel stays in the mosaic (F480M o130 nrcalong
exposure 3, pixel (1525, 141): 1572 MJy/sr against 5-15 MJy/sr in its neighbours, the
lower left neighbour NaN with DO_NOT_USE).  Env ``POSITIVE_SPIKE_MASK_MAX_UNUSABLE``
(``find_positive_spikes(max_unusable=...)``, default 0) lets up to that many of the
eight neighbours be missing: non-finite, DO_NOT_USE or outside the array.  The
isolation is then judged on the remaining usable neighbours, whose brightest must stay
below ``SPIKE_ISOLATION`` of the excess as before.  A SATURATED neighbour is never
tolerated: a pixel beside a saturated core is the rim of a star or a snowball.  The
tolerance is at most ``SPIKE_MAX_UNUSABLE_LIMIT``, one, so a pixel needs seven usable
neighbours: with the brightest neighbours of the star cores of eight tile-bands removed,
the first star that another frame confirms passes the isolation test when two are removed
(``docs/evidence/spike_tolerance/README.md``).
"""
import os
from dataclasses import dataclass, field

import numpy as np
from astropy.io import fits
from scipy import ndimage

DO_NOT_USE = 1
SATURATED = 2

#: environment switch of the mosaic builders; off unless set
ENV_FLAG = 'SINGLE_EXPOSURE_EVENT_MASK'

#: environment switch of the extreme-negative-pixel mask; off unless set
ENV_NEGATIVE = 'NEGATIVE_PIXEL_MASK_NSIGMA'

#: environment switch of the isolated positive spike mask (MJy/sr); off unless set
ENV_SPIKE = 'POSITIVE_SPIKE_MASK_MIN_EXCESS'

#: environment switch of the spike finder's tolerance for missing neighbours: the
#: number of the eight neighbours of a candidate that may be unusable; 0 (the finder
#: as shipped) unless set
ENV_SPIKE_UNUSABLE = 'POSITIVE_SPIKE_MASK_MAX_UNUSABLE'

#: the most neighbours a candidate may lack.  Over the 8 tile-bands of o105, o118, o130 and
#: o138 (F212N and F480M, every frame), no star core that another frame confirms (half of
#: its excess or more at the same sky position) passes the isolation test with its brightest
#: neighbour removed; one such core passes with its two brightest removed (o138 F480M,
#: brightest neighbours at 0.363 and 0.183 of the excess), and 62 do with three removed
#: (``docs/evidence/spike_tolerance/README.md``).  A Gaussian star of PSF sigma 0.95 pixel
#: on a pixel corner looks isolated only when seven neighbours are gone (second faintest
#: 0.33 of the peak pixel, faintest 0.109), a weaker bound than the data give.
SPIKE_MAX_UNUSABLE_LIMIT = 1

#: the brightest neighbour of a spike rises at most this fraction of the spike's own
#: excess over the local background; the brightest neighbour of a star is at 0.3 or
#: more of its peak (the 1.5 micron PSF core, the sharpest of the filters observed)
SPIKE_ISOLATION = 0.15

#: annulus in pixels of the local background of a spike candidate
SPIKE_BKG_R_IN = 3.0
SPIKE_BKG_R_OUT = 8.0

#: a spike also rises at least this many sigma (the MAD-sigma of its annulus) over
#: the local background: in the noisy halo of a saturated star the annulus scatter is
#: hundreds of MJy/sr, and a pixel one sigma up whose neighbours happen to be calm is
#: noise (the same structure sits at the same sky position in the other exposures).
#: Between 300 MJy/sr and the 0.15 isolation the 10678 tiles give halo pixels at 0.2-1.1
#: sigma and defects seen in one exposure only at 10.8 sigma or more
SPIKE_MIN_SIGMA = 5.0


def event_mask_enabled():
    """``SINGLE_EXPOSURE_EVENT_MASK``: unset or blank is off; ``1/true/yes/on`` and
    ``0/false/no/off`` are accepted in any case.  Any other value raises
    ``ValueError`` naming the variable, so a typo is not read as either state."""
    raw = os.environ.get(ENV_FLAG)
    if raw is None or not raw.strip():
        return False
    word = raw.strip().lower()
    if word in ('1', 'true', 'yes', 'on'):
        return True
    if word in ('0', 'false', 'no', 'off'):
        return False
    raise ValueError(f'{ENV_FLAG}={raw!r}: expected 1/true/yes/on or 0/false/no/off')


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


def _env_threshold(name):
    """The number in environment variable ``name``: ``None`` (off) when unset,
    blank or 0.  A value that is not a finite number >= 0 raises ``ValueError``
    naming the variable, so a typo is not read as off."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        value = float(raw)
    except ValueError:
        value = np.nan
    if not (np.isfinite(value) and value >= 0.0):
        raise ValueError(f'{name}={raw!r}: expected a finite number >= 0 '
                         f'(0 is off)')
    return value or None


def negative_pixel_nsigma():
    """``NEGATIVE_PIXEL_MASK_NSIGMA``: ``None`` (off) when unset, blank or 0;
    otherwise the number of ERR below zero at which a pixel is masked.  A value
    that is not a finite number >= 0 raises ``ValueError`` naming the variable,
    so a typo is not read as off."""
    return _env_threshold(ENV_NEGATIVE)


def positive_spike_min_excess():
    """``POSITIVE_SPIKE_MASK_MIN_EXCESS``: ``None`` (off) when unset, blank or 0;
    otherwise the excess in MJy/sr over the local background that an isolated
    one-pixel spike must reach to be masked.  A value that is not a finite
    number >= 0 raises ``ValueError`` naming the variable, so a typo is not read
    as off."""
    return _env_threshold(ENV_SPIKE)


def positive_spike_max_unusable():
    """``POSITIVE_SPIKE_MASK_MAX_UNUSABLE``: 0 (all eight neighbours of a spike must be
    usable, the finder as shipped) when unset or blank; otherwise the number of
    neighbours that may be unusable (0 to ``SPIKE_MAX_UNUSABLE_LIMIT``, one).  A value that
    is not an integer in that range raises ``ValueError`` naming the variable, so a
    typo is not read as 0."""
    raw = os.environ.get(ENV_SPIKE_UNUSABLE)
    if raw is None or not raw.strip():
        return 0
    try:
        value = int(raw.strip())
    except ValueError:
        value = -1
    if not 0 <= value <= SPIKE_MAX_UNUSABLE_LIMIT:
        raise ValueError(f'{ENV_SPIKE_UNUSABLE}={raw!r}: expected an integer from 0 to '
                         f'{SPIKE_MAX_UNUSABLE_LIMIT} (0 is off)')
    return value


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


def find_positive_spikes(sci, dq, min_excess, isolation=SPIKE_ISOLATION,
                         bkg_r_in=SPIKE_BKG_R_IN, bkg_r_out=SPIKE_BKG_R_OUT,
                         min_sigma=SPIKE_MIN_SIGMA, max_unusable=0):
    """The isolated one-pixel spikes of one frame, as a list of dicts ``x, y, peak,
    excess, bkg, sigma, neighbour`` (and ``unusable`` when ``max_unusable`` > 0).

    A spike is a pixel that is at least ``min_excess`` and at least ``min_sigma``
    sigma over the local background (the median and MAD-sigma of the finite pixels
    between ``bkg_r_in`` and ``bkg_r_out`` of it; ``min_sigma`` 0 drops the second
    condition) and whose brightest neighbour of the eight is at most ``isolation``
    of that excess over the same background (``neighbour`` is that neighbour's
    excess).  By default all nine pixels of the 3x3 box must be finite and carry
    neither SATURATED nor DO_NOT_USE: a pixel beside a saturated core, a dropped
    pixel or the edge of the array is not judged, so the rim of a snowball and a
    star cut by the edge are left to the event test and the satstar handling.
    A star is never isolated: its PSF puts 0.3 or more of the peak into the
    brightest neighbour.

    ``max_unusable`` (an integer from 0 to ``SPIKE_MAX_UNUSABLE_LIMIT``; 0 is the
    finder as described above) lets that many of the eight neighbours be missing:
    non-finite, DO_NOT_USE or outside the array.  The neighbour that limits the
    isolation is then the brightest of the usable ones, and ``unusable`` is the
    number of missing neighbours of the spike.  A SATURATED neighbour is never
    tolerated, and a pixel with more missing neighbours than ``max_unusable`` is
    not judged, as before."""
    sci = np.asarray(sci, dtype=float)
    dq = np.asarray(dq)
    max_unusable = int(max_unusable)
    if not 0 <= max_unusable <= SPIKE_MAX_UNUSABLE_LIMIT:
        raise ValueError(f'max_unusable={max_unusable}: expected an integer from 0 to '
                         f'{SPIKE_MAX_UNUSABLE_LIMIT}')
    usable = np.isfinite(sci) & ((dq & (DO_NOT_USE | SATURATED)) == 0)
    ring_weights = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int32)
    n_usable = ndimage.convolve(usable.astype(np.int32), ring_weights,
                                mode='constant', cval=0)
    n_saturated = ndimage.convolve(((dq & SATURATED) != 0).astype(np.int32),
                                   ring_weights, mode='constant', cval=0)
    # the array edge counts as missing: cval 0 of the convolution
    complete = usable & (n_saturated == 0) & (n_usable >= 8 - max_unusable)
    filled = np.where(usable, sci, -np.inf)
    ring = np.ones((3, 3), dtype=bool)
    ring[1, 1] = False
    neighbour_max = ndimage.maximum_filter(filled, footprint=ring, mode='constant',
                                           cval=-np.inf)
    # excess - neighbour excess = peak - brightest neighbour, whatever the
    # background, and it must be at least (1 - isolation) * min_excess
    with np.errstate(invalid='ignore'):
        step = filled - neighbour_max
        pre = complete & (step >= (1.0 - float(isolation)) * float(min_excess))
    out = []
    for y, x in zip(*np.nonzero(pre)):
        bkg, sig = _annulus_stats(sci, x, y, bkg_r_in, bkg_r_out)
        if not np.isfinite(bkg):
            continue
        excess = float(sci[y, x] - bkg)
        neighbour = float(neighbour_max[y, x] - bkg)
        if (excess < min_excess or neighbour > isolation * excess
                or excess < min_sigma * sig):
            continue
        spike = dict(x=float(x), y=float(y), peak=float(sci[y, x]), excess=excess,
                     bkg=bkg, sigma=sig, neighbour=neighbour)
        if max_unusable:
            spike['unusable'] = int(8 - n_usable[y, x])
        out.append(spike)
    return out


def positive_spike_mask(shape, spikes):
    """Bool mask of the 3x3 boxes around ``spikes`` (the dicts of
    ``find_positive_spikes``), an array of ``shape``.  A charge that one pixel
    holds leaks into its neighbours through the inter-pixel capacitance, so the
    box, not the pixel, is dropped."""
    mask = np.zeros(shape, dtype=bool)
    for s in spikes:
        y, x = int(s['y']), int(s['x'])
        mask[max(y - 1, 0):y + 2, max(x - 1, 0):x + 2] = True
    return mask
