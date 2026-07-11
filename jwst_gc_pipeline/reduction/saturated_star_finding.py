# original file : https://github.com/keflavich/brick-jwst-2221/blob/main/brick2221/reduction/saturated_star_finding.py
import os
if not os.getenv('STPSF_PATH'):
    raise ValueError("STPSF_PATH must be specified")

import glob
from astropy.io import fits
from scipy.ndimage import label, find_objects, center_of_mass, sum_labels, binary_dilation
from astropy.modeling.fitting import LevMarLSQFitter
from jwst.datamodels import dqflags
import matplotlib.pyplot as plt

#os.environ['stpsf_PATH'] = '/orange/adamginsburg/jwst/stpsf-data/'
import stpsf
from stpsf.utils import to_griddedpsfmodel


try:
    # version >=1.7.0, doesn't work: the PSF is broken (https://github.com/astropy/photutils/issues/1580?)
    from photutils.psf import PSFPhotometry, IterativePSFPhotometry, SourceGrouper
except ImportError:
    # version 1.6.0, which works
    from photutils.psf import BasicPSFPhotometry as PSFPhotometry, IterativelySubtractedPSFPhotometry as IterativePSFPhotometry, DAOGroup as SourceGrouper
try:
    from photutils.background import MMMBackground, MADStdBackgroundRMS, MedianBackground, Background2D, LocalBackground
except ImportError:
    from photutils.background import MMMBackground, MADStdBackgroundRMS, MedianBackground, Background2D
    from photutils.background import MMMBackground as LocalBackground


from tqdm.notebook import tqdm
from tqdm import tqdm
from astropy import wcs
from astropy.wcs import WCS
import numpy as np
from scipy import ndimage
from astropy.table import Table, QTable
from astropy import table
from astropy import log
from astropy.coordinates import SkyCoord
from astropy import units as u
from .filtering import get_filtername, get_fwhm
import functools
import requests
import urllib3
import builtins

def get_psf(header, path_prefix='.', use_merged_psf_for_merged=False, fov_pixels=None):
    if header['INSTRUME'].lower() == 'nircam':
        psfgen = stpsf.NIRCam()
        fwhm, fwhm_pix = get_fwhm(header, instrument_replacement='NIRCam')
    elif header['INSTRUME'].lower() == 'miri':
        psfgen = stpsf.MIRI()
        fwhm, fwhm_pix = get_fwhm(header, instrument_replacement='MIRI')
    instrument = header['INSTRUME']
    filtername = get_filtername(header)
    try:
        module = header['MODULE']
    except KeyError:
        module = header['DETECTOR']
    detector = header['DETECTOR']

    ww = wcs.WCS(header)
    try:
        assert ww.wcs.cdelt[1] != 1, "This is not a valid WCS!!! CDELT is wrong!! how did this HAPPEN!?!?  (might happen if fitting a non-i2d file)"
    except AssertionError as ex:
        print(ex)
        print("ignoring WCS failure so check that stuff is right...")

    psfgen.filter = filtername
    obsdate = header['DATE-OBS']

    with open(os.path.expanduser('~/.mast_api_token'), 'r') as fh:
        api_token = fh.read().strip()

    npsf = 16
    oversample = 2
    # Default fov_pixels: 512 if not overridden by caller.  In-FOV satstar
    # subtraction passes fov_pixels=1024 for LW (NRC?5) detectors per the
    # PSF-size-vs-flux study (fovp512 underestimates bright LW star flux
    # by 50-70%).
    if fov_pixels is None:
        fov_pixels = 512
    if detector == 'NRCALONG':
        detector = 'nrca5'
    elif detector == 'NRCBLONG':
        detector = 'nrcb5'
    if detector.lower() == 'mirimage':
        detector = 'mirim'

    psfgen.detector = detector.upper()

    psf_fn = f'{path_prefix}/{instrument.lower()}_{detector.lower()}_{filtername.lower()}_fovp{fov_pixels}_samp{oversample}_npsf{npsf}.fits'

    if module == 'merged':
        project_id = header['PROGRAM'][1:5]
        obs_id = header['OBSERVTN'].strip()
        merged_psf_fn = f'{basepath}/psfs/{filtername.upper()}_{project_id}_{obs_id}_merged_PSFgrid.fits'
        if use_merged_psf_for_merged and os.path.exists(merged_psf_fn):
            psf_fn = merged_psf_fn
            log.info(f"Using merged PSF grid {psf_fn}")
        else:
            print("Using detector-specific WebbPSF grid for this frame", flush=True)

    if os.path.exists(str(psf_fn)):
        # As a file
        log.info(f"Loading grid from psf_fn={psf_fn}")
        big_grid = to_griddedpsfmodel(psf_fn)  # file created 2 cells above
        if isinstance(big_grid, list):
            print(f"PSF IS A LIST OF GRIDS!!!", flush=True)
            big_grid = big_grid[0]
    else:
        log.info(f'PSF file {psf_fn} does not exist; downloading from MAST')
        from astroquery.mast import Mast

        print(f"Attempting to load PSF for {obsdate}")
        try:
            Mast.login(api_token.strip())
            os.environ['MAST_API_TOKEN'] = api_token.strip()

            psfgen.load_wss_opd_by_date(f'{obsdate}T00:00:00')
        except (urllib3.exceptions.ReadTimeoutError,
                urllib3.exceptions.ProtocolError,
                requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectionError,
                requests.HTTPError,
                ConnectionError) as ex:
            # Transient MAST hiccup (incl. RemoteDisconnected wrapped in
            # ConnectionError); proceed with whatever PSF state we have
            # rather than crashing out of a multi-hour satstar grid build.
            print(f"Failed to load WSS OPD from MAST: {type(ex).__name__}: {ex}")

        log.info(f"starfinding: Calculating grid for psf_fn={psf_fn}")
        # https://github.com/spacetelescope/webbpsf/blob/cc16c909b55b2a26e80b074b9ab79ed9a312f14c/webbpsf/webbpsf_core.py#L640
        # https://github.com/spacetelescope/webbpsf/blob/cc16c909b55b2a26e80b074b9ab79ed9a312f14c/webbpsf/gridded_library.py#L424
        big_grid = psfgen.psf_grid(num_psfs=npsf, oversample=oversample,
                       all_detectors=False, fov_pixels=fov_pixels,
                                   outdir=path_prefix,
                                   save=True, outfile=None, overwrite=True)
        # now the PSF should be written
        assert glob.glob(psf_fn.replace(".fits", "*"))
        if isinstance(big_grid, list):
            print(f"PSF FROM PSF_GEN IS A LIST OF GRIDS!!!", flush=True)
            big_grid = big_grid[0]
            # if we really want to get this right, we need to create a new grid of PSF models
            # that is some sort of average of the PSF model grid.
            # There's no way to do it _right_ right without going back to the original data,
            # which is untenable with this approach.  It's a huge project.

    return big_grid


def debug_wrap(function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        print(function.__name__, flush=True)
        return function(*args, **kwargs)
    return wrapper


def _merge_spike_satellites(saturated, sources, nsource, gap, ratio):
    """Merge small saturated diffraction-spike satellites into a dominant core.

    A bright saturated star's six diffraction spikes saturate in DISCONNECTED
    segments ~1-1.5" out from the core; ``scipy.ndimage.label`` splits each
    into its own component, so each is fit as a separate satstar -> a HEX RING
    of spurious duplicate models around the real star (brick F2550W o002: a
    316-px core + five ~50-px satellites at ~13 px).  Group components by a
    ``gap``-px dilation, and within each group fold every SMALL satellite into
    the group's dominant core -- but ONLY when the core is >= ``ratio`` x the
    satellite, so two genuinely-separate saturated stars of comparable size
    (crowded GC fields) are NEVER fused.  Satellite pixels keep the core's
    label even though they are spatially disconnected (no connectivity
    relabel), so the core's center-of-mass/area absorb them.
    """
    if gap <= 0 or nsource <= 1:
        return sources, nsource
    comp_sizes = sum_labels(saturated, sources, np.arange(nsource) + 1)
    grp, ngrp = label(binary_dilation(saturated, iterations=int(gap)))
    new_sources = sources.copy()
    n_merged = 0
    for g in range(1, ngrp + 1):
        members = np.unique(sources[(grp == g) & saturated])
        members = members[members > 0]
        if len(members) <= 1:
            continue
        msizes = comp_sizes[members - 1]
        big = int(members[int(np.argmax(msizes))])
        bigsz = float(msizes.max())
        for c, sz in zip(members, msizes):
            if int(c) == big:
                continue
            if bigsz >= ratio * float(sz):
                new_sources[sources == int(c)] = big
                n_merged += 1
    if n_merged:
        # value-remap surviving labels to a contiguous 1..K (do NOT re-label by
        # connectivity -- that would re-split the disconnected satellites).
        uniq = np.unique(new_sources[new_sources > 0])
        remap = np.zeros(int(new_sources.max()) + 1, dtype=int)
        remap[uniq] = np.arange(1, len(uniq) + 1)
        new_sources = remap[new_sources]
        print(f"Saturated starfinding: merged {n_merged} spike-tip satellite(s) "
              f"into dominant cores (gap={gap}px, ratio={ratio}); "
              f"nsources {nsource} -> {len(uniq)}", flush=True)
        return new_sources, len(uniq)
    return sources, nsource


def _merge_overlapping_components(saturated, sources, nsource, overlap_frac,
                                  min_big_px=250):
    """Merge saturated components whose cores OVERLAP into one physical star.

    A bright saturated star embedded in nebulosity (W51 darkfil F480M) fragments
    into 2+ COMPARABLE-size DQ-SATURATED components split by a thin unsaturated
    gap; ``scipy.ndimage.label`` gives each its own label, each is fit as a
    separate satstar, and the overlapping PSF models over-subtract into a crater.
    ``_merge_spike_satellites`` only folds a SMALL satellite into a >=ratio-larger
    core, so it leaves a 1.78x-area pair (blob1: 659 + 370 px) split.

    Merge b into the LARGER component a when b's centroid lies within
    ``overlap_frac`` x a's footprint radius (r = sqrt(area/pi)) -- i.e. b's centre
    is inside a's saturated core, so they are one physical saturation region.
    Gated on a being genuinely large (>= ``min_big_px``): two DISTINCT compact
    stars have small radii, so their containment distance is small and they are
    NEVER fused (a real crowded cluster of separate cores is preserved); only
    large overlapping cores (whose saturation is physically blended and
    unresolvable anyway) merge.  Size-ratio-agnostic (complements the spike
    merge).  OFF by default (overlap_frac<=0); the driver enables it for
    extended-emission NIRCam fields.  Disconnected pixels keep the root label (no
    connectivity relabel), so the merged COM/area absorb them.
    """
    if overlap_frac <= 0 or nsource <= 1:
        return sources, nsource
    labels = np.arange(1, nsource + 1)
    sizes = sum_labels(saturated, sources, labels)
    coms = center_of_mass(saturated, labels=sources, index=labels)  # list of (y,x)
    radii = np.sqrt(np.maximum(sizes, 1.0) / np.pi)
    parent = {int(l): int(l) for l in labels}

    def _find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    order = [int(l) for l in labels[np.argsort(-sizes)]]   # largest first
    for ii in range(len(order)):
        a = order[ii]
        if sizes[a - 1] < min_big_px:
            continue
        ya, xa = coms[a - 1]
        if not (np.isfinite(ya) and np.isfinite(xa)):
            continue
        for jj in range(ii + 1, len(order)):
            b = order[jj]
            yb, xb = coms[b - 1]
            if not (np.isfinite(yb) and np.isfinite(xb)):
                continue
            if np.hypot(ya - yb, xa - xb) <= overlap_frac * radii[a - 1]:
                ra_root, rb_root = _find(a), _find(b)
                if ra_root != rb_root:
                    # attach smaller root under the larger
                    if sizes[ra_root - 1] >= sizes[rb_root - 1]:
                        parent[rb_root] = ra_root
                    else:
                        parent[ra_root] = rb_root
    new_sources = sources.copy()
    n_merged = 0
    for l in labels:
        root = _find(int(l))
        if root != int(l):
            new_sources[sources == int(l)] = root
            n_merged += 1
    if n_merged:
        uniq = np.unique(new_sources[new_sources > 0])
        remap = np.zeros(int(new_sources.max()) + 1, dtype=int)
        remap[uniq] = np.arange(1, len(uniq) + 1)
        new_sources = remap[new_sources]
        print(f"Saturated starfinding: merged {n_merged} overlapping-core "
              f"component(s) into their host (overlap_frac={overlap_frac}, "
              f"min_big_px={min_big_px}); nsources {nsource} -> {len(uniq)}",
              flush=True)
        return new_sources, len(uniq)
    return sources, nsource


# Per-filter DATA-value floor (MJy/sr) for the saturated-star finder.  A pixel
# flagged DQ-SATURATED whose data (and charge-migration wings) sit BELOW this
# floor is a SPURIOUS flag (persistence / JUMP mis-tag / bad pixel), NOT a real
# saturated star: the finder would otherwise invent a satstar there and
# extrapolate an enormous flux from a faint pixel (W51 F480M: ~80% of the
# satstars around bright sources had i2d data < 500; a fake at flux 104948 was
# fit onto a 127-count DQ-SATURATED pixel, with cfit=0 because satstars bypass
# the daophot fit -- so nothing downstream rejected it).  Genuine saturated cores
# read low/NaN in the core but have NaN-variance (unrecoverable) centers and/or
# bright saturated wings, so they clear the floor.  0 = off (default for unlisted
# filters -> behaviour unchanged).  Override per run with env SATSTAR_DATA_FLOOR.
_SATSTAR_DATA_FLOOR = {
    'f140m': 1000., 'f162m': 1000., 'f182m': 1000., 'f187n': 1000., 'f210m': 1000.,
    'f335m': 800.,  'f360m': 800.,  'f405n': 1000., 'f410m': 800.,  'f480m': 1000.,
}


def _resolve_satstar_data_floor(filtername, explicit=None):
    """Data floor for the satstar finder: explicit arg > env SATSTAR_DATA_FLOOR >
    per-filter default > 0 (off)."""
    if explicit is not None and float(explicit) > 0:
        return float(explicit)
    _env = os.environ.get('SATSTAR_DATA_FLOOR')
    if _env is not None:
        try:
            return float(_env)
        except ValueError:
            pass
    return float(_SATSTAR_DATA_FLOOR.get(str(filtername).lower(), 0.0))


# Per-filter SATURATION-SEVERITY floor (MJy/sr): the data level at which the
# filter genuinely saturates -- the same physical level the NORMAL fit uses to
# decide a SATURATED-DQ pixel is truly unusable (cataloging imports this table
# as its --saturation-data-floor auto default; single source of truth).  Used
# by the finder's severity gate: a DQ-SATURATED component whose brightest
# in-component pixel is BELOW this level, with NO unrecoverable (NaN-variance)
# core, cannot contain a saturated star -- it is the any-group SAT-bit
# over-flag (recoverable later-group pixels on an ordinary star).  Such
# components were seeded as satstars, wing-only fit 0.4-2.2 mag too FAINT
# (their valid core is masked out of the fit), and force-substituted over the
# correct daophot flux: Brick F182M carried 362 fake satstars at mag 16-18
# with ZERO saturated pixels in the frames (45% of the satstar catalog; 44%
# F187N/F212N) -- the sat<->unsat CMD discontinuity and most of the ~0.2-0.4"
# satstar 'centroid jitter' (fake wing fits wander; real satstars repeat to
# 0.077" cross-band).  f140m..f480m values = W51 per-frame crf SATURATED-pixel
# plateau p99 (same provenance as the normal-fit floor); the other filters are
# channel-sibling estimates (marked); unlisted (incl. all MIRI) -> 0 = gate
# off.  Override per run with env SATSTAR_SEVERITY_FLOOR (0 disables).
SAT_SEVERITY_FLOOR = {
    'f140m': 5000., 'f162m': 5000., 'f182m': 4000., 'f187n': 8000., 'f210m': 4000.,
    'f335m': 2500., 'f360m': 2500., 'f405n': 5000., 'f410m': 2500., 'f480m': 5000.,
    # channel-sibling estimates (not yet measured from per-field crf data):
    'f212n': 4000., 'f466n': 5000., 'f115w': 4000., 'f200w': 4000.,
    'f356w': 2500., 'f444w': 2500.,
}


def _resolve_satstar_severity_floor(filtername, explicit=None):
    """Severity floor: explicit arg > env SATSTAR_SEVERITY_FLOOR > per-filter
    SAT_SEVERITY_FLOOR > 0 (gate off)."""
    if explicit is not None and float(explicit) > 0:
        return float(explicit)
    _env = os.environ.get('SATSTAR_SEVERITY_FLOOR')
    if _env is not None:
        try:
            return float(_env)
        except ValueError:
            pass
    return float(SAT_SEVERITY_FLOOR.get(str(filtername).lower(), 0.0))


def find_saturated_stars(fitsdata, min_sep_from_edge=5, edge_npix=10000,
                         spike_merge_gap=0, spike_merge_ratio=3.0,
                         sat_data_floor=0.0, severity_floor=0.0):
    """
    Identify candidate saturated stars from the DQ plane.

    This helper builds a boolean mask of saturated pixels from
    ``dqflags.pixel['SATURATED']``, explicitly removes pixels flagged as cosmic
    rays (``dqflags.pixel['JUMP_DET']``), labels connected components, and
    suppresses large edge-adjacent saturated regions.

    Parameters
    ----------
    fitsdata : astropy.io.fits.HDUList
        Open FITS HDU list containing a ``DQ`` extension.
    min_sep_from_edge : int, optional
        Dilation iterations applied when masking edge-associated saturated
        structures.
    edge_npix : int, optional
        Minimum connected saturated area (pixels) used to classify a component
        as an edge source.
    spike_merge_gap : int, optional
        If >0, merge small saturated diffraction-spike satellite components into
        a dominant nearby core within this many px (see
        ``_merge_spike_satellites``).  0 = off (legacy connectivity labeling).
    spike_merge_ratio : float, optional
        A satellite is folded into the core only if core_area >= this x
        satellite_area, protecting comparable-size separate stars.

    Returns
    -------
    saturated : numpy.ndarray
        Boolean mask of saturated, non-cosmic-ray pixels after edge masking.
    sources : numpy.ndarray
        Integer connected-component label image returned by
        ``scipy.ndimage.label``.
    coms : list[tuple[float, float]]
        Centers of mass (y, x) for labeled saturated components.
    """

    dq = fitsdata['DQ'].data
    saturated = (dq & dqflags.pixel['SATURATED']) > 0
    # TRULY-LOST vs FRAME0-RECOVERED saturation, and why SEEDING and FIT-MASKING
    # must be DECOUPLED (2026-07-04).  A pixel flagged SATURATED in a late group
    # but with a good group 0 is RECOVERED by the ramp fit (valid rate, NO
    # DO_NOT_USE); only 0-good-group pixels carry DO_NOT_USE (== NaN VAR_POISSON).
    #   - FIT MASK: only the truly-lost (unrecoverable) pixels should be excluded
    #     from the PSF fit; the recovered wings are valid data and improve the
    #     centroid/flux.  This is done in get_saturated_stars (the fit masks
    #     ``saturated & _unrecoverable``).
    #   - SEEDING: a saturated STAR must still be SEEDED from its full saturated
    #     core even when the core was frame0-recovered.  Moderate saturated stars
    #     (recovered, NO truly-lost core) are otherwise never seeded and never
    #     cataloged -- daofind cannot fit their saturated cores -- which dropped
    #     real W51 cluster stars (A/B) when seeding was restricted to the truly-
    #     lost core.  So SEED on the full SATURATED mask.
    # The earlier truly-lost SEED restriction remains available (debugging /
    # special cases) via SATSTAR_SEED_REQUIRE_DO_NOT_USE=1, default OFF.
    _truly_lost_restricted = False
    if int(os.environ.get('SATSTAR_SEED_REQUIRE_DO_NOT_USE', 0)):
        _truly_lost = saturated & ((dq & dqflags.pixel['DO_NOT_USE']) > 0)
        if np.any(_truly_lost):
            saturated = _truly_lost
            _truly_lost_restricted = True
    cosmic_rays = (dq & dqflags.pixel['JUMP_DET']) > 0
    # JUMP_DET vs SATURATED disambiguation: the JWST ramp fitter sets
    # JUMP_DET on saturated ramps because non-linear ramp curvature looks
    # like a jump.  So saturated cores of bright stars come back with
    # *every* pixel also flagged JUMP_DET.  Naively masking out (sat &
    # ~cr) erases entire saturated stars from the satstar fitter — that
    # was the cause of unsubtracted stars in 6/12 sickle F480M frames
    # for source (17:46:16.090,-28:47:49.47) on 2026-05-15.
    # Real cosmic rays are typically 1-2 pixels; saturated star cores
    # are 5+ pixel clusters.  Keep saturated pixels that belong to a
    # cluster of size >= 3 even when JUMP_DET is set (assume the JUMP_DET
    # is the ramp nonlinearity from saturation, not a CR).  Isolated
    # 1-2 px saturated+JUMP_DET pixels are removed as before.
    if np.any(saturated):
        _sat_labels_for_cr, _n = label(saturated)
        if _n > 0:
            _sat_sizes_for_cr = sum_labels(
                saturated, _sat_labels_for_cr, np.arange(_n) + 1)
            _cluster_size_at_pix = np.where(
                _sat_labels_for_cr > 0,
                _sat_sizes_for_cr[_sat_labels_for_cr - 1],
                0,
            )
            remove_as_cr = cosmic_rays & (_cluster_size_at_pix < 3)
        else:
            remove_as_cr = cosmic_rays
    else:
        remove_as_cr = cosmic_rays
    saturated = saturated & (~remove_as_cr)

    sources, nsource = label(saturated)
    print('Saturated starfinding: nsources=', nsource, flush=True)
    # NOTE (2026-07-11): zero DQ-SATURATED components must NOT short-circuit
    # here -- the PEAK-BASED and SUB-FLOOR seeding blocks below can still add
    # unflagged charge-migration/suppression-strip components on a DQ-clean
    # frame (previously such frames were never seeded at all).  The label-
    # indexed gate blocks below are guarded with ``nsource`` instead, and the
    # empty short-circuit happens after seeding (the spike-merge / size /
    # edge logic below it cannot take an empty label set: ``sum_labels`` on
    # an empty index reduces with ``np.amin`` over nothing -> ValueError).

    # MIN LOST-CORE SIZE.  A genuine saturated star's truly-lost core is a COMPACT
    # cluster (W51 F480M deep-saturated star: 75 px core); scattered 1-4 px
    # truly-lost fragments on bright extended emission are isolated bad/DNU
    # pixels, not a star core.  Drop components below the floor so they never seed
    # a phantom satstar.  Applied ONLY when the truly-lost (DO_NOT_USE) restriction
    # is active -- in the SATURATED-only fallback we cannot tell a small real core
    # from a fragment, so the legacy behaviour is preserved.  Tune/disable with
    # SATSTAR_MIN_LOST_CORE (default 5; 0/1 disables).
    _min_core = int(os.environ.get('SATSTAR_MIN_LOST_CORE', 5))
    if _truly_lost_restricted and _min_core > 1 and nsource:
        _sz = sum_labels(saturated, sources, np.arange(nsource) + 1)
        _small_lab = (np.arange(nsource) + 1)[_sz < _min_core]
        if len(_small_lab):
            saturated = saturated & (~np.isin(sources, _small_lab))
            sources, nsource = label(saturated)
            print(f"Saturated starfinding: min-core ({_min_core}px) dropped "
                  f"{len(_small_lab)} scattered truly-lost fragment(s) -> "
                  f"nsources={nsource}", flush=True)
            if nsource == 0:
                return saturated, sources, []

    # DATA-VALUE FLOOR: drop DQ-SATURATED components sitting on FAINT data (a
    # spurious flag), keeping only genuine saturated stars (bright wings, or a
    # NaN-variance unrecoverable core).  See _SATSTAR_DATA_FLOOR docstring.  Runs
    # BEFORE spike-merge/edge logic so spurious components never participate.
    if sat_data_floor and sat_data_floor > 0 and nsource:
        sci = np.asarray(fitsdata['SCI'].data, dtype=float)
        unrec = (np.isnan(fitsdata['VAR_POISSON'].data)
                 if 'VAR_POISSON' in [h.name for h in fitsdata]
                 else np.zeros(sci.shape, dtype=bool))
        # 5px max-filter picks up the charge-migration wings just outside the
        # (low/NaN) saturated core of a real star.
        sci_wing = ndimage.maximum_filter(
            np.where(np.isfinite(sci), sci, -np.inf), size=5)
        _idx = np.arange(1, nsource + 1)
        comp_wingmax = np.asarray(
            ndimage.maximum(sci_wing, labels=sources, index=_idx), dtype=float)
        comp_unrec = np.asarray(
            ndimage.maximum(unrec.astype(np.uint8), labels=sources, index=_idx)) > 0
        drop_lab = _idx[(~comp_unrec) & np.isfinite(comp_wingmax)
                        & (comp_wingmax < float(sat_data_floor))]
        if len(drop_lab):
            saturated = saturated & (~np.isin(sources, drop_lab))
            sources, nsource = label(saturated)
            print(f"Saturated starfinding: data-floor ({sat_data_floor:g} MJy/sr) "
                  f"dropped {len(drop_lab)} spurious low-data DQ-SATURATED "
                  f"component(s) -> nsources={nsource}", flush=True)
            if nsource == 0:
                return saturated, sources, []

    # SATURATION-SEVERITY GATE: a component with NO unrecoverable (NaN-variance)
    # core whose brightest IN-COMPONENT pixel is below the filter's true
    # saturation level cannot be a saturated star -- it is the any-group
    # SATURATED over-flag on an ordinary star (recoverable later-group pixels).
    # Seeding it produces a wing-only satstar fit 0.4-2.2 mag too FAINT that
    # then OVERWRITES the star's correct daophot flux (the sat<->unsat CMD
    # jump; Brick F182M: 362 fakes = 45% of the satstar catalog, zero saturated
    # pixels in-frame).  Distinct from the wing data-floor above: that guard
    # kills faint spurious flags; this one kills BRIGHT-star over-flags whose
    # wings easily clear the wing floor.  See SAT_SEVERITY_FLOOR.
    if severity_floor and severity_floor > 0 and nsource:
        sci = np.asarray(fitsdata['SCI'].data, dtype=float)
        unrec = (np.isnan(fitsdata['VAR_POISSON'].data)
                 if 'VAR_POISSON' in [h.name for h in fitsdata]
                 else np.zeros(sci.shape, dtype=bool))
        _idx = np.arange(1, nsource + 1)
        comp_coremax = np.asarray(
            ndimage.maximum(np.where(np.isfinite(sci), sci, -np.inf),
                            labels=sources, index=_idx), dtype=float)
        comp_unrec = np.asarray(
            ndimage.maximum(unrec.astype(np.uint8), labels=sources,
                            index=_idx)) > 0
        drop_lab = _idx[(~comp_unrec) & np.isfinite(comp_coremax)
                        & (comp_coremax < float(severity_floor))]
        if len(drop_lab):
            saturated = saturated & (~np.isin(sources, drop_lab))
            sources, nsource = label(saturated)
            print(f"Saturated starfinding: severity gate ({severity_floor:g} "
                  f"MJy/sr) dropped {len(drop_lab)} unsaturated over-flagged "
                  f"component(s) (no NaN-variance core, peak below the true "
                  f"saturation level) -> nsources={nsource}", flush=True)
            if nsource == 0:
                return saturated, sources, []

    # PEAK-BASED SEEDING (2026-07-10): stars whose pixels EXCEED the filter's
    # true-saturation level but were never DQ-SATURATED-flagged (brick F182M:
    # 96% of the 13.5<mB<14.25 red-offset stars have ZERO SAT pixels yet peak
    # ~7300 MJy/sr > the 4000 floor).  Their cores are charge-migration
    # suppressed (flux bled to the rim, outside daophot's 5x5 stamp), biasing
    # daophot ~0.3 mag faint -- the residual CMD red bump between the floor
    # and the DQ-flagging level.  Seed them into the satstar channel, whose
    # masked-core + calibrated-wing machinery handles migration correctly.
    if severity_floor and severity_floor > 0:
        sci_ps = np.asarray(fitsdata['SCI'].data, dtype=float)
        bright_unflagged = (np.isfinite(sci_ps) & (sci_ps > float(severity_floor))
                            & (~saturated))
        if bright_unflagged.any():
            add, nadd = label(bright_unflagged)
            # keep components of >=2 px (single hot pixels are CRs/artifacts)
            szs = np.asarray(ndimage.sum_labels(bright_unflagged, add,
                                                np.arange(1, nadd + 1)))
            keep_lab = np.arange(1, nadd + 1)[szs >= 2]
            if len(keep_lab):
                # DILATE by 2 px: charge migration suppresses the star's
                # shoulder BEYOND the pixels that exceed the floor; without
                # dilation the wing fit anchors on the partially-suppressed
                # shoulder and stays ~0.1 mag faint (brick 13.5<mB<14.25
                # residual after seeding: +0.12 vs +0.34 unseeded).
                _addmask = binary_dilation(np.isin(add, keep_lab), iterations=2)
                saturated = saturated | _addmask
                sources, nsource = label(saturated)
                print(f"Saturated starfinding: peak-based seeding added "
                      f"{len(keep_lab)} unflagged charge-migration component(s) "
                      f"(data > {severity_floor:g}) -> nsources={nsource}",
                      flush=True)

    # SUB-FLOOR SEEDING (2026-07-11): core suppression is CONTINUOUS in well
    # fill, not a step at the severity floor.  Stars peaking at ~0.4-1.0x the
    # floor carry up to ~0.4 mag of unflagged core suppression (Brick F410M
    # 12.2<m<13.3: the F405N-F410M color plunges -0.10 -> -0.49 exactly below
    # the flagging floor; wing-annulus photometry shows the catalog fluxes are
    # >=0.17 mag too faint while F405N at the same mags is clean).  Seed
    # components whose peak lies in [frac*floor, floor) so their suppressed
    # cores are masked and the flux comes from the linear-regime wings.  The
    # mask is AMPLITUDE-DERIVED by construction (pixels above frac*floor plus
    # the 2-px shoulder dilation, r ~ 2-4 px) -- NOT the DQ-saturated area,
    # which is zero for these stars and previously produced ~0.6-px masks
    # whose "wing" fits just returned the same suppressed flux.  The post-fit
    # implied-peak gate arbitrates naturally: a genuinely suppressed star's
    # wing fit implies a peak above the gate threshold (kept), an unsuppressed
    # star implies its observed sub-floor peak (rejected -> daophot row kept).
    # Size-capped so extended emission crossing frac*floor cannot seed.
    _subfloor_frac = float(os.environ.get('SATSTAR_SUBFLOOR_SEED_FRAC', 0.35))
    if severity_floor and severity_floor > 0 and 0 < _subfloor_frac < 1:
        sci_ps = np.asarray(fitsdata['SCI'].data, dtype=float)
        sub_bright = (np.isfinite(sci_ps)
                      & (sci_ps > _subfloor_frac * float(severity_floor))
                      & (~saturated))
        if sub_bright.any():
            add, nadd = label(sub_bright)
            _lab_idx = np.arange(1, nadd + 1)
            szs = np.asarray(ndimage.sum_labels(sub_bright, add, _lab_idx))
            pk = np.asarray(ndimage.maximum(
                np.where(np.isfinite(sci_ps), sci_ps, -np.inf),
                labels=add, index=_lab_idx))
            # >=2 px (reject hot pixels/CRs), <=50 px (reject extended
            # emission), peak below the floor (above-floor components were
            # already seeded by the block above).
            keep_lab = _lab_idx[(szs >= 2) & (szs <= 50)
                                & (pk < float(severity_floor))]
            if len(keep_lab):
                _addmask = binary_dilation(np.isin(add, keep_lab), iterations=2)
                saturated = saturated | _addmask
                sources, nsource = label(saturated)
                print(f"Saturated starfinding: sub-floor seeding added "
                      f"{len(keep_lab)} suppression-strip component(s) "
                      f"(data > {_subfloor_frac:g} x {severity_floor:g}) "
                      f"-> nsources={nsource}", flush=True)

    if nsource == 0:
        # Nothing DQ-flagged and nothing seeded: empty result (the logic
        # below cannot take an empty label set).
        return saturated, sources, []

    # Fold disconnected diffraction-spike satellites into their dominant core
    # BEFORE edge/size logic so one star -> one component -> one satstar fit
    # (kills the per-frame hex of spike-tip duplicates).  Size-gated so two
    # comparable separate stars are never fused.
    sources, nsource = _merge_spike_satellites(
        saturated, sources, nsource, spike_merge_gap, spike_merge_ratio)
    # Merge COMPARABLE-size overlapping saturated cores (one bright star split by
    # a DQ gap into fragments the spike merge's ratio gate leaves separate ->
    # per-frame overlapping-PSF crater).  Ext-emission NIRCam only (driver sets
    # SATSTAR_COMPONENT_OVERLAP_FRAC); 0 = off -> other fields unchanged.
    _ovl_frac = float(os.environ.get('SATSTAR_COMPONENT_OVERLAP_FRAC', 0.0))
    _ovl_minpx = int(os.environ.get('SATSTAR_COMPONENT_OVERLAP_MIN_PX', 250))
    sources, nsource = _merge_overlapping_components(
        saturated, sources, nsource, _ovl_frac, min_big_px=_ovl_minpx)
    sizes = sum_labels(saturated, sources, np.arange(nsource)+1)
    msfe = min_sep_from_edge

    # which sources are edge sources?  Anything w/ more than edge_npix contiguous "saturated" pixels
    edge_ids = np.where(sizes > edge_npix)[0]
    # id 0 is the non-saturated zone that we've excluded [but reading this code 3/28/2026, I'm skeptical this makes sense]
    edge_ids = edge_ids[1:]
    edge_mask = np.isin(sources, edge_ids)
    _edge_remove = ndimage.binary_dilation(edge_mask, iterations=msfe)
    # PRESERVE genuine-saturation cores (2026-06-23).  The >edge_npix suppression
    # targets large detector-edge saturated bleeds, but bright EXTENDED EMISSION
    # (2526 cloud-c filament) produces large spurious DQ-SATURATED regions
    # (14717-px component) that ENGULF real saturated stars.  Erasing the whole
    # component deletes the star -- it is then never seeded or fit (the
    # "unidentified saturated star" failure).  The real star is the
    # ``unrecoverable`` (NaN-variance) core, which is genuine even inside a giant
    # spurious-DQ blob.  Keep those core pixels; only the finite spurious-DQ
    # emission is removed, so the surviving component collapses onto the real core
    # (correct COM + sat_area downstream).
    if 'VAR_POISSON' in [h.name for h in fitsdata]:
        _unrec = np.isnan(fitsdata['VAR_POISSON'].data)
        _edge_remove = _edge_remove & (~_unrec)
    saturated = saturated & (~_edge_remove)

    coms = center_of_mass(saturated, labels=sources, index=np.arange(nsource)+1)

    return saturated, sources, coms


def zeroframe_recover_saturated(data, dq, group0, *, R_g0_min=2000.0,
                                g0_sat_frac=0.9, sat_dilate=3, infl_tol=0.10,
                                R=None):
    """Recover the saturated-star RIM from the ramp first read (group-0).

    A bright star's DQ-SATURATED region reads wrong in the calibrated frame: the
    deep core is clipped LOW, but the rim is INFLATED above the true flux because
    charge migrates/blooms outward from the saturating core during the
    integration (a near-saturation well-overflow effect -- NOT the classical
    brighter-fatter effect, and not IPC).  Verified
    on sickle F210M (NIRCam): at the rim the saturated-frame cal sits ~15% ABOVE the ramp
    first read (recovered/cal ~ 0.85).  Subtracting a fixed PSF model from that
    inflated rim leaves a positive ring (the "+7587 dot" the user sees on the
    most-saturated stars).

    The ramp GROUP-0 (single first read, before charge migration) samples the
    TRUE stellar profile wherever group-0 is itself unsaturated.  Per frame:

        R = median(cal / group0)  over BRIGHT unsaturated pixels (group0>R_g0_min)

    is ~constant (calibration-free: cal[MJy/sr] ~ rate*photom*flat and
    group0[DN] ~ rate*tframe, so R ~ photom*flat/tframe; measured 5% scatter for
    group0>2000).  Then at saturated pixels whose group-0 is UNSATURATED (the
    rim), replace the inflated cal with R*group0 (de-inflated truth) so the
    PSF-subtracted residual collapses to ~0.  Where group-0 ALSO saturates (the
    deep core), it cannot be recovered -> returned in ``deep_core_mask``.

    Charge migration spreads a pixel or two BEYOND the hard DQ-SATURATED flag, so
    the search region is the SATURATED mask dilated by ``sat_dilate``; within the
    (non-DQ-flagged) dilation buffer a pixel is only rewritten if it is actually
    inflated (cal > R*group0*(1+infl_tol)), leaving clean pixels untouched.

    Parameters
    ----------
    data : 2-D float array
        Calibrated frame SCI (MJy/sr), full detector grid.
    dq : 2-D int array or None
        DQ plane (same grid).  SATURATED bit selects the region.
    group0 : 2-D float array or None
        Ramp first read (DN), same detector grid as ``data``.
    R_g0_min : float
        Minimum group-0 (DN) for a pixel to enter the R estimate (bright regime
        where R is stable).
    g0_sat_frac : float
        Fraction of the group-0 pile-up ceiling (99.9th pct) above which group-0
        is itself treated as saturated (deep core, unrecoverable).
    sat_dilate : int
        Dilation (px) of the DQ-SATURATED mask to catch the charge-migration rim.
    infl_tol : float
        In the dilation buffer (not DQ-flagged), only rewrite pixels inflated by
        more than this fraction above R*group0.
    R : float or None
        Precomputed ratio; if None it is measured from ``data``/``group0``.

    Returns
    -------
    recovered : 2-D float array
        Copy of ``data`` with rim pixels replaced by R*group0.
    rim_mask : 2-D bool array
        Pixels that were rewritten (recovered).
    deep_core_mask : 2-D bool array
        Saturated pixels whose group-0 is also saturated (unrecoverable).
    R : float
        The ratio used (NaN if it could not be estimated -> no-op).
    """
    shp = np.shape(data)
    if group0 is None or np.shape(group0) != shp:
        return data, np.zeros(shp, dtype=bool), np.zeros(shp, dtype=bool), np.nan
    sat = ((dq & dqflags.pixel['SATURATED']) != 0) if dq is not None \
        else np.zeros(shp, dtype=bool)
    if not sat.any():
        return data, np.zeros(shp, dtype=bool), np.zeros(shp, dtype=bool), np.nan
    sat_buf = binary_dilation(sat, iterations=int(sat_dilate)) if sat_dilate else sat
    g0_finite = np.isfinite(group0)
    # group-0 SATURATION CEILING: percentile-of-the-whole-frame breaks on a
    # mostly-dark frame (brick F182M: global p99.9 = 2.6k while the true
    # first-read pile-up plateau sits at ~48k -> everything bright was called
    # "deep core").  The pile-up plateau IS visible at the DQ-SATURATED
    # pixels whose group-0 went to the rail, so estimate the ceiling there;
    # fall back to the global percentile when no positive saturated pixels
    # exist.  Some products also ZERO group-0 at flagged pixels (ramp
    # ZEROFRAME ext) -- a non-positive group-0 is INVALID, never "clean".
    _g0sat = group0[sat & g0_finite & (group0 > 0)] if sat.any() else np.array([])
    if _g0sat.size >= 10:
        ceiling = g0_sat_frac * np.nanpercentile(_g0sat, 99.0)
    elif g0_finite.any():
        ceiling = g0_sat_frac * np.nanpercentile(group0[g0_finite], 99.9)
    else:
        ceiling = np.inf
    g0_clean = g0_finite & (group0 > 0) & (group0 < ceiling)
    _Rcurve = None
    if R is None or not np.isfinite(R):
        # cal/group0 DRIFTS ~25% from faint to bright pixels even after the
        # linearity step (0.229 at group0~2k -> 0.167 at ~20k on brick F182M
        # nrca1; residual single-read nonlinearity / charge migration).  A
        # scalar R therefore mis-scales the recovered rim by up to 25%
        # depending on each pixel's brightness -- a 0.1-0.3 mag flux-scale
        # systematic on zeroframe-anchored satstar fits.  Calibrate R AS A
        # FUNCTION OF group0 from this frame's bright unsaturated pixels
        # (binned medians in log g0, interpolated per rim pixel).
        good = ((~sat) & np.isfinite(data) & g0_finite & (data > 0)
                & (group0 > R_g0_min) & (group0 < ceiling))
        R = np.nan
        if int(good.sum()) >= 50:
            _g = group0[good]
            _r = data[good] / _g
            _edges = np.geomspace(R_g0_min, max(ceiling, R_g0_min * 1.01), 9)
            _ctr, _med = [], []
            for _k in range(len(_edges) - 1):
                _inb = (_g >= _edges[_k]) & (_g < _edges[_k + 1])
                if int(_inb.sum()) >= 20:
                    _ctr.append(np.sqrt(_edges[_k] * _edges[_k + 1]))
                    _med.append(float(np.nanmedian(_r[_inb])))
            if len(_ctr) >= 2:
                _Rcurve = (np.array(_ctr), np.array(_med))
                R = float(_med[-1])   # bright-end value, for logging/back-compat
            else:
                R = float(np.nanmedian(_r))
    recovered = np.array(data, dtype=float, copy=True)
    rim_mask = np.zeros(shp, dtype=bool)
    if np.isfinite(R):
        if _Rcurve is not None:
            recov_val = np.interp(np.log(np.clip(group0, 1, None)),
                                  np.log(_Rcurve[0]), _Rcurve[1]) * group0
        else:
            recov_val = R * group0
        # always rewrite genuinely-saturated rim pixels (group-0 clean); in the
        # (non-DQ-flagged) dilation buffer rewrite only BRIGHT (group0>R_g0_min,
        # so R*group0 is reliable) and INFLATED (charge-migration) pixels, leaving
        # faint neighbours untouched.
        rim_mask = (sat & g0_clean) | (
            sat_buf & ~sat & g0_clean & np.isfinite(data)
            & (group0 > R_g0_min)
            & (data > recov_val * (1.0 + infl_tol)))
        recovered[rim_mask] = recov_val[rim_mask]
    deep_core_mask = sat_buf & ~g0_clean
    return recovered, rim_mask, deep_core_mask, R


def _refine_coms_by_data(coms, data, sources, shift_warn_thresh_pix=3.0,
                         unrecoverable=None):
    """Refine DQ_SATURATED-mask centroids using the cluster bounding-box
    center as a regularizer.

    The mask centre-of-mass can be biased away from the true star centre
    when the saturated cluster shape is irregular -- JUMP_DET-contaminated
    extra pixels along the spike direction, asymmetric clipping at the
    detector edge, or partial merging with a neighbouring star.  In those
    cases the COM moves toward the longer "tail" of the irregular shape,
    away from the actual star centre.

    The cluster's bounding-box center is much more robust: real saturated
    stars produce roughly circular sat-clusters around the data peak, so
    the bounding-box center is close to the star centre.  Asymmetric
    distortion biases the bbox center much less than COM (a few extra
    saturated pixels on one side shift bbox by 0.5 px, but can shift COM
    by several px when the cluster is large).

    Centroid-from-data was tried first but pulls toward bright neighbour
    sources within ~10 px (e.g. on sickle F480M frame 0310g_00001 the
    donut centroid moved 2.8 px toward Star B's flux when computed with
    a windowed flux-weighted centroid).  Bounding-box center is immune.

    Parameters
    ----------
    coms : list of (cy, cx)
        Mask centroids from ``center_of_mass`` on the saturated mask.
    data : ndarray
        SCI image (unused now but kept in the signature for future
        data-driven refinements).
    sources : ndarray
        Labelled saturated-cluster image (output of ``scipy.ndimage.label``).
    shift_warn_thresh_pix : float
        Threshold above which a refined centroid is logged.

    Returns
    -------
    list of (cy, cx)
        Refined centroids.
    """
    refined = []
    for ii, (cy, cx) in enumerate(coms):
        if not (np.isfinite(cy) and np.isfinite(cx)):
            refined.append((cy, cx))
            continue
        cluster_id = ii + 1
        clmask = sources == cluster_id
        ys, xs = np.where(clmask)
        if len(ys) == 0:
            refined.append((cy, cx))
            continue
        # GENUINE-SATURATION CORE refinement (2026-06-23).  A real saturated star
        # buried in a LARGE spurious DQ-SATURATED emission blob (2526 cloud-c
        # filament: a 45-px unrecoverable NaN-variance core inside a 14717-px DQ
        # component of bright FINITE emission) has its mask-COM and eroded-COM
        # dragged onto the emission, several px off the star -- the fit then lands
        # on emission and is gated out, so the star is NEVER found.  The genuine
        # saturated core is the ``unrecoverable`` (NaN-variance) sub-region,
        # distinct from the finite spurious-DQ emission filling the rest of the
        # component.  When such a core exists, recentre on its LARGEST sub-cluster
        # (the real star).  This is the most reliable centre available and takes
        # priority over the eroded-mask heuristic below.
        if unrecoverable is not None:
            _core = clmask & unrecoverable
            if int(_core.sum()) >= 3:
                _cl, _cn = ndimage.label(_core)
                if _cn >= 1:
                    _sz = ndimage.sum_labels(_core, _cl, np.arange(1, _cn + 1))
                    _big = _cl == (int(np.argmax(_sz)) + 1)
                    _ncy, _ncx = ndimage.center_of_mass(_big)
                    if np.isfinite(_ncy) and np.isfinite(_ncx):
                        _sh = float(np.hypot(_ncy - cy, _ncx - cx))
                        if _sh > shift_warn_thresh_pix:
                            print(f"  [satstar centroid refine] cluster "
                                  f"{cluster_id}: mask_COM=({cx:.2f},{cy:.2f}) -> "
                                  f"genuine-core=({_ncx:.2f},{_ncy:.2f}) "
                                  f"shift={_sh:.2f} px (sat_area={len(ys)}, "
                                  f"core={int(_core.sum())})", flush=True)
                        refined.append((float(_ncy), float(_ncx)))
                        continue
        # ERODED-CORE CENTROID (2026-06-20).  The bbox centre is biased when the
        # saturated cluster is asymmetric -- the brightest 6-pointed DIFFRACTION
        # SPIKE saturates farther out than the others, or a thin saturated bleed/
        # bridge to a neighbour stretches the bounding box.  Both push the bbox
        # centre several px OFF the true core (sickle F770W star B: bbox 3.5 px
        # from the data peak -> the locked PSF model landed on a spike, core left
        # unsubtracted).  Erode the cluster a couple of px: thin spikes / bridges
        # (1-2 px wide) vanish, leaving the compact CORE; the centre-of-mass of
        # that eroded core sits on the star.  Robust to BOTH the asymmetric tails
        # the bbox was meant to handle AND the bbox's own off-centre bias.  Falls
        # back to the bbox centre when erosion empties the mask (small cores).
        _bbox_cy = (float(ys.min()) + float(ys.max())) / 2.0
        _bbox_cx = (float(xs.min()) + float(xs.max())) / 2.0
        eroded = ndimage.binary_erosion(clmask, iterations=2)
        if eroded.any():
            # largest eroded sub-blob (drops detached spike fragments)
            _el, _en = ndimage.label(eroded)
            if _en > 1:
                _sz = ndimage.sum_labels(eroded, _el, np.arange(1, _en + 1))
                eroded = _el == (int(np.argmax(_sz)) + 1)
            _ecy, _ecx = ndimage.center_of_mass(eroded)
            new_cy, new_cx = float(_ecy), float(_ecx)
        else:
            new_cy, new_cx = _bbox_cy, _bbox_cx
        shift = float(np.hypot(new_cy - cy, new_cx - cx))
        if shift > shift_warn_thresh_pix:
            print(f"  [satstar centroid refine] cluster {cluster_id}: "
                  f"mask_COM=({cx:.2f},{cy:.2f}) -> "
                  f"bbox_center=({new_cx:.2f},{new_cy:.2f}) "
                  f"shift={shift:.2f} px", flush=True)
        refined.append((new_cy, new_cx))
    return refined


def _nearest_window_bounds(center, full_size, window_size):
    """Return [start, stop) bounds of the nearest window to a given center."""
    window_size = int(min(max(1, window_size), full_size))
    if window_size >= full_size:
        return 0, full_size
    start = int(round(center - window_size / 2))
    start = max(0, min(start, full_size - window_size))
    stop = start + window_size
    return start, stop


def compute_adaptive_mask_buffer(sat_area, mask_buffer_min=2, cap=6, scale=0.4):
    """
    Return a brightness-dependent dilation radius for the saturation mask.

    Deeper saturation produces a larger saturated pixel area and a wider
    non-linear transition zone at the core boundary.  We scale the buffer
    sub-linearly with the square root of the saturated area (proxy for
    perimeter length) so that more of the non-linear fringe is excluded
    without over-masking mildly saturated sources.

    Parameters
    ----------
    sat_area : int
        Number of pixels flagged SATURATED for this source.
    mask_buffer_min : int
        Minimum buffer size (applied even for small saturated areas).
        The production default is 2 (validated against synthetic recovery
        tests; old default was 1).
    cap : int
        Maximum buffer to avoid masking all useful pixels for very bright
        stars.

    Returns
    -------
    int
        Effective dilation iterations to pass to ``binary_dilation``.

    Notes
    -----
    Calibrated against NIRCam SW synthetic recovery tests (see
    synthetic_source_recovery/run_recovery_tests.py).  Typical values:

    =========  ===========  ==============
    sat_area   sat_radius   buffer
    =========  ===========  ==============
    5          1.3          2  (= minimum)
    20         2.5          2
    80         5.0          2
    200        8.0          4
    500        12.6         5
    =========  ===========  ==============
    """
    sat_radius = np.sqrt(sat_area / np.pi)
    adaptive = int(np.ceil(sat_radius * scale))
    return int(min(cap, max(mask_buffer_min, adaptive)))


def compute_adaptive_bkg_annulus(sat_area, bkg_inner_min=15, bkg_inner_max=50):
    """
    Brightness-dependent background annulus radii for saturated-source fitting.

    Deeply saturated (bright) sources have extended PSF wings that contaminate
    the background annulus at close radii; this function scales the inner radius
    outward with the saturation extent so that the background estimate is not
    biased by unmasked PSF flux.  For marginally saturated sources the standard
    narrow annulus (15, 30) is returned.

    Parameters
    ----------
    sat_area : int
        Number of pixels flagged SATURATED for this source.
    bkg_inner_min : float
        Inner radius (pixels) used for marginally saturated sources.  Default 15.
    bkg_inner_max : float
        Upper cap on the inner radius (pixels).  Default 50.

    Returns
    -------
    bkg_inner, bkg_outer : int, int
        Inner and outer radii of the background annulus.  The outer radius is
        always 2 × inner.

    Notes
    -----
    Calibrated against NIRCam SW synthetic recovery tests: ``sat_area=37``
    (F200W mag ≈ 10.5, Δmag ≈ +4) maps to ``bkg_inner=25``, matching the
    optimal parameters found in the sweep.  The power-law exponent 0.75
    (≈ 3/4) approximates the expected scaling from a PSF wing profile that
    falls as r^{−2.8}: to maintain constant contamination,
    bkg_inner ∝ flux^{1/2.8} ∝ sat_radius^{2/2.8 × 2} = sat_radius^{1.4/2} ≈ sat_radius^{0.75}.

    Typical values for NIRCam SW at ~2 µm:

    =========  ===========  ============  ============
    sat_area   sat_radius   bkg_inner     bkg_outer
    =========  ===========  ============  ============
    1–5        0.6–1.3      15            30
    20         2.5          21            42
    37         3.4          25            50
    100        5.6          37            74
    =========  ===========  ============  ============
    """
    sat_radius = np.sqrt(sat_area / np.pi)
    bkg_inner = int(np.clip(np.round(10 * sat_radius ** 0.75), bkg_inner_min, bkg_inner_max))
    bkg_outer = 2 * bkg_inner
    return bkg_inner, bkg_outer


def reconcile_outside_fov_satstar_fluxes(per_frame, match_radius=1.0 * u.arcsec,
                                         disagree_factor=2.0, min_frames=2,
                                         high_outlier_factor=3.0,
                                         low_floor_factor=4.0,
                                         min_detector_diversity=None):
    # ``min_detector_diversity`` is DEPRECATED/ignored: reconcile no longer
    # drops stars for lack of detector diversity (that deleted real off-FOV
    # sources).  ``high_outlier_factor``/``low_floor_factor`` tune the robust-low
    # flux estimate (reject runaway corner mis-fits; floor against a single
    # spurious-low frame).
    """Reconcile the flux of OUT-OF-FIELD (forced) saturated stars across frames.

    A star outside the field of view, if bright enough, has its PSF detected in
    the target frames.  It is fit independently in every frame whose PSF
    footprint it touches.  Detectors *close* to the (off-field) star catch its
    high-S/N diffraction spikes and recover a correct flux; detectors that only
    catch a spike-less *corner* of the footprint fit scattered light and return
    a badly wrong flux that over-contributes to the model.  We trust the frame
    whose detector footprint centre is CLOSEST to the star (it sees the spikes)
    and override the discordant far-frame fluxes with it.

    This nearest-detector premise REQUIRES detector diversity: the frames must
    sample the off-field star from materially different detector positions, so
    that the closest one genuinely sees more spike than the others.  For a
    SINGLE-detector filter (e.g. NIRCam LW = one nrcblong detector, only
    dithered), every frame is the same detector at ~the same place: "nearest"
    is meaningless, and if the spikes never enter that one detector's FOV then
    EVERY frame fits spike-less scattered light and no trustworthy flux exists.
    Pinning any of those (mutually-discordant, all-inflated) values just spreads
    a garbage flux to every frame -> the off-field star over-contributes a fake
    background.  When a discordant group shows no detector diversity (all member
    detector centres fall within ``min_detector_diversity``), the source is
    UNFITTABLE here: we DROP it instead of pinning a bad flux.

    Parameters
    ----------
    per_frame : list of (key, table)
        One entry per frame.  ``key`` is any hashable frame identifier; ``table``
        must have columns ``skycoord_fit`` (SkyCoord), ``flux`` (float), and
        ``outside_fov_seed`` (bool), and ``table.meta['det_center']`` must be the
        SkyCoord of that frame's detector footprint centre.
    match_radius : Quantity
        Cross-frame association radius for "same star".
    disagree_factor : float
        A far-frame flux is overridden only if it differs from the nearest-frame
        flux by more than this multiplicative factor (max(a,b)/min(a,b) > factor),
        so in-family agreement is left untouched.
    min_frames : int
        A star must be fit in at least this many frames to reconcile (else there
        is nothing to compare against and the single fit is kept as-is).
    min_detector_diversity : Quantity
        Minimum spread (max pairwise separation) of member detector centres for
        the nearest-detector premise to hold.  A discordant group whose detector
        centres are all within this radius (single detector, only dithered) has
        no trustworthy reference frame -> its source is dropped, not overridden.
        Dithers are << a detector; distinct detectors are >> a dither, so 30"
        cleanly separates "same detector dithered" from "different detectors".

    Returns
    -------
    (overrides, drops) : tuple
        ``overrides`` is a list of (SkyCoord, flux): the reconciled
        (nearest-detector) flux for every out-of-field star that had a
        cross-frame disagreement AND enough detector diversity to trust one
        frame.  The caller forwards it as ``flux_overrides`` to
        ``get_saturated_stars`` so every frame pins that star to the trusted
        flux on the next phase.
        ``drops`` is a list of SkyCoord: out-of-field stars that disagreed
        across frames but had NO detector diversity (unfittable -- no frame
        caught the spikes).  The caller forwards it as ``flux_drops`` so every
        frame SKIPS the forced source instead of contributing a garbage model.
        Both empty if nothing needed reconciling.  Pure function: it does not
        mutate the input tables.
    """
    # Gather every out-of-field forced measurement as
    # (key, row_index, SkyCoord, flux, det_center_SkyCoord).
    recs = []
    for key, tbl in per_frame:
        if tbl is None or len(tbl) == 0 or 'outside_fov_seed' not in tbl.colnames:
            continue
        # det_center is stored as float meta keys (DET_RA/DET_DEC, deg) so it
        # survives the FITS round-trip; reconstruct the SkyCoord here.
        dc = tbl.meta.get('det_center', None)
        if dc is None:
            _ra = tbl.meta.get('DET_RA', None)
            _dec = tbl.meta.get('DET_DEC', None)
            if _ra is None or _dec is None:
                continue
            dc = SkyCoord(float(_ra) * u.deg, float(_dec) * u.deg)
        oof = np.asarray(tbl['outside_fov_seed'], dtype=bool)
        sc = SkyCoord(tbl['skycoord_fit'])
        _fluxcol = 'flux_fit' if 'flux_fit' in tbl.colnames else 'flux'
        flux = np.asarray(tbl[_fluxcol], dtype=float)
        for ri in np.where(oof)[0]:
            if np.isfinite(flux[ri]):
                recs.append((key, int(ri), sc[ri], float(flux[ri]), dc))
    if len(recs) < min_frames:
        return [], []

    all_sc = SkyCoord([r[2] for r in recs])
    # Greedy single-linkage grouping by sky position (n is small: #forced fits).
    n = len(recs)
    unassigned = list(range(n))
    groups = []
    while unassigned:
        seed = unassigned.pop(0)
        seps = all_sc[unassigned].separation(all_sc[seed]) if unassigned else None
        members = [seed]
        if unassigned:
            within = [unassigned[i] for i, s in enumerate(seps) if s < match_radius]
            for w in within:
                members.append(w)
                unassigned.remove(w)
        groups.append(members)

    overrides = []
    drops = []  # retained for API compatibility; reconcile NEVER drops a star
                # that is present in the data (that would delete real flux).
    for members in groups:
        if len(members) < min_frames:
            continue
        fluxes = np.array([recs[m][3] for m in members], dtype=float)
        fluxes = fluxes[np.isfinite(fluxes) & (fluxes > 0)]
        if len(fluxes) < min_frames:
            continue
        # Act only when the cross-frame fits disagree badly; a consistent family
        # is already fine and is left untouched (each frame keeps its own fit).
        if float(fluxes.max()) / float(fluxes.min()) <= disagree_factor:
            continue
        # Robust-LOW estimate of the off-FOV star's true flux.  A saturated star
        # fit in the spike-less CORNER of a frame can only OVER-estimate it
        # (scattered light + unmodelled neighbours get attributed to the point
        # source), so the credible flux sits at the LOW end and the runaway
        # 1e9-1e11 values are corner mis-fits to reject.  We:
        #   1. reject high runaway outliers (flux > median * high_outlier_factor),
        #   2. take the MINIMUM of the survivors (least-contaminated),
        #   3. floored at survivor_median / low_floor_factor so a single
        #      anomalously-low frame cannot drive a gross UNDER-subtraction.
        # This biases LOW (avoids the over-subtracted negative holes) while still
        # modelling the star, and it NEVER drops a present source -- earlier
        # "drop when no detector diversity" wrongly deleted real, fittable
        # off-FOV stars (e.g. sickle 17:46:12.67 in F480M/F470N).
        med = float(np.median(fluxes))
        kept = fluxes[fluxes <= med * high_outlier_factor]
        if len(kept) == 0:
            kept = fluxes
        truth = max(float(np.min(kept)), float(np.median(kept)) / low_floor_factor)
        if not (truth > 0):
            continue
        # Key the override at the GROUP CENTROID (not an extreme member): the
        # per-frame forced positions scatter across the whole group (~1.2" for
        # the sickle off-FOV stars) and the caller matches each frame's seed
        # against ONE key within match_radius.  The centroid halves the max
        # key-to-member distance so every frame -- including the runaway ones
        # that most need pinning -- matches and gets the robust-low flux.
        member_sc = SkyCoord([recs[m][2] for m in members])
        centroid = SkyCoord(member_sc.cartesian.mean(), frame=member_sc.frame).icrs
        overrides.append((centroid, truth))
    return overrides, drops


# Fixed small footprint (~11px radius) for the secondary, neighbour-robust
# prominence/core check in the post-fit gate.  Independent of a source's
# (possibly spuriously huge) DQ-SATURATED sat_area so a fake on extended
# emission cannot inflate its prominence by reaching a neighbouring real star.
_SEED_SMALL_SAT_AREA = 400


def _seed_prominence(data, com, sat_area, robust=False):
    """Adaptive-radius prominence of a saturated-star SEED on the SCI data.

    ``robust``: the default denominator is the annulus MAD, which a bright
    NEIGHBOUR or structured emission in the annulus inflates -- on emission-rich
    fields (cloudc 2526 F770W) this crushes a real faint saturated star's
    prominence below the gate (28 by-eye-real stars dropped, AG 2026-06-28).
    ``robust=True`` instead measures the background as the annulus 25th-percentile
    (the emission FLOOR between neighbours) and the spread from the lower half of
    the annulus (1.4826*MAD of pixels <= the annulus median), so a single bright
    neighbour cannot inflate the denominator.  This lifts genuine faint saturated
    stars (cloudc: prominence median 7.1->16.6, min 2.3->6.0) while diffuse
    emission stays low.  Pair with a field-lowered ``seed_core_min``.

    Distinguishes a genuine saturated STAR (compact bright core + bright PSF
    wings) from a DQ-SATURATED patch of bright EXTENDED EMISSION (no compact
    core; wings ~ local background).  MIRI F770W (and other MIRI broadbands)
    saturate the detector on the Sickle's bright nebulosity, so the DQ plane
    grows many non-stellar saturated components; fitting each as a PSF point
    source produces phantom bright stars that over-subtract the data (deep
    negative residual pits) and show up in the display model.

    The core/annulus radii are scaled to the saturated footprint so the metric
    survives deeply-saturated stars whose ENTIRE core is zeroed (NaN-variance
    pixels are set to 0 upstream): a fixed r=4-10 annulus would sit inside such
    a star's zeroed core and report ~0 prominence, dropping the brightest real
    stars.  Sampling the core just OUTSIDE the saturated footprint catches the
    bright PSF wing ring instead.

    Returns ``(prominence, core)`` where ``core`` is the brightest data value in
    the wing ring (an ABSOLUTE brightness, not background-relative).  Both are
    NaN when the metric cannot be measured (too close to an edge, too few finite
    annulus pixels) -- callers treat NaN as "keep" so a real star is never
    dropped on an inability to measure it.

    Two criteria are needed because they fail on different phantoms:
      * prominence  -- (core-med)/MAD; drops DIFFUSE extended-emission saturation
        (the bulk, ~50/frame).  But a small COMPACT bump on SMOOTH emission has a
        tiny annulus MAD, so a modest absolute excess still scores high (12-55).
      * core (absolute) -- drops those compact bumps: a real saturated star's
        wing ring is intrinsically bright (>=~1191 on F770W; phantom bumps ~800).

    Calibrated on F770W Sickle o002: real hand-selected saturated stars score
    prominence >=15 (median ~130) AND core >=1191 (median ~4700); phantoms score
    prominence ~1-3 (diffuse) or have core ~800 (compact bumps).
    """
    y, x = com
    # com can map to NaN near a tile edge (sickle F1500W o002 exp00003);
    # unmeasurable -> NaN so the gate KEEPS it instead of crashing on int(NaN).
    if not (np.isfinite(x) and np.isfinite(y)):
        return np.nan, np.nan
    xi, yi = int(round(float(x))), int(round(float(y)))
    ny, nx = data.shape
    r_sat = max(2.0, np.sqrt(max(int(sat_area or 0), 1) / np.pi))
    r_core = r_sat + 3.0       # bright PSF-wing ring just outside the sat core
    r_in = r_sat + 5.0         # background annulus, beyond the wing ring
    r_out = r_sat + 15.0
    pad = int(np.ceil(r_out)) + 1
    if not (pad < xi < nx - pad and pad < yi < ny - pad):
        return np.nan, np.nan
    yy, xx = np.mgrid[yi - pad:yi + pad + 1, xi - pad:xi + pad + 1]
    r = np.hypot(xx - xi, yy - yi)
    sub = data[yi - pad:yi + pad + 1, xi - pad:xi + pad + 1]
    # exclude exactly-zero pixels: those are the zeroed (unrecoverable) sat core
    cm = (r <= r_core) & np.isfinite(sub) & (sub != 0)
    am = (r >= r_in) & (r <= r_out) & np.isfinite(sub) & (sub != 0)
    if cm.sum() < 3 or am.sum() < 10:
        return np.nan, np.nan
    core = float(np.nanmax(sub[cm]))
    a = sub[am]
    if robust:
        # emission FLOOR (between neighbours) + lower-half robust spread, so a
        # bright neighbour / structured emission in the annulus cannot inflate
        # the denominator and crush a real faint saturated star's prominence.
        bg = np.percentile(a, 25)
        lower = a[a <= np.median(a)]
        spread = (1.4826 * np.median(np.abs(lower - np.median(lower)))
                  if lower.size > 5 else np.std(a))
        if not (spread > 0):
            return np.nan, core
        return (core - bg) / spread, core
    med = np.median(a)
    mad = np.median(np.abs(a - med)) * 1.4826
    if not (mad > 0):
        return np.nan, core
    return (core - med) / mad, core


def _seed_concentration(data, com, sat_area=None):
    """Concentration of a saturated-star seed = core peak / median just OUTSIDE
    the saturated core, measured on the deep coadd.  Radii scale to the
    saturated footprint (like _seed_prominence).

    Separates a real (saturated) STAR -- a point source whose PSF falls steeply
    once you pass the (possibly large, clipped) saturated core -- from a bright
    EXTENDED EMISSION KNOT, whose profile stays flat across the same radii.

    CRITICAL: the radii MUST scale with the saturated footprint.  A fixed
    r=3-6px annulus sits INSIDE the clipped plateau of a MASSIVE saturated star
    (e.g. a 478-px core, R~12px), giving peak/median ~2 -- indistinguishable
    from a flat knot -- which wrongly DROPS the brightest real stars (sickle
    F770W o002 star at 17:46:17.515 scored 2.3 at fixed radii, 24.6 adaptive).
    With ``sat_area`` the core peak is taken within r<=Rsat+1.5 and the
    comparison annulus at Rsat+3..Rsat+8, i.e. always just outside the core.

    Calibrated on F770W Sickle o002 (adaptive): real saturated stars score
    >=1.6 (median ~12, big ones 6-25); the flattest emission knots ~1.0.
    Returns NaN when unmeasurable (near edge, core entirely zeroed) -> callers
    treat NaN as KEEP so a real star is never dropped on inability to measure.
    """
    y, x = com
    # com can map to NaN near a tile edge; unmeasurable -> NaN so the gate KEEPS
    # it (NaN=keep) instead of crashing on int(NaN).
    if not (np.isfinite(x) and np.isfinite(y)):
        return np.nan
    xi, yi = int(round(float(x))), int(round(float(y)))
    ny, nx = data.shape
    r_sat = max(2.0, np.sqrt(max(int(sat_area or 0), 1) / np.pi))
    # legacy fixed behaviour when no footprint given (Rsat clamps to 2 -> r_core
    # 3.5, ring 5-10): kept for any caller that omits sat_area.
    r_core = r_sat + 1.5
    r_in = r_sat + 3.0
    r_out = r_sat + 8.0
    pad = int(np.ceil(r_out)) + 2
    if not (pad < xi < nx - pad and pad < yi < ny - pad):
        return np.nan
    yy, xx = np.mgrid[yi - pad:yi + pad + 1, xi - pad:xi + pad + 1]
    r = np.hypot(xx - xi, yy - yi)
    sub = data[yi - pad:yi + pad + 1, xi - pad:xi + pad + 1]
    cen = (r <= r_core) & np.isfinite(sub) & (sub != 0)
    ring = (r >= r_in) & (r <= r_out) & np.isfinite(sub) & (sub != 0)
    if cen.sum() < 1 or ring.sum() < 5:
        return np.nan
    peak = float(np.nanmax(sub[cen]))
    mid = float(np.nanmedian(sub[ring]))
    if not (mid > 0):
        return np.nan
    return peak / mid


def satstar_implied_peak(flux, psf_model, x_fit, y_fit):
    """Peak pixel value the fitted satstar model implies at its own position.

    A saturated star's model must, at minimum, reach the level at which the
    filter saturates -- otherwise the "saturated star" cannot be saturated.
    Evaluated by setting the fitted (flux, x_0, y_0) on a copy of the fit PSF
    and sampling it at its centre.  Returns NaN on any failure (gate then
    stays out of the way)."""
    try:
        m = psf_model.copy()
        m.flux = float(flux)
        m.x_0 = float(x_fit)
        m.y_0 = float(y_fit)
        return float(m(float(x_fit), float(y_fit)))
    except Exception:
        return np.nan


def accept_satstar_fit(*, result_is_none, fluxerr, snr, flux, qfit,
                       sidelobe_resid_sigma, ssr_ratio, is_miri,
                       qfit_max_keep, sidelobe_min_keep, ssr_ratio_max_keep,
                       snr_min_keep, ssr_trust_snr=10.0):
    """Decide whether a fitted saturated-star candidate is accepted.

    Pure predicate factored out of ``get_saturated_stars`` so the keep-logic is
    unit-testable (see tests/test_satstar_accept_gate.py).  Behaviour is
    IDENTICAL to the former inline expressions.

    The ssr_ratio gate is the subtle one: it is unreliable for bright saturated
    stars (STPSF wing-amplitude mismatch pushes ssr_ratio>1 even for excellent
    fits with qfit<1, snr>>10), so it is DROPPED for MIRI entirely and, for
    NIRCam, applied ONLY to low-confidence fits.  A high-snr (> ``ssr_trust_snr``)
    good-qfit fit is trusted regardless of ssr_ratio -- otherwise real
    must-detect bright stars get silently deleted from the satstar catalog and
    linger in the residual.  The white-point/no-source failure ssr was built to
    catch had snr~2, qfit=27, so the snr and qfit gates already reject it.
    """
    if result_is_none or not np.isfinite(fluxerr):
        return False
    if not (snr > snr_min_keep) or not (flux > 0):
        return False
    if is_miri:
        # MIRI: ssr dropped; require a finite POSITIVE qfit below the cap.
        return bool(np.isfinite(qfit) and 0.0 < qfit < qfit_max_keep
                    and (not np.isfinite(sidelobe_resid_sigma)
                         or sidelobe_resid_sigma > sidelobe_min_keep))
    # NIRCam
    if np.isfinite(qfit) and not (qfit < qfit_max_keep):
        return False
    if (np.isfinite(sidelobe_resid_sigma)
            and not (sidelobe_resid_sigma > sidelobe_min_keep)):
        return False
    ssr_trustworthy = (np.isfinite(snr) and snr > ssr_trust_snr
                       and np.isfinite(qfit) and qfit < qfit_max_keep)
    if ssr_trustworthy:
        return True
    return bool((not np.isfinite(ssr_ratio)) or (ssr_ratio < ssr_ratio_max_keep))


def is_fake_bright(model_peak, local_peak, *, model_min=1.0e4, localpk_max=3.5e3,
                   has_saturated_core=False):
    """Pure predicate: is a satstar fit a FAKE bright star on faint emission?

    The pathology (2526 cloud-c filament, sickle phantoms): a satstar is fit with
    a HUGE amplitude (model peak 1e4-6e5) at a position where the image is faint,
    smooth extended emission with NO compact source -- gouging a bright fake star
    (and a NaN/deep-negative pit in the residual) where there is none.

    Two signals, both needed to be field-general (an ABSOLUTE local-peak cut alone
    does NOT generalise: a real saturated star in a faint field can have a local
    peak of ~2000 while a fake in a bright field reaches ~2300):

      * ``model_peak > model_min`` AND ``local_peak < localpk_max`` -- the fit
        claims a very bright source the local image cannot support.
      * ``not has_saturated_core`` -- a GENUINE saturated star has an
        unrecoverable NaN (DO_NOT_USE) core in the per-frame SCI; a fake sits on
        FINITE smooth emission (MIRI's DQ-SATURATED flag there is ~99% spurious).
        So any fit with a genuine NaN core nearby is a real star and is EXEMPT
        from the fake cut regardless of how bright its model is -- this is what
        protects faint-field saturated stars (2526 SAT-faint: NaN core, local
        peak 2065) while still killing fakes (2526 FAKE: finite core, peak ~400).

    Returns False when inputs are non-finite or thresholds are disabled (<=0),
    so an unmeasurable fit is never dropped as fake.
    """
    if has_saturated_core:
        return False
    if not (model_min and model_min > 0 and localpk_max and localpk_max > 0):
        return False
    if not (np.isfinite(model_peak) and np.isfinite(local_peak)):
        return False
    return bool(model_peak > model_min and local_peak < localpk_max)


def is_bright_flux_phantom(flux, flux_init, ssr_ratio, *, flux_floor=1.0e5,
                           ssr_max=50.0, ratio_max=50.0):
    """Pure predicate: is a VERY BRIGHT satstar fit a spurious super-bright phantom
    on saturated extended emission?

    The pathology (W51 F770W, and any MIRI field where bright nebulosity saturates
    the detector): real stars and pure-emission knots saturate into ONE connected
    ~10^5-px DQ-SATURATED blob, so every PRE-fit metric fails -- an emission knot is
    a compact bright peak that the seed gate scores as MORE stellar than a genuine
    saturated star whose own core is zeroed (prominence/core/conc/data-floor/NaN-
    core/morphology all rank the phantom >= the real star; verified W51 F770W).  The
    two signals that DO separate them are POST-fit, and only for very bright fits
    (the ``flux_floor`` precondition is load-bearing -- it protects genuine deeply-
    saturated stars whose faint wing-seed gives a large extrapolation ratio, e.g. a
    real spiky star at flux 2e4 with flux/flux_init=61):

      * ``ssr_ratio > ssr_max`` -- the model leaves a badly STRUCTURED residual
        (model-on-emission: the PSF has no counterpart in the flat data; W51 SPUR1
        ssr=139 vs a real bright star ssr=12).
      * ``flux / flux_init > ratio_max`` -- the fit EXTRAPOLATED an enormous flux
        from a faint seed (W51 SPUR2: flux_init 6677 -> flux_fit 7.0e6, 1054x).

    Requires ``flux > flux_floor`` AND (bad ssr OR bad ratio).  Verified on W51
    F770W: flags 7/153 satstars, all confirmed emission phantoms by eye, zero real-
    star false drops (the user's real well-fit bright star: ssr=12, ratio=14 ->
    kept).  Thresholds are F770W-surface-brightness-calibrated; ``flux_floor`` is
    field/filter dependent -> override per run (MIRI_SATSTAR_PHANTOM_*).  Returns
    False when disabled (flux_floor<=0) or inputs non-finite, so a fit is never
    dropped on an inability to measure it.  See project_miri_partialsat_divot /
    the W51 F770W spurious-super-bright investigation.
    """
    if not (flux_floor and flux_floor > 0):
        return False
    if not (np.isfinite(flux) and flux > flux_floor):
        return False
    bad_ssr = (ssr_max and ssr_max > 0 and np.isfinite(ssr_ratio)
               and ssr_ratio > ssr_max)
    ratio = (flux / flux_init) if (np.isfinite(flux_init) and flux_init > 0) else np.nan
    bad_ratio = (ratio_max and ratio_max > 0 and np.isfinite(ratio)
                 and ratio > ratio_max)
    return bool(bad_ssr or bad_ratio)


def is_small_radius_emission_phantom(prom_small, core_small, *,
                                     prom_min=8.0, core_min=1000.0):
    """Pure predicate: is a fit an extended-emission phantom by its SMALL-fixed-
    radius prominence/core (neighbour-robust, independent of a spuriously large
    DQ-SATURATED sat_area)?

    The sat_area-scaled prominence is required for genuinely huge saturated stars
    (their core is zeroed to ~22px; a small ring sits inside it -> NaN -> kept),
    but on a FAKE the equally-large spurious footprint makes the scaled ring reach
    a NEIGHBOURING real star and inflates prominence past the cut (2526 FAKE-1:
    0.9 at r~11px vs 49 at r~22px).  Measuring at a small FIXED radius separates
    them: a fake reads low, a normal star reads high, and a huge real star reads
    NaN (zeroed core).  Reject only when the small-radius value is FINITE and
    below threshold (NaN -> not a phantom -> kept).
    """
    if (prom_min and prom_min > 0 and np.isfinite(prom_small)
            and prom_small < prom_min):
        return True
    if (core_min and core_min > 0 and np.isfinite(core_small)
            and core_small < core_min):
        return True
    return False


def flattop_satstar_model(model_image, data_bg_sub, plateau_frac=0.15,
                          core_mask=None):
    """Replace a saturated-star PSF model by the DATA over its flat-topped core.

    STPSF is sharply PEAKED, but a charge-bled saturated core is a flat-topped
    PLATEAU.  Subtracting ``amp*PSF`` therefore UNDER-subtracts the core (a
    bright ring at r~3px; cloudc F770W "every saturated-core star undersubtracted")
    and, when the amplitude is pushed up to clear the core, OVER-subtracts it (a
    central divot / negative pit).  Over the plateau footprint the model is
    instead set to the (background-subtracted) DATA itself, so ``data - model``
    -> ~0 there -- flat, no ring, no divot.  The PSF WINGS are left untouched
    (they are peaked-PSF-correct outside the saturated core).

    Plateau footprint (on the cutout):
      * NaN core        -- data saturated/masked; keep the PSF (only estimate
                          available; those pixels are masked downstream anyway).
      * ``core_mask``    -- the geometric core+shoulder region (a radius from the
                          fit centre scaled by the saturated footprint + PSF
                          FWHM).  This is what actually covers the under-sub RING:
                          the pipeline's peak-cap keeps the model peak AT the
                          coadd-core level, so ``model>data`` never fires in the
                          core and the shoulder ring sits just below
                          ``plateau_frac*peak`` -- a purely amplitude-based mask
                          misses it.  The geometric mask is amplitude-independent.
      * model > data     -- the PSF over-predicts; capping to data prevents the
                          over-subtraction pit (a global guard, also in the wings).
      * data > frac*peak -- the bright plateau (kept as a floor when no core_mask).

    Parameters
    ----------
    model_image : ndarray
        Rendered ``amp*PSF`` model (>=0), cutout-sized.
    data_bg_sub : ndarray
        Background-subtracted data on the SAME cutout (``cutout_fit``); NaN in
        the saturated core.
    plateau_frac : float
        Bright-plateau threshold as a fraction of the finite-data peak.
    core_mask : ndarray of bool, optional
        Geometric core+shoulder footprint (same shape).  Inside it, finite data
        is subtracted flat.  When None, only the amplitude-based terms apply
        (used by the unit tests / synthetic flat cores).

    Returns
    -------
    ndarray
        Flat-topped model (>=0), same shape.  If ``data_bg_sub`` has no finite
        pixel (a fully-saturated core with nothing to subtract) the input model
        is returned unchanged, so the caller's peaked-PSF behaviour is preserved.
    """
    db = np.asarray(data_bg_sub, dtype=float)
    mpsf = np.asarray(model_image, dtype=float)
    finite = np.isfinite(db)
    if not finite.any():
        return model_image
    peak = float(np.max(db[finite]))
    if not (np.isfinite(peak) and peak > 0):
        return model_image
    # NaN core / geometric core+shoulder / model-over-data / bright plateau all
    # get flat-topped; NaN comparisons resolve False, so ``~finite`` covers the
    # saturated core and ``core_mask`` covers the under-sub RING.
    plateau = (~finite) | (mpsf > db) | (db > plateau_frac * peak)
    if core_mask is not None:
        plateau = plateau | np.asarray(core_mask, dtype=bool)
    # In the plateau use the DATA where it exists (subtract it flat); keep the
    # PSF in the NaN core (no data to subtract) and everywhere outside.
    out = np.where(plateau & finite, db, mpsf)
    return np.maximum(out, 0)


def get_saturated_stars(fitsdata, path_prefix='/orange/adamginsburg/jwst/w51/psfs/', pad=81, size=None, min_sep_from_edge=5, edge_npix=10000, mask_buffer=2, adaptive_mask_buffer_scale=True, adaptive_bkg_annulus=True, plot=True, rindsz=3, use_merged_psf_for_merged=False, outside_star_pixels=None, outside_star_fit_box=512, forced_grid_search_radius=5, satstar_central_downweight_sigma=0.0, flux_overrides=None, flux_drops=None, oversub_clamp_percentile=10.0, seed_prominence_min=8.0, seed_core_min=1000.0, seed_conc_min=1.3, seed_prominence_robust=False, seed_oversub_ratio=3.0, seed_fake_model_min=1.0e4, seed_fake_localpk_max=3.5e3, seed_gate_image=None, seed_gate_wcs=None, zeroframe=None, zeroframe_deblend=False, deblend_daophot_xy=None, deblend_confirm_xy=None, sat_data_floor=None, satstar_severity_floor=None, phantom_flux_floor=0.0, phantom_ssr_max=50.0, phantom_ratio_max=50.0):
    # ``flux_drops``: optional list of SkyCoord.  An out-of-field (forced) source
    # whose seed sky position matches a drop within ~1.0" is SKIPPED entirely
    # (not fit, not contributed): cross-frame reconciliation found no trustworthy
    # reference for it (single detector, spikes never in FOV -> unfittable), so
    # contributing any model would over-add a fake background.
    # ``flux_overrides``: optional list of (SkyCoord, flux) pairs.  An out-of-field
    # (forced) source whose seed sky position matches an override within ~0.2" uses
    # that fixed flux instead of fitting -- the cross-frame reconciliation
    # (reconcile_outside_fov_satstar_fluxes) supplies the nearest-detector flux so a
    # far frame (which would fit the spike-less corner badly) renders the correct
    # amplitude.  See run_manual_pipeline.
    """
    Detect and PSF-fit saturated sources in a JWST image.

    This routine identifies connected saturated-pixel regions using the ``DQ``
    extension, excludes large edge-associated saturated structures, and then
    fits one source per remaining region with ``PSFPhotometry``.  Fits are
    performed on local cutouts with saturated pixels masked, and accepted
    results are stacked into a single output table.

    Parameters
    ----------
    fitsdata : astropy.io.fits.HDUList
        Open FITS HDU list containing at least ``SCI``, ``DQ``, and
        ``VAR_POISSON`` extensions.
    path_prefix : str, optional
        Directory used to load or cache PSF grid files.
    pad : int, optional
        Half-size (pixels) of the square cutout centered on each saturated
        source.
    size : int or tuple, optional
        Fit shape passed to ``PSFPhotometry``.
    min_sep_from_edge : int, optional
        Number of dilation iterations used to mask around large edge-saturated
        regions.
    edge_npix : int, optional
        Minimum saturated-pixel area used to classify a region as an edge
        source to be excluded.
    mask_buffer : int, optional
        Minimum dilation iterations applied to saturated masks before fitting.
        Default changed from 1 → 2; synthetic recovery tests show this reduces
        flux bias from ~6 % to ~3 % by excluding more of the non-linear
        transition zone at the saturation boundary.
    adaptive_mask_buffer_scale : bool, optional
        If ``True`` (default), scale the mask buffer with the size of the
        saturated region so that deeply saturated (bright) sources receive a
        larger buffer.  ``mask_buffer`` acts as the minimum.  See
        ``compute_adaptive_mask_buffer`` for the scaling formula.
    adaptive_bkg_annulus : bool, optional
        If ``True`` (default), scale the background annulus radii with the
        saturation area so that deeply saturated sources use a wider annulus
        (avoiding PSF wing contamination) while marginally saturated sources
        use the standard narrow annulus (15, 30).  See
        ``compute_adaptive_bkg_annulus`` for the calibration.
    plot : bool, optional
        If ``True``, display per-source diagnostic plots (cutout, model,
        residual, mask, thresholded model).
    rindsz : int, optional
        Reserved/legacy parameter; currently unused.
    use_merged_psf_for_merged : bool, optional
        If ``True`` and a merged PSF grid file exists, use it for merged
        mosaics.  Default is ``False`` to prefer detector-specific WebbPSF
        grids for individual frame fitting.

    Returns
    -------
    astropy.table.Table or None
        Stacked table of accepted saturated-source fits, or ``None`` if no
        valid sources are found.  Typical columns include fit parameters such
        as ``x_fit``, ``y_fit``, ``flux_fit``, uncertainties, and derived
        ``xcentroid``, ``ycentroid``, and ``skycoord_fit``.

    Notes
    -----
    - Requires ``STPSF_PATH`` to be defined before this module is imported.
    - Source acceptance currently requires finite flux uncertainty,
      ``snr > 1``, and positive fitted flux.
    - Large contiguous saturated edge structures are removed prior to fitting.
    """
    header = fitsdata[0].header
    data = fitsdata['SCI'].data
    assert data is not None

    # Per-run seed-gate overrides (env).  The seed_core_min=1000 / prom_min=8
    # defaults are F770W-Sickle-calibrated; emission-rich fields with fainter
    # saturated stars (cloudc 2526 F770W: real wing-ring cores 300-760, dropped
    # by core<1000) need a lower core floor + the neighbour-robust prominence
    # (MIRI_SATSTAR_SEED_PROM_ROBUST) so the annulus MAD inflated by neighbouring
    # stars/emission does not crush a real faint saturated star below the gate.
    # Defaults preserve the prior behaviour; set per-field in the launcher.
    # See project_cloudc_f770w_satstar_gate_miscalib.
    _env_core = os.environ.get('MIRI_SATSTAR_SEED_CORE_MIN')
    if _env_core is not None:
        seed_core_min = float(_env_core)
    _env_prom = os.environ.get('MIRI_SATSTAR_SEED_PROM_MIN')
    if _env_prom is not None:
        seed_prominence_min = float(_env_prom)
    _env_conc = os.environ.get('MIRI_SATSTAR_SEED_CONC_MIN')
    if _env_conc is not None:
        seed_conc_min = float(_env_conc)
    _env_robust = os.environ.get('MIRI_SATSTAR_SEED_PROM_ROBUST')
    if _env_robust is not None:
        seed_prominence_robust = bool(int(_env_robust))
    # Post-fit BRIGHT-PHANTOM gate (env).  Rejects spurious super-bright satstars
    # on saturated extended emission that evade every pre-fit gate (W51 F770W).
    # OFF by default (flux_floor=0); F770W-calibrated floor 1e5 / ssr 50 / ratio 50
    # set in the launcher.  See is_bright_flux_phantom.
    _env_pff = os.environ.get('MIRI_SATSTAR_PHANTOM_FLUX_FLOOR')
    if _env_pff is not None:
        phantom_flux_floor = float(_env_pff)
    _env_pssr = os.environ.get('MIRI_SATSTAR_PHANTOM_SSR_MAX')
    if _env_pssr is not None:
        phantom_ssr_max = float(_env_pssr)
    _env_prat = os.environ.get('MIRI_SATSTAR_PHANTOM_RATIO_MAX')
    if _env_prat is not None:
        phantom_ratio_max = float(_env_prat)

    # FLAT-TOPPED SATURATED-CORE MODEL (env, MIRI).  When enabled, an accepted
    # in-FOV satstar's model is replaced INSIDE its flat-topped plateau by the
    # (bg-subtracted) data, so the residual there -> ~0 (no under-sub ring, no
    # over-sub divot).  OFF by default; launcher opts in with MIRI_SATSTAR_FLATTOP=1.
    # plateau_frac tunes the bright-plateau threshold (fraction of data peak).
    # See flattop_satstar_model + the accept block below.
    _env_ft = os.environ.get('MIRI_SATSTAR_FLATTOP')
    satstar_flattop = bool(int(_env_ft)) if _env_ft is not None else False
    _env_ftf = os.environ.get('MIRI_SATSTAR_FLATTOP_PLATEAU_FRAC')
    satstar_flattop_frac = float(_env_ftf) if _env_ftf is not None else 0.15
    if satstar_flattop:
        print(f"[flat-top cfg] MIRI flat-topped saturated cores ENABLED "
              f"(frac={satstar_flattop_frac})", flush=True)

    # nan_to_num data to avoid fitting NaNs
    data[np.isnan(fitsdata['VAR_POISSON'].data)] = 0
    dq = fitsdata['DQ'].data
    full_model_image = np.zeros_like(data, dtype=float)

    # MIRI: fold saturated diffraction-spike satellites into the core so a
    # bright star is ONE component (not a hex of spike-tip duplicates).  Gated
    # by MIRI_SATSTAR_SPIKE_MERGE (gap px, default 3) + ..._RATIO (default 3.0).
    # NIRCam keeps legacy connectivity labeling (gap=0).
    _spike_gap = (int(os.environ.get('MIRI_SATSTAR_SPIKE_MERGE', 3))
                  if header['INSTRUME'].lower() == 'miri' else 0)
    _spike_ratio = float(os.environ.get('MIRI_SATSTAR_SPIKE_MERGE_RATIO', 3.0))
    _sat_floor = _resolve_satstar_data_floor(header.get('FILTER', ''),
                                             explicit=sat_data_floor)
    _sev_floor = _resolve_satstar_severity_floor(header.get('FILTER', ''),
                                                 explicit=satstar_severity_floor)
    saturated, sources, coms = find_saturated_stars(
        fitsdata, min_sep_from_edge=min_sep_from_edge, edge_npix=edge_npix,
        spike_merge_gap=_spike_gap, spike_merge_ratio=_spike_ratio,
        sat_data_floor=_sat_floor, severity_floor=_sev_floor)

    # Diagnostic flag image (uint8 bitmask), written by remove_saturated_stars:
    #   bit 0 (1) = partly saturated (DQ SATURATED but recoverable / nonlinear,
    #               finite variance)
    #   bit 1 (2) = totally saturated (unrecoverable: NaN variance -> data zeroed)
    #   bit 2 (4) = pixel INCLUDED in an accepted saturated-star fit (unmasked
    #               within that source's fit window)
    _unrecoverable = np.isnan(fitsdata['VAR_POISSON'].data)
    flag_img = np.zeros(data.shape, dtype=np.uint8)
    flag_img[saturated & ~_unrecoverable] |= 1
    flag_img[saturated & _unrecoverable] |= 2
    # Refine centroids: the mask centre-of-mass can be biased by
    # JUMP_DET-contamination, asymmetric edge clipping, or merged
    # neighbour saturation.  Re-centre to the flux-weighted centroid of
    # the unsaturated wings -- much more reliable as a star-centre
    # estimate (the saturated mask should be centred on the star
    # centroid).  Added 2026-05-28.
    coms = _refine_coms_by_data(coms, data, sources, unrecoverable=_unrecoverable)

    # Precompute sat_area per labeled component so we can order in-FOV
    # source_records brightest-first for iterative-subtraction fitting
    # (see ``data_working`` setup + post-accept subtraction below).
    if sources.size and sources.max() > 0:
        _sizes_by_label = sum_labels(saturated, sources,
                                     np.arange(int(sources.max())) + 1)
    else:
        _sizes_by_label = np.array([], dtype=float)

    # ZEROFRAME DEBLEND (opt-in): when a frame-zero image is supplied, expand a
    # single merged-saturated component into one seed PER STAR.  In crowded GC
    # fields (gc2211) bright stars' saturated cores TOUCH and label as one
    # component, so the single bbox centroid lands BETWEEN two stars.  The
    # ZEROFRAME (raw first read, saturates ~Ngroup higher) resolves the cores; see
    # satstar_deblend.build_deblended_source_records.  When ``zeroframe is None``
    # this block is skipped and the behaviour is identical to before (one seed per
    # component at its refined centroid).
    if zeroframe is not None and zeroframe_deblend:
        from .satstar_deblend import build_deblended_source_records
        _inst_repl = 'MIRI' if header['INSTRUME'].lower() == 'miri' else 'NIRCam'
        _, _fwhm_pix_db = get_fwhm(header, instrument_replacement=_inst_repl)
        source_records = build_deblended_source_records(
            saturated, sources, coms, _sizes_by_label, zeroframe, data,
            _fwhm_pix_db, daophot_xy=deblend_daophot_xy,
            confirm_xy=deblend_confirm_xy)
        print(f"satstar ZEROFRAME deblend: {len(coms)} components -> "
              f"{len(source_records)} star seeds", flush=True)
    else:
        source_records = []
        for ii, com in enumerate(coms):
            sat_area_ii = int(_sizes_by_label[ii]) if ii < len(_sizes_by_label) else 0
            source_records.append({'com': com, 'label': ii + 1,
                                   'forced': False, 'sat_area': sat_area_ii})

    # MIRI-only seed gate: drop DQ-SATURATED components that are NOT compact
    # bright stars (i.e. saturated patches of bright EXTENDED EMISSION).  F770W
    # and the other MIRI broadbands saturate on the Sickle's nebulosity, so the
    # DQ plane sprouts dozens of non-stellar saturated components; fitting each
    # as a PSF point source produced phantom bright stars (flux_init<0 -> runaway
    # flux_fit~1e6, model peak ~1e4-5e4 where the data peak is ~600-900) that
    # over-subtract the data into deep negative pits and pollute the display
    # model.  TWO criteria, because they catch different phantoms:
    #   - prominence < seed_prominence_min  -> diffuse extended-emission sat
    #     (the bulk, ~50/frame; real >=15, phantom ~1-3).
    #   - core < seed_core_min  -> compact bumps on SMOOTH emission whose tiny
    #     annulus MAD inflates prominence to 12-55 but whose wing ring is faint
    #     (real >=1191, phantom ~800).  seed_core_min is in the data's surface-
    #     brightness units; the 1000 default is F770W-calibrated -- revisit for
    #     other MIRI broadbands (set 0 to disable just this criterion).
    # The gate is measured on the DEEP coadded ``seed_gate_image`` (the filter's
    # data_i2d) when supplied, NOT this single frame: a single-frame noise spike
    # in the wing ring can push a phantom's per-frame core >threshold in ONE of
    # its overlapping frames, and the cross-frame satstar merge then keeps it.
    # The coadd is noise-averaged and frame-invariant, so the same verdict
    # applies in every frame and matches what the user sees in the final i2d
    # (validated: phantom cores 573-966 on the coadd vs real >=1200).  Falls back
    # to the per-frame ``data`` if no coadd is provided.
    # NaN (unmeasurable, e.g. near-edge) is KEPT so no real star is dropped on an
    # inability to measure it.  NIRCam untouched (gate is MIRI-only).
    _is_miri_gate = (header.get('INSTRUME', '').lower() == 'miri')
    if _is_miri_gate and ((seed_prominence_min and seed_prominence_min > 0)
                          or (seed_core_min and seed_core_min > 0)):
        _frame_wcs = wcs.WCS(fitsdata['SCI'].header)
        _gate_img = seed_gate_image if seed_gate_image is not None else data
        _use_coadd = seed_gate_image is not None and seed_gate_wcs is not None
        kept_records, n_prom, n_core, n_conc = [], 0, 0, 0
        # Off-footprint guard: an AUTO-detected saturated component whose centroid
        # lands in a BORDER-CONNECTED NaN region of the gate image (off the data
        # footprint, partly off-field) has no data to constrain it -> the PSF fit
        # runs away to a huge flux (brick F2550W: 215k-5.8M) and renders a bright
        # blob hugging the mosaic edge in the model_i2d.  Drop those.  INTERIOR
        # NaN (a saturated star's own zeroed core, surrounded by data) is filled
        # by binary_fill_holes, so it stays inside the footprint and the real star
        # survives.  USER-SEEDED outside-FOV satstars are added later via
        # ``outside_star_pixels`` and never enter this source_records loop, so
        # they are explicitly NOT affected (per user: keep user-seeded off-fp).
        _drop_offfp = int(os.environ.get('MIRI_DROP_OFFFP_SATSTAR', 1))
        _gate_fp = None
        if _drop_offfp:
            from scipy.ndimage import binary_fill_holes as _bfh
            _gate_fp = _bfh(np.isfinite(_gate_img))
        _gh, _gw = _gate_img.shape
        n_offfp = 0
        for rec in source_records:
            cy, cx = rec['com']
            if _use_coadd:
                # map this frame's component centroid onto the coadd grid
                try:
                    sky = _frame_wcs.pixel_to_world(float(cx), float(cy))
                    gx, gy = seed_gate_wcs.world_to_pixel(sky)
                    gate_com = (float(gy), float(gx))
                except Exception:
                    gate_com = rec['com']
            else:
                gate_com = rec['com']
            # drop auto-detected components off the data footprint (border-connected
            # NaN) -- runaway edge satstars; interior saturated cores survive.
            if _gate_fp is not None:
                _gy, _gx = gate_com
                # NaN gate_com == the component's sky maps OFF the coadd grid
                # (world_to_pixel returned NaN) == off-footprint -> drop.
                if not (np.isfinite(_gy) and np.isfinite(_gx)):
                    n_offfp += 1
                    continue
                _gyi = int(round(_gy))
                _gxi = int(round(_gx))
                if (not (0 <= _gyi < _gh and 0 <= _gxi < _gw)
                        or not _gate_fp[_gyi, _gxi]):
                    n_offfp += 1
                    continue
            # cap sat_area so a giant spurious-DQ cluster's prominence radius
            # cannot grow to grab a distant real star (see post-fit gate note).
            _seed_sa = rec.get('sat_area')
            _seed_sa_cap = (min(int(_seed_sa), 1600)
                            if _seed_sa is not None else _seed_sa)
            prom, core = _seed_prominence(_gate_img, gate_com, _seed_sa_cap,
                                          robust=seed_prominence_robust)
            conc = _seed_concentration(_gate_img, gate_com, _seed_sa_cap)
            drop_prom = (seed_prominence_min and seed_prominence_min > 0
                         and np.isfinite(prom) and prom < seed_prominence_min)
            drop_core = (seed_core_min and seed_core_min > 0
                         and np.isfinite(core) and core < seed_core_min)
            drop_conc = (seed_conc_min and seed_conc_min > 0
                         and np.isfinite(conc) and conc < seed_conc_min)
            if drop_prom or drop_core or drop_conc:
                n_prom += int(bool(drop_prom))
                n_core += int(bool(drop_core and not drop_prom))
                n_conc += int(bool(drop_conc and not drop_prom and not drop_core))
            else:
                kept_records.append(rec)
        n_dropped = len(source_records) - len(kept_records)
        if n_dropped:
            _on = 'coadd i2d' if _use_coadd else 'this frame'
            print(f"satstar seed gate (MIRI, on {_on}): dropped {n_dropped} "
                  f"non-stellar saturated components (extended-emission "
                  f"phantoms; {n_prom} by prominence<{seed_prominence_min}, "
                  f"{n_core} by faint-core<{seed_core_min}, {n_conc} by "
                  f"flat-profile/conc<{seed_conc_min}, {n_offfp} off-footprint "
                  f"edge runaways); kept "
                  f"{len(kept_records)} of {len(source_records)}", flush=True)
        source_records = kept_records

    # Sort in-FOV sources by sat_area descending (brightest saturated cores
    # first) so iterative subtraction handles them in brightness order.
    # Without this, two adjacent saturated stars (e.g. Sickle 0310g_00002
    # pair at (555,272)+(560,280), sep=0.61") fit independently against the
    # full data and each absorbs the other's PSF wings → both fluxes ~50-
    # 100% inflated → 2× over-subtraction when models are accumulated.
    # Subtracting the brighter star's PSF before fitting the fainter one
    # removes wing contamination at the cost of one PSF-model evaluation
    # per source.
    source_records.sort(key=lambda s: -int(s.get('sat_area') or 0))

    if outside_star_pixels is not None:
        for xy in outside_star_pixels:
            if xy is None:
                continue
            if len(xy) != 2:
                continue
            x_extra, y_extra = float(xy[0]), float(xy[1])
            if np.isfinite(x_extra) and np.isfinite(y_extra):
                source_records.append({'com': (y_extra, x_extra),
                                       'label': None, 'forced': True,
                                       'sat_area': None})

    nsource = len(source_records)

    # ZEROFRAME-ANCHORED FIT DATA (2026-07-10): when the ramp first read is
    # available, replace the clipped/NaN/charge-inflated DQ-SATURATED rim with
    # its group-0 truth (R * group0, self-calibrated per frame -- see
    # zeroframe_recover_saturated) BEFORE fitting.  For marginally saturated
    # stars (SW mag ~12-15.5) group-0 is unsaturated across most of the core,
    # so the amplitude is anchored by real profile pixels instead of a
    # wing-only extrapolation -- the wing extrapolation left those stars
    # 0.5-3.6 mag too faint (or, unreplaced, kept clipped daophot fluxes),
    # producing the unphysical red jog / gap / split clouds at the CMD
    # saturation boundary.  Only the group-0-SATURATED deep core stays
    # unmeasurable: ``zf_deep_core`` replaces the full any-group blob in the
    # fit mask below.  No zeroframe (or no usable R) -> zf_deep_core = None,
    # behaviour unchanged.
    zf_deep_core = None
    if zeroframe is not None:
        _dqarr_zf = (fitsdata['DQ'].data
                     if 'DQ' in [h.name for h in fitsdata] else None)
        _rec, _rim, _deep, _R = zeroframe_recover_saturated(
            data, _dqarr_zf, zeroframe)
        if np.isfinite(_R) and _rim.any():
            print(f"satstar ZEROFRAME fit-anchor: recovered {int(_rim.sum())} "
                  f"rim/core pixels from group-0 (R={_R:.4g}); "
                  f"{int(_deep.sum())} deep-core pixels remain masked",
                  flush=True)
            data = _rec
            zf_deep_core = _deep

    # Working copy of the data that gets per-source PSF models subtracted
    # after each accepted fit, so subsequent fits see a cleaner field.
    # ``data`` itself is preserved for the final ``data - full_model_image``
    # residual and for downstream callers that read it.
    data_working = data.astype(float, copy=True)

    # Per-pixel 1-sigma uncertainty for INVERSE-VARIANCE weighting of the
    # satstar PSF fit.  Without it (error=None) PSFPhotometry weights every
    # pixel equally, so the linear-LSQ amplitude f = sum(d*p)/sum(p^2) is
    # dominated by the bright inner-wing / first-sidelobe pixels (large p).
    # That set the amplitude too high and over-subtracted the fainter SECOND
    # sidelobes (negative ~-17 MJy/sr rings seen on sickle pillar satstars,
    # 2026-06-09).  Inverse-variance weighting (w = 1/ERR^2) down-weights the
    # high-Poisson-noise bright pixels and up-weights the low-noise faint
    # sidelobes/outer wings, so the fit matches the full PSF shape instead of
    # just the bright core, removing the amplitude overshoot.  Non-finite or
    # non-positive ERR (saturated / unrecoverable pixels) is set huge so those
    # pixels carry ~zero weight even if the mask were to miss them.
    err_working = np.array(fitsdata['ERR'].data, dtype=float)
    # recovered rim pixels carry no valid ERR (NaN at SATURATED px in crf);
    # assign the ~5% per-pixel scatter of the R calibration so the fit can
    # inverse-variance weight them instead of dropping/inf-weighting.
    if zf_deep_core is not None:
        _rimfix = _rim & ~np.isfinite(err_working)
        err_working[_rimfix] = np.abs(data[_rimfix]) * 0.05
    _bad_err = ~np.isfinite(err_working) | (err_working <= 0)
    err_working[_bad_err] = 1e10

    # LW detectors (NRCALONG/NRCBLONG, internally NRCA5/NRCB5) get a larger
    # 1024-px PSF grid for in-FOV satstar fits.  PSF-size-vs-flux study
    # showed fovp512 underestimates bright LW star flux by 50-70%; fovp1024
    # plateaus and matches catalog flux.  SW detectors keep 512 (Sickle
    # subarray data cannot diagnose larger sizes; full-frame fields may
    # need re-evaluation).
    _det_for_fov = header.get('DETECTOR', '').upper()
    _is_lw_for_satstar = _det_for_fov in ('NRCALONG', 'NRCBLONG', 'NRCA5', 'NRCB5')
    _satstar_fov_pixels = 1024 if _is_lw_for_satstar else 512
    big_grid = get_psf(header, path_prefix=path_prefix,
                       use_merged_psf_for_merged=use_merged_psf_for_merged,
                       fov_pixels=_satstar_fov_pixels)

    # Large PSF grid for forced (outside-FOV) sources: their diffraction spikes
    # extend ~40" into the FOV — that's 635 px (LW) to 1290 px (SW), much
    # larger than the default fov_pixels=512 grid (256 px radius).  Required
    # whenever any forced source is present; if missing, abort — the small
    # grid silently fails to model off-edge spikes and produces wrong outputs.
    # defined here (before the large-PSF block uses it); re-set identically below
    _is_miri = header['INSTRUME'].lower() == 'miri'
    big_grid_large = None
    forced_sources_present = any(s.get('forced') for s in source_records)
    # 2026-06-20: also load the large (fovp1024, spike-resolving) PSF for MIRI
    # IN-FOV saturated stars, not just forced/off-FOV ones.  A bright saturated
    # star's amplitude is set by its unsaturated 6-pointed DIFFRACTION SPIKES;
    # the small (fovp101) grid omits them, so the masked-core LSQ has nothing to
    # pin the amplitude and under-fits (sickle A/B fit ~10-30x too low).  Fitting
    # the large PSF lets the spikes constrain the amplitude (B->2.7e5, A->~1e6,
    # matching the user's by-eye).  For IN-FOV use we fall back to the small grid
    # if the large file is missing (only forced sources hard-require it).
    if forced_sources_present or _is_miri:
        from .filtering import get_filtername
        _inst = header['INSTRUME'].lower()   # 'nircam' or 'miri'
        _det = header['DETECTOR']
        if _det == 'NRCALONG':
            _det = 'NRCA5'
        elif _det == 'NRCBLONG':
            _det = 'NRCB5'
        elif _det.upper() == 'MIRIMAGE':
            _det = 'mirim'
        _filt = get_filtername(header)
        # PSF grid fov_pixels sized so off-FOV diffraction spikes are modelled:
        # NIRCam SW (NRC?1-4) 2048, NIRCam LW (NRC?5) 1024; MIRI (0.11"/px,
        # spikes reach ~40"=364px) 1024.
        if _inst == 'miri':
            _fovp = 1024
        else:
            _is_lw = _det.upper().endswith('5')
            _fovp = 1024 if _is_lw else 2048
        _lg_fn = (f"{path_prefix}/{_inst}_{_det.lower()}_{_filt.lower()}"
                  f"_fovp{_fovp}_samp2_npsf16.fits")
        if not os.path.exists(_lg_fn):
            if forced_sources_present:
                raise FileNotFoundError(
                    f"Forced source(s) present but large PSF grid is missing: {_lg_fn}. "
                    f"Build it with /orange/adamginsburg/jwst/sickle/build_large_psf_grids.py "
                    f"before running with outside-FOV seeds.  Falling back to the small "
                    f"grid silently produces wrong outputs."
                )
            else:
                print(f"Large PSF grid {_lg_fn} missing; MIRI in-FOV satstars fall "
                      f"back to the small grid (amplitude may under-fit).", flush=True)
        else:
            print(f"Loading large PSF grid: {_lg_fn}", flush=True)
            big_grid_large = to_griddedpsfmodel(_lg_fn)
            if isinstance(big_grid_large, list):
                big_grid_large = big_grid_large[0]

    # WCS for ALL pixel->sky conversions below (sky_centroid, skycoord_fit, etc.)
    # MUST come from the SCI header, which carries the full RA---TAN-SIP
    # distortion.  ``header`` is the PRIMARY header (fitsdata[0]) and only has a
    # WCS at all because remove_saturated_stars copies one in -- historically via
    # a plain to_header() that DROPPED the SIP terms, so every satstar position
    # was ~0.1" off at the detector edges.  Read SCI directly (relax=True keeps
    # SIP) so the fit coordinates are correct regardless of the primary copy.
    ww = wcs.WCS(fitsdata['SCI'].header, relax=True)

    # We force the centroid to be fixed b/c the fitter doesn't do a great job with this...
    # ....this is not optimal...
    #big_grid.fixed['x_0'] = True
    #big_grid.fixed['y_0'] = True

    # daogroup should be set super high to avoid fitting lots of "stars"... if there are a lot of saturated pixels near each other, they're probably all junk
    #daogroup = SourceGrouper(min_separation=25)

    #resid = data

    #results = []

    lmfitter = LevMarLSQFitter()
    # def levmarverbosewrapper(self, *args, **kwargs):
    #     print("Running lmfitter")
    #     log.info(f"Running lmfitter with args {args} and kwargs {kwargs}")
    #     return self(*args, **kwargs)
    # #lmfitter.__call__ = levmarverbosewrapper
    # lmfitter._run_fitter = levmarverbosewrapper

    if header['INSTRUME'].lower() == 'nircam':
        psfgen = stpsf.NIRCam()
        fwhm, fwhm_pix = get_fwhm(header, instrument_replacement='NIRCam')
    elif header['INSTRUME'].lower() == 'miri':
        psfgen = stpsf.MIRI()
        fwhm, fwhm_pix = get_fwhm(header, instrument_replacement='MIRI')

    # The accept gates below (sidelobe_resid_sigma, ssr_ratio, qfit) were tuned
    # on NIRCam, where IPC makes the STPSF first-sidelobe brighter than the
    # real saturated-star PSF.  MIRI has no such sidelobe mismatch and a much wider
    # diffraction pattern at 7-25um, so the NIRCam-tuned cuts spuriously reject
    # real saturated stars (e.g. ~10 of the 34 F770W hand-selected stars are
    # DQ-saturated and were being dropped here).  Loosen for MIRI.
    _is_miri = header['INSTRUME'].lower() == 'miri'
    _qfit_max_keep = 15.0 if _is_miri else 5.0
    _sidelobe_min_keep = -40.0 if _is_miri else -10.0
    _ssr_ratio_max_keep = 2.0 if _is_miri else 1.0
    _snr_min_keep = 2.0 if _is_miri else 3.0

    # (removed dead `slices = find_objects(saturated)`: the result was never
    # used, and passing the BOOL `saturated` mask -- rather than the int-labelled
    # `sources` -- raises under newer scipy/numpy ("'numpy.bool' object cannot be
    # interpreted as an integer").  The per-source bounding boxes are taken from
    # `sources`/`source_records` below, so this call was pure dead weight.)
    # NOTE: a former ``slices = find_objects(saturated)`` lived here but its
    # result was never used (per-source windows are derived from ``src['com']``
    # below).  Newer scipy rejects the boolean ``saturated`` array
    # (find_objects needs an int label image), raising
    # "'numpy.bool' object cannot be interpreted as an integer" and aborting
    # every frame.  The dead call is removed.

    if size is None:
        size = pad

    index = 0
    print(f"Found {nsource} saturated sources to process", flush=True)
    # Per-component bounding boxes (one pass over the label image) -> a STABLE
    # sky anchor per saturated component.  The fitted satstar position of a large
    # component (extended-emission core) scatters by ~r_sat across exposures, so
    # the consolidation dedup cannot tell "one blob seen N times" from "N stars";
    # the component bbox-centre barely moves frame-to-frame, so tagging each
    # per-exposure satstar with it lets the dedup merge same-component detections
    # regardless of the (unstable) fitted flux/position.
    _comp_slices = (ndimage.find_objects(sources)
                    if sources is not None and getattr(sources, 'size', 0) else [])
    for ii, src in enumerate(source_records):
        # get the center of pixels with this label

        com = src['com']
        src_label = src['label']
        forced_source = src['forced']
        # defaults (overridden in the in-FOV fit branch); guard later references
        _use_large_infov = False
        _infov_psf = big_grid
        _psf_for_fit = big_grid
        #center_of_mass(saturated, labels=sources, index=ii+1)
        # center_of_mass can return (nan, nan) for degenerate labels; guard against that
        if com is None:
            print(f"Source {ii+1}: center_of_mass returned None; skipping", flush=True)
            continue
        yf, xf = com
        if not (np.isfinite(yf) and np.isfinite(xf)):
            print(f"Source {ii+1}: center_of_mass returned NaN or infinite values ({yf}, {xf}); skipping", flush=True)
            continue
        ycen = int(round(yf))
        xcen = int(round(xf))
        print(f"Source {ii+1}: center at (x, y) = ({xcen}, {ycen}), forced={forced_source}")

        if forced_source:
            # Cross-frame reconciliation may have flagged this off-field star as
            # unfittable here (no detector saw its spikes -> any flux is garbage).
            # Skip it entirely so it contributes no (fake-background) model.
            if flux_drops:
                try:
                    _wpos_drop = ww.pixel_to_world(xcen, ycen)
                    _dropped = any(
                        _wpos_drop.separation(_dsc).arcsec < 1.5 for _dsc in flux_drops)
                except Exception:
                    _dropped = False
                if _dropped:
                    print(f"Source {ii+1}: out-of-field forced source at "
                          f"(x={xcen},y={ycen}) DROPPED by cross-frame reconciliation "
                          f"(no detector diversity -> no trustworthy flux; "
                          f"unfittable here, skipping to avoid fake-background model).",
                          flush=True)
                    continue
            # DROP forced satstars the deep coadd does NOT cover (user choice
            # 2026-06-19).  An off-FOV forced star whose TRUE (seed) position
            # falls outside the joint coadd footprint has NO data anywhere to
            # constrain its flux: the LSQ locks onto edge scattered light and
            # returns a ~1e9 runaway whose wings gouge deep negative pits at the
            # tile edge (sickle joint o001+o002: 25 such at 5-7e9 -> the -70k
            # pit).  With no coadd core to cap against, the only correct action is
            # NOT to fit it (contribute no fake-background model); the mild real
            # scattered light it leaves is an honest POSITIVE edge residual, not a
            # deep pit.  Stars the coadd DOES cover (in-FOV in some frame, e.g.
            # star A) map onto finite coadd data and are KEPT (then flux-capped to
            # the coadd core downstream).  MIRI only; needs the deep coadd.
            if (_is_miri and seed_gate_image is not None
                    and seed_gate_wcs is not None):
                try:
                    _scov = ww.pixel_to_world(xcen, ycen)
                    _cxg, _cyg = seed_gate_wcs.world_to_pixel(_scov)
                    _cxi, _cyi = int(round(float(_cxg))), int(round(float(_cyg)))
                    _gny2, _gnx2 = seed_gate_image.shape
                    _covered = False
                    if 0 <= _cxi < _gnx2 and 0 <= _cyi < _gny2:
                        _yl, _yh = max(0, _cyi - 3), min(_gny2, _cyi + 4)
                        _xl, _xh = max(0, _cxi - 3), min(_gnx2, _cxi + 4)
                        _cw = seed_gate_image[_yl:_yh, _xl:_xh]
                        _covered = bool((np.isfinite(_cw) & (_cw != 0)).any())
                    if not _covered:
                        print(f"Source {ii+1}: forced outside-FOV star at "
                              f"(x={xcen},y={ycen}) maps OUTSIDE the deep coadd "
                              f"footprint -> DROPPED (no data to constrain flux; "
                              f"avoids ~1e9 edge runaway / deep pit).", flush=True)
                        continue
                except Exception:
                    pass
            y0, y1 = _nearest_window_bounds(ycen, data.shape[0], outside_star_fit_box)
            x0, x1 = _nearest_window_bounds(xcen, data.shape[1], outside_star_fit_box)
            size_saturated = max(5, int(3 * fwhm_pix))
        else:
            y0 = int(max(0, ycen - pad))
            y1 = int(min(data.shape[0], ycen + pad))
            x0 = int(max(0, xcen - pad))
            x1 = int(min(data.shape[1], xcen + pad))
            size_saturated = int(np.sqrt(sum_labels(saturated, labels=sources, index=src_label))/2)

        # area_saturated = sum_labels(saturated, labels=sources, index=ii+1)
        # Fit each source against ``data_working`` rather than the raw
        # ``data`` — earlier accepted fits' PSF models have been
        # subtracted from ``data_working`` already, so this fit sees a
        # field with brighter neighbour satstars removed.
        cutout = data_working[y0:y1, x0:x1]
        err_cutout = err_working[y0:y1, x0:x1]
        init_params = QTable()
        # For outside-FOV forced sources the star center may be outside the
        # cutout bounds (xcen < x0 or xcen > x1).  DO NOT clip to [0, width-1]:
        # clipping forces both seeds to (0,0), they fit the same corner bright
        # source, and produce duplicate catalog rows with the wrong position.
        # Allow the unclipped offset so the PSF is correctly initialised at its
        # true (possibly negative) position in cutout coordinates.
        if forced_source:
            x_init = float(xcen - x0)
            y_init = float(ycen - y0)
        else:
            x_init = float(np.clip(xcen - x0, 0, max(0, cutout.shape[1] - 1)))
            y_init = float(np.clip(ycen - y0, 0, max(0, cutout.shape[0] - 1)))
        init_params['x'] = [x_init]
        init_params['y'] = [y_init]
        # PSFPhotometry derives flux_init from aperture photometry by default.
        # For outside-FOV forced sources the aperture circle is mostly off-image
        # → 0 pixels → flux_init = NaN → LevMarLSQ starts from NaN → flux_fit = NaN.
        # Provide an explicit large positive flux_init so the fitter has a real
        # starting point and can scale down to the right amplitude.  Saturated
        # JWST stars are typically 1e6-1e9 counts; 1e7 is a reasonable seed.
        if forced_source:
            init_params['flux'] = [1.0e7]
        cutout[np.isnan(cutout)] = 0.0
        # if isinstance(grid, list):
        #     print(f"Grid is a list: {grid}")
        #     psf_model = WrappedPSFModel(grid[0])
        #     dao_psf_model = grid[0]
        # else:

        #psf_model = WrappedPSFModel(grid, stampsz=(size,size))

        # Compute sat_area once; used by both adaptive mask buffer and adaptive bkg annulus.
        if not forced_source:
            src_sat_area = int(sum_labels(saturated, labels=sources, index=src_label))
            # stable component anchor = bbox centre of this labeled component
            _sl = (_comp_slices[src_label - 1]
                   if 0 < src_label <= len(_comp_slices) else None)
            if _sl is not None and _sl[0] is not None:
                src_comp_cy = 0.5 * (_sl[0].start + _sl[0].stop - 1)
                src_comp_cx = 0.5 * (_sl[1].start + _sl[1].stop - 1)
            else:
                src_comp_cy = src_comp_cx = np.nan
        else:
            src_sat_area = None
            src_comp_cy = src_comp_cx = np.nan

        # Brightness-dependent mask dilation: scale buffer with saturated area
        # so deeply saturated sources exclude more of the non-linear fringe.
        # MIRI: the nonlinear/charge-bleed fringe around a saturated core is
        # WIDER than NIRCam (broad PSF, more charge spread).  The NIRCam-tuned
        # buffer (scale 0.4, cap 6) under-masks it, leaving corrupt near-core
        # pixels that drive the wing fit to overshoot (ssr_ratio~230, nan
        # flux_err on the F770W hand-selected saturated stars).  Mask farther
        # out for MIRI so only trustworthy outer wings set the amplitude.
        if adaptive_mask_buffer_scale and src_sat_area is not None:
            if _is_miri:
                effective_buffer = compute_adaptive_mask_buffer(
                    src_sat_area, mask_buffer_min=max(mask_buffer, 4),
                    cap=12, scale=0.8)
            else:
                effective_buffer = compute_adaptive_mask_buffer(
                    src_sat_area, mask_buffer_min=mask_buffer)
        else:
            effective_buffer = mask_buffer

        # Brightness-dependent background annulus: wider for brighter sources to
        # avoid PSF-wing contamination of the background estimate.
        if adaptive_bkg_annulus and src_sat_area is not None:
            bkg_inner, bkg_outer = compute_adaptive_bkg_annulus(src_sat_area)
        else:
            bkg_inner, bkg_outer = 25, 50   # fixed wide annulus as fallback

        # For forced outside-FOV sources the star center may be hundreds of
        # pixels outside the image.  A LocalBackground annulus of 25–50 px
        # centred there contains zero image pixels, causing the estimator to
        # fail silently.  Skip local background for those sources; the frames
        # are already bgsub'd so a zero-background assumption is appropriate.
        if forced_source:
            localbkg_estimator = None
            print(f"  forced source: no local-bkg (source is off-edge)  "
                  f"(sat_area=forced)", flush=True)
        else:
            print(f"  mask_buffer={effective_buffer}  bkg=({bkg_inner},{bkg_outer})"
                  f"  (sat_area={src_sat_area})", flush=True)
            localbkg_estimator = LocalBackground(bkg_inner, bkg_outer)

        # Use the large-FOV PSF grid for forced (outside-FOV) sources so the
        # diffraction spike pattern (extending ~40" into the FOV) is actually
        # represented in the model.  Default 512-px grid (256 px radius) is
        # far too small for stars 200-600 px off-edge.
        # with a zeroframe anchor the recovered rim holds real data -- only
        # the group-0-saturated deep core is unmeasurable and masked.
        _sat_for_mask = zf_deep_core if zf_deep_core is not None else saturated
        saturated_mask = _sat_for_mask[y0:y1, x0:x1]
        # Only dilate the CURRENT source's saturated component.  Other
        # saturated stars in the same cutout were fit earlier (sorted by
        # sat_area desc) and their PSF models were subtracted from
        # ``data_working`` — so their wing flux has been removed.  If we
        # kept dilating those neighbours' saturated pixels by
        # ``effective_buffer`` we'd shield the fit from the cleaned wing
        # region, defeating iterative subtraction.  Their raw saturated
        # cores still need to be masked (subtracted data there is wrong:
        # the data was the saturated-clipped value, not the true peak).
        if not forced_source and src_label is not None:
            this_source_sat = (sources[y0:y1, x0:x1] == src_label) & saturated_mask
            this_source_sat_expanded = binary_dilation(
                this_source_sat, iterations=effective_buffer)
            other_sources_sat = saturated_mask & (~this_source_sat)
            satmask_combined = this_source_sat_expanded | other_sources_sat
            # effective circular-equivalent mask radius of THIS source's core
            # (drives the wing self-calibration: the wing-fit bias grows with
            # how much core is masked -- see _wing_selfcal)
            _wingcal_rmask = float(np.sqrt(
                max(int(this_source_sat_expanded.sum()), 1) / np.pi))
        else:
            # Forced sources or no src_label: fall back to dilating the
            # whole saturated mask (legacy behaviour).
            satmask_combined = binary_dilation(saturated_mask,
                                               iterations=effective_buffer)
            _wingcal_rmask = np.nan   # forced/off-FOV: no wing self-cal
        # NB: np.logical_or takes only TWO array operands; a 3rd positional arg
        # is interpreted as ``out``.  The previous
        # ``np.logical_or(cutout==0, np.isnan(cutout), satmask_combined)`` therefore
        # silently DISCARDED satmask_combined (used it as the output buffer), so
        # the saturated core pixels were never masked -- the fit matched their
        # clipped-low values and badly UNDER-fit the amplitude of saturated stars
        # (half-subtracted residuals).  OR all three explicitly.
        mask = (cutout == 0) | np.isnan(cutout) | satmask_combined

        # Extra down-weighting of pixels NEAR the saturated core.  Physical
        # model (user, 2026-06-09): the central lobe is SUPPRESSED near
        # saturation (charge overflow / nonlinearity), so the bright near-core
        # pixels read systematically LOW.  Inverse-variance weighting alone
        # still lets them constrain the amplitude (their formal ERR is modest),
        # and because they sit low the fit raises the amplitude to compensate,
        # over-subtracting the (well-validated) faint SECOND sidelobes.  Ramp an
        # extra weight from ~0 at the saturated-mask edge up to 1 by a few px
        # out (Gaussian in distance-to-saturation, scale =
        # satstar_central_downweight_sigma px) and fold it into the per-pixel
        # error: err_eff = err / sqrt(w_prox).  Only reshapes weights among
        # UNMASKED pixels, so the trustworthy outer wings / sidelobes set the
        # amplitude instead of the suppressed inner lobe.  Sidelobes are NOT
        # down-weighted (they are far from the saturated pixels).
        #
        # DEFAULT OFF (sigma=0): empirically (sickle pillar satstars, 2026-06-09)
        # this made the second-sidelobe oversubtraction WORSE (-9 -> -30 MJy/sr).
        # The suppressed near-core pixels read LOW and were holding the fitted
        # amplitude DOWN; down-weighting them let the amplitude rise to match the
        # wings, overshooting the second sidelobe more.  Kept as a tunable knob,
        # but inverse-variance weighting alone (above) is the win.
        err_cutout_eff = err_cutout
        if satstar_central_downweight_sigma and satstar_central_downweight_sigma > 0:
            dist_to_sat = ndimage.distance_transform_edt(~satmask_combined)
            _sig = float(satstar_central_downweight_sigma)
            w_prox = 1.0 - np.exp(-(dist_to_sat ** 2) / (2.0 * _sig ** 2))
            w_prox = np.clip(w_prox, 1e-3, 1.0)
            err_cutout_eff = err_cutout / np.sqrt(w_prox)

        # MIRI 2D LOCAL BACKGROUND (2026-06-14).  MIRI-ONLY -- NIRCam backgrounds
        # are always low.  The scalar-annulus ``LocalBackground`` cannot remove
        # MIRI's EXTENDED EMISSION, so the masked-core wing fit sits on the
        # emission and inflates the satstar amplitude ~50x (ssr_ratio median ~45;
        # residual pits to -156000 MJy/sr).  Subtract a masked 2D-median-filter
        # local background BEFORE the fit so the wings see a flat field.  The bg
        # is removed only for the FIT (the star model alone is later subtracted
        # from data_working); the emission stays in the data.  Experiment
        # (scripts/miri_reduction/experiment_satstar_bottomup.py): this drops
        # ssr 45->~1 and the amplitude 5.2e7->~1e6, leaving only mild (~-130)
        # saturated-core residuals.
        cutout_fit = cutout
        bg2d = None
        if _is_miri and not forced_source:
            _srcmask = satmask_combined | (~np.isfinite(cutout)) | (cutout == 0)
            _fillval = (float(np.nanmedian(cutout[~_srcmask]))
                        if (~_srcmask).any() else 0.0)
            _filled = np.where(_srcmask, _fillval, cutout)
            _bgbox = int(max(15, round(6 * fwhm_pix)))
            bg2d = ndimage.median_filter(_filled, size=_bgbox)
            cutout_fit = cutout - bg2d
            localbkg_estimator = None  # 2D bg already removed; don't double-subtract

        if forced_source:
            # Custom 3-parameter (x, y, flux) fit for off-edge sources.
            # photutils PSFPhotometry's fit_shape is centered at source
            # position and clipped to image bounds, which gives 0 fit
            # pixels for stars whose seed is far off-edge — so it always
            # returns NaN.  Instead we evaluate the (large) PSF at every
            # cutout pixel for each candidate (dx, dy) sub-pixel offset
            # in [-FORCED_SHIFT_RADIUS, +FORCED_SHIFT_RADIUS] (integer
            # 1-pixel grid), do a linear-LSQ amplitude fit at each, and
            # pick the (dx, dy, flux) triple that minimises chi^2.
            # Iterative 3-sigma clipping then refines the amplitude to
            # suppress contributions from bright neighbour stars and
            # other unmodeled structure (otherwise the over-subtraction
            # observed empirically pulls the fitted flux too high).
            if big_grid_large is None:
                raise RuntimeError(
                    "forced_source=True but big_grid_large is None — "
                    "earlier require-large-PSF check should have raised."
                )
            FORCED_SHIFT_RADIUS = int(forced_grid_search_radius)  # pixels, per-axis (0 = single-point flux-only fit at seed)
            FORCED_SIGMA_CLIP = 3.0
            FORCED_CLIP_ITERS = 3
            cy_, cx_ = cutout.shape
            yy, xx = np.mgrid[0:cy_, 0:cx_]

            # Background sigma from cutout edges — used for chi^2 metric
            # and for sigma-clipping.  Evaluate PSF at seed once just to
            # define the "background" region (where PSF is negligible).
            psf_center = big_grid_large(xx - x_init, yy - y_init)
            if psf_center.shape != cutout.shape:
                raise RuntimeError(
                    f"PSF eval shape {psf_center.shape} != cutout shape "
                    f"{cutout.shape}; PSF model is misconfigured."
                )
            psf_thresh = max(1e-10, float(np.nanmax(psf_center)) * 1e-6)
            # PSF significance of the in-frame footprint: brightest in-frame PSF
            # value vs the model's global peak.  For an off-FOV seed only the
            # faint outer tail clips the frame, so this ratio is tiny; a seed
            # whose bright core is in-frame gives a ratio near 1.  Used below to
            # tell negligible off-FOV overlap (skip) from a real in-frame failure.
            _psf_global_peak = float(np.nanmax(np.asarray(big_grid_large.data)))
            _inframe_peak = float(np.nanmax(psf_center))
            _inframe_psf_frac = (_inframe_peak / _psf_global_peak
                                 if _psf_global_peak > 0 else 0.0)
            bg_mask = (~mask) & np.isfinite(cutout) & (psf_center <= psf_thresh)
            if int(bg_mask.sum()) >= 100:
                bg_pix = cutout[bg_mask].astype(float)
                sigma = float(1.4826 * np.median(np.abs(bg_pix - np.median(bg_pix))))
            else:
                # No room for a background sample — pixels mostly in PSF
                # support.  Fall back to MAD of all unmasked cutout
                # pixels (less robust but defined).
                all_pix = cutout[~mask & np.isfinite(cutout)].astype(float)
                if all_pix.size < 10:
                    raise ValueError(
                        f"Forced source {ii+1}: <10 unmasked finite cutout "
                        f"pixels; cannot estimate background sigma."
                    )
                sigma = float(1.4826 * np.median(np.abs(all_pix - np.median(all_pix))))
            sigma = max(sigma, 1e-30)

            # Cross-frame reconciliation: if this out-of-field star's seed sky
            # position matches a supplied override (the nearest-detector flux
            # measured in a frame that caught its diffraction spikes), use that
            # fixed flux at the seed position instead of fitting the spike-less
            # corner this frame sees.  Rendering with the (large) PSF then gives
            # the correct wing amplitude here (or ~0 where the PSF doesn't reach).
            # Match tolerance must cover the CROSS-FRAME CENTROID SCATTER of a
            # saturated off-FOV star: each frame fits the spike-less corner and
            # the forced positions wander ~1.2" across the whole group, while the
            # override is keyed at ONE position.  The old 0.2" tolerance was
            # tighter than that scatter, so the worst (runaway) frame -- the one
            # MOST in need of the override -- was the farthest from it and never
            # matched, leaving its 1e9-1e11 flux -> fake-background model.  The
            # reconcile now keys at the GROUP CENTROID (max key-to-member ~half
            # the group diameter, ~0.6"), plus the seed-vs-fit offset (~0.3");
            # 1.5" gives margin and is still far below the spacing of (sparse)
            # off-FOV saturated stars, so there is no risk of grabbing a
            # different star.
            _ovr_flux = None
            if flux_overrides:
                try:
                    _wpos = ww.pixel_to_world(xcen, ycen)
                    for _osc, _of in flux_overrides:
                        if np.isfinite(_of) and _wpos.separation(_osc).arcsec < 1.5:
                            _ovr_flux = float(_of)
                            break
                except Exception:
                    _ovr_flux = None

            best = None   # (chi2, dx, dy, flux, flux_err, n_pix, qfit, red_chi2, resid_at_d, p_vec_at_d)
            # Track the largest number of IN-FRAME pixels where the tabulated PSF
            # rises above threshold (ignoring the data mask).  This is the PSF's
            # real reach: the stamp half-size overestimates it because the PSF
            # model is ~0 over its outer support.  Used below to tell a genuine
            # off-FOV non-overlap (PSF support never reaches the frame -> skip)
            # from a real in-frame data failure (PSF reaches but data unusable).
            max_psf_inframe = 0
            for dy in range(-FORCED_SHIFT_RADIUS, FORCED_SHIFT_RADIUS + 1):
                for dx in range(-FORCED_SHIFT_RADIUS, FORCED_SHIFT_RADIUS + 1):
                    psf_try = big_grid_large(xx - (x_init + dx),
                                             yy - (y_init + dy))
                    psf_hot = psf_try > psf_thresh
                    max_psf_inframe = max(max_psf_inframe, int(psf_hot.sum()))
                    usable = (~mask) & np.isfinite(cutout) & psf_hot
                    n_use = int(usable.sum())
                    if n_use < 10:
                        continue
                    d_vec = cutout[usable].astype(float)
                    p_vec = psf_try[usable].astype(float)
                    denom = float(np.sum(p_vec * p_vec))
                    if denom <= 0:
                        continue
                    flux_try = float(np.sum(d_vec * p_vec) / denom)
                    # Iterative 3-sigma clip to suppress bright neighbours
                    # and unmodeled flux from biasing the amplitude high.
                    keep = np.ones_like(d_vec, dtype=bool)
                    for _ in range(FORCED_CLIP_ITERS):
                        resid_try = d_vec - flux_try * p_vec
                        new_keep = np.abs(resid_try) < FORCED_SIGMA_CLIP * sigma
                        new_keep &= keep
                        if int(new_keep.sum()) < 100 or np.array_equal(new_keep, keep):
                            keep = new_keep
                            break
                        keep = new_keep
                        d_k = d_vec[keep]; p_k = p_vec[keep]
                        denom_k = float(np.sum(p_k * p_k))
                        if denom_k <= 0:
                            break
                        flux_try = float(np.sum(d_k * p_k) / denom_k)
                    # Final residual + chi^2 on the unclipped pixel set
                    resid_try = d_vec - flux_try * p_vec
                    chi2_try = float(np.sum((resid_try / sigma) ** 2))
                    red_chi2 = chi2_try / max(n_use - 1, 1)
                    qfit_try = (float(np.sum(np.abs(resid_try)))
                                / max(abs(flux_try), 1e-30))
                    flux_err_try = sigma / np.sqrt(denom)
                    if best is None or chi2_try < best[0]:
                        best = (chi2_try, dx, dy, flux_try, flux_err_try,
                                n_use, qfit_try, red_chi2,
                                float(resid_try[0]))

            # Apply cross-frame override: pin flux to the nearest-detector value
            # at the seed position (dx=dy=0).  The grid search above this frame
            # would otherwise lock onto the spike-less corner and mis-fit; here
            # we keep the geometry (PSF at the seed) but force the trusted flux,
            # so the rendered wings carry the correct amplitude into this frame.
            if _ovr_flux is not None:
                psf_seed = big_grid_large(xx - x_init, yy - y_init)
                usable = ((~mask) & np.isfinite(cutout) & (psf_seed > psf_thresh))
                n_use = int(usable.sum())
                if n_use >= 10:
                    d_vec = cutout[usable].astype(float)
                    p_vec = psf_seed[usable].astype(float)
                    denom = float(np.sum(p_vec * p_vec))
                    if denom > 0:
                        resid_ovr = d_vec - _ovr_flux * p_vec
                        chi2_ovr = float(np.sum((resid_ovr / sigma) ** 2))
                        red_chi2_ovr = chi2_ovr / max(n_use - 1, 1)
                        qfit_ovr = (float(np.sum(np.abs(resid_ovr)))
                                    / max(abs(_ovr_flux), 1e-30))
                        flux_err_ovr = sigma / np.sqrt(denom)
                        best = (chi2_ovr, 0, 0, float(_ovr_flux), flux_err_ovr,
                                n_use, qfit_ovr, red_chi2_ovr,
                                float(resid_ovr[0]))
                        print(f"  Forced-source flux OVERRIDDEN by cross-frame "
                              f"reconciliation: flux={_ovr_flux:.3e} (n_pix={n_use})",
                              flush=True)

            if best is None:
                # Force-fit at this seed position produced no candidate (dx,dy)
                # with enough usable pixels.  Two cases, distinguished by whether
                # the tabulated PSF actually has support reaching into the frame:
                #   1. NON-OVERLAP: the PSF model is ~0 over the in-frame region
                #      (max_psf_inframe < 10 above-threshold pixels at ANY offset).
                #      The geometric stamp half-size overestimates the PSF reach
                #      (the model is effectively zero over its outer support), so
                #      a seed can sit "within the stamp" yet contribute nothing.
                #      Nothing to fit, nothing to subtract -- skip cleanly.
                #   2. DATA FAILURE: the PSF DOES have >=10 above-threshold pixels
                #      in-frame, but the data there are all masked/non-finite, so
                #      no usable pixels remain.  A real problem (corrupt data, bad
                #      mask) -- raise.
                _ny_frame, _nx_frame = data.shape
                _dx_out = max(0.0, -x_init, x_init - (_nx_frame - 1))
                _dy_out = max(0.0, -y_init, y_init - (_ny_frame - 1))
                _dist_outside = float(np.hypot(_dx_out, _dy_out))
                # Negligible-overlap (skip) vs real in-frame failure (raise).
                # Skip when EITHER the PSF support barely reaches the frame
                # (<10 above-threshold pixels) OR the in-frame footprint is only
                # the faint outer tail (in-frame peak < 0.1% of the PSF peak):
                # such an off-FOV seed contributes nothing fittable here even if
                # its tail technically clips a few (possibly masked) pixels.
                _SIG_FRAC = 1e-3
                if max_psf_inframe < 10 or _inframe_psf_frac < _SIG_FRAC:
                    print(f"  Forced source {ii+1} at (x={x_init:.1f},y={y_init:.1f}): "
                          f"in-frame PSF is faint tail only "
                          f"(peak {_inframe_psf_frac:.1e} of global, "
                          f"{max_psf_inframe} px above threshold; seed {_dist_outside:.0f} "
                          f"px outside the {_nx_frame}x{_ny_frame} frame); negligible "
                          f"overlap -- skipping.", flush=True)
                    continue
                raise ValueError(
                    f"Forced source {ii+1} at (x={x_init:.1f},y={y_init:.1f}): "
                    f"in-frame PSF reaches {_inframe_psf_frac:.1e} of the global peak "
                    f"({max_psf_inframe} px above threshold) but no candidate (dx,dy) in "
                    f"+/-{FORCED_SHIFT_RADIUS} px gave >= 10 USABLE (unmasked, finite) "
                    f"cutout pixels.  Seed is {_dist_outside:.0f} px outside the "
                    f"{_nx_frame}x{_ny_frame} frame; a SIGNIFICANT part of the PSF "
                    f"overlaps the data region, so the data there are all masked/"
                    f"non-finite -- a real fit failure, not an off-FOV non-overlap."
                )

            (chi2_val, best_dx, best_dy, flux_fit_val, flux_err_val,
             n_usable, qfit_val, red_chi2_val, resid0_val) = best
            x_fit_val = x_init + best_dx
            y_fit_val = y_init + best_dy

            # ---- INDEPENDENT over-subtraction check (forced/off-FOV sources) ----
            # The amplitude was fit from the data above.  SEPARATELY (this is a
            # post-fit validation, NOT a constraint folded into the fit), verify
            # the rendered model does not exceed the data in its significant
            # (>5 sigma) footprint.  An off-FOV star whose bright core never
            # enters the frame has an amplitude only weakly constrained by the
            # faint outer wings, so it can run away (1e8-1e11) and gouge a deep
            # negative pit (model >> data).  For a CORRECT fit the model is the
            # star's contribution, which is <= the data (data = star + bg +
            # noise), so data/model >= ~1 and nothing is clamped.  Where the
            # model over-subtracts, scale the amplitude DOWN to the largest value
            # that keeps model <= data across the footprint (robust 10th-pct so a
            # few noisy pixels can't zero it).  Under-subtracting slightly (some
            # residual contamination) is preferred to a worse over-subtraction.
            # Logged, never silent.  ``oversub_clamp_scale`` is recorded so the
            # check is auditable.  (The position prior from Spitzer + the spike-
            # constrained grid search should make this rarely trigger; it is the
            # safety net for the residual unconstrained-amplitude cases.)
            _oversub_scale = 1.0
            if forced_source and np.isfinite(flux_fit_val) and flux_fit_val > 0:
                _pmod = big_grid_large(xx - x_fit_val, yy - y_fit_val) * flux_fit_val
                _foot = (~mask) & np.isfinite(cutout) & (_pmod > 5.0 * sigma)
                _nf = int(_foot.sum())
                if _nf >= 10:
                    _ratio = cutout[_foot].astype(float) / _pmod[_foot]  # data/model
                    _frac_over = float(np.mean(_ratio < 1.0))
                    # ``oversub_clamp_percentile`` controls how strictly model<=data
                    # is enforced over the >5sigma footprint: the scale is the Nth
                    # percentile of data/model, so a smaller N forces model<=data on
                    # a LARGER fraction of pixels (N=10 -> 90% of pixels; N=1 -> 99%;
                    # N=0 -> every pixel).  Single-detector LW off-FOV stars over a
                    # BRIGHT background (e.g. F335M at 3.35um) have no spike leverage,
                    # so every per-frame fit over-estimates and the weak 10th-pct
                    # ceiling still leaves deep negative spike residuals; a smaller
                    # percentile under-subtracts slightly instead (preferred).
                    _pct = float(np.clip(oversub_clamp_percentile, 0.0, 50.0))
                    _scale = float(np.percentile(_ratio, _pct))
                    if _scale < 1.0:
                        _oversub_scale = max(_scale, 0.0)
                        print(f"  OVER-SUB CHECK (forced src {ii+1}): model>data on "
                              f"{100*_frac_over:.0f}% of {_nf}-px >5sig footprint; "
                              f"clamping flux {flux_fit_val:.3e} -> "
                              f"{flux_fit_val*_oversub_scale:.3e} (scale="
                              f"{_oversub_scale:.3f}, pct={_pct:.1f}) so model<=data", flush=True)
                        flux_fit_val = flux_fit_val * _oversub_scale
                        flux_err_val = flux_err_val * _oversub_scale

            print(f"  Custom forced-source fit: flux={flux_fit_val:.3e}  "
                  f"flux_err={flux_err_val:.3e}  n_pix={n_usable}  "
                  f"sigma_bg={sigma:.3e}  "
                  f"(dx,dy)=({best_dx},{best_dy})  qfit={qfit_val:.3f}  "
                  f"reduced_chi2={red_chi2_val:.3f}", flush=True)
            result = QTable()
            result['id'] = [1]
            result['group_id'] = [1]
            result['group_size'] = [1]
            result['local_bkg'] = [0.0]
            result['x_init'] = [x_init]
            result['y_init'] = [y_init]
            result['flux_init'] = [float(init_params['flux'][0])]
            result['x_fit'] = [x_fit_val]
            result['y_fit'] = [y_fit_val]
            result['flux_fit'] = [flux_fit_val]
            result['x_err'] = [0.0]   # grid search; no formal positional error
            result['y_err'] = [0.0]
            result['flux_err'] = [flux_err_val]
            result['n_pixels_fit'] = [n_usable]
            result['qfit'] = [qfit_val]
            result['cfit'] = [resid0_val]
            result['reduced_chi2'] = [red_chi2_val]
            result['flags'] = [0]
            bad_fit_rows = np.zeros(1, dtype=bool)
        else:
            # MIRI in-FOV saturated stars: fit the LARGE (spike-resolving) PSF so
            # the unsaturated diffraction spikes pin the amplitude (the small grid
            # omits them -> masked-core LSQ under-fits).  Fall back to small grid.
            # ``_infov_psf`` is reused for the envelope + render so the flux stays
            # in ONE PSF's normalization (big_grid vs big_grid_large differ).
            # ONLY heavily-saturated stars (large DQ-SATURATED core) use the
            # large-PSF / uncapped path: their amplitude is set by the unsaturated
            # spikes and the coadd core is clipped/diluted (sickle A=1572, B=478,
            # the bright pillar star=8310 sat px).  MILDLY-saturated stars (C=38,
            # D=28) have a finite, trustworthy coadd core -- the small PSF + coadd-
            # core cap fits them well; the large PSF OVER-predicts them.  Threshold
            # 200 cleanly separates (A/B/bright >>200, C/D <40).
            _SAT_AREA_LARGE = 200
            _use_large_infov = (_is_miri and (not forced_source)
                                and big_grid_large is not None
                                and src_sat_area is not None
                                and int(src_sat_area) >= _SAT_AREA_LARGE)
            _infov_psf = big_grid_large if _use_large_infov else big_grid
            _psf_for_fit = _infov_psf
            psfphot = PSFPhotometry(
                                    localbkg_estimator=localbkg_estimator,
                                    fitter=lmfitter,
                                    psf_model=_psf_for_fit,
                                    fit_shape=size,
                                    aperture_radius=15*fwhm_pix)
            # get the underlying model
            model = getattr(psfphot, "psf_model", None)
            if model is None:
                raise RuntimeError("psfphot.psf_model is None — can't set parameter bounds")

            # MIRI bounded fit (2026-06-27): reuse NIRCam's bounded-position
            # infrastructure for MIRI in-FOV satstars instead of hard-locking.
            # The DQ-mask eroded-core seed is NOT always ~0.05" accurate: for
            # an ASYMMETRIC saturated cluster (diffraction spike/merge) it lands
            # ~0.7" (~6px) off the true centre (brick F2550W o002), and the
            # locked PSF then subtracts at the wrong place -> mispositioned +
            # UNDER-subtracted satstar, and the per-frame seed scatter renders a
            # ~1.3" HEX of duplicate models.  A bounded fit lets the unsaturated
            # diffraction-spike wings pull x_0/y_0 to the true centre.
            # CRITICAL bound choice: NIRCam uses max(size_saturated, 1.5*FWHM),
            # but size_saturated = int(sqrt(sat_area)/2) is exactly the term that
            # drove the historical MIRI drift (~1.5-2" on heavily-saturated star
            # A).  So MIRI uses ONLY the 1.5*FWHM term -- the value NIRCam cites
            # for a non-singular covariance / finite flux_err -- which for
            # F2550W (~1.2") still covers the ~0.7" seed bias without the
            # core-size runaway.  Gated by MIRI_SATSTAR_BOUNDED_FIT (default 1;
            # set 0 to restore the legacy hard lock).
            _miri_bounded = (_is_miri
                             and int(os.environ.get('MIRI_SATSTAR_BOUNDED_FIT', 1)))
            if _is_miri and not _miri_bounded:
                # legacy hard lock (MIRI_SATSTAR_BOUNDED_FIT=0)
                model.x_0.fixed = True
                model.y_0.fixed = True
                print(f"MIRI: locked satstar position to seed "
                      f"(x={xcen}, y={ycen}); fitting flux only")
            else:
                # bounded position fit (>=1.5 FWHM keeps covariance
                # non-singular; tighter pegs x_0/y_0 -> NaN flux_err -> good fits
                # tossed, the dominant gc2211 baseline rejection mode).  MIRI
                # drops the size_saturated term (see above); NIRCam keeps it.
                #
                # NIRCAM extended-emission tight bound (user 2026-07-06): on a
                # bright saturated core embedded in nebulosity, size_saturated =
                # int(sqrt(sat_area)/2) is HUGE (blob sat_area~600 -> 12px =
                # 0.76") so the fit slides that far per frame across the NaN-
                # masked core.  The per-frame subtracted satstar model then lands
                # at 8 different places and the COADD is a multi-peak smear ->
                # oversubtraction crater (user: "modeled as a cluster") even after
                # the catalog/consolidation dedup collapse it to one row.  This is
                # the SAME failure MIRI fixed by dropping size_saturated; the
                # DQ-component seed is stable to ~0.1" here, so the 1.5*FWHM bound
                # is enough.  Gated by NIRCAM_SATSTAR_TIGHT_BOUND (the driver sets
                # it for extended-emission NIRCam fields); default preserves the
                # size_saturated bound for star-dominated fields.
                _nc_tight = (not _is_miri
                             and int(os.environ.get('NIRCAM_SATSTAR_TIGHT_BOUND', 0)))
                if _is_miri or _nc_tight:
                    pos_bound = 1.5 * fwhm_pix
                else:
                    pos_bound = max(size_saturated, 1.5 * fwhm_pix)
                low_x  = xcen - x0 - pos_bound
                high_x = xcen - x0 + pos_bound
                low_y  = ycen - y0 - pos_bound
                high_y = ycen - y0 + pos_bound
                for pname, bounds in (("x_0", (low_x, high_x)), ("y_0", (low_y, high_y))):
                    if not hasattr(model, pname):
                        raise AttributeError(
                            f"PSF model has no parameter '{pname}'; "
                            f"param_names={getattr(model, 'param_names', None)}"
                        )
                    param = getattr(model, pname)
                    # Set bounds via the supported astropy.modeling API.  If
                    # this ever raises, the photutils/astropy contract has
                    # changed and we want to know — silently falling back to
                    # ``_bounds`` could leave a fitter completely unbounded
                    # and corrupt every subsequent forced photometry result.
                    param.bounds = bounds
                    print(f"Set {pname}.bounds = {bounds}")

            result = psfphot(cutout_fit, init_params=init_params, mask=mask,
                             error=err_cutout_eff)

            if len(result) == 0:
                # Empty result is a real fit failure for an existing
                # in-FOV saturated source — do not silently skip; raise
                # so the cause gets investigated.
                raise ValueError(
                    f"PSF photometry returned 0 rows for source {ii+1} at "
                    f"(x={xcen},y={ycen}).  Empty result means LSQ did not "
                    f"converge for an in-FOV source."
                )

            if hasattr(result['x_fit'], 'mask'):
                bad_fit_rows = result['x_fit'].mask | result['y_fit'].mask
            else:
                bad_fit_rows = (~np.isfinite(result['x_fit'])) | (~np.isfinite(result['y_fit']))

        # (Forced sources construct ``result`` + ``bad_fit_rows`` directly
        #  in the custom-fit branch above; no masked-x_fit recovery needed.)

        if np.any(bad_fit_rows):
            print(f"Removing {bad_fit_rows.sum()} invalid fit rows for source {ii+1}", flush=True)
            result = result[~bad_fit_rows]

        if len(result) == 0:
            print(f"All fit rows were invalid for source {ii+1}; skipping", flush=True)
            continue

        result['outside_fov_seed'] = np.full(len(result), forced_source, dtype=bool)

        # Saturated-footprint area (px) of the DQ-SATURATED component this seed
        # came from.  Carried into the per-exposure catalog so the consolidation
        # dedup can scale its merge radius to the footprint: a bright star's
        # per-frame satstar position scatters ~r_sat, so a LARGE footprint (e.g.
        # a saturated core embedded in extended emission, ~370px -> r~11px) needs
        # a ~0.7" merge radius to collapse to one row, while compact stars keep a
        # tight radius (so distinct close stars are not fused).  -1 == forced /
        # unmeasured seed.
        result['sat_area'] = np.full(
            len(result), int(src_sat_area) if src_sat_area else -1, dtype=int)

        # stable saturated-component sky anchor (bbox centre) -> lets the
        # consolidation dedup merge same-component per-frame detections even when
        # the fitted position/flux is unstable (see _comp_slices note above).
        if np.isfinite(src_comp_cx) and np.isfinite(src_comp_cy):
            _canchor = ww.pixel_to_world(float(src_comp_cx), float(src_comp_cy))
            if isinstance(_canchor, SkyCoord):
                result['sat_com_ra'] = np.full(len(result), float(_canchor.ra.deg))
                result['sat_com_dec'] = np.full(len(result), float(_canchor.dec.deg))
            else:
                result['sat_com_ra'] = np.full(len(result), np.nan)
                result['sat_com_dec'] = np.full(len(result), np.nan)
        else:
            result['sat_com_ra'] = np.full(len(result), np.nan)
            result['sat_com_dec'] = np.full(len(result), np.nan)

        result['xcentroid'] = result['x_fit'] + x0
        result['ycentroid'] = result['y_fit'] + y0
        x_centroid = np.asarray(result['xcentroid'], dtype=float)
        y_centroid = np.asarray(result['ycentroid'], dtype=float)
        world_fit = ww.pixel_to_world(x_centroid, y_centroid)
        if isinstance(world_fit, SkyCoord):
            result['skycoord_fit'] = world_fit
        else:
            log.warning("pixel_to_world did not return SkyCoord; setting skycoord_fit to None")
            result['skycoord_fit'] = [None] * len(result)

        # MIRI BOTTOM-UP ENVELOPE AMPLITUDE (2026-06-14).  The masked-core LSQ
        # (even on the 2D-bg-subtracted cutout) still OVER-fits some saturated
        # stars -- amplitude inflated by residual emission / wing structure ->
        # the model exceeds the data and leaves deep negative pits (-156k..-197k).
        # Instead of trusting LSQ, set the amplitude to the BOTTOM-UP ENVELOPE:
        # the value at which the model PSF just begins to exceed the data at the
        # inner wings (a low percentile of (data-bg)/psf).  By construction the
        # model then sits at/below the data across the inner wings -> it can only
        # MILDLY over-subtract (a few % of pixels), never produce a deep pit, and
        # it still subtracts the bulk of the star (100% effect).  LSQ position
        # (x_fit/y_fit) is kept; only the amplitude is replaced.  MIRI-only.
        if (_is_miri and not forced_source and bg2d is not None and len(result)):
            _yy, _xx = np.mgrid[0:cutout.shape[0], 0:cutout.shape[1]]
            _xf = float(result['x_fit'][0]); _yf = float(result['y_fit'][0])
            # use the SAME psf the fit used (flux normalization must match)
            _psfu = np.clip(_infov_psf(_xx - _xf, _yy - _yf), 0, None)
            # SPIKE-WEIGHTED AMPLITUDE (2026-06-21).  Nearly all saturated stars
            # came out UNDER-subtracted (model peak 25-45% of data) because the
            # amplitude was capped at the p10 of the inner-wing (which samples the
            # near-core SATURATED-flagged pixels: valid but fewer-group / slightly
            # suppressed rates that drag the fit DOWN), while the bright UNSATURATED
            # diffraction-spike ridges -- which carry the true amplitude -- were
            # masked out by the ~99%-spurious DQ SATURATED flag.  Instead, estimate
            # the amplitude as the optimal flux Sum(d*p)/Sum(p^2) over the SPIKE
            # pixels: finite-data pixels OUTSIDE the GENUINE NaN core (+4px buffer,
            # to drop the suppressed near-core) with non-negligible PSF.  The p^2
            # weighting puts the weight on the bright spike ridges (high local PSF)
            # and ~0 on faint inter-spike / outer tails; a 3-sigma clip rejects
            # residual neighbours / CRs.  This includes the DQ-SATURATED-but-finite
            # spike pixels (their early-group rate is valid), which is exactly the
            # "underweight cores / overweight spikes" the under-sub demands.
            _data_cut_raw = data[y0:y1, x0:x1]
            _genuine_core = ~np.isfinite(_data_cut_raw)
            _core_dist = (ndimage.distance_transform_edt(~_genuine_core)
                          if _genuine_core.any()
                          else np.full(cutout.shape, np.inf, dtype=float))
            _spike_set = (np.isfinite(cutout_fit) & np.isfinite(_data_cut_raw)
                          & (_psfu > _psfu.max() * 1e-3) & (_core_dist >= 4))
            _spike_flux = None
            if int(_spike_set.sum()) >= 8:
                _sp = _psfu[_spike_set].astype(float)
                _sd = cutout_fit[_spike_set].astype(float)
                _den = float(np.sum(_sp * _sp))
                if _den > 0:
                    _sf = float(np.sum(_sd * _sp) / _den)
                    for _ in range(3):
                        _r = _sd - _sf * _sp
                        _sg = 1.4826 * np.median(np.abs(_r - np.median(_r))) + 1e-9
                        _kp = np.abs(_r) < 4.0 * _sg
                        if int(_kp.sum()) < 8 or _kp.all():
                            break
                        _spk, _sdk = _sp[_kp], _sd[_kp]
                        _dn = float(np.sum(_spk * _spk))
                        if _dn <= 0:
                            break
                        _sf = float(np.sum(_sdk * _spk) / _dn)
                    if np.isfinite(_sf) and _sf > 0:
                        _spike_flux = _sf
            # Measure the inner-wing band from the SAT-MASK BOUNDARY (distance
            # transform), NOT a fixed centre radius: the brightest stars have
            # large saturated cores whose wings sit well beyond r=25px, so a
            # fixed r<25 window falls entirely inside the (dilated) mask -> 0
            # wing pixels -> envelope skipped -> the runaway LSQ amplitude
            # (seen at 1.3e9 -> -183k pit) survives.  A boundary-referenced
            # annulus always has pixels and always lands on the inner wings,
            # whatever the core size.  Widen the band until >=8 px qualify.
            _dist = ndimage.distance_transform_edt(~satmask_combined)
            _psfok = (_psfu > _psfu.max() * 1e-5)
            # Prefer the INNERMOST wing band (dist 2-5 px from the sat mask):
            # the PSF is high there so data/psf pins the amplitude correctly.
            # A wide band (out to 12+ px) is dominated by faint outer pixels
            # whose data/psf is inflated (extended emission / noise over a tiny
            # PSF), which pushes the percentile -- hence amplitude -- too high
            # and OVER-predicts the inner wing (model ~3-14k vs data ~700 at
            # r~5 -> -280k core pit).  Start tight; widen only if <8 px qualify.
            _wing = None
            for _wid in (5, 8, 12, 20, 30, 45, 60):
                _w = ((~satmask_combined) & np.isfinite(cutout_fit)
                      & (_dist >= 2) & (_dist <= _wid) & _psfok)
                if _w.sum() >= 8:
                    _wing = _w
                    break
            # FALLBACK: edge / few-pixel stars can have no boundary band with
            # >=8 px (the runaway 1.3e9 -> -183k pit was exactly this case: an
            # at-edge fit whose wing band was empty so the envelope was skipped
            # and the LSQ amplitude survived).  Fall back to ALL unmasked finite
            # pixels with non-negligible PSF so the envelope ALWAYS applies and
            # can cap any runaway.
            if _wing is None or _wing.sum() < 8:
                _wing = ((~satmask_combined) & np.isfinite(cutout_fit) & _psfok)
            if _wing.sum() >= 1:
                _rat = (cutout_fit[_wing] / _psfu[_wing])
                _rat = _rat[np.isfinite(_rat) & (_rat > 0)]
                if _rat.size:
                    # p10 of (data-bg)/psf over the INNER wing = the bottom-up
                    # amplitude where the model just touches the data from below
                    # at the inner wing (model exceeds data at ~10% of px -> only
                    # micro over-sub).  ALWAYS cap the amplitude at this envelope:
                    # the masked-core LSQ frequently over-predicts the inner wing
                    # (flux 3.3e6 -> model peak ~282k vs data ~750 -> -280k core
                    # pit) yet sits BELOW a 99th-pct cap, so a "lower only if
                    # >99th-pct" rule let it through.  Capping at p10 fixes the
                    # over-prediction; a genuinely-fainter LSQ is kept (min).
                    _env = float(np.percentile(_rat, 10))
                    _f0 = float(result['flux_fit'][0])
                    if _spike_flux is not None and not _use_large_infov:
                        # PREFERRED (small-PSF / moderately-saturated): spike-
                        # weighted amplitude (overweight spikes, underweight core).
                        # Fixes the systemic under-subtraction (model 25-45% ->
                        # ~75% of data peak).  Downstream peak-cap still guards any
                        # over-shoot.  NOT used for the LARGE-PSF path: the fovp1024
                        # spike pattern spans ~hundreds of px, so the spike_set
                        # (n~650) sweeps in background/noise and the estimate is
                        # unstable (broke A->0, B->4.6x over-sub).  Large-PSF keeps
                        # the LSQ + env guard below.
                        _new = _spike_flux
                        _amp_src = f'spike Sum(dp)/Sum(p2) n={int(_spike_set.sum())}'
                    elif _use_large_infov:
                        # LARGE-PSF: trust LSQ, env guard (unchanged behaviour).
                        _new = (min(_f0, 3.0 * _env)
                                if (np.isfinite(_f0) and _f0 > 0) else _env)
                        _amp_src = f'env_guard p10={_env:.2e}'
                    else:
                        # small-PSF fallback: original behaviour (cap at p10)
                        _new = (min(_f0, _env) if (np.isfinite(_f0) and _f0 > 0)
                                else _env)
                        _amp_src = f'env_cap p10={_env:.2e}'
                    if _new != _f0:
                        result['flux_fit'][0] = _new
                        print(f"  [miri amp {'large' if _use_large_infov else 'small'}] "
                              f"LSQ {_f0:.2e} -> {_new:.2e} ({_amp_src}, "
                              f"wing_px={int(_wing.sum())})", flush=True)
            # PEAK-MATCHING CAP (2026-06-18).  The p10 envelope still OVER-predicts
            # the central peak by 2-5x on many MIRI saturated stars: the masked
            # core forces the amplitude to be extrapolated from wing pixels, and
            # the STPSF concentrates more flux into the peak than a real (clipped /
            # charge-bled) saturated star, so (data-bg)/psf at the inner wing lands
            # above the amplitude that makes the MODEL PEAK match the DATA PEAK.
            # Result: model peak 2-5x the data -> deep negative pits (e.g. sickle
            # F770W o002 stars at 17:46:17.84/.515/18.92/20.21).  Directly bound it:
            # the rendered model peak (flux * unit-PSF peak) must not exceed the
            # brightest UNSATURATED data pixel just outside the masked core by more
            # than SEED_PEAK_CAP_FACTOR.  This is a physical limit on how negative
            # the subtraction can drive the residual (the only data we can subtract
            # outside the clipped core is what is actually there).  Caps the
            # runaways to ~1.5x (mild, infillable) while leaving well-fit stars
            # (model<=data) untouched, and brings the brightest star's model/data
            # under the post-fit oversub gate so it is SUBTRACTED instead of
            # deleted.  MIRI in-FOV only.
            # 2026-06-20: factor 1.5 -> 1.0.  A saturated star's TRUE peak is
            # unmeasurable (clipped); the best we can subtract is its measurable
            # coadd core.  1.5x over-subtracted the core by 50% (a negative pit at
            # the centre -- user: "must not oversubtract").  Cap model peak to
            # EXACTLY the coadd core: clean subtraction, and the small unmeasurable
            # saturated excess is left as an HONEST positive residual (never a pit).
            _peak_cap_factor = 1.0
            _psf_peak = float(np.nanmax(_psfu)) if np.isfinite(_psfu).any() else np.nan
            # Reference the cap to the DEEP COADD core peak (the star's real
            # brightness), NOT the per-frame unsaturated wing.  The per-frame
            # core is saturated/masked, and the brightest wing pixel just outside
            # it is far fainter than the true peak -- referencing the wing capped
            # bright saturated stars to ~1.5x wing and UNDER-subtracted them
            # (model peak ~1130 while their coadd cores are 5000-12000).  The
            # coadd cores are filled by dithers (not clipped), so the coadd peak
            # at the fit position IS the real core brightness.  Cap model peak <=
            # 1.5x that: subtracts the star to its core (mild, infillable
            # residual) without the runaway over-subtraction.  Fall back to the
            # per-frame wing only when the coadd is unavailable.
            _dpk = np.nan
            if seed_gate_image is not None and seed_gate_wcs is not None:
                try:
                    _xcg = x0 + float(result['x_fit'][0])
                    _ycg = y0 + float(result['y_fit'][0])
                    _skyc = ww.pixel_to_world(_xcg, _ycg)
                    _gx, _gy = seed_gate_wcs.world_to_pixel(_skyc)
                    _gxi, _gyi = int(round(float(_gx))), int(round(float(_gy)))
                    _rr = int(max(3, np.ceil(np.sqrt(
                        max(int(src_sat_area or 0), 1) / np.pi)) + 2))
                    _gny, _gnx = seed_gate_image.shape
                    if _rr < _gxi < _gnx - _rr and _rr < _gyi < _gny - _rr:
                        _wd = seed_gate_image[_gyi - _rr:_gyi + _rr + 1,
                                              _gxi - _rr:_gxi + _rr + 1]
                        _wf = np.isfinite(_wd) & (_wd != 0)
                        if _wf.any():
                            _dpk = float(np.nanmax(_wd[_wf]))
                except Exception:
                    _dpk = np.nan
            if not (np.isfinite(_dpk) and _dpk > 0):
                _near = ((~satmask_combined) & np.isfinite(cutout)
                         & (_dist >= 1) & (_dist <= 8) & _psfok)
                if _near.sum() < 3:
                    _near = ((~satmask_combined) & np.isfinite(cutout) & _psfok)
                _dpk = float(np.nanmax(cutout[_near])) if _near.any() else np.nan
            # LARGE-PSF fit (_use_large_infov): do NOT cap to the coadd core.  The
            # coadd core is the SATURATION-CLIPPED, dither-DILUTED value (sickle A:
            # coadd ~3378 while the true per-frame peak ~1.8e6); capping to it
            # under-fits the star 10-500x.  The spike-constrained LSQ + envelope
            # guard already bound the amplitude.  Keep the cap only for the
            # small-PSF fallback.
            if (not _use_large_infov
                    and np.isfinite(_psf_peak) and _psf_peak > 0
                    and np.isfinite(_dpk) and _dpk > 0):
                _flux_cap = _peak_cap_factor * _dpk / _psf_peak
                _fcur = float(result['flux_fit'][0])
                if np.isfinite(_fcur) and _fcur > _flux_cap:
                    result['flux_fit'][0] = _flux_cap
                    print(f"  [miri peak cap] flux {_fcur:.2e} -> {_flux_cap:.2e} "
                          f"(model peak {_fcur*_psf_peak:.0f} -> "
                          f"{_flux_cap*_psf_peak:.0f} vs coadd core {_dpk:.0f}; "
                          f"{_peak_cap_factor}x)", flush=True)
                # NOTE: a previous "amplitude FLOOR" here (force model peak up to
                # the coadd core when the wing-LSQ under-fit) was REVERTED
                # 2026-06-19: it sampled _dpk over a sat_area-scaled radius that
                # grabbed a BRIGHT NEIGHBOUR's coadd core, then floored the flux to
                # it -> injected a model peak ~70k on faint data (-70k pit at
                # 17:46:13.05 -28:48:18.5 in the joint o001+o002 run) and
                # over-subtracted medium stars (B/C).  Well-fit saturated stars
                # reach their core via the normal 2D-bg+envelope fit (joint B:
                # model 8730 ~ data 10547) without the floor.  The huge-core
                # under-fit it targeted (star A) is actually an OFF-FOV / seed
                # problem, not a wing-LSQ amplitude problem -- handled elsewhere.
            # AMPLITUDE FLOOR (2026-06-21, user-requested).  The spike-weighted
            # small-PSF fit reaches only ~75-90% of the coadd core, so the bright
            # cores still show ~10-27% residual.  Push the model peak up to the
            # MEASURABLE coadd core so they fully clear.  Unlike the reverted
            # 2026-06-19 floor (which used a sat_area-scaled radius that grabbed a
            # bright NEIGHBOUR's core and injected -70k pits), this uses a TIGHT
            # <=4px box at the fit position (this star's own core) and is gated to
            # spike-fit small-PSF stars.  Floor to 0.95x the coadd core: model peak
            # never exceeds the data core -> a small POSITIVE residual, never a pit.
            if (not _use_large_infov and _spike_flux is not None
                    and np.isfinite(_psf_peak) and _psf_peak > 0
                    and seed_gate_image is not None and seed_gate_wcs is not None):
                # Reference the coadd value at the EXACT fit-centre pixel (3x3
                # MEDIAN, robust).  A box-MAX (2-4px) over-referenced a 1px-off
                # bright pixel -> the floor pushed the model above the centre value
                # -> mild core OVER-sub (-130..-500).  The 3x3 median ~ the centre
                # level (not a single hot pixel), so 0.90x it leaves a guaranteed
                # small POSITIVE core residual ("must not oversubtract").
                _rc = 1
                _dpk_core = np.nan
                try:
                    _skc = ww.pixel_to_world(x0 + float(result['x_fit'][0]),
                                             y0 + float(result['y_fit'][0]))
                    _gxc, _gyc = seed_gate_wcs.world_to_pixel(_skc)
                    _gxic, _gyic = int(round(float(_gxc))), int(round(float(_gyc)))
                    _gnyc, _gnxc = seed_gate_image.shape
                    if _rc < _gxic < _gnxc - _rc and _rc < _gyic < _gnyc - _rc:
                        _wdc = seed_gate_image[_gyic - _rc:_gyic + _rc + 1,
                                               _gxic - _rc:_gxic + _rc + 1]
                        _wfc = np.isfinite(_wdc) & (_wdc != 0)
                        if _wfc.any():
                            _dpk_core = float(np.median(_wdc[_wfc]))
                except Exception:
                    _dpk_core = np.nan
                if np.isfinite(_dpk_core) and _dpk_core > 0:
                    # 0.90 (was 0.95): leave a ~10% POSITIVE core residual margin
                    # so a 1px model-vs-data centroid offset can't drive the core
                    # negative ("must not oversubtract").
                    _floor_flux = 0.90 * _dpk_core / _psf_peak
                    _fcur3 = float(result['flux_fit'][0])
                    if np.isfinite(_fcur3) and _fcur3 < _floor_flux:
                        result['flux_fit'][0] = _floor_flux
                        print(f"  [miri amp floor] flux {_fcur3:.2e} -> "
                              f"{_floor_flux:.2e} (model peak "
                              f"{_fcur3*_psf_peak:.0f} -> {_floor_flux*_psf_peak:.0f} "
                              f"vs coadd core {_dpk_core:.0f})", flush=True)
            # ALWAYS (MIRI in-FOV): the masked-core LSQ frequently returns a
            # NaN flux_err (singular covariance from x_0/y_0 pegging the bound),
            # which would make the accept gate reject an otherwise-good fit
            # (correct 2D-bg amplitude, qfit~0.1).  Saturated stars MUST be
            # subtracted, and the amplitude is trusted (2D-bg + envelope bound),
            # so synthesize a finite positive flux_err (snr~20) when LSQ gives
            # NaN/<=0.  Keeps every saturated star through the accept gate.
            _ff = float(result['flux_fit'][0])
            if (np.isfinite(_ff) and _ff > 0
                    and (not np.isfinite(float(result['flux_err'][0]))
                         or float(result['flux_err'][0]) <= 0)):
                result['flux_err'][0] = _ff * 0.05
            if ('qfit' in result.colnames
                    and not (0 < float(result['qfit'][0]) < 1e3)):
                result['qfit'][0] = 1.0

        # FORCED-SOURCE COADD-CORE CAP (2026-06-19).  The forced / outside-FOV
        # LSQ (and the cross-frame reconcile that OVERRIDES the discordant frames
        # to the "reference" flux) is amplitude-UNCONSTRAINED when only a faint
        # spike-less PSF corner is in-frame: it locks onto scattered light and
        # returns flux ~1e9 (sickle joint o001+o002: 25 forced accepts at
        # 1.18e9, ssr_ratio 12-61), whose wings gouge deep pits and whose
        # reconcile-propagated value poisons the merged catalog.  The in-FOV
        # satstars already get a coadd-core peak cap; apply the SAME physical
        # bound to forced sources -- the rendered model peak (flux * unit-PSF
        # peak) must not exceed ~1.5x the DEEP COADD core at the source's sky
        # position (its real, dither-filled brightness).  Maps the (possibly
        # off-frame) fit position through the frame WCS to the coadd; if the
        # coadd does not cover it (truly off-everything star) _dpk is NaN and the
        # flux is left as-is (those rare cases are handled by the outside-FOV
        # region file).  MIRI only.
        if _is_miri and forced_source and len(result):
            print(f"  [miri forced cap DIAG] src {ii+1}: seed_gate="
                  f"{seed_gate_image is not None}/{seed_gate_wcs is not None}, "
                  f"big_grid_large={big_grid_large is not None}, "
                  f"flux={float(result['flux_fit'][0]):.2e}, sat_area={src_sat_area}",
                  flush=True)
        if (_is_miri and forced_source and seed_gate_image is not None
                and seed_gate_wcs is not None and big_grid_large is not None
                and len(result)):
            try:
                _fpk = float(np.nanmax(big_grid_large(np.zeros(1), np.zeros(1))))
                _gny, _gnx = seed_gate_image.shape
                for _ri in range(len(result)):
                    # Use the SEED center (xcen,ycen = the star's TRUE projected
                    # position, possibly off this frame) NOT the fitted x_fit/y_fit:
                    # for a forced off-FOV source the cutout is clamped to the frame
                    # EDGE, so x0+x_fit maps to the coadd edge (off-coadd) -- the
                    # seed position maps to the star's real coadd core.
                    _skyc = ww.pixel_to_world(xcen, ycen)
                    _gx, _gy = seed_gate_wcs.world_to_pixel(_skyc)
                    _gxi, _gyi = int(round(float(_gx))), int(round(float(_gy)))
                    # Use a min radius of 5px so the peak window reaches the
                    # star's bright WING, not just its (possibly still-saturated,
                    # low) coadd CENTER -- a forced star like A is saturated even
                    # in the deep coadd (core ~689) while its wing peaks ~3377;
                    # capping to the center would badly under-subtract.  Centered
                    # on the TRUE seed position, so a 5px window is the star's own
                    # wing, not a neighbour (off-coadd stars are already dropped).
                    _rr = int(max(5, np.ceil(np.sqrt(
                        max(int(src_sat_area or 0), 1) / np.pi)) + 2))
                    # Off-FOV / forced stars map to the COADD EDGE, so don't
                    # require the full +/-_rr window to fit -- clip it to the
                    # coadd bounds and use whatever core pixels are present (the
                    # star's real, dither-filled core).  Skip only if the
                    # position is entirely off the coadd.
                    _ylo, _yhi = max(0, _gyi - _rr), min(_gny, _gyi + _rr + 1)
                    _xlo, _xhi = max(0, _gxi - _rr), min(_gnx, _gxi + _rr + 1)
                    if _yhi <= _ylo or _xhi <= _xlo:
                        continue
                    _wd = seed_gate_image[_ylo:_yhi, _xlo:_xhi]
                    _wf = np.isfinite(_wd) & (_wd != 0)
                    if not _wf.any():
                        continue
                    _dpk = float(np.nanmax(_wd[_wf]))
                    if not (np.isfinite(_fpk) and _fpk > 0 and _dpk > 0):
                        continue
                    _cap = 1.5 * _dpk / _fpk
                    _fc = float(result['flux_fit'][_ri])
                    if np.isfinite(_fc) and _fc > _cap:
                        result['flux_fit'][_ri] = _cap
                        print(f"  [miri forced coadd-core cap] flux {_fc:.2e} -> "
                              f"{_cap:.2e} (model peak {_fc*_fpk:.0f} -> {_dpk*1.5:.0f}"
                              f" = 1.5x coadd core {_dpk:.0f})", flush=True)
            except Exception as _capex:
                print(f"  [miri forced coadd-core cap] skipped: {_capex}", flush=True)

        # NIRCam (and any non-MIRI in-FOV): the masked-core LSQ has the SAME
        # singular-covariance failure -> NaN flux_err -> NaN snr -> the snr
        # accept-gate tosses an otherwise-good saturated fit.  The gc2211
        # baseline-vs-deblend comparison (2026-06-18) showed this dominates
        # rejections (nan_fluxerr/snr = 387/604 components at BASELINE).  The MIRI
        # branch above already synthesises a finite flux_err for this case; mirror
        # it for NIRCam, but CONSERVATIVELY: only rescue a finite, positive-flux
        # fit whose qfit already passes the quality gate, so we convert
        # "good-qfit-but-unmeasurable-snr -> accept" WITHOUT admitting bad fits
        # (qfit/ssr still decide).  snr~20 (flux_err = 5% of flux).
        if (not _is_miri and not forced_source and len(result)
                and 'qfit' in result.colnames):
            _ff = float(result['flux_fit'][0])
            _qf = float(result['qfit'][0])
            _fe = float(result['flux_err'][0])
            if (np.isfinite(_ff) and _ff > 0 and np.isfinite(_qf)
                    and _qf < _qfit_max_keep
                    and (not np.isfinite(_fe) or _fe <= 0)):
                result['flux_err'][0] = _ff * 0.05
                print(f"  [satstar nan-fluxerr rescue] source {ii+1}: "
                      f"qfit={_qf:.2f} < {_qfit_max_keep}, flux={_ff:.3e} -> "
                      f"flux_err={_ff*0.05:.3e} (snr~20)", flush=True)

        result.pprint(max_width=-1)

        ny = cutout.shape[0]
        nx = cutout.shape[1]
        model_image = np.zeros_like(cutout)


        for x_fit, y_fit, flux in zip(result['x_fit'], result['y_fit'], result['flux_fit']):
            # Make a local grid around the source
            if np.isnan(flux):
                # NaN flux from photutils means the LSQ fit did not converge.
                # That is a real bug — silently skipping the row would produce
                # an iter3 residual image that still has the unsubtracted
                # source (the original problem we're trying to fix).  Raise
                # so the failure is loud and gets investigated.
                raise ValueError(
                    f"Fit returned NaN flux for source {ii+1} at "
                    f"(x={x_fit}, y={y_fit}), forced={forced_source}, "
                    f"flux_init={init_params['flux'][0] if 'flux' in init_params.colnames else 'auto'}. "
                    f"Investigate the fit configuration (fit_shape, PSF model "
                    f"size, position bounds) — do not silently skip."
                )
            y, x = np.mgrid[0:ny, 0:nx]
            #psf_eval = big_grid(x, y, flux=flux, x_0=x0, y_0=y0)  # works for analytic PSF
            # Use large PSF grid for forced sources so the spike pattern at
            # off-edge offsets is represented in the model image.  Required
            # to exist for forced sources; raise if it isn't (no silent
            # fallback to small grid).
            if forced_source:
                if big_grid_large is None:
                    raise RuntimeError(
                        "forced_source=True but big_grid_large is None at model "
                        "build time — earlier require-large-PSF check should have "
                        "raised already."
                    )
                _psf_for_model = big_grid_large
            else:
                # in-FOV: render with the SAME psf the amplitude was fit against
                # (big_grid_large for MIRI satstars, else big_grid) so the model
                # flux normalization matches the fit.
                _psf_for_model = _infov_psf
            psf_eval = _psf_for_model(x-x_fit, y-y_fit) * flux  # works for GriddedPSFModel
            # Stars are physically nonnegative.  GriddedPSFModel bicubic
            # interpolation produces small negative pixel values at large
            # offsets (interpolation overshoot between tabulated grid
            # points); multiplied by a bright neighbour's flux these
            # produce *negative* model holes — e.g. star at flux=8.9e4
            # gave PSF wing values of -3e-4 → −27 in the model image,
            # leaving negative residuals at locations where no star
            # exists in the catalog.  Clip to zero so the model is
            # physically valid.
            psf_eval = np.maximum(psf_eval, 0)
            # cut psf_eval to the image size
            model_image += psf_eval[0:ny, 0:nx]

        # Residual-quality gates (computed BEFORE accumulating into
        # full_model_image so rejected fits cannot corrupt downstream data):
        #   * sidelobe_resid_sigma — median residual in the first-sidelobe
        #     annulus (r=8-25 px from fit centre), normalised by the
        #     local background sigma.  A strongly-negative value means
        #     the PSF model amplitude is too high (model > data by many
        #     sigma); diagnosed cases on Sickle 0310g_00002 show model ≈
        #     2× data in this annulus when the fit is bad.
        #   * ssr_ratio — sum-of-squares of residual / sum-of-squares of
        #     (data - median).  > 1 means the fit makes the cutout worse
        #     than just subtracting a constant (the "white-point / no
        #     real star" failure mode the user diagnosed).
        # ``sidelobe_resid_sigma`` is computed from the per-source cutout
        # and uses MAD on the non-source, non-saturated pixels as the
        # local-bkg sigma estimate.
        resid_cutout = cutout - model_image
        cy_cut, cx_cut = cutout.shape
        _yy_c, _xx_c = np.mgrid[0:cy_cut, 0:cx_cut]
        # Use first accepted fit position as centre (typically only one
        # row per source); fall back to the initial COM if empty.
        if len(result) > 0:
            _xf_cut = float(result['x_fit'][0])
            _yf_cut = float(result['y_fit'][0])
        else:
            _xf_cut = float(x_init)
            _yf_cut = float(y_init)
        _r2_c = (_xx_c - _xf_cut) ** 2 + (_yy_c - _yf_cut) ** 2
        _bg_pix = resid_cutout[(~mask) & (_r2_c > 50 ** 2) & np.isfinite(resid_cutout)]
        if _bg_pix.size > 100:
            _bg_sigma_local = float(1.4826 * np.median(np.abs(_bg_pix - np.median(_bg_pix))))
        else:
            _bg_sigma_local = float('nan')
        _bg_sigma_local = max(_bg_sigma_local, 1e-30)
        # First-sidelobe annulus: r=2-10 px from fit centre.  NIRCam LW
        # F480M PSF FWHM ≈ 2.5 px (λ/D=0.156"/0.063"·px-1), first Airy
        # ring at r≈3 px, sidelobe peak r≈4-7 px.  Sickle 0310g_00002
        # over-subtracted cyan-marker stars had min resid -2085 / -950 /
        # -231 etc. at r=2-10 (worst pixel at distance 2.8 px from the
        # source centroid).  An r=8-25 annulus completely misses this
        # zone and fires on background noise instead.
        _ann = ((_r2_c >= 2 ** 2) & (_r2_c < 10 ** 2)
                & (~mask) & np.isfinite(resid_cutout))
        if _ann.any():
            sidelobe_resid_sigma = float(np.median(resid_cutout[_ann]) / _bg_sigma_local)
        else:
            sidelobe_resid_sigma = float('nan')
        # SSR ratio: fit-vs-no-fit sum-of-squared-residual.
        # LOCALIZED (2026-06-08): evaluate over a DISK centred on the star
        # (radius SSR_RADIUS_PIX), using the UNMASKED pixels.  The previous
        # whole-cutout SSR was dominated by bright NEIGHBOURS and background, so
        # a real saturated star next to a brighter one (sickle
        # pillar_with_satstar A, 2"=~32 px from a 600k-count star) got
        # ssr_ratio>1 and was wrongly rejected -> never subtracted.  A disk (not
        # a thin ring) is essential: it spans the full core->background gradient,
        # so ssr_before (cutout-median) is large and a good fit reduces it
        # (ratio<1); a thin wing ring has little pre-fit variance, so subtracting
        # the steep PSF wing ADDS variance (ratio>1) and rejects good fits.  The
        # 15-px radius captures the LW PSF wings while excluding the 32-px
        # neighbour.  Falls back to the whole cutout if too few disk pixels.
        # EXPERIMENTAL: SSR_RADIUS_PIX may need per-filter tuning.
        SSR_RADIUS_PIX = 15.0
        _ssr_region = (_r2_c < SSR_RADIUS_PIX ** 2)
        _fit_pix = (_ssr_region & (~mask)
                    & np.isfinite(resid_cutout) & np.isfinite(cutout))
        if int(_fit_pix.sum()) < 10:
            # too few local pixels -> fall back to the whole-cutout SSR
            _fit_pix = (~mask) & np.isfinite(resid_cutout) & np.isfinite(cutout)
        if _fit_pix.any():
            _ssr_after = float(np.sum(resid_cutout[_fit_pix] ** 2))
            _med_cut = float(np.median(cutout[_fit_pix]))
            _ssr_before = float(np.sum((cutout[_fit_pix] - _med_cut) ** 2))
            ssr_ratio = _ssr_after / max(_ssr_before, 1e-30)
        else:
            ssr_ratio = float('nan')

        result['sidelobe_resid_sigma'] = [sidelobe_resid_sigma] * len(result)
        result['ssr_ratio'] = [ssr_ratio] * len(result)

        threshold_image = np.zeros_like(cutout)

        #count the number of pixels above local background in the model_image
        threshold = np.nanpercentile(cutout, 99)
        threshold_image[model_image>threshold]=1
        num_pixels_above_threshold = np.nansum(threshold_image)
        print(np.nanmax(model_image), flush=True)
        print(f"Number of pixels above threshold ({threshold}): {num_pixels_above_threshold}", flush=True)

        if len(result) > 0:
            flux = result['flux_fit'][0]
            fluxerr = result['flux_err'][0]
            snr = result['flux_fit'][0] / result['flux_err'][0]
            qfit = result['qfit'][0]
            cfit = result['cfit'][0]

        if plot:
            fig = plt.figure(figsize=(12,12))
            ax1 = fig.add_subplot(2,3,1)
            ax1.imshow(cutout, origin='lower', cmap='viridis', vmin=0, vmax=np.nanpercentile(cutout, 99))
            ax1.set_title('Cutout')
            ax2 = fig.add_subplot(2,3,2)
            ax2.set_title('Model')
            ax2.imshow(model_image, origin='lower', cmap='viridis', vmin=0, vmax=np.nanpercentile(cutout, 99))
            ax3 = fig.add_subplot(2,3,3)
            resid_image = cutout - model_image
            ax3.imshow(resid_image, origin='lower', cmap='viridis', vmin=0, vmax=np.nanpercentile(cutout, 99))
            ax3.set_title('Residual')
            ax4 = fig.add_subplot(2,3,4)
            ax4.imshow(mask, origin='lower', cmap='gray')
            ax4.set_title('Mask')
            ax5 = fig.add_subplot(2,3,5)
            ax5.imshow(threshold_image, origin='lower', cmap='gray')
            ax5.set_title('Thresholded Model Pixels')
            # print flux, fluxerr, snr, cfit in the title
            if len(result) > 0:


                ax1.set_title(f'Cutout\nFlux={flux:.2f}, FluxErr={fluxerr:.2f}, SNR={snr:.2f}, qfit={qfit:.2f}, cfit={cfit:.2f}')

            plt.show()
            plt.close()

        # check whether the pixels with dqflag = saturated are also flagged as HOT, DEAD, and RC
        # this is a sanity check to make sure that the saturated pixels are not being used in the fit
        # modified 2026-04-18 to only check if _all_ pixels are flagged as HOT or DEAD, since some pixels may be flagged as both SATURATED and HOT/DEAD because a saturated star happened to land on top of a hot/dead pixel
        idx_saturated_in_cutout = (dq[y0:y1, x0:x1] & dqflags.pixel['SATURATED']) > 0
        saturated_dqflags = dq[y0:y1, x0:x1][idx_saturated_in_cutout]
        if saturated_dqflags.size > 0:
            if np.all((saturated_dqflags & dqflags.pixel['HOT'])!=0):
                print(f"Warning: Some saturated pixels are flagged as HOT; skipping source", flush=True)
                continue
            if np.all((saturated_dqflags & dqflags.pixel['DEAD'])!=0):
                print(f"Warning: Some saturated pixels are flagged as DEAD; skipping source", flush=True)
                continue
        #if np.any((saturated_dqflags & dqflags.pixel['RC'])!=0):
        #    print(f"Warning: Some saturated pixels are flagged as RC; skipping source", flush=True)
        #    continue



        # compare FWHM of the model and the size of saturated pixels
        #if area_saturated > num_pixels_above_threshold:
        #    print(f"Warning: Saturated mask area ({area_saturated}) is larger than number of pixels above threshold ({num_pixels_above_threshold}); skipping source", flush=True)
        #    continue

        # process the result
        # Reject blatantly bad fits before they corrupt the satstar model:
        #  - flux <= 0 : negative star
        #  - snr <= 3  : barely above noise; LSQ unconstrained
        #  - qfit > 5  : photutils' quality-of-fit metric well above the
        #                ~1 typical "OK" value (qfit=27 catastrophic case
        #                in 0310g_00002 had snr~2)
        #  - sidelobe_resid_sigma < -10 : in the first-sidelobe annulus
        #                (r=8-25 px), the median residual is more than
        #                10σ below zero — i.e. the model amplitude is
        #                badly over-fit (Sickle 0310g_00002 cyan stars
        #                show ~2× over-subtraction with values < -30).
        #                STPSF first-sidelobe is brighter than the real
        #                NIRCam PSF for saturated stars (IPC); a
        #                fit that matches the wing amplitude leaves a
        #                strong negative residual at this radius.
        #  - ssr_ratio > 1 : the fit makes the cutout WORSE than just
        #                subtracting a constant (white-point / "no
        #                star" failure mode in 0310g_00002).  Indicates
        #                no real source at the proposed position.
        # MIRI drops the ssr gate entirely (gridded-PSF ring mismatch makes ssr
        # run 17-400 even for excellent fits -> rejected every real bright
        # star); NIRCam applies ssr ONLY to low-confidence fits (a high-snr,
        # good-qfit fit is a real star regardless of ssr -- the ssr gate was
        # silently deleting must-detect bright stars).  See accept_satstar_fit.
        accept_source = accept_satstar_fit(
            result_is_none=(result is None), fluxerr=fluxerr, snr=snr,
            flux=flux, qfit=qfit, sidelobe_resid_sigma=sidelobe_resid_sigma,
            ssr_ratio=ssr_ratio, is_miri=_is_miri,
            qfit_max_keep=_qfit_max_keep, sidelobe_min_keep=_sidelobe_min_keep,
            ssr_ratio_max_keep=_ssr_ratio_max_keep, snr_min_keep=_snr_min_keep)

        # POST-FIT SATURATION-SEVERITY GATE (NIRCam, in-FOV; 2026-07-10): the
        # cal any-group SATURATED bit over-flags recoverable pixels on ordinary
        # stars; in the crf products EVERY SATURATED pixel is blanked
        # (SCI=NaN, VAR_POISSON=NaN), so neither the frame data nor the
        # variance can distinguish a real saturated core from the over-flag --
        # but the FIT can: a genuine saturated star's model must at minimum
        # reach the filter's saturation level at its own peak.  Brick F182M:
        # 362 fakes (45% of the satstar catalog, mag 16-18, wing-fit 0.4-2.2
        # mag too faint, positions wandering 0.2-0.4") all imply peaks far
        # below the floor; real satstars imply peaks >> floor.  Margin 0.5x
        # allows wing-fit scatter for genuinely borderline saturation.
        # Rejected stars fall through to the normal daophot channel, whose
        # photometry for them was correct all along.  MIRI is excluded (its
        # own envelope/cap machinery + flat-top model handle amplitude), as
        # are forced/off-FOV sources (flux pinned externally).
        if (accept_source and not _is_miri and not forced_source
                and result is not None and _sev_floor and _sev_floor > 0):
            try:
                _xf = float(np.atleast_1d(result['x_fit'])[0])
                _yf = float(np.atleast_1d(result['y_fit'])[0])
            except (KeyError, IndexError, TypeError, ValueError):
                _xf = _yf = np.nan
            _implied = (satstar_implied_peak(flux, _psf_for_fit, _xf, _yf)
                        if np.isfinite(_xf) and np.isfinite(_yf) else np.nan)
            # OBSERVED-PEAK second chance: narrow-band wing fits UNDER-estimate
            # (F187N/F212N bias 0.5-0.7 at deep masks), halving the implied
            # peak and wrongly rejecting REAL deep satstars (brick F187N: only
            # 157 satstars survived vs ~300 real).  A genuinely saturated star
            # always shows near-full-well pixels in its cutout (recovered rim
            # or unflagged shoulder), so require BOTH the model-implied AND
            # the observed unmasked peak to fall below the threshold before
            # rejecting.  Fakes stay rejected (their observed peaks are far
            # below the floor by construction).
            try:
                _obs_pk = float(np.nanmax(np.where(mask, np.nan, cutout)))
            except (ValueError, TypeError):
                _obs_pk = np.nan
            if (np.isfinite(_implied) and _implied < 0.5 * float(_sev_floor)
                    and not (np.isfinite(_obs_pk)
                             and _obs_pk >= 0.5 * float(_sev_floor))):
                print(f"Satstar severity gate: REJECT fitted 'satstar' whose "
                      f"model peak {_implied:.0f} MJy/sr cannot reach the "
                      f"{_sev_floor:g} saturation level (flux_fit={flux:.3g}; "
                      f"over-flagged unsaturated star -> daophot channel)",
                      flush=True)
                accept_source = False
        if forced_source and result is not None:
            xcent = np.asarray(result['xcentroid'], dtype=float)
            ycent = np.asarray(result['ycentroid'], dtype=float)
            accept_source = np.all(np.isfinite(xcent)) and np.all(np.isfinite(ycent))

        # POST-FIT coadd gate (MIRI, in-FOV): the pre-fit seed gate measures at
        # the DQ-component centroid, but the fit can DRIFT several px onto a
        # faint extended-emission pit (the component centroid is pulled toward a
        # bright neighbour or merged DQ blob, so the seed passes, then the fit
        # settles on the phantom).  Re-check the prominence+faint-core gate on
        # the deep coadd at the FITTED position so a drifted phantom is rejected
        # where its model actually lands.  Real saturated stars (incl. deeply-
        # saturated zeroed cores) keep core>=1191 via the adaptive wing ring;
        # phantoms that drifted off-seed read core~700.  Forced sources are
        # exempt (their position is locked, not fit-driven).
        if (_is_miri and accept_source and not forced_source
                and result is not None):
            # Per-frame FALLBACK (2026-06-23): on a field's FIRST run the deep
            # coadd data_i2d does not exist yet, so seed_gate_image is None and
            # this whole post-fit gate used to be SKIPPED -- letting FAKE bright
            # satstars (huge model on faint emission) survive (2526 cloud-c
            # filament: flux 4.4e6 on data ~340; sickle worked only because its
            # coadd pre-existed).  The FAKE-BRIGHT check is robust on the
            # per-frame SCI too (a real bright star's local peak is bright there
            # as well), so fall back to ``data``/``ww`` for it.  The noisier
            # prominence/core/conc/oversub coadd metrics stay coadd-ONLY.
            _pf_img = seed_gate_image if seed_gate_image is not None else data
            _pf_ww = seed_gate_wcs if seed_gate_wcs is not None else ww
            _have_coadd = (seed_gate_image is not None and seed_gate_wcs is not None)
            try:
                _xc = float(np.atleast_1d(result['xcentroid'])[0])
                _yc = float(np.atleast_1d(result['ycentroid'])[0])
                _sky = ww.pixel_to_world(_xc, _yc)
                _cx, _cy = _pf_ww.world_to_pixel(_sky)
                # CAP sat_area for the prominence radius (2026-06-21).  A spurious
                # DQ-SATURATED cluster (detector artifact: sat_area 8000-20000 of
                # flagged-but-finite ~1100 pixels) makes r_sat=sqrt(area/pi) grow
                # to 50-80 px, so the prominence "core" max grabs a DISTANT real
                # star -> prom inflates to 30-106 and the phantom (140k-727k model
                # on ~1100 data) slips the gate.  Cap at 1600 (~22px radius): real
                # saturated stars (A/B sat_area<=1572, pillar bright within ~22px)
                # are unaffected, but a fake's 22px-radius core is just its own
                # ~1100 data -> prom<8 -> rejected.
                _capped_sa = (min(int(src_sat_area), 1600)
                              if src_sat_area is not None else src_sat_area)
                # prominence/core/conc are calibrated on the noise-averaged COADD;
                # on the noisier per-frame fallback they over-reject real stars,
                # so measure them ONLY when the coadd is available.
                if _have_coadd:
                    _pp, _cc = _seed_prominence(seed_gate_image, (float(_cy), float(_cx)),
                                                _capped_sa, robust=seed_prominence_robust)
                    _kk = _seed_concentration(seed_gate_image, (float(_cy), float(_cx)),
                                              _capped_sa)
                    # SMALL-RADIUS prominence/core (2026-06-23).  The sat_area-
                    # scaled radius is REQUIRED for genuinely huge saturated stars
                    # (their core is zeroed out to ~22px; a small ring sits inside
                    # the zeroed core -> NaN -> kept).  But for a FAKE on extended
                    # emission the spurious DQ-SATURATED footprint is just as large,
                    # so the capped (1600 -> ~22px) ring reaches a NEIGHBOURING real
                    # star and inflates prominence past the cut (2526 FAKE-1: prom
                    # 0.9 at r~11px but 49 at r~22px -> wrongly kept).  Also measure
                    # at a small FIXED radius: a fake reads low here (its own faint
                    # emission), while a huge real star reads NaN (zeroed core ->
                    # kept) and a normal star reads high.  Reject if the small-radius
                    # value is FINITE and below threshold.
                    _pp_s, _cc_s = _seed_prominence(seed_gate_image,
                                                    (float(_cy), float(_cx)),
                                                    _SEED_SMALL_SAT_AREA,
                                                    robust=seed_prominence_robust)
                else:
                    _pp = _cc = _kk = _pp_s = _cc_s = np.nan
                # OVER-SUBTRACTION ratio: the fitted PSF model PEAK vs the deep
                # coadd DATA at the fit position.  This is the most direct measure
                # of the negative-fake-star pathology -- a satstar whose model
                # vastly exceeds the data gouges a deep negative pit.  On the
                # coadd (cores filled by dithers, not clipped) a well-fit real
                # star has model <= data (ratio <=~0.5); EVERY over-fit phantom /
                # bright knot / amplitude-runaway reads ratio >=4 (huge gap, no
                # overlap).  Catches the cases concentration/core cannot (a bright
                # compact knot scores conc~3.3 like a faint real star, but its
                # model/data ratio is ~100).
                # DENOMINATOR = the adaptive CORE peak (``_cc`` from
                # _seed_prominence above), NOT a 3x3 window at the fit centre.
                # FIX 2026-06-20: a star saturated in MOST overlapping frames
                # (sickle A/B) has a CLIPPED centre even in the coadd (A centre
                # ~689, B brighter but still depressed), so a 3x3-centre _dpk is
                # artificially low and model/_dpk trips the oversub gate even for
                # a CORRECTLY peak-capped model -> the real saturated star is
                # DELETED and left unsubtracted (B's core stayed; daophot then fit
                # its spike).  ``_cc`` is the max over the adaptive wing ring (B
                # =10547, A=3378 -- the star's true brightness, what the cap also
                # references), so a capped model (<=1.5x core) gives ratio ~1.5
                # and passes, while a genuine runaway/phantom (model>>core) still
                # reads >=4 and is rejected.  Falls back to the 3x3 centre when
                # _cc is unmeasurable.
                _ix, _iy = int(round(float(_cx))), int(round(float(_cy)))
                _gny, _gnx = _pf_img.shape
                _dpk = float(_cc) if (np.isfinite(_cc) and _cc > 0) else np.nan
                if not (np.isfinite(_dpk) and _dpk > 0):
                    if 1 < _ix < _gnx - 1 and 1 < _iy < _gny - 1:
                        _dwin = _pf_img[_iy - 1:_iy + 2, _ix - 1:_ix + 2]
                        _fin = np.isfinite(_dwin) & (_dwin != 0)
                        if _fin.any():
                            _dpk = float(np.nanmax(_dwin[_fin]))
                _mpk = float(np.nanmax(model_image)) if np.isfinite(model_image).any() else np.nan
                # FAKE-BRIGHT gate (2026-06-22).  A class of phantoms sits on
                # smooth diffuse emission (NOT a local peak) yet is fit with a
                # HUGE amplitude (model peak 1e4-6e5 on coadd data ~700-1700) --
                # severe over-additions that gouge bright fake stars where there
                # is no star at all.  They evade every prior gate: a big spurious
                # sat_area pushes them onto the large-PSF branch (oversub skipped),
                # their wandering adaptive core/prominence can grab a real
                # neighbour, and seed_core_min=1000 is below their ~1000-2700
                # ring core.  The robust discriminator is a SMALL-radius LOCAL
                # coadd peak: a genuine bright saturated star (A/B) has a bright
                # near-core (>=6000 within 6px even when the very centre is NaN);
                # these fakes peak <=~2300 within 6px.  So a source claiming a
                # very bright model MUST be backed by a bright local coadd peak,
                # else it is a fake.  Real under-subtracted stars are exempt:
                # their models are all <1e4 (the model_min cut), so only the
                # "claims to be very bright" sources are tested.  Uses a small
                # fixed radius (not the adaptive ring) so an inflated sat_area
                # cannot make the core wander onto a distant real star.
                _lpk = np.nan
                _lr = 6
                if _lr < _ix < _gnx - _lr and _lr < _iy < _gny - _lr:
                    _lwin = _pf_img[_iy - _lr:_iy + _lr + 1,
                                    _ix - _lr:_ix + _lr + 1]
                    _lfin = np.isfinite(_lwin) & (_lwin != 0)
                    if _lfin.any():
                        _lpk = float(np.nanmax(_lwin[_lfin]))
                # Genuine-saturation exemption: a real saturated star has an
                # unrecoverable NaN/zero (DO_NOT_USE) core in the PER-FRAME SCI
                # (``data`` retains it; the coadd fills it via dithers).  Check a
                # small radius around the FIT position in the frame.
                _has_satcore = False
                _fx, _fy = int(round(_xc)), int(round(_yc))
                _dny, _dnx = data.shape
                if 2 < _fx < _dnx - 2 and 2 < _fy < _dny - 2:
                    _ccut = data[_fy - 2:_fy + 3, _fx - 2:_fx + 3]
                    _has_satcore = bool(np.any(~np.isfinite(_ccut))
                                        or np.any(_ccut == 0))
                _fake_bright = is_fake_bright(
                    _mpk, _lpk, model_min=seed_fake_model_min,
                    localpk_max=seed_fake_localpk_max,
                    has_saturated_core=_has_satcore)
                # The over-subtraction ratio compares the model PEAK to the coadd
                # core.  For a LARGE-PSF in-FOV fit the model peak is the star's
                # TRUE (unclipped) brightness while the coadd core is saturation-
                # clipped + dither-diluted, so model/core is legitimately huge --
                # this gate would wrongly DELETE every correctly-fit bright
                # saturated star.  Skip it for _use_large_infov (prom/core/conc
                # still apply; ssr_ratio guards genuine over-fits).
                _oversub = (_have_coadd and (not _use_large_infov)
                            and seed_oversub_ratio and seed_oversub_ratio > 0
                            and np.isfinite(_mpk) and np.isfinite(_dpk) and _dpk > 0
                            and _mpk > seed_oversub_ratio * _dpk)
                # coadd-only metrics (prom/core/conc/oversub) gate ONLY when the
                # coadd is present; the fake-bright check runs on the per-frame
                # fallback too (it is the #1-harm failure and is robust there).
                _coadd_bad = (_have_coadd and (
                        (seed_prominence_min and seed_prominence_min > 0
                         and np.isfinite(_pp) and _pp < seed_prominence_min)
                        or (seed_core_min and seed_core_min > 0
                            and np.isfinite(_cc) and _cc < seed_core_min)
                        or (seed_conc_min and seed_conc_min > 0
                            and np.isfinite(_kk) and _kk < seed_conc_min)
                        # small-radius prominence/core: catches fakes whose
                        # inflated sat_area ring reaches a neighbour (NaN here for
                        # huge real stars -> kept)
                        or is_small_radius_emission_phantom(
                            _pp_s, _cc_s, prom_min=seed_prominence_min,
                            core_min=seed_core_min)
                        or _oversub))
                _bad = _coadd_bad or _fake_bright
                if _bad:
                    accept_source = False
                    print(f"Post-fit {'coadd' if _have_coadd else 'per-frame'} "
                          f"gate (MIRI): rejecting source {ii+1} "
                          f"-- phantom/knot/over-fit (prom={_pp:.1f}, "
                          f"core={_cc:.0f}, conc={_kk:.1f}, localpk={_lpk:.0f}, "
                          f"model/data="
                          f"{(_mpk/_dpk if (np.isfinite(_mpk) and np.isfinite(_dpk) and _dpk>0) else float('nan')):.1f}"
                          f"{' FAKE-BRIGHT' if _fake_bright else ''}"
                          f"; thresh prom{seed_prominence_min}/core{seed_core_min}/"
                          f"conc{seed_conc_min}/oversub{seed_oversub_ratio}/"
                          f"fake(model>{seed_fake_model_min:.0f}&localpk<{seed_fake_localpk_max:.0f}))",
                          flush=True)
            except Exception:
                pass

        # POST-FIT BRIGHT-PHANTOM gate (MIRI, in-FOV, auto-detected only).  A
        # spurious super-bright satstar on saturated extended emission (W51 F770W:
        # real stars + emission knots saturate into one connected ~1e5-px blob, so
        # every pre-fit gate ranks the phantom >= a real star) is caught only by
        # its very bright flux PLUS a badly structured residual (ssr_ratio) or an
        # enormous flux-from-faint-seed extrapolation (flux/flux_init).  Forced /
        # user-seeded sources are exempt (position locked, trusted).  OFF unless
        # phantom_flux_floor>0 (launcher-set for MIRI).  See is_bright_flux_phantom.
        if (_is_miri and accept_source and not forced_source
                and result is not None
                and phantom_flux_floor and phantom_flux_floor > 0):
            _pf_init = (float(np.atleast_1d(result['flux_init'])[0])
                        if 'flux_init' in result.colnames else float('nan'))
            if is_bright_flux_phantom(flux, _pf_init, ssr_ratio,
                                      flux_floor=phantom_flux_floor,
                                      ssr_max=phantom_ssr_max,
                                      ratio_max=phantom_ratio_max):
                accept_source = False
                _pf_ratio = (flux / _pf_init
                             if (np.isfinite(_pf_init) and _pf_init > 0)
                             else float('nan'))
                print(f"Post-fit bright-phantom gate (MIRI): rejecting source "
                      f"{ii+1} -- spurious super-bright emission phantom "
                      f"(flux={flux:.2e} > {phantom_flux_floor:.0e}; "
                      f"ssr_ratio={ssr_ratio:.1f} (>{phantom_ssr_max:g}?) or "
                      f"flux/flux_init={_pf_ratio:.0f} (>{phantom_ratio_max:g}?))",
                      flush=True)

        if accept_source:
            # FLAT-TOPPED CORE (MIRI in-FOV, env-gated).  The gates above used
            # the peaked amp*PSF model_image; now, before it is subtracted,
            # replace it inside the saturated PLATEAU by the (bg-subtracted)
            # data so the core residual -> ~0 (no under-sub ring / over-sub
            # divot).  bg2d is required (defines the bg-subtracted data =
            # cutout_fit) and forced sources are exempt (off-FOV; no core to
            # flat-top).  OFF unless MIRI_SATSTAR_FLATTOP=1.
            if (satstar_flattop and _is_miri and not forced_source
                    and bg2d is not None):
                # Geometric core+shoulder mask (amplitude-independent): a radius
                # from the fit centre = saturated-footprint radius + k*FWHM.  The
                # peak-cap holds the model at the coadd-core level so an
                # amplitude-only plateau misses the under-sub RING just outside
                # the saturated core; this covers it.  k via
                # MIRI_SATSTAR_FLATTOP_SHOULDER_FWHM (default 2.0).
                _ft_k = float(os.environ.get(
                    'MIRI_SATSTAR_FLATTOP_SHOULDER_FWHM', 2.0))
                _r_core = np.sqrt(max(int(src_sat_area or 0), 1) / np.pi)
                _r_ft = _r_core + _ft_k * float(fwhm_pix)
                _cy_ft = (float(result['y_fit'][0]) if len(result)
                          else float(y_init))
                _cx_ft = (float(result['x_fit'][0]) if len(result)
                          else float(x_init))
                _yy_ft, _xx_ft = np.mgrid[0:cutout.shape[0], 0:cutout.shape[1]]
                _core_mask_ft = (np.hypot(_xx_ft - _cx_ft, _yy_ft - _cy_ft)
                                 <= _r_ft)
                _mi_ft = flattop_satstar_model(model_image, cutout_fit,
                                               satstar_flattop_frac,
                                               core_mask=_core_mask_ft)
                _dpk_ft = float(np.nanmax(model_image)) if np.isfinite(model_image).any() else np.nan
                print(f"  [miri flat-top] src {ii+1}: plateau data-subtract "
                      f"(R={_r_ft:.1f}px, model peak {_dpk_ft:.0f} -> "
                      f"{float(np.nanmax(_mi_ft)):.0f}, frac={satstar_flattop_frac})",
                      flush=True)
                model_image = _mi_ft
            # Only accumulate the per-source model into the global
            # full_model_image AFTER passing all gates.  Previously this
            # accumulation happened before the accept check, so a rejected
            # bad fit still corrupted the cumulative satstar model.
            full_model_image[y0:y1, x0:x1] += model_image
            # BRIGHTNESS-SCALED MODEL FOOTPRINT (MIRI).  The model is rendered
            # only over the small +-pad FIT cutout, so a BRIGHT star's PSF wings
            # are truncated at the box -> a hard SQUARE edge in the model/residual
            # (user-reported, sgrb2).  The FIT stays in the small box (stable),
            # but the bright star should be SUBTRACTED with a footprint big enough
            # to contain its wings.  Re-render the accepted model out to where its
            # wing falls below a noise floor (radius grows with brightness, capped
            # at the PSF grid FOV) and add ONLY the region exterior to the cutout
            # (interior already added above -> no double count).  Gated by
            # MIRI_SATSTAR_RENDER_FOOTPRINT (default 1).
            if (_is_miri and not forced_source
                    and int(os.environ.get('MIRI_SATSTAR_RENDER_FOOTPRINT', 1))):
                try:
                    _wfloor = float(os.environ.get('MIRI_SATSTAR_WING_FLOOR', 5.0))
                    _psf0 = float(_psf_for_model(np.array([0.0]),
                                                 np.array([0.0]))[0])
                    _maxhalf = int(min(512,
                        min(_psf_for_model.data.shape[-2:])
                        // (2 * int(max(1, getattr(_psf_for_model,
                                                   'oversampling', [1])[0])))))
                    for _xf, _yf, _fl in zip(result['x_fit'], result['y_fit'],
                                             result['flux_fit']):
                        if not np.isfinite(_fl):
                            continue
                        _peak = _psf0 * float(_fl)
                        # ~r^-3 diffraction wing reaches _wfloor at this radius
                        _rh = (int(pad * (max(1.0, _peak / _wfloor)) ** (1.0 / 3.0))
                               if _peak > _wfloor else pad)
                        _rh = int(np.clip(_rh, pad, _maxhalf))
                        if _rh <= pad:
                            continue
                        _gcx = x0 + float(_xf)
                        _gcy = y0 + float(_yf)
                        _Y0 = int(max(0, _gcy - _rh)); _Y1 = int(min(data.shape[0], _gcy + _rh))
                        _X0 = int(max(0, _gcx - _rh)); _X1 = int(min(data.shape[1], _gcx + _rh))
                        _yb, _xb = np.mgrid[_Y0:_Y1, _X0:_X1]
                        _wing = np.maximum(_psf_for_model(_xb - _gcx, _yb - _gcy)
                                           * float(_fl), 0)
                        _ext = np.ones(_wing.shape, dtype=bool)
                        _iy0 = max(_Y0, y0); _iy1 = min(_Y1, y1)
                        _ix0 = max(_X0, x0); _ix1 = min(_X1, x1)
                        if _iy1 > _iy0 and _ix1 > _ix0:
                            _ext[_iy0 - _Y0:_iy1 - _Y0, _ix0 - _X0:_ix1 - _X0] = False
                        full_model_image[_Y0:_Y1, _X0:_X1][_ext] += _wing[_ext]
                        data_working[_Y0:_Y1, _X0:_X1][_ext] -= _wing[_ext]
                except (ValueError, TypeError, IndexError) as _wex:
                    print(f"satstar wing-render skipped src {ii+1}: {_wex}",
                          flush=True)
            # mark pixels this accepted fit actually used (unmasked in its window)
            try:
                flag_img[y0:y1, x0:x1][~mask] |= 4
            except (ValueError, NameError):
                pass
            # Also subtract from the working copy so subsequent fits in
            # this loop don't see this source's PSF wings (iterative
            # subtraction; pairs ordered by ``sat_area`` desc so the
            # brightest fits first and gets clean data).
            data_working[y0:y1, x0:x1] -= model_image
            if forced_source:
                print(f"Accepting forced outside-FOV source {ii+1} with flux={flux}, fluxerr={fluxerr}, snr={snr}, "
                      f"sidelobe_resid_sigma={sidelobe_resid_sigma:.2f}, ssr_ratio={ssr_ratio:.3f}", flush=True)
            else:
                print(f"Accepting source {ii+1} with flux={flux}, fluxerr={fluxerr}, snr={snr}, "
                      f"sidelobe_resid_sigma={sidelobe_resid_sigma:.2f}, ssr_ratio={ssr_ratio:.3f}", flush=True)
            result['wingcal_rmask'] = _wingcal_rmask
            if index == 0:
                base_tab = result
            else:
                base_tab = table.vstack([base_tab, result])

            index += 1
        else:
            print(f"Skipping source {ii+1}: "
                  f"snr={snr}, fluxerr={fluxerr}, qfit={qfit}, "
                  f"sidelobe_resid_sigma={sidelobe_resid_sigma:.2f}, "
                  f"ssr_ratio={ssr_ratio:.3f}", flush=True)

    # NOTE: a pass-2 leave-one-out refit was attempted on 2026-05-14 but
    # produced catastrophically low fluxes for the brightest sources
    # (e.g. 1.37e6 -> 6.7e3 on the 0310g_00002 demo) — total model flux
    # dropped 60%.  Root cause not fully diagnosed; suspect that adding
    # back a single source's pass-1 PSF model to a working image whose
    # other sources had wing-contaminated (over-fit) fluxes does NOT
    # reconstruct the leave-one-out data correctly.  Reverted; single-
    # pass iter-subtract + r=2-10 gate handles 5/7 of the user's cyan
    # markers cleanly.  Remaining close-pair members ((560,280) and
    # (503,142)) still need attention; future work = proper joint fit
    # via PSFPhotometry multi-row init_params + SourceGrouper.

    # if base_tab is not defined, return None
    # this happens if no saturated stars are found
    if index == 0:
        print('No saturated stars found after processing all sources', flush=True)
        return None
    else:
        if 'x_0' not in base_tab.colnames and 'xcentroid' in base_tab.colnames:
            base_tab['x_0'] = base_tab['xcentroid']
        if 'y_0' not in base_tab.colnames and 'ycentroid' in base_tab.colnames:
            base_tab['y_0'] = base_tab['ycentroid']
        # Record this frame's footprint center so cross-frame reconciliation
        # (reconcile_outside_fov_satstar_fluxes) can pick the nearest detector
        # to an out-of-field star.  Use the full frame's pixel center.
        # Store as float RA/Dec (deg) keys so they survive the FITS round-trip
        # (a SkyCoord object cannot serialise to a FITS header card).
        try:
            _ny_full, _nx_full = data.shape
            _dc = ww.pixel_to_world((_nx_full - 1) / 2.0, (_ny_full - 1) / 2.0)
            if isinstance(_dc, SkyCoord):
                _icrs = _dc.icrs
                base_tab.meta['DET_RA'] = float(_icrs.ra.deg)
                base_tab.meta['DET_DEC'] = float(_icrs.dec.deg)
        except Exception as ex:
            log.warning(f"Could not record det_center in satstar table meta: {ex}")
        # WING SELF-CALIBRATION: correct the catalog fluxes for the frame's
        # measured wing-fit bias (STPSF wing deficit); the model/residual keep
        # the raw amplitudes (they subtract the observed wings cleanly).
        try:
            _inst_wc = 'MIRI' if header['INSTRUME'].lower() == 'miri' else 'NIRCam'
            _, _fwhm_wc = get_fwhm(header, instrument_replacement=_inst_wc)
        except Exception:
            _fwhm_wc = 2.0
        base_tab = apply_wing_selfcal(
            base_tab, data - full_model_image, err_working, saturated,
            big_grid, fwhm_pix=float(_fwhm_wc),
            severity_floor=float(_sev_floor or 0.0))
        builtins.satstar_table = base_tab
        builtins.satstar_model = full_model_image
        builtins.satstar_resid = data - full_model_image
        builtins.satstar_flagimg = flag_img
        return base_tab


def _wing_selfcal(data, err, sat_mask, psf_grid, radii, *, fwhm_pix=2.0,
                  peak_lo=None, peak_hi=None, severity_floor=0.0,
                  n_stars_max=30, min_stars=8, edge=110):
    """Per-frame WING-FIT self-calibration: measure the bias of masked-core
    PSF fitting on BRIGHT UNSATURATED stars in this very frame.

    The satstar amplitude comes from fitting the STPSF model to the wings
    (core masked).  Measured on Brick F182M (two detectors, ~60 stars): the
    real PSF carries 10-25% MORE wing flux relative to its core than the
    STPSF model (the model Airy troughs are too deep), so wing-only fits are
    systematically too BRIGHT -- bias(r_mask) = 1.02/1.11/1.20/1.27 at
    r_mask = 3/4/8/12 px and still rising, while the unmasked control fit is
    1.004.  The bias is per-filter (F187N shows none at small radii) and
    varies ~3-10% frame-to-frame, so a static correction table is unsafe:
    calibrate on each frame from its own bright unsaturated stars, fitting
    each once normally (truth) and once with the core masked to each satstar
    effective mask radius.  Median ratio precision ~ madstd/sqrt(K) ~= 3-5%
    for K ~= 20-40, removing a 0.1-0.5+ mag systematic.

    Returns dict {radius_px: (median_ratio, n_stars, madstd)} for the
    requested ``radii`` (empty dict when calibration is impossible).
    """
    from photutils.detection import DAOStarFinder
    from astropy.stats import sigma_clipped_stats, mad_std as _madstd

    if peak_hi is None:
        peak_hi = 0.875 * severity_floor if severity_floor > 0 else 0.0
    if peak_lo is None:
        peak_lo = 0.12 * severity_floor if severity_floor > 0 else 0.0
    if not (peak_hi > peak_lo > 0):
        print("wing-selfcal: no usable peak window (severity floor unset); "
              "skipping", flush=True)
        return {}

    bad = ~np.isfinite(data)
    _, med, std = sigma_clipped_stats(data[~bad], sigma=3.0, maxiters=5)
    dao = DAOStarFinder(threshold=10 * std, fwhm=fwhm_pix, exclude_border=True)
    cat = dao(np.nan_to_num(data - med), mask=bad)
    if cat is None or len(cat) == 0:
        print("wing-selfcal: no detections; skipping", flush=True)
        return {}
    xc = 'x_centroid' if 'x_centroid' in cat.colnames else 'xcentroid'
    yc = 'y_centroid' if 'y_centroid' in cat.colnames else 'ycentroid'
    x = np.asarray(cat[xc], dtype=float)
    y = np.asarray(cat[yc], dtype=float)
    peak = np.asarray(cat['peak'], dtype=float)

    ny, nx = data.shape
    cand = np.where((peak > peak_lo) & (peak < peak_hi)
                    & (x > edge) & (x < nx - edge)
                    & (y > edge) & (y < ny - edge))[0]
    sy, sx = np.nonzero(sat_mask) if sat_mask is not None else ([], [])
    sel = []
    for i in cand:
        d2 = (x - x[i]) ** 2 + (y - y[i]) ** 2
        d2[i] = np.inf
        if np.any((d2 < 20 ** 2) & (peak > 0.05 * peak[i])):
            continue
        if np.any((d2 < 60 ** 2) & (peak > 0.3 * peak[i])):
            continue
        if len(sx) and np.any((np.asarray(sx) - x[i]) ** 2
                              + (np.asarray(sy) - y[i]) ** 2 < 40 ** 2):
            continue
        sel.append(i)
    if len(sel) < min_stars:
        print(f"wing-selfcal: only {len(sel)} isolated bright unsaturated "
              f"star(s) (need {min_stars}); skipping", flush=True)
        return {}
    sel = np.array(sel)[np.argsort(peak[np.array(sel)])[::-1]][:n_stars_max]
    localbkg = LocalBackground(25, 50)

    def _fit(xi, yi, fmask, fit_shape, fixed):
        m = psf_grid.copy()
        # the satstar loop sets position BOUNDS on the shared grid object;
        # a stale bound anchored to the last satstar makes every fit here
        # fail silently -- reset all constraints.
        for par in ('x_0', 'y_0', 'flux'):
            getattr(m, par).bounds = (None, None)
            getattr(m, par).fixed = False
        m.x_0.fixed = fixed
        m.y_0.fixed = fixed
        phot = PSFPhotometry(psf_model=m, fit_shape=(fit_shape, fit_shape),
                             localbkg_estimator=localbkg, aperture_radius=5)
        init = table.Table({'x_init': [xi], 'y_init': [yi]})
        try:
            res = phot(data, error=err, mask=fmask, init_params=init)
            f = float(res['flux_fit'][0])
            return f if np.isfinite(f) and f > 0 else np.nan
        except Exception:
            return np.nan

    yy, xx = np.mgrid[0:ny, 0:nx]
    truths = {}
    for i in sel:
        truths[i] = _fit(x[i], y[i], bad, 11, False)
    good = [i for i in sel if np.isfinite(truths[i])]
    print(f"wing-selfcal: {len(sel)} calibration star(s), {len(good)} good "
          f"truth fit(s)", flush=True)
    out = {}
    for r in sorted(set(int(round(min(max(rr, 3), 30))) for rr in radii
                        if np.isfinite(rr))):
        fs = int(min(4 * r + 21, 101)) | 1
        ratios = []
        for i in good:
            cm = ((xx - x[i]) ** 2 + (yy - y[i]) ** 2) < r ** 2
            f = _fit(x[i], y[i], bad | cm, fs, True)
            if np.isfinite(f) and truths[i] > 0:
                ratios.append(f / truths[i])
        if len(ratios) >= min_stars:
            ratios = np.array(ratios)
            out[r] = (float(np.median(ratios)), len(ratios),
                      float(_madstd(ratios)))
        else:
            print(f"wing-selfcal: r={r}px only {len(ratios)} usable masked "
                  f"fit(s) (<{min_stars}); bucket dropped", flush=True)
    if out:
        print("wing-selfcal: " + "; ".join(
            f"r={r}px ratio={v[0]:.3f}+/-{v[2]:.3f} (n={v[1]})"
            for r, v in sorted(out.items())), flush=True)
    return out


def apply_wing_selfcal(base_tab, data_sub, err, sat_mask, psf_grid, *,
                       fwhm_pix=2.0, severity_floor=0.0):
    """Correct satstar CATALOG fluxes for the wing-fit bias measured on this
    frame (see _wing_selfcal).  The rendered/subtracted MODEL keeps the raw
    fitted amplitude on purpose: the fit matched the observed wings, so
    model-with-raw-flux subtracts the wings cleanly; only the total-flux
    NUMBER is biased by the model's wing deficit.  Adds columns
    ``flux_fit_raw``, ``wingcal_ratio``; divides ``flux_fit`` and
    ``flux_err`` by the per-star interpolated ratio.  Env
    SATSTAR_WINGCAL=0 disables."""
    if os.environ.get('SATSTAR_WINGCAL', '1') in ('0', 'false', 'False'):
        return base_tab
    if base_tab is None or 'wingcal_rmask' not in base_tab.colnames:
        return base_tab
    rmask = np.asarray(base_tab['wingcal_rmask'], dtype=float)
    want = rmask[np.isfinite(rmask)]
    if want.size == 0:
        print("wing-selfcal: no in-FOV satstars with a mask radius; skipping",
              flush=True)
        return base_tab
    print(f"wing-selfcal: calibrating {len(base_tab)} satstar(s), mask radii "
          f"{np.nanmin(want):.1f}-{np.nanmax(want):.1f} px, severity floor "
          f"{severity_floor:g}", flush=True)
    cal = _wing_selfcal(data_sub, err, sat_mask, psf_grid,
                        radii=list(want), fwhm_pix=fwhm_pix,
                        severity_floor=severity_floor)
    base_tab['flux_fit_raw'] = np.asarray(base_tab['flux_fit'], dtype=float)
    base_tab['wingcal_ratio'] = np.ones(len(base_tab))
    if not cal:
        return base_tab
    rs = np.array(sorted(cal))
    vs = np.array([cal[r][0] for r in rs])
    ratio = np.interp(np.clip(rmask, rs.min(), rs.max()), rs, vs)
    ratio = np.where(np.isfinite(rmask), ratio, 1.0)
    base_tab['wingcal_ratio'] = ratio
    base_tab['flux_fit'] = base_tab['flux_fit_raw'] / ratio
    if 'flux_err' in base_tab.colnames:
        base_tab['flux_err'] = np.asarray(base_tab['flux_err'],
                                          dtype=float) / ratio
    n = int(np.sum(ratio != 1.0))
    print(f"wing-selfcal: corrected {n}/{len(base_tab)} satstar catalog "
          f"flux(es); median ratio "
          f"{np.median(ratio[ratio != 1.0]) if n else 1.0:.3f}", flush=True)
    return base_tab



def _find_ramp_for(filename):
    """Return the path to the sibling ``_ramp.fits`` for a cal/crf ``filename``,
    or None.  Shared by the ZEROFRAME loader and the first-group saturation map."""
    import re
    base = os.path.basename(filename)
    m = re.match(r'(jw\d+_\d+_\d+_[a-z0-9]+)', base)
    if not m:
        return None
    stem = m.group(1)
    pipedir = os.path.dirname(filename)
    cands = [os.path.join(pipedir, f'{stem}_ramp.fits'),
             os.path.join(pipedir, 'pipeline', f'{stem}_ramp.fits'),
             os.path.join(os.path.dirname(pipedir), 'pipeline', f'{stem}_ramp.fits')]
    for rf in cands:
        if os.path.exists(rf):
            return rf
    return None


def first_group_saturation_mask(filename):
    """Boolean mask of pixels saturated in the FIRST ramp group (the only
    TRULY-unrecoverable pixels), read from the sibling ``_ramp.fits`` GROUPDQ.

    The cal/crf ``SATURATED`` flag marks any pixel that saturates in ANY group.
    On bright MIRI emission that floods enormous regions (cloudc 2526 F770W:
    69852 px / 904 connected components, max 16231 px -- fusing many real point
    sources into ONE giant DQ "blob") even though the ramp fitter recovers a
    valid flux for every pixel that has >=1 good (pre-saturation) group.  Only a
    pixel whose FIRST group already saturates has no good read and is genuinely
    unrecoverable (cloudc: 720 px / 26 components, max 54 px).  Using this map in
    place of the cal SATURATED flag stops real stars-on-bright-emission from
    being swept into the satstar channel / vetoed from daophot.

    Returns the (ny, nx) boolean mask, or None if no ramp / no GROUPDQ."""
    rf = _find_ramp_for(filename)
    if rf is None:
        return None
    with fits.open(rf) as rh:
        if 'GROUPDQ' not in [e.name for e in rh]:
            return None
        gdq = rh['GROUPDQ'].data
        if gdq is None or getattr(gdq, 'ndim', 0) != 4:
            return None
        # (nint, ngroup, ny, nx): first group of each integration; a pixel is
        # truly saturated if its first read is saturated in ANY integration.
        return ((gdq[:, 0] & dqflags.pixel['SATURATED']) > 0).any(axis=0)


def correct_dq_first_group_saturation(dq, filename, instrument=''):
    """Clear the ``SATURATED`` bit on pixels that saturate only in LATER ramp
    groups (recoverable by the ramp fitter), leaving it set only where the FIRST
    group saturates.  See ``first_group_saturation_mask`` for the rationale.

    Opt-in via env ``MIRI_FIRSTGROUP_SAT_DQ`` (default off); MIRI-only.  Returns
    ``dq`` unchanged when disabled, non-MIRI, or no sibling ramp/GROUPDQ exists."""
    if dq is None or not int(os.environ.get('MIRI_FIRSTGROUP_SAT_DQ', 0)):
        return dq
    if 'MIRI' not in str(instrument).upper():
        return dq
    fg = first_group_saturation_mask(filename)
    if fg is None or fg.shape != dq.shape:
        return dq
    SAT = dqflags.pixel['SATURATED']
    clear = ((dq & SAT) > 0) & (~fg)
    n = int(clear.sum())
    if n:
        dq = dq.copy()
        dq[clear] &= ~np.array(SAT, dtype=dq.dtype)
        print(f"first-group-sat DQ correction ({os.path.basename(filename)}): cleared "
              f"SATURATED on {n} later-group-only px; kept {int(fg.sum())} "
              f"first-group-saturated", flush=True)
    return dq


def _find_zeroframe_for(filename):
    """Locate and load the ZEROFRAME (frame zero) for a cal/crf ``filename``.

    The satstar step runs on the 2D cal/crf image, which has no ZEROFRAME; it
    lives in the matching ``_ramp.fits`` produced by Detector1 (same exposure
    stem, in this target's ``pipeline/`` directory).  Returns the ZEROFRAME 2D
    array (raw DN) or ``None`` if no ramp file is found.  Used to opt the
    ZEROFRAME deblender into ``remove_saturated_stars``."""
    rf = _find_ramp_for(filename)
    if rf is not None:
        with fits.open(rf) as rh:
            names = [e.name for e in rh]
            # PREFER the ramp SCI cube's FIRST READ (first integration, first
            # group): it is superbias/refpix/linearity-corrected by Detector1
            # (measured cal/group0 scatter 2.7-4.9% in the bright regime) and,
            # unlike the ZEROFRAME extension, is NOT zeroed at flagged pixels
            # (brick F182M ramps: ZEROFRAME ext reads 0 at every DQ-SATURATED
            # pixel -> R*0 'recoveries').  Validated on cloudc F770W: 9/10
            # blob-embedded saturated stars are distinct + unsaturated in the
            # first read.
            if 'SCI' in names:
                _sci = rh['SCI'].data
                if _sci is not None and getattr(_sci, 'ndim', 0) == 4:
                    zf = np.asarray(_sci[0, 0], dtype=float)
                    print(f"satstar zeroframe: using ramp first read SCI[0,0] "
                          f"from {rf}", flush=True)
                    return zf
            if 'ZEROFRAME' in names:
                zf = np.asarray(rh['ZEROFRAME'].data[0], dtype=float)
                print(f"satstar zeroframe: loaded ZEROFRAME ext from {rf} "
                      f"(no 4-D SCI cube)", flush=True)
                return zf
    print(f"satstar deblend: no _ramp.fits ZEROFRAME found for "
          f"{os.path.basename(filename)} (deblend disabled for this frame)", flush=True)
    return None


def remove_saturated_stars(filename, save_suffix='_unsatstar', overwrite=True,
                           file_suffix='', deblend_with_zeroframe=False, **kwargs):
    """
    ``file_suffix`` is inserted into the output filenames *before* the
    ``_satstar_{catalog,model,residual}`` suffix so that concurrent runs
    that differ only by post-processing options (e.g. ``--bgsub``,
    ``--iteration-label=iter2``) write to distinct files and do not race
    on ``os.remove`` during ``overwrite=True``.  Pass an empty string
    (default) to preserve the pre-existing filename scheme.
    """
    print(f"Removing saturated stars from {filename}", flush=True)
    fh = fits.open(filename)
    data = fh['SCI'].data

    # Correct the SATURATED DQ bit to FIRST-group saturation (env-gated, MIRI).
    # The cal/crf SATURATED flag floods bright emission and fuses real point
    # sources into giant DQ blobs; only first-group-saturated pixels are truly
    # unrecoverable.  Correcting it here shrinks the saturated components the
    # satstar finder sees to the genuine cores.  See first_group_saturation_mask.
    if 'DQ' in fh:
        _instr = fh[0].header.get('INSTRUME', '')
        fh['DQ'].data = correct_dq_first_group_saturation(
            fh['DQ'].data, filename, _instr)

    # there are examples, especially in F405, where the variance is NaN but the value
    # is negative
    print(f"Setting NaN variance to 0", flush=True)
    #data[np.isnan(fh['VAR_POISSON'].data)] = 0

    header = fh[0].header
    if 'CRPIX1' not in header:
        # Copy the SCI WCS -- INCLUDING SIP distortion -- into the satstar
        # catalog/model/residual header.  ``WCS(...).to_header()`` drops the
        # SIP A_i_j/B_i_j terms by default (SIP is non-standard, omitted unless
        # relax=True on BOTH read and write), so for the common JWST case where
        # the SCI CTYPE is ``RA---TAN-SIP`` the distortion is silently lost.
        # The satstar model/residual live on the SCI pixel grid, so a SIP-less
        # header makes them reproject a few 0.1" off (the SIP magnitude at the
        # detector edges) and contaminates the merged image.  relax=True keeps
        # the SIP coefficients verbatim.
        header.update(wcs.WCS(fh['SCI'].header, relax=True).to_header(relax=True))
    # ZEROFRAME auto-discovery.  Two independent consumers:
    #  * FIT ANCHORING (default ON; env SATSTAR_ZEROFRAME_FIT=0 disables):
    #    recovered group-0 rim pixels anchor the satstar amplitude, fixing the
    #    0.5-3.6 mag wing-extrapolation deficit for marginally saturated stars
    #    (the CMD saturation-boundary jog/gap).
    #  * DEBLEND seeding (opt-in via deblend_with_zeroframe, experimental).
    _zf_fit_on = os.environ.get('SATSTAR_ZEROFRAME_FIT', '1') not in ('0', 'false', 'False')
    if kwargs.get('zeroframe') is None and (_zf_fit_on or deblend_with_zeroframe):
        zf = _find_zeroframe_for(filename)
        if zf is not None:
            kwargs['zeroframe'] = zf
    if deblend_with_zeroframe:
        kwargs['zeroframe_deblend'] = True
    print("Running get_saturated_stars", flush=True)
    satstar_table = get_saturated_stars(fh, **kwargs)
    if satstar_table is not None:
        satstar_table.meta.update(header)
        print("Finished get_saturated_stars", flush=True)

        satstar_catalog_filename = filename.replace(".fits", f'{file_suffix}_satstar_catalog.fits')
        satstar_model_filename = filename.replace(".fits", f'{file_suffix}_satstar_model.fits')
        satstar_residual_filename = filename.replace(".fits", f'{file_suffix}_satstar_residual.fits')

        satstar_table.write(satstar_catalog_filename, overwrite=overwrite)
        print(f"Saved saturated star catalog to {satstar_catalog_filename}", flush=True)

        if hasattr(builtins, 'satstar_model'):
            fits.PrimaryHDU(data=builtins.satstar_model, header=header).writeto(
                satstar_model_filename, overwrite=overwrite
            )
            print(f"Saved saturated star model image to {satstar_model_filename}", flush=True)

        if hasattr(builtins, 'satstar_resid'):
            fits.PrimaryHDU(data=builtins.satstar_resid, header=header).writeto(
                satstar_residual_filename, overwrite=overwrite
            )
            print(f"Saved saturated star residual image to {satstar_residual_filename}", flush=True)

        if hasattr(builtins, 'satstar_flagimg'):
            # uint8 bitmask: 1=partly saturated (nonlinear), 2=totally saturated
            # (unrecoverable NaN), 4=included in a saturated-star fit.
            fh_flag = fits.PrimaryHDU(data=builtins.satstar_flagimg, header=header)
            fh_flag.header['FLAGBIT1'] = (1, 'partly saturated (nonlinear, recoverable)')
            fh_flag.header['FLAGBIT2'] = (2, 'totally saturated (unrecoverable NaN)')
            fh_flag.header['FLAGBIT4'] = (4, 'included in saturated-star fit')
            fh_flag.writeto(filename.replace(".fits", f'{file_suffix}_satstar_flags.fits'),
                            overwrite=overwrite)
            print(f"Saved saturated star flag image", flush=True)
    else:
        print("No saturated stars found", flush=True)
        return



def main():
    if not os.get('STPSF_PATH'):
        raise ValueError("STPSF_PATH must be specified")

    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-f", "--filter", dest="filter",
                      default='F140M',
                      help="filter list", metavar="filter")
    parser.add_option("--target", dest="target",
                      default='w51',
                      help="target name", metavar="target")
    (options, args) = parser.parse_args()
    filt = options.filter
    if int(filt[1:4]) < 500:
        modules = ('nrca', 'nrcb')
    else:
        modules = ('mirim',)

    for module in modules:
        if int(filt[1:4]) < 500:
            globlist = glob.glob(f"/orange/adamginsburg/jwst/{options.target}/{filt}/pipeline/*{module}*align*crf.fits")
        else:
            globlist = glob.glob(f"/orange/adamginsburg/jwst/{options.target}/{filt}/pipeline/*mirimage_cal.fits")
        for i, fn in enumerate(globlist):
            print()
            print(fn)
            if True:
                remove_saturated_stars(fn)

    #for module in ('nrca', 'nrcb', 'merged'):
    #    for fn in glob.glob(f"/orange/adamginsburg/jwst/w51/F*/pipeline/*{module}*crf.fits"):
    #        print()
    #        print(fn)
    #        remove_saturated_stars(fn, verbose=True)




if __name__ == "__main__":
    main()
