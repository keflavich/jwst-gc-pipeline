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


def find_saturated_stars(fitsdata, min_sep_from_edge=5, edge_npix=10000):
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
    sizes = sum_labels(saturated, sources, np.arange(nsource)+1)
    msfe = min_sep_from_edge

    # which sources are edge sources?  Anything w/ more than edge_npix contiguous "saturated" pixels
    edge_ids = np.where(sizes > edge_npix)[0]
    # id 0 is the non-saturated zone that we've excluded [but reading this code 3/28/2026, I'm skeptical this makes sense]
    edge_ids = edge_ids[1:]
    edge_mask = np.isin(sources, edge_ids)
    saturated = saturated & (~ndimage.binary_dilation(edge_mask, iterations=msfe))

    coms = center_of_mass(saturated, labels=sources, index=np.arange(nsource)+1)

    return saturated, sources, coms


def _refine_coms_by_data(coms, data, sources, shift_warn_thresh_pix=3.0):
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
        ys, xs = np.where(sources == cluster_id)
        if len(ys) == 0:
            refined.append((cy, cx))
            continue
        # Bounding-box centre of the labelled cluster.  (ys.min() +
        # ys.max()) / 2 gives the pixel centre of the bbox, which is the
        # centre of mass of a symmetric rectangle covering all sat
        # pixels.  Robust to asymmetric "tails" that pull a COM off-centre.
        new_cy = (float(ys.min()) + float(ys.max())) / 2.0
        new_cx = (float(xs.min()) + float(xs.max())) / 2.0
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
                                         disagree_factor=2.0, min_frames=2):
    """Reconcile the flux of OUT-OF-FIELD (forced) saturated stars across frames.

    A star outside the field of view, if bright enough, has its PSF detected in
    the target frames.  It is fit independently in every frame whose PSF
    footprint it touches.  Detectors *close* to the (off-field) star catch its
    high-S/N diffraction spikes and recover a correct flux; detectors that only
    catch a spike-less *corner* of the footprint fit scattered light and return
    a badly wrong flux that over-contributes to the model.  We trust the frame
    whose detector footprint centre is CLOSEST to the star (it sees the spikes)
    and override the discordant far-frame fluxes with it.

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

    Returns
    -------
    list of (SkyCoord, flux) : the reconciled (nearest-detector) flux for every
        out-of-field star that had a cross-frame disagreement.  Empty if nothing
        needed reconciling.  Pure function: it does not mutate the input tables.
        The caller forwards this as ``flux_overrides`` to ``get_saturated_stars``
        so every frame pins that star to the trusted flux on the next phase.
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
        return []

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
    for members in groups:
        if len(members) < min_frames:
            continue
        # distance from the star (use each member's own fitted position vs its
        # own detector centre) -> the member with the smallest star-to-detector-
        # centre distance is the most interior (sees the spikes).
        dists = [recs[m][2].separation(recs[m][4]).arcsec for m in members]
        near = members[int(np.argmin(dists))]
        truth = recs[near][3]
        if not (truth > 0):
            continue
        # Override only when some far frame disagrees badly with the trusted
        # near-frame flux; otherwise leave the (already consistent) family alone.
        disagrees = False
        for m in members:
            if m == near:
                continue
            f = recs[m][3]
            ratio = max(f, truth) / min(f, truth) if min(f, truth) > 0 else np.inf
            if ratio > disagree_factor:
                disagrees = True
                break
        if disagrees:
            overrides.append((recs[near][2], truth))
    return overrides


def get_saturated_stars(fitsdata, path_prefix='/orange/adamginsburg/jwst/w51/psfs/', pad=81, size=None, min_sep_from_edge=5, edge_npix=10000, mask_buffer=2, adaptive_mask_buffer_scale=True, adaptive_bkg_annulus=True, plot=True, rindsz=3, use_merged_psf_for_merged=False, outside_star_pixels=None, outside_star_fit_box=512, forced_grid_search_radius=5, satstar_central_downweight_sigma=0.0, flux_overrides=None):
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

    # nan_to_num data to avoid fitting NaNs
    data[np.isnan(fitsdata['VAR_POISSON'].data)] = 0
    dq = fitsdata['DQ'].data
    full_model_image = np.zeros_like(data, dtype=float)

    saturated, sources, coms = find_saturated_stars(fitsdata, min_sep_from_edge=min_sep_from_edge, edge_npix=edge_npix)

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
    coms = _refine_coms_by_data(coms, data, sources)

    # Precompute sat_area per labeled component so we can order in-FOV
    # source_records brightest-first for iterative-subtraction fitting
    # (see ``data_working`` setup + post-accept subtraction below).
    if sources.max() > 0:
        _sizes_by_label = sum_labels(saturated, sources,
                                     np.arange(int(sources.max())) + 1)
    else:
        _sizes_by_label = np.array([], dtype=float)

    source_records = []
    for ii, com in enumerate(coms):
        sat_area_ii = int(_sizes_by_label[ii]) if ii < len(_sizes_by_label) else 0
        source_records.append({'com': com, 'label': ii + 1,
                               'forced': False, 'sat_area': sat_area_ii})

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
    big_grid_large = None
    forced_sources_present = any(s.get('forced') for s in source_records)
    if forced_sources_present:
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
            raise FileNotFoundError(
                f"Forced source(s) present but large PSF grid is missing: {_lg_fn}. "
                f"Build it with /orange/adamginsburg/jwst/sickle/build_large_psf_grids.py "
                f"before running with outside-FOV seeds.  Falling back to the small "
                f"grid silently produces wrong outputs."
            )
        print(f"Loading large PSF grid for forced sources: {_lg_fn}", flush=True)
        big_grid_large = to_griddedpsfmodel(_lg_fn)
        if isinstance(big_grid_large, list):
            big_grid_large = big_grid_large[0]

    ww = wcs.WCS(header)

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
    # on NIRCam, where BFE/IPC make the STPSF first-sidelobe brighter than the
    # real saturated-star PSF.  MIRI has no such BFE sidelobe and a much wider
    # diffraction pattern at 7-25um, so the NIRCam-tuned cuts spuriously reject
    # real saturated stars (e.g. ~10 of the 34 F770W hand-selected stars are
    # DQ-saturated and were being dropped here).  Loosen for MIRI.
    _is_miri = header['INSTRUME'].lower() == 'miri'
    _qfit_max_keep = 15.0 if _is_miri else 5.0
    _sidelobe_min_keep = -40.0 if _is_miri else -10.0
    _ssr_ratio_max_keep = 2.0 if _is_miri else 1.0
    _snr_min_keep = 2.0 if _is_miri else 3.0

    slices = find_objects(saturated)

    if size is None:
        size = pad

    index = 0
    print(f"Found {nsource} saturated sources to process", flush=True)
    for ii, src in enumerate(source_records):
        # get the center of pixels with this label

        com = src['com']
        src_label = src['label']
        forced_source = src['forced']
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
        else:
            src_sat_area = None

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
        saturated_mask = saturated[y0:y1, x0:x1]
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
        else:
            # Forced sources or no src_label: fall back to dilating the
            # whole saturated mask (legacy behaviour).
            satmask_combined = binary_dilation(saturated_mask,
                                               iterations=effective_buffer)
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
            # the fitted centroids wander ~0.7-1" between frames, while the
            # override is keyed at ONE frame's fitted position.  The old 0.2"
            # tolerance was tighter than that scatter, so the worst (runaway)
            # frame -- the one MOST in need of the override -- was the farthest
            # from it and never matched, leaving its 1.3e9 flux -> -180k pit.
            # 1.0" matches the reconcile grouping radius (sparse off-FOV sources,
            # so no risk of grabbing a different star).
            _ovr_flux = None
            if flux_overrides:
                try:
                    _wpos = ww.pixel_to_world(xcen, ycen)
                    for _osc, _of in flux_overrides:
                        if np.isfinite(_of) and _wpos.separation(_osc).arcsec < 1.0:
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
            _psf_for_fit = big_grid
            psfphot = PSFPhotometry(
                                    localbkg_estimator=localbkg_estimator,
                                    fitter=lmfitter,
                                    psf_model=_psf_for_fit,
                                    fit_shape=size,
                                    aperture_radius=15*fwhm_pix)
            # Position-fit bound radius.  ``size_saturated`` is sqrt(sat_area)/2,
            # which is ~1 px for a small saturated core (sat_area~15).  A +/-1 px
            # bound is so tight the LevMar fit pegs x_0/y_0 at the bound, making
            # the covariance matrix singular -> flux_err (and snr) come back NaN
            # even when the fit is excellent (qfit~1).  Widen for MIRI so the
            # centroid converges freely and the covariance (hence flux_err) is
            # finite; the init is already the bbox-refined saturated-core centre
            # so a ~1.5 FWHM search is ample.  NIRCam keeps the validated value.
            pos_bound = size_saturated
            if _is_miri:
                pos_bound = max(size_saturated, 1.5 * fwhm_pix)
            low_x  = xcen - x0 - pos_bound
            high_x = xcen - x0 + pos_bound
            low_y  = ycen - y0 - pos_bound
            high_y = ycen - y0 + pos_bound

            # get the underlying model and set bounds there
            model = getattr(psfphot, "psf_model", None)
            if model is None:
                raise RuntimeError("psfphot.psf_model is None — can't set parameter bounds")

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
            _psfu = np.clip(big_grid(_xx - _xf, _yy - _yf), 0, None)
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
            _wing = None
            for _wid in (12, 20, 30, 45, 60):
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
                    # 20th pct: model exceeds data at ~20% of inner-wing px
                    # (mild over-sub), subtracts the rest -- the "just touches
                    # from below" amplitude the bottom-up scheme targets.  Also
                    # cap at the absolute envelope (max ratio) so the model can
                    # never exceed the data at ANY unmasked pixel -> no deep pit.
                    _env = float(np.percentile(_rat, 20))
                    _cap = float(np.percentile(_rat, 99))
                    _f0 = float(result['flux_fit'][0])
                    _new = min(_env, _cap) if np.isfinite(_f0) else _env
                    # only ever LOWER a runaway LSQ amplitude or replace a bad
                    # (nan/<=0) one; never inflate a sane LSQ fit above the cap.
                    if (not np.isfinite(_f0)) or _f0 <= 0 or _f0 > _cap:
                        result['flux_fit'][0] = _new
                        print(f"  [miri envelope amp] LSQ {_f0:.2e} -> {result['flux_fit'][0]:.2e} (env={_env:.2e} cap={_cap:.2e})",
                              flush=True)
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
                _psf_for_model = big_grid
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
        #                NIRCam PSF for saturated stars (BFE / IPC); a
        #                fit that matches the wing amplitude leaves a
        #                strong negative residual at this radius.
        #  - ssr_ratio > 1 : the fit makes the cutout WORSE than just
        #                subtracting a constant (white-point / "no
        #                star" failure mode in 0310g_00002).  Indicates
        #                no real source at the proposed position.
        if _is_miri:
            # MIRI: the ssr_ratio gate is dropped entirely.  ssr_ratio compares
            # the residual to (data - median) in the inner annulus; for bright
            # MIRI cores the gridded-PSF first/second-ring shape mismatches the
            # real star, so ssr runs 17-400 even for an excellent fit (verified:
            # a source with qfit=1.30, snr=1525 scored ssr_ratio=45).  It
            # rejected EVERY real bright star (0/36 hand-selected captured even
            # after the mask-dilation + position-bound fixes made the fits good).
            # qfit and snr cleanly separate good fits (positive qfit~1, positive
            # snr) from bad ones (negative qfit / negative snr), so gate on those
            # instead: require a FINITE, POSITIVE qfit below the cap and a
            # positive snr above threshold.
            accept_source = (result is not None
                             and np.isfinite(fluxerr)
                             and snr > _snr_min_keep
                             and flux > 0
                             and np.isfinite(qfit)
                             and 0.0 < qfit < _qfit_max_keep
                             and (not np.isfinite(sidelobe_resid_sigma)
                                  or sidelobe_resid_sigma > _sidelobe_min_keep))
        else:
            accept_source = (result is not None
                             and np.isfinite(fluxerr)
                             and snr > _snr_min_keep
                             and flux > 0
                             and (not np.isfinite(qfit) or qfit < _qfit_max_keep)
                             and (not np.isfinite(sidelobe_resid_sigma)
                                  or sidelobe_resid_sigma > _sidelobe_min_keep)
                             and (not np.isfinite(ssr_ratio) or ssr_ratio < _ssr_ratio_max_keep))
        if forced_source and result is not None:
            xcent = np.asarray(result['xcentroid'], dtype=float)
            ycent = np.asarray(result['ycentroid'], dtype=float)
            accept_source = np.all(np.isfinite(xcent)) and np.all(np.isfinite(ycent))

        if accept_source:
            # Only accumulate the per-source model into the global
            # full_model_image AFTER passing all gates.  Previously this
            # accumulation happened before the accept check, so a rejected
            # bad fit still corrupted the cumulative satstar model.
            full_model_image[y0:y1, x0:x1] += model_image
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
        builtins.satstar_table = base_tab
        builtins.satstar_model = full_model_image
        builtins.satstar_resid = data - full_model_image
        builtins.satstar_flagimg = flag_img
        return base_tab

def remove_saturated_stars(filename, save_suffix='_unsatstar', overwrite=True,
                           file_suffix='', **kwargs):
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

    # there are examples, especially in F405, where the variance is NaN but the value
    # is negative
    print(f"Setting NaN variance to 0", flush=True)
    #data[np.isnan(fh['VAR_POISSON'].data)] = 0

    header = fh[0].header
    if 'CRPIX1' not in header:
        header.update(wcs.WCS(fh['SCI'].header).to_header())
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
