import os
from astropy.io import fits
from scipy.ndimage import median_filter, map_coordinates
import numpy as np
from astropy.wcs import WCS
import scipy
import scipy.ndimage

from ..mast_names import proposal_id_from_program, filtername_from_header

basepath = '/orange/adamginsburg/jwst/brick/'

#: Where ``background_map_path`` looks for a region's maps when the caller does
#: not name a directory: ``<data root>/<regionname>/images/``.
BACKGROUND_MAP_ROOT = '/orange/adamginsburg/jwst'


class DestreakWouldDeleteSky(ValueError):
    """``destreak_data`` was asked to subtract the row percentile and restore nothing.

    Raised instead of executing the bare-subtraction branch.  See
    ``destreak_data`` for what that branch does and how a caller that really
    does restore the large scales says so.
    """

# these were created in notebooks/MedianFilterBackground.ipynb
background_mapping = { '2221':
                      { '001':
                       {
                        'regionname': 'brick',
                        'f212n': 'jw02221-o001_t001_nircam_clear-f212n_i2d_medfilt256.fits',
                        'f187n': 'jw02221-o001_t001_nircam_clear-f187n_i2d_medfilt256.fits',
                        'f410m': 'jw02221-o001_t001_nircam_clear-f410m_i2d_medfilt128.fits',
                        'f405n': 'jw02221-o001_t001_nircam_f405n-f444w_i2d_medfilt128.fits',
                        'f182m': 'jw02221-o001_t001_nircam_clear-f182m_i2d_medfilt256.fits',
                        'f466n': 'jw02221-o001_t001_nircam_f444w-f466n_i2d_medfilt128.fits',
                        # UNREACHABLE, and the three files below do not exist on
                        # disk (only *.fits_stale).  F444W/F356W/F200W/F115W are
                        # brick's proposal-1182 obs-004 bands, not 2221/001's, and
                        # 1182 has no key in this mapping at all -- so these
                        # entries have never been read.  They are kept only as a
                        # record of which maps someone meant to build; with
                        # background_map_path()'s existence check, wiring 1182 in
                        # without building them now fails loudly instead of
                        # silently subtracting the sky.
                        # 2026-07-11: was '..._merged_nodestreak_realigned-to-refcat_background.fits'
                        # (realign retired; that name was already .fits_stale since 2023 -> destreak
                        # was silently degraded).  Point at the real mosaic background that exists on
                        # disk (the _i2d background map).  Missing -> add_background_map warns + skips.
                        'f444w': 'jw01182-o004_t001_nircam_clear-f444w-merged_i2d_background.fits',
                        'f356w': 'jw01182-o004_t001_nircam_clear-f356w-merged_i2d_background.fits',
                        'f200w': 'jw01182-o004_t001_nircam_clear-f200w-merged_i2d_background.fits',
                        #'f115w': 'jw01182-o004_t001_nircam_clear-f115w-merged_nodestreak_i2d_medfilt128.fits',
                       },
                        '002':
                       {
                        'regionname': 'cloudc',
                        # 2026-06-22: was '..._realigned-to-vvv_i2d_medfilt128.fits' (missing on
                        # disk -> destreak crash). Point at the background map that actually exists.
                        # NOTE: the other 5 Cloud C filters (f410m/f466n/f182m/f187n/f212n) have no
                        # background map -> add_background_map warns + skips (degraded destreak).
                        # 2026-06-24: the plain '..._medfilt128.fits' map has ~13% NaN holes (the
                        # medfilt builder's zero->NaN spread) even though the source i2d is fully
                        # covered; bilinear sampling propagated those NaNs into the destreaked frames
                        # -> NaN blob in the merged image (~17:46:21.9 -28:37:34).  Use the
                        # nearest-finite-filled (hole-free) map instead.  add_background_map also
                        # nan_to_num-guards the map now as defence-in-depth.
                        'f405n': 'jw02221-o002_t001_nircam_clear-f405n-merged-nodestreak_i2d_medfilt128_filled.fits',
                       }
                      }
                     }


def compute_zero_spacing_approximation(filename, ext=('SCI', 1), dx=128,
                                       smooth=True,
                                       percentile=10, regs=None, progressbar=lambda x: x):
    """
    Use a local, large-scale percentile to estimate the "zero spacing"
    background level across the image.

    We'll then use this to replace the missing zero-spacing lost from
    the destreaking process.


    smooth: use percentile_Filter
    """
    img = fits.getdata(filename, ext=ext)
    header = fits.getheader(filename, ext=ext)
    ww = WCS(header)

    img[img == 0] = np.nan

    if regs is not None:
        for reg in regs:
            preg = reg.to_pixel(ww)
            mask = preg.to_mask()
            slcs,smslcs = mask.get_overlap_slices(img.shape)
            img[slcs][mask.data.astype('bool')[smslcs]] = np.nan


    if smooth:
        y, x = np.mgrid[:dx, :dx]
        circle = ((x-dx/2)**2 + (y-dx/2)**2) < (dx/2)**2
        arr = scipy.ndimage.percentile_filter(img, percentile,
                                              #size=(dx, dx),
                                              footprint=circle,
                                              mode='reflect',
                                             )
        return fits.PrimaryHDU(data=arr, header=header)
    else:
        # the bottom-left pixel will be centered at (dx/2 + 1) in FITS coordinates if we start at 0
        # so we start at -dx/4 so that the bottom-left pixel is centered at 1,1
        # (BLC of image is at -0.5, -0.5 in FITS, pixel size is dx/2, so offset is dx/4)
        # we don't want to wrap, so we use max(pixel, 0)
        # the percentile will be over a smaller region, but that should be OK
        chunks = [[img[(slice(max(sty, 0), sty+dx), slice(max(stx, 0), stx+dx))]
                for stx in range(-dx//4, img.shape[1]+dx//2, dx//2)]
                for sty in range(-dx//4, img.shape[0]+dx//2, dx//2)
                ]

        # only include positive values (actually no that didn't work)
        arr = np.array(
            [[np.nanpercentile(ch, percentile)  # if np.any(ch > 0) else 0
            for ch in row]
            for row in progressbar(chunks)]
        )

        # I can never remember how to do this, but I'm *certain* this is wrong (independent of what this next line says:)
        # but empirically I'm _pretty_ sure dx/4 + 0.5 looks like it matches maybe
        # with revised version, we drop the shift
        wwsl = ww[::dx//2, ::dx//2]

        return fits.PrimaryHDU(data=arr, header=wwsl.to_header())


def nozero_percentile(arr, pct, **kwargs):
    """
    nanpercentile([nan, nan, nan]) gives nan, but we want zero, so this function
    returns zero if everything is nan
    """
    arr = arr.copy()
    arr[arr == 0] = np.nan
    rslt = np.nanpercentile(arr, pct, **kwargs)

    # sometimes whole rows are zero.  We want to retain these as zero.
    return np.nan_to_num(rslt)


def destreak_data(data, percentile=10, median_filter_size=256, add_smoothed=True,
                  caller_restores_large_scales=False):
    """Subtract the 1/f streaks from one NIRCam frame, in place.

    The streak estimator is a per-ROW percentile taken independently inside
    each of the four 512-column amplifier chunks, so the correction is
    ``f(y, x//512)`` -- four independent row profiles, not one.

    ``add_smoothed=True`` (the safe mode) subtracts that profile and adds back
    a median-filtered copy of it, so only the structure finer than
    ``median_filter_size`` rows is removed.  The frame keeps its sky pedestal
    and its large angular scales.

    ``add_smoothed=False`` subtracts the profile and adds back NOTHING.  That
    drives every chunk's row percentile to exactly zero, which deletes the sky
    pedestal and the large-scale sky along with the streaks and sends
    ``percentile``% of the pixels negative.  It is only correct when the caller
    restores those scales from an external background map on the very next
    line, so a caller that intends it must say so with
    ``caller_restores_large_scales=True``.  Anything else raises
    ``DestreakWouldDeleteSky`` rather than quietly returning a frame with no
    sky in it.

    That guard is the point of this function's current shape.  Both production
    call sites passed ``use_background_map=True``, which
    ``destreak()`` wired to ``add_smoothed=False``; when no background map was
    registered for the frame's (proposal, observation, filter) -- the case for
    all but seven (field, observation, filter) combinations in the archive --
    the restore step silently did not happen and the bare subtraction was the
    whole operation.  Measured on the products that produced: the per-frame sky
    pedestal went to zero (e.g. Cloud E/F F210M 6.37 -> 1.45 MJy/sr, p10
    4.36 -> -0.001), 10.06% of pixels went negative in every affected frame,
    and the mosaic-level transfer of diffuse sky fell to a slope of 0.14-0.47
    against 0.95 where a map existed.

    ``median_filter_size`` is a window in ROWS.  It is clipped to the profile
    length, which also removes the old ``np.ones(2048)`` hardcode.  The branch
    it replaces read ``if median_filter_size >= 2048: median_filter(...) else:
    np.ones(2048) * np.median(pct)``, i.e. it returned a flat scalar for every
    window smaller than the whole detector -- including this function's own
    default of 256 -- and only did the filtering the docstring promised at the
    one setting where the window spans the array.  ``median_filter(pct, 2048)``
    on a length-2048 profile is not flat (``mode='reflect'`` keeps ~60% of the
    input std), which is why that setting worked and the default did not.
    """
    if not add_smoothed and not caller_restores_large_scales:
        raise DestreakWouldDeleteSky(
            "destreak_data(add_smoothed=False) subtracts a per-row, per-512-column "
            "percentile and adds nothing back, which deletes the sky pedestal and "
            "the large-scale sky along with the 1/f streaks and drives "
            f"~{percentile}% of the pixels negative.  Pass add_smoothed=True to "
            "destreak without an external background map.  Only a caller that "
            "restores the large scales itself immediately afterwards -- i.e. one "
            "that has already resolved a background map with "
            "background_map_path() -- may ask for this, and it must say so with "
            "caller_restores_large_scales=True.")

    if data.shape[1] != 2048:
        # The 512-column chunking IS the NIRCam amplifier layout.  On a
        # narrower detector `range(0, 2048, 512)` silently produced empty
        # slices for the chunks past the edge and destreaked only part of the
        # frame; on a wider one it would ignore the remainder.
        raise ValueError(
            f"destreak_data is written for the 2048-column NIRCam detector and its "
            f"four 512-column amplifiers; got {data.shape[1]} columns.  MIRI and "
            f"NIRISS do not have this readout structure and are not destreaked.")

    for start in range(0, 2048, 512):
        chunk = data[:, slice(start, start + 512)]
        pct = nozero_percentile(chunk, percentile, axis=1)
        if add_smoothed:
            smoothed_pct = median_filter(pct, min(int(median_filter_size), pct.size))
            data[:, slice(start, start + 512)] = chunk - pct[:, None] + smoothed_pct[:, None]
        else:
            data[:, slice(start, start + 512)] = chunk - pct[:, None]

    return data


def background_map_path(header, background_mapping=background_mapping,
                        bgmap_path=None):
    """Path of the background map registered for this frame, or ``None``.

    ``None`` means no map is registered for the frame's (proposal,
    observation, filter) -- a field nobody has built a map for yet.  Callers
    treat that as "destreak without one" (``add_smoothed=True``), not as
    permission to subtract the sky.

    A map that IS registered but whose file is absent raises
    ``FileNotFoundError``.  That distinction is deliberate: a stale entry is a
    configuration error, and the one time it degraded silently instead it went
    unnoticed for two years -- the brick-1182 entries pointed at
    ``..._background.fits`` names that had been renamed ``.fits_stale`` in
    2023, so those filters were plain sky subtractions the whole time.

    ``bgmap_path`` defaults to ``<BACKGROUND_MAP_ROOT>/<regionname>/images/``
    using the ``regionname`` recorded beside the filter entries.
    """
    # `background_mapping` is keyed on the UNPADDED proposal ('2221'), while
    # PROGRAM carries MAST's five-character padded form ('02221', '10678').
    # The slice [1:5] this replaces read '0678' off a 10678 frame, missed the
    # mapping, and returned the frame with no background added -- a warning,
    # not an error (issue #414).
    proposal_id = proposal_id_from_program(header['PROGRAM'])
    obsid = header['OBSERVTN'].strip()

    if (proposal_id not in background_mapping
            or obsid not in background_mapping[proposal_id]):
        return None

    bgm = background_mapping[proposal_id][obsid]
    # The FILTER/PUPIL rule lives in mast_names now.  The copy that used to be
    # here only recognised six hardcoded narrow/medium bands, so every
    # CLEAR-pupil WIDE band (F115W/F200W/F356W/F444W) resolved to the literal
    # 'CLEAR' and could never match a mapping key.
    filtername = filtername_from_header(header).lower()
    if filtername not in bgm:
        return None

    if bgmap_path is None:
        bgmap_path = os.path.join(BACKGROUND_MAP_ROOT, bgm['regionname'], 'images')
    path = os.path.join(bgmap_path, bgm[filtername])
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"background_mapping registers {bgm[filtername]!r} for proposal "
            f"{proposal_id} obs {obsid} filter {filtername.upper()}, but "
            f"{path} does not exist.  Either build the map or drop the entry -- "
            f"leaving it in place would destreak this filter with no background "
            f"restoration.")
    return path


def add_background_map(data, hdu, background_mapping=background_mapping,
                       bgmap_path=f'{basepath}/images/',
                       verbose=False,
                       ext=('SCI', 1),
                       return_background=False,
                       bgfile=None):
    """Add the mosaic background back onto a fully-subtracted destreaked frame.

    ``bgfile`` may be a path already resolved by ``background_map_path`` (what
    ``destreak`` does, so the decision to subtract fully and the decision about
    which map to add are the same decision).  Left ``None``, this resolves it
    itself and keeps the historical warn-and-return behaviour for a frame with
    no registered map.
    """
    if bgfile is None:
        filtername = filtername_from_header(hdu[0].header)
        bgfile = background_map_path(hdu[0].header,
                                     background_mapping=background_mapping,
                                     bgmap_path=bgmap_path)
        if bgfile is None:
            print(f"WARNING: filter {filtername} is not in background mapping {background_mapping}.  "
                  "This likely means you haven't made it yet!")
            return data

    if verbose:
        print(f'Background filename: {bgfile}')

    ww = WCS(hdu[ext].header)

    bg = fits.getdata(bgfile)

    # Defence-in-depth: a background map with NaN holes (e.g. coverage gaps, or
    # a medfilt builder that spread zeros->NaN) would otherwise propagate NaNs
    # into the destreaked frame via bilinear sampling + `data += bg_sampled`,
    # punching NaN blobs into the merged mosaic.  Fill holes with the map median
    # so the added background is always finite.
    if not np.all(np.isfinite(bg)):
        bg = np.nan_to_num(bg, nan=np.nanmedian(bg))

    # we want the middles of the columns
    for start in range(0, 2048, 512):
        # pixel coordinates (px)
        pxy = np.arange(2048)
        pxx = np.ones(2048) * (start + 256)
        crds = ww.pixel_to_world(pxx, pxy)

        wwbg = WCS(fits.getheader(bgfile))
        bgx, bgy = wwbg.world_to_pixel(crds)

        bg_sampled = map_coordinates(bg, [bgy, bgx], order=1)
        if verbose:
            print(f'bg_sampled shape: {bg_sampled.shape}, nanmedian: {np.nanmedian(bg_sampled)}')

        data[:, slice(start, start + 512)] += bg_sampled[:, None]

    return data


def destreak(frame, percentile=10, median_filter_size=256, overwrite=True, write=True,
             background_folder=None,
             background_mapping=background_mapping,
             use_background_map=False
             ):
    """
    "Massimo's Destreaker" - subtract off the median (or percentile)
    of each row, but put the large angular scales back so the frame keeps its
    sky.

    For some filters, there are zeros, so we use a 'nozero percentile' to
    mask out the zeros before calculating the percentile

    There are two ways to put the large scales back, and which one is used
    depends on whether this frame's (proposal, observation, filter) has a
    background map registered in ``background_mapping``:

    * **A map is registered** -- subtract the row percentile in full
      (``add_smoothed=False``) and add the mosaic background back on top.  The
      restored background is a real map of the field, so this is the better of
      the two.  Seven (field, observation, filter) combinations in the archive
      qualify: brick 2221/001 in six filters, and Cloud C 2221/002 in F405N.
    * **No map is registered** -- destreak in place with ``add_smoothed=True``,
      subtracting only the structure finer than ``median_filter_size`` rows.
      The frame keeps its pedestal and most of its large angular scales.

    ``use_background_map`` now selects only whether the first branch is
    *allowed*, not whether the second one is skipped.  It used to be wired
    straight to ``add_smoothed = not use_background_map``, which meant a
    frame with ``use_background_map=True`` and no registered map -- the case
    for all but those seven combinations -- got the full subtraction with
    nothing added back at all.  ``destreak_data`` now refuses that
    combination outright (``DestreakWouldDeleteSky``); this function no longer
    asks for it.

    NOTE this changes the pixels of every destreaked product outside those
    seven combinations: they previously came out with their sky pedestal set
    to zero and ~10% of pixels negative, and now keep it.  Products made
    before this commit and after it must not be mixed in one mosaic or one
    catalog.

    ``background_folder`` overrides the directory the map is looked up in.  It
    used to default to brick's ``images/`` and was then never read -- the
    lookup directory was rebuilt from ``regionname`` a few lines further down,
    so passing it did nothing.  It now defaults to ``None`` (keep deriving the
    directory from ``regionname``) and is honoured when set.
    """
    assert frame.endswith('_cal.fits')
    print(f"Destreaking {frame}")
    hdu = fits.open(frame)

    data = hdu[('SCI', 1)].data

    # Resolve the map BEFORE destreaking, so the choice to subtract the sky in
    # full and the ability to put it back are the same decision.  Raises if an
    # entry exists but its file does not.
    bgfile = (background_map_path(hdu[0].header, background_mapping=background_mapping,
                                  bgmap_path=background_folder)
              if use_background_map else None)
    if use_background_map and bgfile is None:
        print(f"WARNING: no background map registered for {frame}; destreaking "
              f"with add_smoothed=True (large scales kept in-frame) instead of "
              f"subtracting them.", flush=True)

    data = destreak_data(data, percentile=percentile,
                         median_filter_size=median_filter_size,
                         add_smoothed=bgfile is None,
                         caller_restores_large_scales=bgfile is not None,
                         )

    if bgfile is not None:
        data = add_background_map(data, hdu, background_mapping=background_mapping,
                                  bgfile=bgfile)

    hdu[('SCI', 1)].data = data

    if write:
        outname = frame.replace("_cal.fits", "_destreak.fits")
        hdu.writeto(outname, overwrite=overwrite)

        return outname

    else:
        return hdu
