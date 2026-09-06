#!/usr/bin/env python
"""Per-EXPOSURE module-locked VIRAC2 offsets (relative-internal + bulk-absolute).

Motivation (measured 2026-06-20, F115W):
  Per-VISIT locking (one shift/visit) leaves real per-exposure jitter -- most exposures
  ~1.5 mas, but individual exposures up to ~7 mas (e.g. 1182 visit001 exp11/12 at
  dDec ~-8 vs visit -1.8).  That blurs those exposures in the mosaic.  Naive per-exposure
  VIRAC2 solving instead injects ~2.4 mas/exposure VIRAC2 noise (only ~215 matches/exp).

Solution -- decouple the two so we get BOTH:
  1. Per-exposure RELATIVE shift vs the dense INTERNAL per-visit consensus (thousands of
     stars, sub-mas) -> removes jitter precisely, no VIRAC2 noise.
  2. ONE per-VISIT BULK absolute tie consensus->VIRAC2 (all visit stars pooled, ~0.5 mas)
     -> sets the zero point and absorbs the per-visit guide-star pointing error (~17"/~2").
  shift[visit,exp] = bulk[visit] + relative[visit,exp]
  Applied to the pristine assign_wcs (SIAF) cal frame this lands every exposure on VIRAC2
  with jitter removed; module-locked (one shift for all detectors -> SIAF lock intact).

SIAF positions are recovered by UNDOing the recorded per-detector RAOFFSET/DEOFFSET in each
per-frame catalog meta.  The coarse per-visit shift (median RAOFFSET) bridges the large
guide-star offset so the consensus matches VIRAC2.

Output: <basepath>/offsets/Offsets_JWST_Brick<prop>_VIRAC2locked.csv, keyed (Visit, Exposure,
Filter) with Exposure INT (matches fix_alignment's per-exposure lookup).  Rows for OTHER
filters AND other (proposal,field) Visit prefixes are PRESERVED (field-safe), so a per-field
run does not clobber another field that shares the per-proposal table.

General for ALL GC fields (migrated 2026-07-17 from brick-jwst-2221
analysis/build_virac2_locked_perexp.py, PR #39). The coarse absolute tie uses the repo's
sanctioned ``jwst_gc_pipeline.photometry.astrometry_offsets.measure_offset`` (density-immune
all-pairs histogram with an internal window SWEEP + contrast/error, sitting next to the
``assert_sparse_reference_for_nn_median`` guard) -- NEVER a nearest-neighbour match
against the dense mosaic (that fabricates a false 'clean tie at
zero' once the offset exceeds the NN spacing; cloudef obs005 F162M is a real ~7.5" gross
offset an NN read as ~0).  Add a field by adding a REGION entry (proposal/field/basepath/
{filt: (subdir, obs-epoch, mtag)}).  A multi-pointing field needs a VIRAC2 cache covering
ALL its pointings.

``mtag`` names the merge stage of the per-frame catalogs the tie is measured on, keyed
per (OBSERVATION, filter): when two observations catalog into one directory under
the same names, whichever ran last owns each stage, so the mtag must name a stage the
region's OWN observation actually has (cloudef obs005 reached only m2 while obs002 reached
m7 -- see the cloudef entries).  ``_gather`` verifies this from each catalog's crf and
refuses a mismatch, so another observation's tie stays out of this one's table.

Usage:  python -m jwst_gc_pipeline.reduction.build_virac2_offsets --region <key> [filt ...]
"""
import sys, glob, os, re, argparse
from datetime import datetime
import numpy as np
import astropy.units as u
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import mad_std
import warnings
warnings.filterwarnings('ignore')
# GENERATION LOCK: recompute RA/Dec from stable detector x/y through the live crf WCS.
from ..astrometry_utils import _resolve_existing_path
from ..mast_names import jw_prefix
# Sanctioned density-immune, window-swept, guarded bulk-offset estimator (replaces the
# bespoke coarse_xcorr this module used to reimplement -- see brick-jwst-2221 PR #39 review).
from ..photometry.astrometry_offsets import measure_offset
# ONE canonical form for the visit-group token on both sides of the table: the
# builder writes it here, update_offsets_table / lookup_consensus_offset /
# fix_alignment compare against it.  (Both '_vgroup07101' and '_vgroup7101' exist
# on disk for the same group, so keying on the raw token would split one pointing's
# detectors across two rows -- the exact failure the column exists to prevent.)
from ..photometry.astrometry_checkpoint import vgroup_key

V2EP = 2014.0
SEARCH = 0.3 * u.arcsec
CLIP_MAS = 60.0
CLUSTER_MAS = 50.0
# Coarse per-visit bulk tie: large-radius crowding-robust search (cross-correlation
# peak), NOT nearest-neighbour. A nearest-neighbour match with radius SEARCH cannot
# recover an offset larger than SEARCH: in a dense field every star has a chance
# neighbour within SEARCH, so the median collapses to ~0 and the visit is left
# UNcorrected (silently). This was the brick-1182 recurrence: the old coarse bridge
# was taken from each catalog's previously-applied RAOFFSET, which was itself ~0, so
# every rebuild re-confirmed ~0. COARSE_MAXSEP must exceed the largest expected raw
# guide-star pointing error (~2" for brick); QA_FAIL_MAS rejects a bad solve.
COARSE_MAXSEP = 5.0 * u.arcsec   # per-FILTER i2d seed radius; refined per-visit below
COARSE_BIN = 0.08          # arcsec, coarse-histogram bin (refined by the SEARCH fine step)
COARSE_MIN_PEAK_RATIO = 5.0  # i2d xcorr peak/background below this -> tie ambiguous, fail loud
# PER-VISIT coarse radius.  MUST exceed the largest raw per-visit guide-star pointing
# error, NOT the ~2" once thought: brick-1182 visit001 is ~22" off (-17.5"/+13.5") while
# visit002 is ~2".  The single mosaic-wide COARSE_MAXSEP seed captures only the dominant
# visit; the other visit needs its OWN large-radius histogram xcorr (below) or it silently
# inherits the dominant visit's shift (the 2026-07 brick-1182 visit001 corruption).
COARSE_MAXSEP_VISIT = 25.0 * u.arcsec


def coarse_xcorr(sc, ref, maxsep=COARSE_MAXSEP):
    """Robust bulk COORDINATE offset (arcsec Delta-alpha/Delta-delta, NO cosd) to ADD to
    ``sc`` to land on ``ref`` -- via the SANCTIONED ``measure_offset`` (density-immune
    all-pairs histogram with an internal window SWEEP + contrast/error + the
    assert_sparse_reference_for_nn_median guard).  ``sc`` must be a CLEAN, high-SNR source
    list (drizzled-mosaic detections), NOT raw per-frame SIAF positions.

    Returns ``(dra, ddec, npairs, contrast, window_arcsec, swept)`` or ``(None,)*6``.
    NB (why this is not a bespoke histogram / never an NN match): a NN match against the
    dense mosaic piles wrong pairs at ~0 and fabricates a false 'clean tie at zero' once
    the offset exceeds the NN spacing; and a naive max-contrast-across-windows on a coarse
    (0.08") bin can select a spurious wide-window peak (both observed on cloudef obs005
    F162M, a real ~7.5" gross offset that NN read as ~0).  ``measure_offset`` sweeps
    narrow->wide with fine bins and flags ``swept`` when the tie only appeared after
    widening -- the gross-shift tell.  (brick-jwst-2221 PR #39 review.)
    """
    r = measure_offset(sc, ref, maxsep=maxsep, sweep=True, min_pairs=50)
    if r is None or not r.get("ok"):
        return (None,) * 6
    # measure_offset returns ON-SKY mas (ref - sc); convert to COORDINATE arcsec (no cosd).
    cosd = max(np.cos(np.radians(float(np.median(ref.dec.deg)))), 1e-6)
    return (r["dra"] / 1000.0 / cosd, r["ddec"] / 1000.0,
            int(r["npairs"]), float(r["contrast"]), float(r["window_arcsec"]),
            bool(r.get("swept", False)))


from jwst_gc_pipeline.frame_wcs import frame_wcs


def detect_i2d_sources(i2d_path, thr=80.0, fwhm=2.5):
    """Bright high-SNR source SkyCoords from a drizzled mosaic (for the coarse tie)."""
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.stats import sigma_clipped_stats
    from photutils.detection import DAOStarFinder
    sci = fits.open(i2d_path)['SCI']
    w = WCS(sci.header)
    d = sci.data.astype('float32')
    _, med, std = sigma_clipped_stats(d, sigma=3.0)
    t = DAOStarFinder(fwhm=fwhm, threshold=thr * std)(d - med)
    return SkyCoord(w.pixel_to_world(t['xcentroid'], t['ycentroid']))


#: Per-module mosaic labels, i.e. everything that is not ``merged``.
MODULE_MOSAICS = ('nrca', 'nrcb', 'nrcalong', 'nrcblong')


def mosaic_candidates(stem, n_modules=2):
    """``(label, path)`` mosaics that EXIST for a filter, best first.

    ``merged`` is preferred only when the build actually has **two or more
    modules**: there it is the one product the inter-module seam lives in, so it
    is the only correct seed.  On a SINGLE-module field a ``-merged`` product
    cannot be a merge -- there is nothing to merge -- so it can only be a
    leftover from an earlier generation, and preferring it is backwards.

    sickle is exactly that case and it is not hypothetical: its
    ``f210m-merged_i2d.fits`` is dated **2026-04-19** against a current
    ``f210m-nrcb_i2d.fits`` from **2026-08-04**, and the two disagree by ~120 mas
    in ddec.  The seed is not free -- running the fine step from each gives bulk
    (-0.0190, -0.1071) vs (-0.0210, -0.0902), so ~17 mas of the F210M answer is
    decided by which mosaic seeds it.

    Otherwise take the NEWEST mosaic on disk, which is the generation the rest of
    the products came from.
    """
    cands = [(m, f"{stem}-{m}_i2d.fits") for m in ('merged',) + MODULE_MOSAICS]
    present = [(m, p) for m, p in cands if os.path.exists(p)]
    if not present:
        return []
    if n_modules >= 2 and present[0][0] == 'merged':
        return present
    # newest first; ties broken by the fixed order above so the choice is stable
    return sorted(present, key=lambda mp: -os.path.getmtime(mp[1]))


def perframe_matches(basename, mtag):
    """Is ``basename`` the per-frame catalog for merge tag ``mtag``?

    The ``exp*`` glob wildcard swallows any suffix between the exposure number
    and the merge tag, so globbing for ``_m3`` also matches the GROUPED-fit
    variant ``..._exp00001_group_m3_daophot_basic.fits``.  Both then claim the
    same frame and the duplicate check aborts the build, reporting them as "a
    stale duplicate from before the per-frame names carried the observation
    token" -- which they are not; they are two legitimate products of the same
    exposure.  Require the exposure number to be followed IMMEDIATELY by the
    merge tag.  Only sickle carries ``group_`` per-frame catalogs today
    (sgrc/cloudc/brick have none), which is why the over-match went unseen.
    """
    return bool(re.search(rf'_exp\d+{re.escape(mtag)}_daophot_basic\.fits$',
                          basename))


def coarse_from_i2d(filt, rc, ref, n_modules=2):
    """Per-FILTER coarse bulk tie measured on the drizzled mosaic (clean) vs VIRAC2.
    This seeds every visit of the filter; the per-visit/per-exposure fine same-star
    pass then resolves the <SEARCH residual.  Returns (dra, ddec) arcsec, or None.

    ``n_modules`` is how many modules this build actually has catalogs for; it
    decides whether ``merged`` is the right seed -- see ``mosaic_candidates``.
    """
    sub = rc['filts'][filt][0]
    stem = (f"{rc['basepath']}/{sub}/pipeline/"
            f"{jw_prefix(rc['proposal'])}-o{rc['field']}_t001_nircam_clear-{filt}")
    candidates = mosaic_candidates(stem, n_modules=n_modules)
    if not candidates:
        print(f"  [coarse] no i2d for {filt}: nothing matches {stem}-*_i2d.fits")
        return None
    mod, i2d = candidates[0]
    # SAY which mosaic and how old.  The seed decides ~17 mas of the answer on
    # sickle, so "which generation did this come from" has to be in the log.
    stamp = datetime.fromtimestamp(os.path.getmtime(i2d)).strftime('%Y-%m-%d')
    others = ', '.join(f"{m}({datetime.fromtimestamp(os.path.getmtime(p)):%Y-%m-%d})"
                       for m, p in candidates[1:]) or 'none'
    print(f"  [coarse] {filt}: seeding on {os.path.basename(i2d)} [{mod}, {stamp}]; "
          f"also present: {others}", flush=True)
    sc = detect_i2d_sources(i2d)
    # ONE call: coarse_xcorr -> measure_offset sweeps the window internally (narrow->wide,
    # density-immune) and flags `swept` when the tie only appeared after widening (the
    # gross-shift tell -- e.g. cloudef obs005 F162M is really ~7.5" off).
    dra, ddec, n, contrast, window, swept = coarse_xcorr(sc, ref, maxsep=COARSE_MAXSEP)
    if dra is None:
        print(f"  [coarse] {filt}: no coherent i2d tie at any window -> refusing", flush=True)
        return None
    if swept:
        print(f"  [coarse] {filt}: GROSS shift -- tie ({dra:+.3f},{ddec:+.3f})\" only found at "
              f"window {window:g}\" (contrast {contrast:.0f}); investigate this frame", flush=True)
    # refine to mas precision with a fine same-star pass on the CLEAN i2d detections (now
    # within <SEARCH of VIRAC2, so the nearest pair is the RIGHT counterpart -- the sanctioned
    # post-verified-tie refinement, not a coarse NN).
    shifted = SkyCoord((sc.ra.deg + dra / 3600.0) * u.deg, (sc.dec.deg + ddec / 3600.0) * u.deg)
    fine = coord_shift(shifted.ra.deg, shifted.dec.deg, ref)
    if fine is not None:
        dra += fine[0]; ddec += fine[1]
    print(f"  [coarse] {filt}: i2d tie ADD=({dra:+.4f},{ddec:+.4f})\" window={window:g}\" "
          f"contrast={contrast:.0f} npairs={n} "
          f"i2dfine=({(fine[0]*1000 if fine else 0):+.1f},{(fine[1]*1000 if fine else 0):+.1f})mas "
          f"({len(sc)} i2d srcs)", flush=True)
    return dra, ddec

# region -> proposal/field/basepath + {filt: (subdir, obs-epoch, mtag)}
REGION = {
    '1182': dict(proposal='1182', field='004', basepath='/orange/adamginsburg/jwst/brick',
                 filts={'f115w': ('F115W', 2022.703, '_m3'), 'f200w': ('F200W', 2022.703, '_m3'),
                        'f356w': ('F356W', 2022.703, '_m2'), 'f444w': ('F444W', 2022.703, '_m2')}),
    'cloudc': dict(proposal='2221', field='002', basepath='/orange/adamginsburg/jwst/cloudc',
                   filts={'f182m': ('F182M', 2023.30, '_m3'), 'f187n': ('F187N', 2023.30, '_m3'),
                          'f212n': ('F212N', 2023.30, '_m3'), 'f405n': ('F405N', 2023.30, '_m3'),
                          'f410m': ('F410M', 2023.30, '_m3'), 'f466n': ('F466N', 2023.30, '_m3')}),
    # cloudef (2092): Cloud E (obs 002) + Cloud F (obs 005), separate pointings ->
    # separate region keys; combine their VIRAC2locked tables into one Offsets file after.
    #
    # ⚠ THE mtag IS PER-OBSERVATION, NOT PER-FILTER-ONLY (2026-07-29).  Both
    # observations catalog into the SAME directory under the SAME per-frame names
    # (cloudef has no `_oNNN_` token anywhere -- 0 of 11635 catalogs carry one), so
    # whichever run wrote a given stage last OWNS that stage's files.  Census by crf
    # provenance (meta['FILENAME']), F*/*_visit*_vgroup*_exp*<mtag>_daophot_basic:
    #
    #     stage    F162M          F210M          F360M         F480M
    #     _m1      obs005 (72)    obs005 (72)    obs005 (24)   obs005 16 + obs002 8
    #     _m2      obs005 (72)    obs005 (72)    obs005 (24)   obs005 16 + obs002 8
    #     _m3      obs002 (72)    obs002 (64)    obs002 (16)   obs002 (16)
    #     _m4..m7  obs002         obs002         obs002        obs002
    #
    # obs 005 IS cataloged -- through m2, for all four filters.  It simply has no
    # m3+ products, so an `_m3` glob under cloudef5 can only ever match obs 002's
    # files.  Before _gather verified the observation, that silently relabelled obs
    # 002's tie as jw02092005001 -- and cloudef obs005 F162M is a real ~7.5" gross
    # offset, so it would have been badly wrong.  The fix is the mtag, not the data:
    # cloudef5 uses `_m2` (its own deepest stage), cloudef2 keeps `_m3`.
    #
    # Measuring the tie on m2 is not a downgrade: the builder takes positions from
    # x_fit/y_fit through the LIVE crf GWCS (load_siaf) with a qfit<0.4 / flux>0 cut,
    # and later merge stages refine deblending and photometry, not the frame.  m2 is
    # also the stage the pipeline itself measures and CORRECTS astrometry at (m3+ are
    # frozen -- ASTROMETRY_CHECKPOINTS.md), and 1182 already ties f356w/f444w at _m2
    # for the same reason: it is the deepest stage those products reached.
    # Verified 2026-07-29: cloudef5 at _m2 gathers jw02092005001 only -- 8 exposures
    # x 8 detectors in F162M/F210M, 8 x 2 in F360M/F480M, no obs-002 contamination --
    # and all four obs-005 merged i2d mosaics exist for the coarse tie.
    'cloudef2': dict(proposal='2092', field='002', basepath='/orange/adamginsburg/jwst/cloudef',
                     filts={'f162m': ('F162M', 2023.21, '_m3'), 'f210m': ('F210M', 2023.21, '_m3'),
                            'f360m': ('F360M', 2023.21, '_m3'), 'f480m': ('F480M', 2023.21, '_m3')}),
    # obs 005's tree moved to `cloudef_controlfield` with the field split (this
    # PR); 3,991 paths, receipt in
    # cloudef_controlfield/MIGRATION_from_cloudef_20260820T185854Z.json.  Left
    # pointing at `cloudef`, this region reads obs 002's catalogs instead --
    # 72/64/16/16 `_m3` across the four bands, none of them obs 005's -- and
    # would lock obs 002's offsets into obs 005's table.
    #
    # NOTE the control field has no `_m3` catalogs yet (only m1/m2, and only in
    # F360M: 8 each), so `--region cloudef5` cannot be rebuilt until its chain
    # has run to m3 under the new tree.  The mtag stays `_m2` for that reason.
    'cloudef5': dict(proposal='2092', field='005',
                     basepath='/orange/adamginsburg/jwst/cloudef_controlfield',
                     filts={'f162m': ('F162M', 2023.21, '_m2'), 'f210m': ('F210M', 2023.21, '_m2'),
                            'f360m': ('F360M', 2023.21, '_m2'), 'f480m': ('F480M', 2023.21, '_m2')}),
    # sgrc (4147/012).  The VIRAC2locked table was previously authored without a
    # REGION entry and covered only 7 of the 8 reduced bands -- F115W had NO row, so
    # the m2 checkpoint's F115W corrections matched nothing and update_offsets_table
    # hard-failed the whole re-tie (2026-07-27).  Registering the field here makes
    # F115W buildable by the sanctioned path instead of hand-authored.
    # Epoch 2023.72 matches catalogs/gaia_virac2_refcat_epoch2023.72.fits.
    'sgrc': dict(proposal='4147', field='012', basepath='/orange/adamginsburg/jwst/sgrc',
                 filts={'f115w': ('F115W', 2023.72, '_m3'), 'f162m': ('F162M', 2023.72, '_m3'),
                        'f182m': ('F182M', 2023.72, '_m3'), 'f212n': ('F212N', 2023.72, '_m3'),
                        'f360m': ('F360M', 2023.72, '_m3'), 'f405n': ('F405N', 2023.72, '_m3'),
                        'f470n': ('F470N', 2023.72, '_m3'), 'f480m': ('F480M', 2023.72, '_m3')}),
    # sgrb2 (5365/001) and quintuplet (2045/003).  Both had VIRAC2locked tables
    # authored outside the builder and, like sgrc, with NO Module column.  The m2
    # checkpoint emits corrections keyed (visit, exposure, module), and
    # update_offsets_table skips the module narrowing when that column is absent
    # -- so all 8 detectors' corrections for one exposure land on the SAME row and
    # SUM into an N-fold over-correction.  Registering them here lets both tables be
    # rebuilt with --per-module so corrections map 1:1.
    # Epochs from EXPSTART of the obs' own NIRCam frames: sgrb2 60560.715 ->
    # 2024.685, quintuplet 60535.756 -> 2024.617.  (sgrb2's MIRI bands F770W/
    # F1280W/F2550W are deliberately absent: this builder ties NIRCam detectors.)
    'sgrb2': dict(proposal='5365', field='001', basepath='/orange/adamginsburg/jwst/sgrb2',
                  filts={'f150w': ('F150W', 2024.685, '_m3'), 'f182m': ('F182M', 2024.685, '_m3'),
                         'f187n': ('F187N', 2024.685, '_m3'), 'f210m': ('F210M', 2024.685, '_m3'),
                         'f212n': ('F212N', 2024.685, '_m3'), 'f300m': ('F300M', 2024.685, '_m3'),
                         'f360m': ('F360M', 2024.685, '_m3'), 'f405n': ('F405N', 2024.685, '_m3'),
                         'f410m': ('F410M', 2024.685, '_m3'), 'f466n': ('F466N', 2024.685, '_m3'),
                         'f480m': ('F480M', 2024.685, '_m3')}),
    # brick 2221/001.  The second field the
    # `test_every_virac2_locked_field_has_a_builder_region` guard caught: 2221
    # had a region for obs 002 (`cloudc`) but none for obs 001, even though both
    # are VIRAC2 TABLE_LOCKED and they keep SEPARATE tables in separate trees
    # (brick/offsets/ vs cloudc/offsets/, same basename).  Note `1182` above is
    # brick's OTHER observation, obs 004 -- one target, two proposals, two
    # tables, and the shared `brick` basepath is why the filter sets must stay
    # disjoint (o001 F182M/F187N/F212N/F405N/F410M/F466N,
    # o004 F115W/F200W/F356W/F444W).
    #
    # Epoch from DATE-OBS 2022-08-28 = 2022.655.
    'brick2221': dict(proposal='2221', field='001',
                      basepath='/orange/adamginsburg/jwst/brick',
                      filts={'f182m': ('F182M', 2022.655, '_m3'),
                             'f187n': ('F187N', 2022.655, '_m3'),
                             'f212n': ('F212N', 2022.655, '_m3'),
                             'f405n': ('F405N', 2022.655, '_m3'),
                             'f410m': ('F410M', 2022.655, '_m3'),
                             'f466n': ('F466N', 2022.655, '_m3')}),
    # sgra (1939/001), the Galactic Centre pointing.  It had NO region here at
    # all, and that absence was a hard block rather than an inconvenience: its
    # m12 finalize dies every iteration on
    #
    #     OffsetsTableUpdateError: cannot pool corrections for
    #     Offsets_JWST_Brick1939_VIRAC2locked.csv: 8 corrections spanning module
    #     families ['nrca', 'nrcb'] land on the same row(s)
    #
    # which is the guard working as designed -- the table is 36 rows keyed
    # (Visit, Exposure, Filter) with no Module column, so module A's and module
    # B's corrections median together.  The guard's own message names the remedy
    # (`build_virac2_offsets --per-module`), and that remedy could not be run for
    # this field because `--region` had no key for it (issue #409).
    #
    # One visit (jw01939001001), three bands, 216 frames.  Epoch from DATE-OBS
    # 2022-09-19 = 2022.715.  `_m3` because per-exposure catalogs exist through
    # _m7 for all three bands, and _m3 is what every other VIRAC2-tied GC region
    # here reads.
    'sgra': dict(proposal='1939', field='001',
                 basepath='/orange/adamginsburg/jwst/sgra',
                 filts={'f115w': ('F115W', 2022.715, '_m3'),
                        'f212n': ('F212N', 2022.715, '_m3'),
                        'f405n': ('F405N', 2022.715, '_m3')}),
    'quintuplet': dict(proposal='2045', field='003',
                       basepath='/orange/adamginsburg/jwst/quintuplet',
                       filts={'f212n': ('F212N', 2024.617, '_m3'),
                              'f323n': ('F323N', 2024.617, '_m3')}),
    # sickle (3958/007).  The last GC field still tied in the GNS frame while
    # ``refnames`` already called it VIRAC2 -- the live inconsistency
    # alignment_config's module docstring slated for re-measurement.  Registering
    # it here is what makes that possible: step 0 refuses to measure a fresh tie
    # for a field that is already tied ("routing that to MEASURE would record a
    # new tie for a field that already has one"), so the route to VIRAC2 is to
    # BUILD the VIRAC2 table, not to blank the recorded bulk and re-measure.
    # Epoch from EXPSTART of its own NIRCam frames: MJD 60545.503 -> 2024.643,
    # which agrees with the refcat already on disk
    # (catalogs/gaia_virac2_refcat_epoch2024.64.fits).
    # NIRCam is observation 007 and single-module (nrcb only); the 3958 MIRI data
    # are observations 001 and 002 (fields.yaml) and are NOT tied here -- this
    # builder ties NIRCam.
    'sickle': dict(proposal='3958', field='007',
                   basepath='/orange/adamginsburg/jwst/sickle',
                   # NIRCam obs 007 is B-module only.  DECLARED, not inferred: it
                   # is what authorises skipping module A, and it is what tells
                   # coarse_from_i2d that a `-merged` product here can only be a
                   # leftover generation (sickle's f210m-merged is 2026-04-19
                   # against a 2026-08-04 nrcb).  Matches the reducer's
                   # MODULES_BY_PROPOSAL_FIELD_FILTER['3958']['007'].
                   #
                   # BOTH keys are needed: module_key() keeps the full detector
                   # name for LW, so 'nrcb' alone would skip every long-wavelength
                   # filter -- three of sickle's five.
                   modules=('nrcb', 'nrcblong'),
                   filts={'f187n': ('F187N', 2024.643, '_m3'),
                          'f210m': ('F210M', 2024.643, '_m3'),
                          'f335m': ('F335M', 2024.643, '_m3'),
                          'f470n': ('F470N', 2024.643, '_m3'),
                          'f480m': ('F480M', 2024.643, '_m3')}),
    # gc2211 (2211): FIVE observations of one proposal reduced into ONE directory
    # tree, so it needs one region key per observation -- ``field`` is what makes a
    # region, and Offsets_JWST_Brick2211_VIRAC2locked.csv already separates them by
    # Visit (jw02211023001 ... jw02211050001).  Two things make this field the one
    # that most needs the Vgroup column AND the most dangerous to build naively:
    #   * SIX visit groups per filter (02201 04201 06201 08201 10201 12201) with the
    #     exposure number restarting in each -- 437-457 corrections/filter that a
    #     Vgroup-less table cannot express (the refusal this PR lifts);
    #   * the visit-group ids are REUSED across observations (vgroup 02201 exists
    #     under o023, o046, o049 AND o050) and every observation reduces to
    #     ``visit001``, so NEITHER the visit nor the vgroup separates them.  Only
    #     the crf behind each catalog does -- hence otag=True (glob this
    #     observation's own ``_oNNN_`` per-frame products) plus _gather's
    #     crf-observation refusal.  The un-tokened catalogs still in these
    #     directories are stale duplicates: F200W/f200w_nrca1_visit001_vgroup02201_
    #     exp00001_m3 has meta FILENAME = jw02211046001_..., i.e. o046 overwrote
    #     whatever o023/o049/o050 had written to that name.
    # Epochs are EXPSTART of each observation's own NIRCam frames.
    # NB the per-(obs,filter) catalog coverage is incomplete mid-campaign; a build
    # of a pair that has no catalogs (or no merged i2d) fails loudly rather than
    # writing a partial table.
    # gc2211's five observations were split into one tree each on 2026-08-21 --
    # different target fields, different sky, different epochs, sharing only a
    # proposal id.  Left pointing at the old shared tree these regions still
    # glob, they just read a tree with no frames in it.
    'gc2211_023': dict(proposal='2211', field='023', otag=True,
                       basepath='/orange/adamginsburg/jwst/gc2211_o023',
                       filts={'f150w': ('F150W', 2023.707, '_m3'),
                              'f200w': ('F200W', 2023.707, '_m3'),
                              'f277w': ('F277W', 2023.707, '_m3')}),
    'gc2211_028': dict(proposal='2211', field='028', otag=True,
                       basepath='/orange/adamginsburg/jwst/gc2211_o028',
                       filts={'f150w': ('F150W', 2023.703, '_m3'),
                              'f200w': ('F200W', 2023.703, '_m3'),
                              'f277w': ('F277W', 2023.703, '_m3')}),
    'gc2211_046': dict(proposal='2211', field='046', otag=True,
                       basepath='/orange/adamginsburg/jwst/gc2211_o046',
                       filts={'f150w': ('F150W', 2024.316, '_m3'),
                              'f200w': ('F200W', 2024.316, '_m3'),
                              'f277w': ('F277W', 2024.316, '_m3')}),
    'gc2211_049': dict(proposal='2211', field='049', otag=True,
                       basepath='/orange/adamginsburg/jwst/gc2211_o049',
                       filts={'f150w': ('F150W', 2024.633, '_m3'),
                              'f200w': ('F200W', 2024.633, '_m3'),
                              'f277w': ('F277W', 2024.633, '_m3')}),
    'gc2211_050': dict(proposal='2211', field='050', otag=True,
                       basepath='/orange/adamginsburg/jwst/gc2211_o050',
                       filts={'f150w': ('F150W', 2025.302, '_m3'),
                              'f200w': ('F200W', 2025.302, '_m3'),
                              'f277w': ('F277W', 2025.302, '_m3')}),
}
# NIRCam SW (nrca1-4/nrcb1-4) vs LW (nrcalong/nrcblong) split at ~2.4um: F070W..F212N are
# SW, F250M+ are LW.  Classify by filter number so any GC field's bands map to the right
# detector set (the old hardcoded 5-band set silently dropped f150w/f162m/f210m into LW).
def _is_sw(filt):
    m = re.match(r'f(\d{3})', filt.lower())
    return bool(m) and int(m.group(1)) < 240
SW_DETS = ['nrca1', 'nrca2', 'nrca3', 'nrca4', 'nrcb1', 'nrcb2', 'nrcb3', 'nrcb4']
LW_DETS = ['nrcalong', 'nrcblong']

# STAGE-2 JWST<->JWST cross-tie (2026-07-06).  A field tied to VIRAC2 lands with a
# ~15-30 mas residual (2MASS-tie floor + hardcoded per-module shifts in fix_alignment);
# two fields tied INDEPENDENTLY (jw01182 broadbands vs jw02221 narrows -- SAME epoch,
# same brick) therefore disagree ~15 mas, which is unacceptable for JWST<->JWST.  So
# after the VIRAC2 solve we ADD a fixed per-filter shift that lands each secondary-field
# filter on the dense 2221 MASTER (F212N) frame (~1-3 mas).  VIRAC2 stays the absolute
# tie of the master (Gaia too sparse to tie per-visit here).
#
# The shift is a HARDCODED CONSTANT (Δα no-cosδ, Δδ; arcsec), NOT auto-measured each
# build.  Rationale: the 1182<->2221 frame offset is a fixed physical quantity (the
# VIRAC2 tie is deterministic + stable), so a constant is durable and cannot silently
# regress -- whereas re-measuring the LIVE catalog residual each build self-cancels once
# a prior cross-tie has already been drizzled in (catalog already on-frame -> measures 0
# -> writes 0 -> regression).  Re-measure with `--remeasure-crosstie` (flux-vetted; it
# PRINTS suggested constants, does not write) whenever the master 2221 frame moves, then
# paste the new numbers here.  Values below measured flux-vetted vs the m7 F212N catalog
# 2026-07-06 (fluxcorr 0.65-1.00, pk/bg 400-1000, n~22k), validated by simulation to
# bring 1182<->F212N from 11-19 mas to 1-3 mas.  cloudc is single-proposal -> no cross-tie.
CROSSTIE = {
    '1182': dict(master_cat='/orange/adamginsburg/jwst/brick/catalogs/'
                            'f212n_merged_indivexp_merged*_m[0-9]*_dao_basic_vetted.fits',
                 master_name='2221 F212N',
                 shifts={  # per-filter (Δα no-cosδ, Δδ) arcsec to ADD
                     'f115w': (+0.01868, -0.00080),
                     'f200w': (+0.01852, -0.00070),
                     'f356w': (+0.02114, -0.00100),
                     'f444w': (+0.02084, -0.00090),
                 }),
}


def _empty_like(col, n):
    """A length-``n`` fill for a column missing from the other table.

    ``np.nan`` is wrong for string columns: vstack then reports
    ``The 'Module' columns have incompatible types: ['float64', 'str160']``.
    Use an empty string for str/bytes columns, NaN for everything numeric.
    """
    kind = getattr(getattr(col, 'dtype', None), 'kind', 'f')
    if kind in ('U', 'S', 'O'):
        return np.full(n, '', dtype=col.dtype)
    return np.full(n, np.nan)


def _region_key(rc):
    for rk, rv in REGION.items():
        if rv is rc:
            return rk
    return None


def crosstie_constant(filt, rc):
    """Hardcoded flux-vetted-derived (Δα no-cosδ, Δδ) arcsec to ADD to `filt`'s VIRAC2
    tie so it lands on the master frame.  (0,0) if the region/filter has no cross-tie."""
    cfg = CROSSTIE.get(_region_key(rc))
    if cfg is None:
        return 0.0, 0.0
    return cfg['shifts'].get(filt, (0.0, 0.0))
CROSSTIE_SEED_WIN = 0.5      # arcsec, candidate window for the pair histogram
CROSSTIE_RIDGE_MAS = 60.0    # near-peak radius (mas) used to LEARN the cross-band mag ridge
CROSSTIE_CORE_MAS = 50.0     # true-match core (mas) for the final clipped-median shift
CROSSTIE_MIN_N = 200         # refuse (warn, apply 0) below this many vetted core matches


def _load_cat_fluxpos(path_or_glob):
    """(SkyCoord, instrumental mag) from a merged per-band vetted catalog."""
    g = sorted(glob.glob(path_or_glob), key=os.path.getmtime)
    if not g:
        return None
    t = Table.read(g[-1])
    col = 'skycoord' if 'skycoord' in t.colnames else next(
        (c for c in t.colnames if c.startswith('skycoord')), None)
    if col is None:
        return None
    fx = farr(t['flux']) if 'flux' in t.colnames else np.full(len(t), np.nan)
    return SkyCoord(t[col]), -2.5 * np.log10(np.where(fx > 0, fx, np.nan)), os.path.basename(g[-1])


def _offhist_peak(dra_mas, dde_mas, win_mas=500.0, bin_mas=2.0):
    e = np.arange(-win_mas, win_mas + bin_mas, bin_mas)
    H, xe, ye = np.histogram2d(dra_mas, dde_mas, bins=[e, e])
    i, j = np.unravel_index(H.argmax(), H.shape)
    pk = H.max(); bg = np.median(H[H > 0]) if (H > 0).any() else 1.0
    return (xe[i] + xe[i + 1]) / 2, (ye[j] + ye[j + 1]) / 2, pk / max(bg, 1.0)


class CrosstieCatalogMissingError(RuntimeError):
    """A region keyed in ``CROSSTIE`` whose master or src catalog cannot be resolved."""


def _crosstie_cat_note(loaded, pattern):
    """One line saying why ``_load_cat_fluxpos(pattern)`` did or did not resolve."""
    if loaded is not None:
        return f"resolved {loaded[2]}"
    n = len(glob.glob(pattern))
    if n == 0:
        return f"no file matches {pattern}"
    return (f"{n} file(s) match {pattern}, but the newest carries no skycoord column")


def crosstie_offset(filt, rc, allow_missing=False):
    """Flux-vetted coordinate shift (Δα no-cosδ, Δδ; arcsec) to ADD to this filter's
    VIRAC2-locked tie so it lands on the master (2221 F212N) frame.

    An unresolved master/src catalog RAISES ``CrosstieCatalogMissingError``.  This
    function's output is pasted into ``CROSSTIE`` by hand, and a glob that matches
    nothing used to print ``(+0.00000, +0.00000)`` in the same paste-ready form as a
    measured value -- indistinguishable from a real zero, and the brick/1182 constants
    it would replace are 18-21 mas, the size the cross-tie exists to remove.  Pass
    ``allow_missing=True`` (CLI ``--allow-missing-crosstie-catalog``) to restore the
    warn-and-return-0 behaviour for a build run before those catalogs exist.

    A measurement that RAN and was declined (too few pairs, too few vetted core
    matches) still returns (0,0) with a loud warning -- that is a different severity
    and is left alone here.  :func:`crosstie_offset_detail` says WHICH of the two a
    given (0,0) is, and :func:`crosstie_block` marks it in the printed block.
    """
    return crosstie_offset_detail(filt, rc, allow_missing=allow_missing)[:2]


def crosstie_offset_detail(filt, rc, allow_missing=False):
    """:func:`crosstie_offset` plus a third element: ``None`` when the shift was
    MEASURED, else a short string saying why it was not.

    A returned ``(0.0, 0.0, None)`` cannot happen: every zero this function produces
    carries a reason, so a caller that renders the pair can mark it.  Without that,
    a declined measurement prints byte-identically to a measured zero (see
    :func:`crosstie_block`).
    """
    region = None
    for rk, rv in REGION.items():
        if rv is rc:
            region = rk; break
    cfg = CROSSTIE.get(region)
    if cfg is None:
        return 0.0, 0.0, f"region {region!r} has no CROSSTIE entry; nothing to tie"
    master_pat = cfg['master_cat']
    src_pat = (f"{rc['basepath']}/catalogs/"
               f"{filt}_merged_indivexp_merged*_m[0-9]*_dao_basic_vetted.fits")
    master = _load_cat_fluxpos(master_pat)
    src = _load_cat_fluxpos(src_pat)
    if master is None or src is None:
        if not allow_missing:
            raise CrosstieCatalogMissingError(
                f"{filt}: region {region!r} is keyed in CROSSTIE but its cross-tie "
                f"catalogs did not resolve; refusing to report a zero shift that is "
                f"indistinguishable from a measured one.\n"
                f"  master ({cfg['master_name']}): "
                f"{_crosstie_cat_note(master, master_pat)}\n"
                f"  src ({filt}): {_crosstie_cat_note(src, src_pat)}\n"
                f"Re-run with --allow-missing-crosstie-catalog only if you intend the "
                f"printed constants to be zero.")
        print(f"  [crosstie] {filt}: missing master/src catalog -> APPLYING 0 (WARN)", flush=True)
        return 0.0, 0.0, ('catalog did not resolve, --allow-missing-crosstie-catalog '
                          'was passed')
    (msc, mmag, mnm), (ssc, smag, snm) = master, src
    i2, i1, _, _ = msc.search_around_sky(ssc, CROSSTIE_SEED_WIN * u.arcsec)
    if len(i1) < CROSSTIE_MIN_N:
        print(f"  [crosstie] {filt}: only {len(i1)} candidate pairs -> APPLYING 0 (WARN)", flush=True)
        return 0.0, 0.0, (f'only {len(i1)} candidate pairs within {CROSSTIE_SEED_WIN}" '
                          f'(need {CROSSTIE_MIN_N}); an offset larger than that window '
                          f'leaves no true pairs in it')
    # on-sky separations (mas) for the seed peak + ridge learning
    dra_gc = (ssc[i2].ra - msc[i1].ra).to(u.arcsec).value * np.cos(msc[i1].dec.rad) * 1000.0
    dde = (ssc[i2].dec - msc[i1].dec).to(u.arcsec).value * 1000.0
    dm = smag[i2] - mmag[i1]
    ra0, de0, _ = _offhist_peak(dra_gc, dde)
    near = (np.hypot(dra_gc - ra0, dde - de0) < CROSSTIE_RIDGE_MAS) & np.isfinite(dm)
    if near.sum() < 20:
        print(f"  [crosstie] {filt}: too few near-peak for ridge -> APPLYING 0 (WARN)", flush=True)
        return 0.0, 0.0, (f'only {int(near.sum())} pairs within {CROSSTIE_RIDGE_MAS:.0f} mas '
                          f'of the seed peak (need 20) to learn the cross-band mag ridge')
    med = np.median(dm[near]); mad = 1.4826 * np.median(np.abs(dm[near] - med))
    tol = max(3 * mad, 0.5)
    vet = np.isfinite(dm) & (np.abs(dm - med) < tol)          # FLUX VET
    ra1, de1, pr = _offhist_peak(dra_gc[vet], dde[vet])
    core = vet & (np.hypot(dra_gc - ra1, dde - de1) < CROSSTIE_CORE_MAS)
    if core.sum() < CROSSTIE_MIN_N:
        print(f"  [crosstie] {filt}: only {core.sum()} vetted core matches -> APPLYING 0 (WARN)", flush=True)
        return 0.0, 0.0, (f'only {int(core.sum())} flux-vetted core matches within '
                          f'{CROSSTIE_CORE_MAS:.0f} mas (need {CROSSTIE_MIN_N})')
    # coordinate offset (src - master), NO cosδ on RA (table convention); ADD its NEGATION
    dra_nc = (ssc[i2[core]].ra - msc[i1[core]].ra).to(u.arcsec).value
    dde_c = (ssc[i2[core]].dec - msc[i1[core]].dec).to(u.arcsec).value
    add_ra = -float(np.median(dra_nc)); add_de = -float(np.median(dde_c))
    fcorr = np.corrcoef(mmag[i1[core]], smag[i2[core]])[0, 1]
    print(f"  [crosstie] {filt} vs {cfg['master_name']}: residual "
          f"({np.median(dra_nc) * 1000:+.1f},{np.median(dde_c) * 1000:+.1f})mas -> ADD "
          f"({add_ra * 1000:+.1f},{add_de * 1000:+.1f})mas  n={core.sum()} pk/bg={pr:.0f} "
          f"ridgeΔm={med:+.2f} fluxcorr={fcorr:.2f} [{mnm} <- {snm}]", flush=True)
    return add_ra, add_de, None


def crosstie_block(region, rc, filts, allow_missing=False):
    """The paste-ready ``--remeasure-crosstie`` block, as ``(text, n_unmeasured)``.

    Two properties the caller depends on, both of which the streaming print this
    replaced did not have:

    * **All or nothing.**  Every filter is measured BEFORE any of the text exists, so
      a ``CrosstieCatalogMissingError`` on filter *k* propagates with NOTHING
      paste-ready on stdout.  Printing as it went left filters 1..k-1 under the
      header as a block that reads complete while being short one filter -- and a
      filter absent from ``CROSSTIE[...]['shifts']`` is ``(0.0, 0.0)`` in
      :func:`crosstie_constant`, so the miss the raise exists to stop arrives anyway,
      now invisibly.
    * **Every unmeasured zero is marked.**  A rigid offset larger than
      ``CROSSTIE_SEED_WIN`` leaves no true pairs in the window, so the catalogs
      resolve, no raise fires, and the declined measurement used to print
      ``'f115w': (+0.00000, +0.00000),`` byte-identically to a measured line.  Each
      such line now carries a ``# NOT MEASURED -- <why>`` comment that survives the
      paste into the source file, and ``n_unmeasured`` lets the CLI exit nonzero.
    """
    cfg = CROSSTIE[region]
    n_unmeasured = 0
    lines = [f"# flux-vetted cross-tie vs {cfg['master_name']} -- paste into "
             f"CROSSTIE['{region}']['shifts']:"]
    for f in filts:
        ra, de, why = crosstie_offset_detail(f, rc, allow_missing=allow_missing)
        mark = ''
        if why is not None:
            n_unmeasured += 1
            mark = f"  # NOT MEASURED -- {why}"
        lines.append(f"    '{f}': ({ra:+.5f}, {de:+.5f}),{mark}")
    if n_unmeasured:
        lines.append(f"# {n_unmeasured} of {len(filts)} filter(s) WERE NOT MEASURED "
                     f"(marked above): those zeros are not measurements. Fix the cause "
                     f"and re-run rather than pasting them.")
    return '\n'.join(lines), n_unmeasured


def farr(x):
    return np.asarray(np.ma.filled(np.ma.masked_invalid(np.asarray(x, float)), np.nan), float)


def virac2(epoch, cachepath):
    v = Table.read(cachepath)
    ra = farr(v['RAJ2000']); dec = farr(v['DEJ2000'])
    pr = np.where(np.isfinite(farr(v['pmRA'])), farr(v['pmRA']), 0.)
    pd = np.where(np.isfinite(farr(v['pmDE'])), farr(v['pmDE']), 0.)
    dt = epoch - V2EP
    return SkyCoord((ra + (pr * dt / 3.6e6) / np.cos(np.radians(dec))) * u.deg,
                    (dec + pd * dt / 3.6e6) * u.deg)


def coord_shift(ra, dec, ref, peak=(0.0, 0.0)):
    """clipped-median Δα/Δδ COORDINATE offset (arcsec, NO cosδ) to ADD to (ra,dec) to land
    on ref -- the convention adjust_wcs(delta_ra/delta_dec) consumes.  Clip is on-sky.

    SAME-STAR, not nearest-neighbour.  This was ``match_to_catalog_sky`` + clipped
    median, i.e. a NN median against a reference whose median nearest-neighbour
    spacing is **1.10"** (sickle's own ``refcache/virac2.fits``, n=26922, no
    magnitude cut; a Ks<15 cut only reaches ~1.4").  That is below the 3" the
    dense-reference guard draws the line at, and it is not a rounding correction:
    on F480M the fine step moved the tie by (-52.3, -39.3) mas, ~65 mas of a
    ~105 mas answer.

    Injection on the real F480M consensus (PR #268 review) showed the NN form
    under-recovering by 1.8 / 2.4 / 9.7 mas at 50 / 100 / 200 mas and collapsing
    at 300 -- degrading THROUGH the search radius rather than at it.  That
    degradation needs real crowding plus a real unmatched-source background; it
    does not reproduce in a synthetic field, where ``CLIP_MAS`` alone keeps the NN
    median honest.  What DOES reproduce, and is the mechanism, is that the nearest
    neighbour is chosen by sky distance while the right counterpart is the one
    nearest the EXPECTED offset -- see
    ``test_the_counterpart_is_chosen_by_expected_offset_not_sky_distance``.

    So: take EVERY pair within ``SEARCH`` (``search_around_sky``), express each as
    an offset relative to the already-verified ``peak``, and for each source keep
    the single pair closest to that peak.  Pairs further than ``CLIP_MAS`` from
    the peak are dropped and the median of what remains is the refinement.

    With ``peak`` left at zero this reduces to selecting the nearest counterpart
    again, so callers that have a verified tie must pass it.

    ``peak`` is the coordinate-frame offset (arcsec) already applied to
    ``(ra, dec)``; callers that pre-shift onto the tie leave it at zero.
    """
    sc = SkyCoord(ra * u.deg, dec * u.deg)
    ia, ib, _, _ = search_around_sky(sc, ref, SEARCH)
    if len(ia) == 0:
        return None
    cosd = np.cos(np.radians(-28.7))
    dra_c = (ref.ra[ib] - sc.ra[ia]).to(u.arcsec).value
    ddec_c = (ref.dec[ib] - sc.dec[ia]).to(u.arcsec).value
    # distance of each PAIR from the expected offset -- this selection is what
    # makes the counterpart the same star rather than the nearest neighbour.
    off = np.hypot((dra_c - peak[0]) * cosd, ddec_c - peak[1]) * 1000.0
    keep = off < CLIP_MAS
    if int(keep.sum()) < 15:
        return None
    ia, dra_c, ddec_c, off = ia[keep], dra_c[keep], ddec_c[keep], off[keep]
    # one pair per source, the closest to the peak.  Without this a crowded source
    # contributes several times and the median is weighted by local density.
    order = np.argsort(off)
    seen = np.zeros(len(sc), dtype=bool)
    take = np.zeros(len(ia), dtype=bool)
    for j in order:
        if not seen[ia[j]]:
            seen[ia[j]] = True
            take[j] = True
    dra_c, ddec_c = dra_c[take], ddec_c[take]
    n = int(len(dra_c))
    if n < 15:
        return None
    return (float(np.median(dra_c)), float(np.median(ddec_c)),
            mad_std(dra_c * cosd) * 1000.0 / np.sqrt(n),  # arcsec->mas
            mad_std(ddec_c) * 1000.0 / np.sqrt(n), n)


def build_consensus(frames):
    """Welford incremental combine of frame SIAF positions -> consensus SkyCoord (per-visit)."""
    cosd = np.cos(np.radians(-28.70)); rad = CLUSTER_MAS / 1000. / 3600.
    cap = sum(len(fr[0]) for fr in frames) + 10
    g_ra = np.empty(cap); g_dec = np.empty(cap); g_n = np.zeros(cap, int); ng = 0
    for (fra, fdec) in frames:
        if ng == 0:
            n = len(fra); g_ra[:n] = fra; g_dec[:n] = fdec; g_n[:n] = 1; ng = n; continue
        base = SkyCoord(g_ra[:ng] * u.deg, g_dec[:ng] * u.deg)
        idx, sep, _ = SkyCoord(fra * u.deg, fdec * u.deg).match_to_catalog_sky(base)
        mt = sep.deg < rad; gi = idx[mt]
        g_n[gi] += 1
        g_ra[gi] += (fra[mt] - g_ra[gi]) / g_n[gi]
        g_dec[gi] += (fdec[mt] - g_dec[gi]) / g_n[gi]
        um = ~mt; k = int(um.sum())
        g_ra[ng:ng+k] = fra[um]; g_dec[ng:ng+k] = fdec[um]; g_n[ng:ng+k] = 1; ng += k
    sel = g_n[:ng] >= 2
    return SkyCoord(g_ra[:ng][sel] * u.deg, g_dec[:ng][sel] * u.deg)


def load_siaf(f):
    """Recover current-generation SIAF positions. -> (ra,dec,ra0,de0,crf).

    GENERATION LOCK: RA/Dec are recomputed from the STABLE detector x_fit/y_fit
    through the LIVE crf WCS (meta['FILENAME']), not the catalog's cached
    skycoord_centroid (which encodes the WCS at build time and goes stale ~up to
    48 mas across re-drizzle generations).  Then undo the RAOFFSET currently baked
    into that crf to reach SIAF.

    ``crf`` is the exposure this catalog was actually built from (``None`` for a
    legacy catalog with no ``FILENAME`` meta).  It is the ONLY trustworthy source
    of the frame's observation: the catalog basename carries `visit001` for every
    observation of a proposal, so several observations reduced into one directory
    are indistinguishable by name -- see ``_gather``.
    """
    t = Table.read(f)
    crf = None
    if 'x_fit' in t.colnames and 'FILENAME' in t.meta:
        crf = _resolve_existing_path(t.meta['FILENAME'])
        with fits.open(crf) as hl:
            # GWCS, not the SCI header's SIP fit.  This re-projects every
            # per-frame (x_fit, y_fit) to build the VIRAC2 offsets table -- a
            # 5-8 mas position-dependent SIP-fit error here propagates straight
            # into the tie every frame is then corrected by.
            wcs = frame_wcs(hl)
            ra0 = float(hl['SCI'].header.get('RAOFFSET', t.meta.get('RAOFFSET', 0.0)))
            de0 = float(hl['SCI'].header.get('DEOFFSET', t.meta.get('DEOFFSET', 0.0)))
        sc = SkyCoord(wcs.pixel_to_world(farr(t['x_fit']), farr(t['y_fit'])))
        if 'skycoord_centroid' in t.colnames:
            cached = SkyCoord(t['skycoord_centroid'])
            drift = float(np.nanmedian((sc.ra.deg - cached.ra.deg)
                                       * np.cos(np.radians(cached.dec.deg)) * 3.6e6))
            if abs(drift) > 15:
                print(f"    [genlock] {os.path.basename(f)}: cached skycoord {drift:+.0f} mas "
                      f"stale vs live crf WCS -> reprojected from x/y", flush=True)
    else:   # legacy catalog without x/y or FILENAME: fall back to cached positions
        sc = SkyCoord(t['skycoord_centroid'])
        ra0 = float(t.meta.get('RAOFFSET', 0.0)); de0 = float(t.meta.get('DEOFFSET', 0.0))
    fl = farr(t['flux_fit']); q = farr(t['qfit']) if 'qfit' in t.colnames else np.zeros(len(t))
    good = np.isfinite(fl) & (fl > 0) & (q < 0.4) & np.isfinite(sc.ra.deg)
    return (sc.ra.deg[good] - ra0 / 3600.0, sc.dec.deg[good] - de0 / 3600.0,
            ra0, de0, crf)


def module_key(det):
    """fix_alignment (PipelineRerunNIRCAM-LONG.py:1208) matches a 'Module' cell against
    the detector name OR its digit-stripped root.  SW detectors nrca1..4 -> 'nrca',
    nrcb1..4 -> 'nrcb'; LW nrcalong/nrcblong keep their full names (strip('1234') is a
    no-op there).  Grouping by this key gives one tie per PHYSICAL module (A vs B).

    THE MODULE IS THE FINEST GRANULARITY THIS WRITER SOLVES, and #697 is why it has
    not been made finer.  Asked on 2026-09-06 whether per-detector corrections
    should go into the offsets table, the maintainer answered no.  What that
    answers is a per-detector CALIBRATION TERM: a value derived from many
    observations and written as a static per-detector row.  The report the question
    was conditional on, ``reports/per_detector_offsets.md``, finds that term is not
    static -- on sky (the 2026-08-07 run, 34,672 measurements over 11 fields) every
    detector's between-field scatter exceeds its mean and every one changes sign
    from field to field, and in the de-rotated re-run (2026-08-25, 73,673
    measurements) a shuffled-angle control reproduces nearly all of the apparent
    shrinkage.  A term that reverses between observations, written here, is read
    back by the next observation with the wrong sign.  If a static per-detector
    placement term is ever established, its home is the distortion/SIAF layer
    (#689, #299): it is an instrument-frame quantity an on-sky per-(visit,
    exposure, module) row cannot express in any case.

    A ``--per-detector`` counterpart to ``--per-module`` would be a different
    quantity -- each detector's OWN measured tie against VIRAC2, the same category
    as the per-detector rows the m2 ``consensus`` channel already writes and
    ``fix_alignment`` already applies.  #697 did not rule on that, so it is neither
    provided here nor forbidden; it stands or falls on its own merits (chiefly that
    a per-detector tie is measured against a quarter of the stars, which is what
    #386's precision weighting is about), not on #697.
    """
    return det if det in LW_DETS else det[:4]


def parse_vgroup(basename):
    """Canonical visit-group id of a per-frame catalog filename.

    Matches the WHOLE token (``_vgroup<TOKEN>_exp``), not just its digit prefix:
    MIRI and parallel visit groups can carry a trailing letter (sgrb2/F2550W has
    ``vgroup0020210b``), and ``r'_vgroup(\\d+)'`` would silently truncate that to
    ``0020210`` -- a DIFFERENT group id, returned as if it were correct.
    Canonicalised with ``vgroup_key`` so the zero-padded and bare spellings of the
    same group (both ``_vgroup07101`` and ``_vgroup7101`` exist on disk) key and
    compare identically on the table's producer and consumer sides.
    """
    m = re.search(r'_vgroup([^_]+)_exp', basename)
    if m is None:
        raise ValueError(f"cannot parse a visit group from {basename}")
    return vgroup_key(m.group(1))


class WrongObservationError(RuntimeError):
    """A globbed catalog belongs to a different observation than the region."""


class NoPerFrameCatalogsError(RuntimeError):
    """A module produced NO per-frame catalogs at all.

    Distinct from WrongObservationError on purpose.  A single-module field simply
    did not observe the other module and skipping it is correct; a two-module
    field that has merely not finished cataloging module A must NOT be skipped,
    or the build silently locks a table covering only module B and the gap
    resurfaces much later as a match=0 raise at apply time.  Which of the two it
    is comes from the region's declared ``modules``, not from this exception.
    """


def _gather(filt, base, sub, mtag, dets, prop=None, field=None, otag=''):
    """Collect per-(visit,vgroup,exp) and per-visit SIAF positions + legacy coarse.

    Keyed by VGROUP as well as exposure: a visit can dither across several visit
    groups (physically disjoint sky tiles) and the exposure number RESTARTS in
    each, so ``(visit, exposure)`` is ambiguous.  Keying on it alone averaged two
    disjoint pointings into one row -- cloudc has 2 visit groups in every filter,
    sgrb2 F187N has 2, gc2211 has 6.

    The per-VISIT consensus (``byv``) deliberately still pools all vgroups: it is
    a superset of stars, and each exposure only matches the tile it overlaps.

    THE VISIT KEY IS THE ``crf``'s, NOT THE CATALOG BASENAME'S.  A catalog is
    named ``..._visit001_...`` for the FIRST visit of whatever observation it came
    from, so every observation of a proposal reduced into the same directory is
    named identically and the basename cannot tell them apart -- while the row
    this feeds is keyed on the full ``jw<prop><obs><visit>`` token.  Synthesising
    that token from the region's own ``field`` therefore LABELS whatever was
    globbed as the requested observation, whether or not it is:

    * ``--region cloudef5`` (2092/005) globs only ``jw02092002001`` catalogs (obs
      005 has none on disk) and would have written them out as ``jw02092005001``
      -- obs 002's tie applied to obs 005, which is a real ~7.5" gross offset in
      F162M;
    * gc2211's five observations all reduce to ``visit001`` in one directory and
      REUSE the same visit-group ids (``02201`` appears under o023, o046, o049 and
      o050), so neither the visit nor the vgroup separates them.

    So the observation is read off ``meta['FILENAME']`` (the exposure the catalog
    was fit on) and a catalog from another observation is REFUSED, never
    relabelled.  ``otag`` additionally narrows the glob to one observation's
    per-frame products (``_o046_``) for fields that carry the token.
    """
    from collections import defaultdict
    byve = defaultdict(lambda: [[], []]); byv = defaultdict(list); coarse = defaultdict(lambda: [[], []])
    expect = f'{jw_prefix(prop)}{field}' if (prop is not None and field is not None) else None
    wrong_obs = {}
    seen = {}
    unstamped = []
    for det in dets:
        for f in sorted(glob.glob(f'{base}/{sub}/{filt}_{det}{otag}'
                                  f'_visit*_vgroup*_exp*{mtag}_daophot_basic.fits')):
            b = os.path.basename(f)
            # The glob over-matches the grouped-fit variant; see perframe_matches.
            if not perframe_matches(b, mtag):
                continue
            vis3 = b.split('_visit')[1][:3]; exp = int(re.search(r'_exp(\d+)', b).group(1))
            vgr = parse_vgroup(b)
            ra, dec, ra0, de0, crf = load_siaf(f)
            if crf is None:
                # legacy catalog with no FILENAME meta: the observation cannot be
                # verified.  Fall back to the region's own token, but SAY SO --
                # this is the case the refusal above cannot police.
                vis = f'{jw_prefix(prop)}{field}{vis3}' if expect else vis3
                unstamped.append(b)
            else:
                vis = os.path.basename(crf).split('_')[0]
                if expect is not None and not vis.startswith(expect):
                    wrong_obs.setdefault(vis, []).append(b)
                    continue
            if (vis, vgr, exp, det) in seen:
                raise WrongObservationError(
                    f"{filt}: two catalogs claim the SAME frame "
                    f"(visit={vis}, vgroup={vgr}, exp={exp}, det={det}):\n"
                    f"  {seen[(vis, vgr, exp, det)]}\n  {b}\n"
                    f"One of them is a stale duplicate from before the per-frame "
                    f"names carried the observation token; pooling both would "
                    f"double-weight that frame in the consensus.  Remove or "
                    f"archive the stale one, or set the region's otag.")
            seen[(vis, vgr, exp, det)] = b
            byve[(vis, vgr, exp)][0].append(ra); byve[(vis, vgr, exp)][1].append(dec)
            byv[vis].append((ra, dec)); coarse[vis][0].append(ra0); coarse[vis][1].append(de0)
    if unstamped:
        print(f"  [WARNING] {filt}: {len(unstamped)} catalog(s) have no FILENAME "
              f"meta, so their OBSERVATION could not be verified and the region's "
              f"own token was assumed: {unstamped[:3]}", flush=True)
    if wrong_obs:
        raise WrongObservationError(
            f"{filt}: {sum(len(v) for v in wrong_obs.values())} globbed catalog(s) "
            f"belong to observation(s) {sorted(wrong_obs)}, not to this region "
            f"({expect}*).  Writing them would label another observation's tie as "
            f"this one's.  Examples: "
            f"{ {k: v[:2] for k, v in sorted(wrong_obs.items())} }.  Either this "
            f"observation has not been cataloged yet, or the region needs "
            f"otag=True so the glob picks its own per-frame products.")
    if not byve:
        raise NoPerFrameCatalogsError(
            f"{filt}: no per-frame catalogs for {expect or 'this region'} matched "
            f"{base}/{sub}/{filt}_<det>{otag}_visit*_vgroup*_exp*{mtag}_daophot_basic.fits")
    return byve, byv, coarse


def _solve(byve, byv, coarse, c_ra, c_dec, ref, filt, modlabel=None):
    """Per-visit bulk tie (consensus vs VIRAC2, seeded by the merged i2d coarse) + per-exposure
    relative shift vs that consensus.  modlabel=None -> module-LOCKED (one shift/exposure over all
    detectors, no Module column).  modlabel set -> that module's own tie, written with a Module
    cell so fix_alignment applies it per-module (removes a real inter-module A/B offset).

    The ``Visit`` written is the key ``_gather`` built from each catalog's own crf
    (``jw<prop><obs><visit>``) -- NOT a token synthesised from the region, which
    would relabel whatever was globbed as the requested observation."""
    tag = f"[{modlabel}] " if modlabel else ""
    rows = []
    for vis in sorted(byv):
        # legacy coarse (median of previously-applied RAOFFSET) -- diagnostic only.
        c_ra_legacy = float(np.median(coarse[vis][0])); c_dec_legacy = float(np.median(coarse[vis][1]))
        consensus = build_consensus(byv[vis])
        cc_ra = consensus.ra.deg + c_ra / 3600.0; cc_dec = consensus.dec.deg + c_dec / 3600.0
        # PER-VISIT coarse residual on top of the shared per-filter i2d seed.  REQUIRED:
        # visits can carry very different raw guide-star pointing errors (brick-1182
        # visit001 ~22" vs visit002 ~2").  The single mosaic-wide i2d coarse captures only
        # the dominant visit, so the other visit is left mis-seeded and the <SEARCH fine NN
        # below cannot bridge it -> it silently inherits the dominant visit's shift (the
        # cause of the 2026-07 brick-1182 visit001 corruption: all visits got visit002's
        # +1.9" instead of visit001's true -17.5").  A per-visit large-radius histogram
        # xcorr (crowding-proof) recovers each visit's own bulk before the fine step.
        cv_ra, cv_dec = c_ra, c_dec
        # vx = (dra, ddec, npairs, contrast, window, swept) from measure_offset (coord arcsec)
        vx = coarse_xcorr(SkyCoord(cc_ra * u.deg, cc_dec * u.deg), ref, maxsep=COARSE_MAXSEP_VISIT)
        if vx[0] is not None:
            vcontrast = vx[3]
            # only apply a MEANINGFUL per-visit correction (> the fine-NN radius); small
            # residuals are left to the fine step to avoid double counting.
            if vcontrast >= COARSE_MIN_PEAK_RATIO and np.hypot(vx[0], vx[1]) > SEARCH.to(u.arcsec).value:
                cc_ra = cc_ra + vx[0] / 3600.0; cc_dec = cc_dec + vx[1] / 3600.0
                cv_ra = c_ra + vx[0]; cv_dec = c_dec + vx[1]
                print(f"  {tag}visit{vis}: PER-VISIT coarse ADD ({vx[0]:+.3f},{vx[1]:+.3f})\" "
                      f"contrast={vcontrast:.1f} window={vx[4]:g}\" npairs={vx[2]}"
                      f"{' [SWEPT/gross]' if vx[5] else ''}  (visit differs from mosaic seed)",
                      flush=True)
        res = coord_shift(cc_ra, cc_dec, ref)
        if res is None:
            # coarse alone (no per-visit fine refinement available)
            res = (0.0, 0.0, 0.0, 0.0, 0)
            print(f"  visit{vis}: fine tie weak; using coarse alone")
        bulk_ra = cv_ra + res[0]; bulk_dec = cv_dec + res[1]
        print(f"  {tag}visit{vis}: i2d_coarse({c_ra:+.4f},{c_dec:+.4f})\" [legacy {c_ra_legacy:+.4f},"
              f"{c_dec_legacy:+.4f}] pervisit({cv_ra:+.4f},{cv_dec:+.4f}) + fine({res[0]*1000:+.1f},{res[1]*1000:+.1f})mas "
              f"=> BULK ({bulk_ra:.4f},{bulk_dec:.4f})\" SEM {res[2]:.2f}/{res[3]:.2f}mas "
              f"n={res[4]}; consensus={len(consensus)}", flush=True)
        want = sorted((g, e) for (v, g, e) in byve if v == vis)
        failed = []
        for vgr, exp in want:
            ra = np.concatenate(byve[(vis, vgr, exp)][0])
            dec = np.concatenate(byve[(vis, vgr, exp)][1])
            rel = coord_shift(ra, dec, consensus)
            if rel is None:
                # NOT silent: a missing row means fix_alignment finds no offset for
                # that exposure.  With a per-exposure table that is a hard match=0
                # raise at apply time, so it must be visible here too.
                failed.append((vgr, exp))
                print(f"    [WARNING] {tag}visit{vis} vgroup{vgr} exp{exp}: relative "
                      f"tie FAILED -> no row; this exposure will have NO offset",
                      flush=True)
                continue
            tot_ra = bulk_ra + rel[0]; tot_dec = bulk_dec + rel[1]
            row = dict(Visit=str(vis), Vgroup=str(vgr),
                       Exposure=int(exp), Filter=filt.upper(),
                       dra=tot_ra, ddec=tot_dec, nmatch=rel[4],
                       rel_ra_mas=rel[0] * 1000, rel_dec_mas=rel[1] * 1000)
            if modlabel is not None:
                row['Module'] = modlabel
            rows.append(row)
            print(f"    {tag}vgroup{vgr} exp{exp:>2}: rel({rel[0]*1000:+.2f},"
                  f"{rel[1]*1000:+.2f})mas n={rel[4]}"
                  f"  -> total({tot_ra:.4f},{tot_dec:.4f})\"", flush=True)
        if failed and len(failed) == len(want):
            # every exposure of the visit failed to tie -> the consensus itself is
            # broken (bad crop, wrong reference, empty catalogs).  Writing a table
            # with no rows for this visit leaves every one of its frames unaligned.
            raise SystemExit(
                f"[FAIL] {filt} {tag}visit {vis}: ALL {len(want)} exposures failed "
                f"to tie to the visit consensus; refusing to write a table that "
                f"would leave the whole visit without an offset.")
    return rows


def lock_filter(filt, rc, per_module=False):
    sub, ep, mtag = rc['filts'][filt]
    prop, field, base = rc['proposal'], rc['field'], rc['basepath']
    cache = f'{base}/astrometry_diag/refcache/virac2.fits'
    print(f"=== per-exposure relock {filt} ({prop}/{field}, epoch {ep}) "
          f"[{'PER-MODULE' if per_module else 'module-locked'}] ===", flush=True)
    ref = virac2(ep, cache)
    dets = SW_DETS if _is_sw(filt) else LW_DETS
    # How many modules does this region actually have?  Decides whether `merged`
    # is the right coarse seed -- on a single-module field a `-merged` product
    # cannot be a merge, only a leftover generation.
    declared = rc.get('modules')
    n_modules = (len(declared) if declared is not None
                 else len({module_key(d) for d in dets}))
    # PER-FILTER coarse bulk tie, measured ONCE on the clean drizzled mosaic vs VIRAC2.
    # Seeds every visit; the per-visit/per-exposure fine same-star pass below resolves
    # the residual (including any per-module <SEARCH difference).  FAIL LOUD if dirty.
    i2d_coarse = coarse_from_i2d(filt, rc, ref, n_modules=n_modules)
    if i2d_coarse is None:
        raise SystemExit(f"[FAIL] {filt}: could not measure a clean i2d coarse tie; "
                         f"refusing to write a lock table (would re-perpetuate ~0).")
    c_ra, c_dec = i2d_coarse
    otag = f"_o{field}" if rc.get('otag') else ''
    if not per_module:
        byve, byv, coarse = _gather(filt, base, sub, mtag, dets, prop, field, otag)
        return _solve(byve, byv, coarse, c_ra, c_dec, ref, filt, modlabel=None)
    # PER-MODULE: solve a separate tie for each physical module (A=nrca*, B=nrcb*/LW
    # nrcalong/nrcblong).  A single module-locked shift cannot remove a real A/B offset
    # (the ~20 mas Dec-28.71 seam / NRCB distortion residual); two independent ties do.
    groups = {}
    for det in dets:
        groups.setdefault(module_key(det), []).append(det)
    rows = []
    for modlabel, gdets in sorted(groups.items()):
        # Skipping a module must be AUTHORISED by the region's declaration, not
        # inferred from an empty glob.  A two-module field that has merely not
        # finished cataloging module A would otherwise lock a table covering only
        # B, and the gap resurfaces much later as a match=0 raise at apply time.
        if declared is not None and modlabel not in declared:
            print(f"  --- module '{modlabel}': not among this region's declared "
                  f"modules {tuple(declared)}; not observed, skipping ---",
                  flush=True)
            continue
        print(f"  --- module '{modlabel}': {gdets} ---", flush=True)
        try:
            byve, byv, coarse = _gather(filt, base, sub, mtag, gdets, prop, field,
                                        otag)
        except NoPerFrameCatalogsError:
            if declared is not None:
                # DECLARED and yet empty: this field does have the module and its
                # cataloging has not finished.  Locking now writes a table missing
                # it entirely, so refuse.
                raise
            print(f"  --- module '{modlabel}': no per-frame catalogs, and this "
                  f"region declares no module list, so an unobserved module "
                  f"cannot be told from an unfinished catalog run; skipping. Add "
                  f"'modules' to the REGION entry to make this explicit ---",
                  flush=True)
            continue
        rows.extend(_solve(byve, byv, coarse, c_ra, c_dec, ref, filt, modlabel=modlabel))
    if not rows:
        raise NoPerFrameCatalogsError(
            f"{filt}: no module produced a tie -- every module was skipped for "
            f"want of per-frame catalogs. Nothing to lock.")
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--region', default='1182', choices=list(REGION))
    ap.add_argument('--per-module', action='store_true',
                    help='solve a separate per-module (A/B) tie and emit a Module column '
                         '(removes a real inter-module offset; fix_alignment narrows by Module)')
    ap.add_argument('--out', default=None, help='override output path (for validation before '
                    'overwriting the production table)')
    ap.add_argument('--allow-missing-crosstie-catalog', action='store_true',
                    help='with --remeasure-crosstie, print a 0 constant (with a WARN) instead '
                         'of raising when a cross-tie catalog glob resolves to nothing')
    ap.add_argument('--remeasure-crosstie', action='store_true',
                    help='flux-vetted RE-MEASURE of the JWST<->JWST cross-tie vs the 2221 master; '
                         'PRINTS suggested CROSSTIE constants and EXITS (writes nothing). Run this '
                         'when the master 2221 frame moves, then paste the numbers into CROSSTIE. '
                         'Exit 2 = the block printed but one or more filters were NOT MEASURED '
                         '(each such line is marked); exit 0 = every line is a measurement.')
    ap.add_argument('filts', nargs='*', help='filters (default: all of region)')
    args = ap.parse_args()
    rc = REGION[args.region]
    filts = args.filts or list(rc['filts'])

    if args.remeasure_crosstie:
        cfg = CROSSTIE.get(args.region)
        if cfg is None:
            print(f"region {args.region} has no cross-tie master; nothing to measure."); sys.exit(0)
        # measure EVERY filter before printing anything paste-ready: a raise partway
        # through must not leave a block that reads complete but is short a filter.
        text, n_unmeasured = crosstie_block(
            args.region, rc, filts,
            allow_missing=args.allow_missing_crosstie_catalog)
        print(text)
        sys.exit(2 if n_unmeasured else 0)

    rows = []
    for f in filts:
        frows = lock_filter(f, rc, per_module=args.per_module)
        # STAGE-2: JWST<->JWST cross-tie onto the master frame (hardcoded constant; 1182 only).
        ct_ra, ct_de = crosstie_constant(f, rc)
        if ct_ra or ct_de:
            for r in frows:
                r['dra'] += ct_ra; r['ddec'] += ct_de
            print(f"  [crosstie] {f}: applied CONSTANT ({ct_ra*1000:+.1f},{ct_de*1000:+.1f})mas "
                  f"to {len(frows)} rows", flush=True)
        rows.extend(frows)
    if not rows:
        print("no rows produced"); sys.exit(1)
    t = Table(rows)
    t['dra (arcsec)'] = t['dra']; t['ddec (arcsec)'] = t['ddec']
    path = args.out or f"{rc['basepath']}/offsets/Offsets_JWST_Brick{rc['proposal']}_VIRAC2locked.csv"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # FIELD-SAFE merge: replace only rows for the SAME (Filter, proposal+field Visit prefix);
    # preserve every other filter AND every other field that shares this per-proposal table.
    # 11 chars = 'jw' + proposal(5, MAST zero-pads) + observation(3) + 1, i.e.
    # the visit-token prefix through the observation number.
    new_visit_prefixes = set(str(v)[:11] for v in t['Visit'])
    new_filts = set(str(x) for x in t['Filter'])
    if os.path.exists(path):
        old = Table.read(path)
        keepmask = np.array([not (str(r['Filter']) in new_filts and str(r['Visit'])[:11] in new_visit_prefixes)
                             for r in old])
        if keepmask.any():
            old = old[keepmask]
            # A HALF-per-module table is worse than either whole: update_offsets_
            # table narrows on Module as soon as the column exists, so corrections
            # for the filters that were NOT rebuilt would match only the filled
            # rows and hard-fail with "matches NO row".  Refuse rather than write
            # it.  (Rebuilding every filter of the field in one command takes the
            # keepmask.any() == False path below and rewrites cleanly.)
            if ('Module' in t.colnames) != ('Module' in old.colnames):
                have, lack = (('new', 'existing') if 'Module' in t.colnames
                              else ('existing', 'new'))
                # The table is per-PROPOSAL, so the blocking rows may belong to a
                # different FIELD entirely (cloudef2 2092/002 and cloudef5
                # 2092/005 share Offsets_JWST_Brick2092_VIRAC2locked.csv).  In
                # that case "rebuild all filters of this region" is not a remedy
                # -- a region is one field -- so name the prefixes that are
                # actually blocking and every region that writes this table.
                blocking = sorted(set(str(v)[:11] for v in old['Visit']))
                # siblings share this exact FILE, i.e. proposal AND basepath --
                # arches and quintuplet are both 2045 but sit under different
                # basepaths, so they do NOT share a table.
                siblings = sorted(k for k, v in REGION.items()
                                  if v['proposal'] == rc['proposal']
                                  and v['basepath'] == rc['basepath'])
                raise SystemExit(
                    f"REFUSING to merge: the {have} rows have a Module column and "
                    f"the {lack} rows do not, so {os.path.basename(path)} would be "
                    f"half per-module.  update_offsets_table narrows on Module "
                    f"once the column exists, so the rows that keep the other "
                    f"convention would then match no row.\n"
                    f"  blocking visit prefixes: {blocking}\n"
                    f"  filters:                 "
                    f"{sorted(set(str(x) for x in old['Filter']))}\n"
                    f"This table is per-PROPOSAL ({rc['proposal']}), written by "
                    f"region(s): {siblings}.  Rebuild ALL filters of EVERY one of "
                    f"them, in one command each, before any of them is used:\n"
                    + "".join(
                        f"  python -m jwst_gc_pipeline.reduction."
                        f"build_virac2_offsets --region {k} --per-module\n"
                        for k in siblings))
            # NB Vgroup needs no counterpart to the half-per-module refusal above.
            # The Module narrowing treats an unmatched row as "not this module",
            # so a half-filled column strands the rows that were not rebuilt; the
            # Vgroup narrowing treats an EMPTY cell as "group unknown -> applies"
            # (astrometry_checkpoint.vgroup_row_matches), so preserved rows keep
            # matching exactly as they did before the column existed.
            # dtype-aware fill: `np.nan` into a string column makes vstack raise
            # TableMergeError ('float64' vs 'str160').
            for c in t.colnames:
                if c not in old.colnames:
                    old[c] = _empty_like(t[c], len(old))
            for c in old.colnames:
                if c not in t.colnames:
                    t[c] = _empty_like(old[c], len(t))
            # Vgroup is a STRING identifier that happens to look numeric.  A
            # table whose groups are all digits round-trips through CSV as
            # int64, so merging it with freshly-built str rows raises
            #   TableMergeError: 'Vgroup' columns have incompatible types
            #                    ['int64', 'str128']
            # -- which is what stopped the cloudef obs005 rebuild after obs002
            # had just written an all-numeric column.  Canonicalise both sides
            # through vgroup_key (the same normaliser the consumers use, so
            # '06201' and 6201 stay one group) before stacking.
            for tbl in (old, t):
                if 'Vgroup' in tbl.colnames:
                    tbl['Vgroup'] = [vgroup_key(v) for v in tbl['Vgroup']]
            t = vstack([old, t])
    t.write(path, overwrite=True)
    print(f"\nwrote {path}: {len(t)} rows (replaced {sorted(new_filts)} for prefixes {sorted(new_visit_prefixes)})", flush=True)
