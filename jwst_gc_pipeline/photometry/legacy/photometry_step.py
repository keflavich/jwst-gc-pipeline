"""LEGACY crowdsource / "iter2"-"iter3" per-exposure photometry -- BENCHMARKS ONLY.

Sequestered out of ``catalog_long.py``; the active CLI ``main`` and
the shared helpers stay there.  Superseded by the manual-iteration pipeline
(``cataloging.do_photometry_step_manual`` / ``run_manual_pipeline``).  Reached
only via ``--legacy-iterations``; do not wire into active reductions.

FROZEN: bug fixes and threshold tuning land in the manual path only.  In
particular the qfit/peak-over-bkg vetting constants here (and in
``build_filtered_iter2_residual_bg`` / ``_flag_likely_extended_iter4`` in the
host module) are deliberately NOT kept in sync with
``manual_defaults.MANUAL_DEFAULTS`` -- they preserve the legacy behaviour for
benchmarking.  Do not "fix" them to match.

Implementation note: this module reuses the shared helpers, constants and
third-party imports that live in ``catalog_long`` by importing that
module and copying its public module namespace in below, so the relocated code
resolves every name exactly as it did before the move.  Tests that stub those
helpers must patch THIS module (e.g. ``legacy.photometry_step.get_psf_model``),
not the host module.
"""
import jwst_gc_pipeline.photometry.catalog_long as _host
import pylab as pl

# Reproduce the exact module namespace the relocated code had while it lived in
# catalog_long (shared helpers + that module's imports), so every
# bare-name reference below resolves unchanged.
globals().update({_k: _v for _k, _v in vars(_host).items()
                  if not _k.startswith('__')})


# ===========================================================================
# Relocated legacy code follows (verbatim move from catalog_long).
#
# The crowdsource and daophot fitting backends have been split into sibling
# modules fit_crowdsource.py / fit_daophot.py.  Re-import the names that
# do_photometry_step (and existing tests that reference them as attributes of
# this module) use, so every reference resolves unchanged.
# ===========================================================================
from jwst_gc_pipeline.photometry.legacy.fit_crowdsource import (  # noqa: E402,F401
    save_crowdsource_results,
)
from jwst_gc_pipeline.photometry.legacy.fit_daophot import (  # noqa: E402,F401
    _parallel_psfphotometry,
    _FakePhot,
    _parallel_iterative_psfphotometry,
    _make_grouper,
    _subtract_satstar_model,
    _first_pass_daofinder,
)




def _run_cutout_pipeline(options, modules, filternames, nvisits, proposal_id,
                         target, field, basepath, crowdsource_default_kwargs,
                         bg_boxsizes):
    """In-process multi-phase pipeline for a ``--cutout-region`` run.

    A cutout is small enough to run every phase sequentially in one process
    (no SLURM array), so the subsequent-step orchestration that the full-frame
    pipeline does via dependent SLURM jobs is done here as plain calls.

    Phases (single filter):  iter1 -> iter2 -> iter4
      * iter1  : unseeded per-frame photometry; merge per-frame catalogs;
                 build the iter1 residual i2d mosaic.
      * iter2  : re-fit each frame seeded by the MERGED iter1 catalog (the
                 merged catalog fed back in); merge; build iter2 residual i2d.
      * iter4  : median-smooth the iter2 residual mosaic, subtract it from the
                 input image, and re-fit seeded by the MERGED iter2 catalog
                 (residual built against the ORIGINAL data); merge.

    Multi-filter adds an ``iter3`` phase between iter2 and iter4 that seeds
    every filter from the cross-filter union of the iter2 merged catalogs.

    All outputs land under ``<basepath>/cutouts/<label>/`` (disjoint from
    full-frame products).  Frames not overlapping the region are skipped; if
    NO frame overlaps, raises (wrong region/target).
    """
    import copy
    from jwst_gc_pipeline.photometry import merge_catalogs as _merge_catalogs

    cut_bp = _cutout_out_basepath(basepath, options)
    os.makedirs(os.path.join(cut_bp, 'catalogs'), exist_ok=True)
    pupil = 'clear'
    multifilter = len(filternames) > 1

    phases = ['iter1', 'iter2']
    if multifilter:
        phases.append('iter3')
    phases.append('iter4')
    print(f"CUTOUT PIPELINE: label={_cutout_label_for(options)} "
          f"phases={phases} filters={filternames} modules={modules}", flush=True)

    def _merged_iter_path(phase, module, filt, kind='iterative'):
        """Reconstruct the merged minimal catalog path that
        merge_individual_frames writes for ``phase`` (matches its token logic).
        ``kind`` selects the iterative (daoiterative) or basic (dao) catalog."""
        desat = '_unsatstar' if options.desaturated else ''
        bgsub = ('_bgsub' if options.bgsub else '') + ('_resbgsub' if phase == 'iter4' else '')
        blur_ = '_blur' if options.blur else ''
        iter_token = '' if phase == 'iter1' else f'_{phase}'
        method_suffix = 'daoiterative_iterative' if kind == 'iterative' else 'dao_basic'
        return (f'{cut_bp}/catalogs/{filt.lower()}_{module}_indivexp_merged'
                f'{desat}{bgsub}{blur_}{iter_token}_{method_suffix}.fits')

    overlap_total = 0
    # mosaic infilled-paths recorded per (phase, module, filt) for iter4 bg build
    mosaic_paths = {}
    # overlapping-frame list recorded in iter1, reused by later phases
    frame_cache = {}

    for phase in phases:
        is_iter1 = (phase == 'iter1')
        iteration_label = None if is_iter1 else phase
        resbgsub = (phase == 'iter4')

        opts_phase = copy.copy(options)
        opts_phase.iteration_label = iteration_label or ''
        opts_phase.seed_catalog = ''
        # iter4 carries the _resbgsub filename token (it subtracts a residual
        # bg); drives _bgsub_token so per-frame/mosaic/merge names agree.
        opts_phase.use_iter3_residual_bg = resbgsub

        for module in modules:
            for filt in filternames:
                # --- determine seed + resbg for this (phase, module, filt) ---
                seed_catalog = None
                resbg_path = None
                if phase == 'iter2':
                    seed_catalog = _merged_iter_path('iter1', module, filt)
                elif phase == 'iter3':
                    seed_catalog = _build_cutout_union_seed(
                        cut_bp, modules, filternames, options)
                elif phase == 'iter4':
                    seed_src = 'iter3' if multifilter else 'iter2'
                    seed_catalog = _merged_iter_path(seed_src, module, filt)
                    if getattr(options, 'iter4_bg_exclude_badfit', False):
                        # EXPERIMENTAL: smoothed iter2 residual that subtracts
                        # only confident stars, leaving extended-emission false
                        # detections in the background so iter4 doesn't inflate
                        # fluxes to absorb them.  See NOTES_star_vs_extended_emission.md
                        resbg_path = build_filtered_iter2_residual_bg(
                            cut_bp, basepath, filt, proposal_id, field, module,
                            options, frame_cache.get((module, filt), []),
                            pupil=pupil)
                        print(f"iter4: built qfit-filtered smoothed bg "
                              f"{resbg_path}", flush=True)
                    if not resbg_path:
                        # standard smoothed-bg from the seed-source residual mosaic
                        src_infilled = mosaic_paths.get((seed_src, module, filt))
                        if src_infilled is None:
                            raise ValueError(
                                f"iter4 needs the {seed_src} residual mosaic for "
                                f"module={module} filt={filt}; none was produced.")
                        src_residual = src_infilled.replace(
                            '_residual_infilled_i2d.fits', '_residual_i2d.fits')
                        resbg_path = _cutout_smooth_residual_bg(src_residual)
                        print(f"iter4: built smoothed bg {resbg_path}", flush=True)

                if seed_catalog is not None and not os.path.exists(seed_catalog):
                    raise ValueError(
                        f"{phase}: seed catalog {seed_catalog} missing "
                        f"(prior phase merge did not produce it).")

                postprocess = options.postprocess_residuals or (seed_catalog is not None)

                # --- candidate frames ---
                # iter1 (first phase) scans every exposure of every visit and
                # records which ones overlap the cutout region (the overlap test
                # inside do_photometry_step costs ~10 s/frame); later phases reuse
                # that cached overlapping-frame list instead of re-scanning the
                # non-overlapping frames every phase.
                if phase == phases[0]:
                    candidate_frames = []
                    for visitid in range(1, nvisits[proposal_id][target] + 1):
                        candidate_frames.extend(sorted(get_filenames(
                            basepath, filt, proposal_id, field,
                            visitid=f'{visitid:03d}', each_suffix=options.each_suffix,
                            module=module, pupil='clear')))
                else:
                    candidate_frames = frame_cache.get((module, filt), [])

                n_overlap_phase = 0
                overlapping_now = []
                for filename in candidate_frames:
                    exposure_id = filename.split("_")[2]
                    visit_id = filename.split("_")[0][-3:]
                    vgroup_id = filename.split("_")[1]
                    file_detector = filename.split("_")[3]
                    file_module = file_detector if module == 'merged' else module
                    if options.skip_if_done and _expected_output_exists(
                            cut_bp, filt, file_module, opts_phase,
                            visit_id, vgroup_id, exposure_id,
                            iteration_label=iteration_label):
                        print(f'skip-if-done [{phase}]: {filt} {file_module} '
                              f'visit={visit_id} exp={exposure_id}', flush=True)
                        overlapping_now.append(filename)
                        n_overlap_phase += 1
                        continue
                    try:
                        do_photometry_step(
                            opts_phase, filt, file_module, file_detector,
                            field, basepath, filename, proposal_id,
                            crowdsource_default_kwargs,
                            exposurenumber=int(exposure_id),
                            visit_id=visit_id, vgroup_id=vgroup_id,
                            use_webbpsf=True, bg_boxsizes=bg_boxsizes,
                            seed_catalog=seed_catalog,
                            iteration_label=iteration_label,
                            postprocess_residuals=postprocess,
                            residual_negative_threshold=options.residual_negative_threshold,
                            local_snr_threshold=options.local_snr_threshold,
                            daofind_roundlo=options.daofind_roundlo,
                            daofind_roundhi=options.daofind_roundhi,
                            resbg_path=resbg_path)
                    except CutoutNoOverlap as ex:
                        print(f"cutout [{phase}]: skipping non-overlapping "
                              f"frame {filename} ({ex})", flush=True)
                        continue
                    overlapping_now.append(filename)
                    n_overlap_phase += 1

                if phase == phases[0]:
                    frame_cache[(module, filt)] = overlapping_now

                if n_overlap_phase == 0:
                    raise ValueError(
                        f"--cutout-region={options.cutout_region!r} overlapped "
                        f"none of the {filt}/{module} frames in phase {phase}.")
                if phase == phases[0]:
                    overlap_total += n_overlap_phase

                # --- merge per-frame catalogs FIRST (the merged-catalog
                # residual needs the vetted merged catalog) ---
                if options.daophot:
                    _merge_methods = [('dao', '_basic')]
                    if not options.basic_only:
                        _merge_methods.append(('daoiterative', '_iterative'))
                    for _mname, _msuffix in _merge_methods:
                        _merge_catalogs.merge_individual_frames(
                            module=module, filtername=filt.lower(),
                            progid=proposal_id, method=_mname, suffix=_msuffix,
                            target=target, basepath=cut_bp,
                            iteration_label=iteration_label,
                            bgsub=options.bgsub, desat=options.desaturated,
                            epsf=options.epsf, blur=options.blur,
                            resbgsub=resbgsub, fwhm_basepath=basepath)
                        print(f"cutout [{phase}]: merged {_mname} catalog under "
                              f"{cut_bp}/catalogs/", flush=True)
                    # iter4: flag likely-extended (non-star) detections by
                    # comparing iter2->iter4 centroid motion (diagnostic column).
                    if phase == 'iter4' and not options.basic_only:
                        try:
                            _pixscale = float(np.sqrt(np.abs(np.linalg.det(
                                wcs.WCS(fits.getheader(
                                    f'{cut_bp}/{filt}/pipeline/jw0{proposal_id}-o'
                                    f'{field}_t001_{_inst_token(filt)}_clear-'
                                    f'{filt.lower()}-{module}_data_i2d.fits',
                                    extname='SCI')).pixel_scale_matrix))) * 3600.0)
                            _flag_likely_extended_iter4(
                                _merged_iter_path('iter4', module, filt, 'iterative'),
                                _merged_iter_path('iter2', module, filt, 'iterative'),
                                _pixscale)
                        except Exception as ex:
                            print(f"cutout [iter4]: likely_extended flag failed: "
                                  f"{ex}", flush=True)

                # --- residual i2d(s) for this phase ---
                # --residual-source: 'mergedcat' (default), 'rawcat', or 'both'.
                # The merged-catalog residual drops spurious sources (which eat
                # extended emission in the raw residual).  The raw iterative
                # mosaic is also built when it is the iter4 background source
                # (seed_src), regardless of --residual-source.
                if not options.skip_mosaic_each_exposure_residuals and options.daophot:
                    residual_source = getattr(options, 'residual_source', 'mergedcat')
                    kinds = ['basic'] if options.basic_only else ['basic', 'iterative']
                    seed_src = 'iter3' if multifilter else 'iter2'
                    need_raw_iter_for_bg = (phase == seed_src)
                    build_raw = residual_source in ('rawcat', 'both')
                    for residual_kind in kinds:
                        do_raw = build_raw or (residual_kind == 'iterative'
                                               and need_raw_iter_for_bg)
                        if not do_raw:
                            continue
                        infilled = mosaic_each_exposure_residuals(
                            basepath=cut_bp, filtername=filt,
                            proposal_id=proposal_id, field=field, module=module,
                            residual_kind=residual_kind,
                            desat=options.desaturated, bgsub=options.bgsub,
                            epsf=options.epsf, blur=options.blur,
                            group=options.group, pupil=pupil,
                            iteration_label=iteration_label, resbgsub=resbgsub,
                            make_starless=False, crop_to_data=True)
                        if residual_kind == 'iterative':
                            mosaic_paths[(phase, module, filt)] = infilled

                    # merged-catalog residual i2d (default deliverable).  Built
                    # for the science kind (iterative, or basic if --basic-only),
                    # from the matching vetted merged catalog.
                    if residual_source in ('mergedcat', 'both'):
                        mc_kind = 'basic' if options.basic_only else 'iterative'
                        try:
                            # opts_phase (not options) carries the phase's bgsub
                            # token (iter4 -> _resbgsub) so the per-frame raw
                            # product names are reconstructed correctly.
                            build_mergedcat_residuals(
                                cut_bp, basepath,
                                _merged_iter_path(phase, module, filt, mc_kind),
                                filt, proposal_id, field, module, opts_phase,
                                frame_cache.get((module, filt), []),
                                iteration_label, [mc_kind], pupil=pupil)
                        except Exception as ex:
                            print(f"cutout [{phase}]: mergedcat residual failed: "
                                  f"{ex}", flush=True)

                    # original-data i2d (once, during iter1) so catalog sky
                    # positions can be overplotted on real data
                    if phase == phases[0]:
                        try:
                            mosaic_cutout_input_data(
                                cut_bp, filt, proposal_id, field, module,
                                _cutout_label_for(options), pupil=pupil)
                        except Exception as ex:
                            print(f"cutout: data i2d build failed: {ex}", flush=True)
                        try:
                            mosaic_cutout_satstar_flags(
                                cut_bp, filt, proposal_id, field, module,
                                _cutout_label_for(options), pupil=pupil)
                        except Exception as ex:
                            print(f"cutout: satstar flags i2d failed: {ex}", flush=True)

                    # final model image = data_i2d - residual_i2d (same grid),
                    # built for the final phase so the model can be overplotted /
                    # compared in CARTA.  Prefers the mergedcat residual when
                    # present (the vetted-catalog model), else the raw residual.
                    if phase == phases[-1]:
                        try:
                            _build_cutout_model_i2d(
                                cut_bp, filt, proposal_id, field, module,
                                iteration_label, resbgsub, options, pupil=pupil)
                        except Exception as ex:
                            print(f"cutout: model i2d build failed: {ex}", flush=True)

    print(f"CUTOUT PIPELINE DONE: {overlap_total} overlapping frames, "
          f"phases={phases}", flush=True)


def _build_cutout_union_seed(cut_bp, modules, filternames, options):
    """Build the cross-filter union seed for a multi-filter cutout iter3.

    Stacks the iter2 merged catalogs across all filters (and modules) into one
    skycoord seed table, writes it under the cutout catalogs/ dir, and returns
    its path.  Single-filter cutouts skip iter3 and never call this.
    """
    from astropy.table import vstack as _vstack
    desat = '_unsatstar' if options.desaturated else ''
    bgsub = '_bgsub' if options.bgsub else ''
    blur_ = '_blur' if options.blur else ''
    tbls = []
    for module in modules:
        for filt in filternames:
            p = (f'{cut_bp}/catalogs/{filt.lower()}_{module}_indivexp_merged'
                 f'{desat}{bgsub}{blur_}_iter2_daoiterative_iterative.fits')
            if os.path.exists(p):
                t = Table.read(p)
                if 'skycoord' in t.colnames:
                    tbls.append(Table({'skycoord': t['skycoord']}))
    if not tbls:
        raise ValueError("iter3 union seed: no iter2 merged catalogs found "
                         f"under {cut_bp}/catalogs/")
    union = _vstack(tbls, metadata_conflicts='silent')
    out = f'{cut_bp}/catalogs/union_seed_iter2_cutout.fits'
    union.write(out, overwrite=True)
    print(f"iter3: wrote cross-filter union seed {out} (n={len(union)})", flush=True)
    return out




from collections import namedtuple as _namedtuple

_SuffixTokens = _namedtuple(
    '_SuffixTokens',
    'desat bgsub epsf_ exposure_ visitid_ vgroupid_ vgroup_numeric blur_ group iter_')


def _output_suffix_tokens(options, exposurenumber=None, visit_id=None,
                          vgroup_id=None, iteration_label=None):
    """Filename-suffix tokens shared across one exposure's output products.

    Factored out of ``do_photometry_step`` (block B).  Returns a
    ``_SuffixTokens`` namedtuple whose fields unpack in the same order as the
    local variables the function used to assign inline, so all downstream
    filename construction is unchanged.
    """
    vgroupid_, vgroup_numeric = normalize_vgroup_id(vgroup_id)
    return _SuffixTokens(
        desat='_unsatstar' if options.desaturated else '',
        bgsub=_bgsub_token(options),
        epsf_="_epsf" if options.epsf else "",
        exposure_=f'_exp{exposurenumber:05d}' if exposurenumber is not None else '',
        visitid_=f'_visit{int(visit_id):03d}' if visit_id is not None else '',
        vgroupid_=vgroupid_,
        vgroup_numeric=vgroup_numeric,
        blur_="_blur" if options.blur else "",
        group="_group" if options.group else "",
        iter_=_iteration_token(iteration_label),
    )


_SVO_INSTRUMENT_MAP = {'NIRCAM': 'NIRCam', 'NIRISS': 'NIRISS',
                       'NIRSPEC': 'NIRSpec', 'MIRI': 'MIRI'}


def _svo_effective_wavelength(telescope, instrument, filt):
    """Effective wavelength (Quantity) of a filter from the SVO FPS service.

    Factored out of ``do_photometry_step``.  FITS headers use all-caps
    instrument names (NIRCAM); SVO uses mixed case (NIRCam).
    """
    svo_instrument = _SVO_INSTRUMENT_MAP.get(instrument.upper(), instrument)
    filter_table = SvoFps.get_filter_list(facility=telescope,
                                          instrument=svo_instrument)
    filter_table.add_index('filterID')
    return filter_table.loc[
        f'{telescope}/{svo_instrument}.{filt}']['WavelengthEff'] * u.AA




def do_photometry_step(options, filtername, module, detector, field, basepath,
                       filename, proposal_id, crowdsource_default_kwargs, exposurenumber=None,
                       visit_id=None, vgroup_id=None,
                       bg_boxsizes=None,
                       use_webbpsf=False,
                       nsigma=5,
                       local_snr_threshold=5.0,
                       daofind_roundlo=-1.0,
                       daofind_roundhi=1.0,
                       pupil='clear',
                       seed_catalog=None,
                       iteration_label=None,
                       postprocess_residuals=False,
                       residual_negative_threshold=0.0,
                       resbg_path=None):
    """
    LEGACY (benchmarks only).  This is the crowdsource / "iter2"-"iter3" seeded
    per-exposure pipeline, superseded by the manual-iteration path
    (cataloging.do_photometry_step_manual / run_manual_pipeline).  It is kept for
    benchmark reproduction and is being sequestered into photometry/legacy/
    (see REFACTOR_PLAN.md).  Do not wire into active reductions; reach it only via
    --legacy-iterations.

    nsigma is the threshold to multiply the error estimate by to get the detection threshold
    """
    print(f"Starting {field} filter {filtername} module {module} detector {detector} {exposurenumber}", flush=True)

    # Memory profiling is OFF by default everywhere -- it is debugging-only.
    # tracemalloc.start(25) instruments EVERY process-wide allocation and the
    # ~16 _mem_report() snapshots per frame (take_snapshot + statistics, some
    # deep=True) cost tens of seconds each on a multi-GB process, dwarfing the
    # actual photometry (it made each frame ~9 min).  Enable only with the
    # explicit --profile-memory flag when chasing a leak.
    _profile_mem = bool(getattr(options, 'profile_memory', False))
    if _profile_mem and not tracemalloc.is_tracing():
        tracemalloc.start(25)

    def _mem_report(label, deep=False):
        if not _profile_mem:
            return
        snap = tracemalloc.take_snapshot()
        top = snap.statistics('lineno')
        peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        try:
            curr_kb = int(open('/proc/self/status').read().split('VmRSS:')[1].split()[0])
        except (OSError, IndexError, ValueError):
            # /proc/self/status may not exist (non-Linux), or VmRSS may be
            # absent / unparseable; mem report is diagnostic so 0 is fine.
            curr_kb = 0
        print(f"=MEM= {label}: curr={curr_kb/1e6:.2f}GB peak={peak_kb/1e6:.2f}GB", flush=True)
        for s in top[:12]:
            print(f"  {s.size/1e9:.3f}GB {s.traceback[0]}", flush=True)
        if deep and top:
            print(f"  --- traceback for #1 allocator ({top[0].size/1e9:.3f}GB) ---", flush=True)
            for frame in top[0].traceback.format():
                print(f"  {frame}", flush=True)
    fwhm_tbl = Table.read(FWHM_TABLE)
    row = fwhm_tbl[fwhm_tbl['Filter'] == filtername]
    fwhm = fwhm_arcsec = float(row['PSF FWHM (arcsec)'][0])
    fwhm_pix = float(row['PSF FWHM (pixel)'][0])

    # redundant, saves me renaming variables....
    filt = filtername

    # LocalBackground annulus, filter-scaled.  Inner must exceed the
    # photometric aperture (2*fwhm_pix) and sit beyond the first PSF
    # sidelobe (~1.6 FWHM peak; ~2.2 FWHM outer edge).  Floor at the
    # historical NIRCam-LW value (6,10) so existing NIRCam configs
    # are unchanged.
    aperture_radius_pix = 2.0 * fwhm_pix
    localbkg_inner = max(6, int(round(aperture_radius_pix + 0.5 * fwhm_pix)))
    localbkg_outer = localbkg_inner + max(4, int(round(fwhm_pix)))

    # file naming suffixes (factored into _output_suffix_tokens; unpacked in the
    # same order so all downstream filename construction is unchanged)
    (desat, bgsub, epsf_, exposure_, visitid_, vgroupid_, vgroup_numeric,
     blur_, group, iter_) = _output_suffix_tokens(
        options, exposurenumber, visit_id, vgroup_id, iteration_label)

    print(f"Starting cataloging on {filename}", flush=True)
    # ---- Optional small-region cutout ----------------------------------
    # Run the whole per-exposure pipeline on just a hand-specified region.
    # We write a cropped copy of the input into <basepath>/cutouts/<label>/
    # and run on THAT, so every output -- both basepath-derived (catalog,
    # residual, diagnostics) and filename-derived (satstar models, background
    # dumps) -- lands in the cutout tree and never overwrites full-frame
    # products.  basepath itself is redirected to out_basepath further below
    # (after all basepath INPUT reads are done).
    cutout_label = ''
    out_basepath = basepath
    _cutout_x0, _cutout_y0 = 0, 0
    if getattr(options, 'cutout_region', ''):
        cutout_label, filename, out_basepath, _cutout_x0, _cutout_y0 = _prepare_cutout_input(
            filename, basepath, filtername, options)
    _cutout_active = bool(cutout_label)
    if _cutout_active:
        # Disable diagnostic PNGs for cutout runs (whole invocation is a
        # cutout run, so this never affects full-frame output): make the
        # zoom-diagnostic a no-op and suppress all savefig writes.
        # Set the flag on the HOST module: catalog_zoom_diagnostic (which reads
        # _SUPPRESS_DIAGNOSTICS) lives there, so a legacy-local copy wouldn't
        # suppress it.
        _host._SUPPRESS_DIAGNOSTICS = True
        pl.savefig = _host._noop_savefig

    fh, im1, data, wht, err, instrument, telescope, obsdate = load_data(filename)
    background_map = None
    inst_token = instrument.lower()

    # set up coordinate system
    ww = wcs.WCS(im1[1].header)
    pixscale = ww.proj_plane_pixel_area()**0.5
    cen = ww.pixel_to_world(im1[1].shape[1]/2, im1[1].shape[0]/2)

    # iter4resbgrefit builds its residual against the pristine image, so keep a
    # copy of the data *before* any background subtraction.  (The authoritative
    # is_resbg_refit flag is recomputed in the iteration block below.)
    _is_resbg_refit_early = (
        (_strip_chunk(iteration_label) or '').lower() in ('iter4resbgrefit', 'iter4'))
    original_data = data.copy() if _is_resbg_refit_early else None

    if options.bgsub:
        # background subtraction
        # see BackgroundEstimationExperiments.ipynb
        bkg = Background2D(data, box_size=bg_boxsizes[filt.lower()], bkg_estimator=MedianBackground())
        background_map = bkg.background
        fits.PrimaryHDU(data=bkg.background,
                        header=im1['SCI'].header).writeto(filename.replace(".fits",
                                                                           "_background.fits"),
                                                          overwrite=True)

        # subtract background, but then re-zero the edges
        zeros = data == 0
        data = data - bkg.background
        data[zeros] = 0

        fits.PrimaryHDU(data=data, header=im1['SCI'].header).writeto(filename.replace(".fits", "_bgsub.fits"), overwrite=True)

    if resbg_path or getattr(options, 'use_iter3_residual_bg', False):
        # 2026-04-25: alternative background subtraction that uses the
        # iter3 photometry residual (3x3-median-smoothed) as the
        # background estimate.  Built by make_iter3_residual_bgmaps.py
        # and consumed by the iter2-residbg / iter3-residbg cascade.
        #
        # 2026-06-06: use the whole-field iter3 residual *mosaic*, smoothed,
        # instead of the per-exposure residual.  The mosaic co-adds every
        # exposure so its background has much higher S/N.  It lives on the
        # mosaic pixel grid, so reproject it onto this exposure's WCS before
        # subtracting.  Built by make_iter3_residual_bgmaps.py.
        #
        # The mosaic's module token is configurable via
        # ``--resbg-mosaic-module`` (default 'merged').  Targets whose
        # whole-field co-add is a single detector (e.g. sickle LW = 'nrcb')
        # pass that token; SW four-detector co-adds use 'merged'.
        #
        # 2026-06-07: ``resbg_path`` (in-process cutout pipeline) overrides
        # the iter3-filename construction -- the cutout wrapper passes the
        # smoothed iter2 (or iter3) residual mosaic it just built.
        from reproject import reproject_interp
        if resbg_path:
            residbg_path = resbg_path
        else:
            _inst = _inst_token(filtername)
            _bg_module = getattr(options, 'resbg_mosaic_module', '') or 'merged'
            residbg_path = (
                f'{basepath}/{filtername}/pipeline/'
                f'jw0{proposal_id}-o{field}_t001_{_inst}_{pupil}-{filtername.lower()}-'
                f'{_bg_module}_iter3_daophot_iterative_residual_smoothed_bg_i2d.fits'
            )
        if not os.path.exists(residbg_path):
            raise ValueError(
                f"residual-bg subtraction requires the smoothed-bg mosaic "
                f"{residbg_path} to exist; run "
                f"`python make_iter3_residual_bgmaps.py --target=<target>` "
                f"after the iter3 residual mosaic is complete (or, for cutout "
                f"runs, the wrapper builds it from the iter2 residual mosaic)."
            )
        with fits.open(residbg_path) as bgh:
            if 'SCI' in [h.name for h in bgh]:
                bg_hdu = bgh['SCI']
            else:
                bg_hdu = bgh[0]
            bg_wcs = wcs.WCS(bg_hdu.header)
            bg_data = bg_hdu.data.astype(float)
        # Reproject the merged-grid background onto this exposure's grid.
        # Surface-brightness units (MJy/sr) are resolution-independent, so
        # interpolation across the grid change is valid without rescaling.
        bg_reproj, _ = reproject_interp((bg_data, bg_wcs), ww,
                                        shape_out=data.shape)
        n_nan = int(np.sum(~np.isfinite(bg_reproj)))
        bg_finite = np.where(np.isfinite(bg_reproj), bg_reproj, 0.0)
        zeros = data == 0
        data = data - bg_finite
        data[zeros] = 0
        background_map = bg_finite
        print(f"Subtracted merged iter3-residual-smoothed bg ({residbg_path}) "
              f"reprojected onto exposure grid: sum={float(np.nansum(bg_finite)):.3e} "
              f"MJy/sr-equiv, {n_nan} pix outside merged FOV (set to 0)", flush=True)
        # Diagnostics (both cutout and full-frame paths): save the reprojected
        # smoothed residual that was subtracted and the resulting source-finding
        # input (data - smoothed_residual), with the frame's SCI WCS so they
        # overplot against the catalog.  Tokens distinguish iter/bgsub variants.
        _diag_suffix = f"{_bgsub_token(options)}{_iteration_token(iteration_label)}"
        _sci_hdr = im1['SCI'].header
        try:
            fits.PrimaryHDU(data=bg_finite.astype('float32'), header=_sci_hdr).writeto(
                filename.replace('.fits', f'{_diag_suffix}_resbg_reproj.fits'),
                overwrite=True)
            fits.PrimaryHDU(data=data.astype('float32'), header=_sci_hdr).writeto(
                filename.replace('.fits', f'{_diag_suffix}_srcfind_input.fits'),
                overwrite=True)
            print(f"  wrote resbg diagnostics: *{_diag_suffix}_resbg_reproj.fits "
                  f"and *_srcfind_input.fits", flush=True)
        except (OSError, ValueError) as _ex:
            print(f"  WARNING: failed to write resbg diagnostics: {_ex}", flush=True)

    # try to limit memory use before we start photometry
    data = data.astype('float32')

    # Load PSF model
    _mem_report("before PSF load")
    grid, psf_model = get_psf_model(filtername, proposal_id, field,
                                    module=module,
                                    use_webbpsf=use_webbpsf,
                                    # if we're doing each exposure, we want the full grid
                                    use_grid=options.each_exposure,
                                    blur=options.blur,
                                    target=options.target,
                                    obsdate=obsdate,
                                    basepath='/blue/adamginsburg/adamginsburg/jwst/',
                                    psf_cache_dir=os.path.join(basepath, 'psfs'),
                                    instrument=instrument)
    dao_psf_model = grid
    _mem_report("after PSF load")

    if _cutout_active and (_cutout_x0 or _cutout_y0):
        # The spatially-varying PSF grid is indexed in the PARENT frame's
        # pixel coords, but the cutout data is 0-origin.  Re-origin the grid
        # by the cutout offset so a source at cutout pixel (cx, cy) is fit
        # with the SAME PSF the full-frame run would use at parent pixel
        # (cx + x0, cy + y0).  Without this the cutout silently uses a
        # mis-positioned (wrong) PSF.  Exact (verified maxdiff 0).
        from astropy.nddata import NDData as _NDData
        _shifted_xy = [(gx - _cutout_x0, gy - _cutout_y0)
                       for (gx, gy) in dao_psf_model.grid_xypos]
        dao_psf_model = type(dao_psf_model)(_NDData(
            np.asarray(dao_psf_model.data),
            meta={'grid_xypos': _shifted_xy,
                  'oversampling': dao_psf_model.oversampling}))
        grid = dao_psf_model
        print(f"CUTOUT: re-origined PSF grid by (-{_cutout_x0}, -{_cutout_y0}) "
              f"so the spatially-varying PSF matches the parent-frame fit",
              flush=True)

    # bound the flux to be >= 0 (no negative peak fitting)
    dao_psf_model.flux.min = 0

    dq, weight, bad = get_uncertainty(err, data, wht=wht, dq=im1['DQ'].data if 'DQ' in im1 else None)

    eff_wavelength = _svo_effective_wavelength(telescope, instrument, filt)

    # DAO Photometry setup (grouper honors --max-group-size; see _make_grouper).
    grouper = _make_grouper(options, fwhm_pix)
    mmm_bkg = MMMBackground()

    # empirically determined in debugging session with Taehwa on 2025-12-09:
    # with just nan_to_num, setting pixels to zero, some stars got "erased"
    kernel = Gaussian2DKernel(x_stddev=fwhm_pix/2.355)
    mask = np.isnan(data) | bad
    if 'DQ' in im1:
        dqarr = im1['DQ'].data
        is_saturated = (dqarr & dqflags.pixel['SATURATED']) != 0
        # we want original data_ to be untouched for imshowing diagnostics etc.
        data_ = data.copy()
        data_[is_saturated] = np.nan
        mask |= is_saturated
        # Honor the broader bad-DQ bitmask for the instrument.  For MIRI
        # this also drops NON_SCIENCE imager regions and PERSISTENCE
        # latents; for NIRCam it adds DO_NOT_USE coverage that wasn't
        # previously enforced here.
        bad_bitmask = _bad_dq_bitmask(instrument)
        is_baddq = (dqarr & bad_bitmask) != 0
        mask |= is_baddq
    else:
        data_ = data

    nan_replaced_data = interpolate_replace_nans(data_, kernel, convolve=convolve_fft,
                                                  allow_huge=True)

    # Infer a per-exposure ``_daophot_basic.fits`` seed for iter2, but *not*
    # for iter3 -- iter3 must use the cross-band union seed catalog that
    # the caller passes in explicitly.  Silently falling back to the basic
    # per-frame catalog would defeat the purpose of iter3 entirely.
    # Use the *base* iteration label so a chunk-suffixed compound label
    # (e.g. 'iter3_chunk03of08' from --n-seed-chunks > 1) still triggers
    # the iter3-specific code path (explicit-seed requirement, xy_bounds,
    # tighter local-SNR threshold).
    _base_label = _strip_chunk(iteration_label)
    is_iter3 = (_base_label is not None
                and str(_base_label).lower() == 'iter3')
    # iter4resbgrefit: final residual-bg refit step appended after iter3.
    # Re-fits the EXACT per-frame iter3 catalog as seeds with the iter3 tight
    # xy_bounds, on the residual-bg-subtracted data, and writes its residual
    # against the ORIGINAL (non-bg-subtracted) data.  Purely additive -- the
    # iter1/iter2/iter3 code paths are unchanged.
    # 'iter4' is the cutout-pipeline final refit (in-process wrapper): same
    # residual-from-original + tight-bounds behavior as iter4resbgrefit, but it
    # is seeded by an EXPLICIT merged catalog (passed by the wrapper), so the
    # per-frame iter3-seed inference below is gated on 'iter4resbgrefit' only.
    is_resbg_refit = (_base_label is not None
                      and str(_base_label).lower() in ('iter4resbgrefit', 'iter4'))
    if (seed_catalog is None and iteration_label not in (None, '')
            and not is_iter3 and not is_resbg_refit):
        inferred_seed_catalog = (
            f'{basepath}/{filtername}/'
            f'{filtername.lower()}_{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}_daophot_basic.fits'
        )
        if os.path.exists(inferred_seed_catalog):
            seed_catalog = inferred_seed_catalog
    if (is_resbg_refit and seed_catalog is None
            and str(_base_label).lower() == 'iter4resbgrefit'):
        # Seed from this frame's own iter3 iterative catalog (the "exact
        # iter3 catalog").  That file carries NO bgsub token and the
        # ``_iter3`` iter token -- even though this run's bgsub token is
        # ``_resbgsub`` and its iter token is ``_iter4resbgrefit``.
        inferred_iter3_catalog = (
            f'{basepath}/{filtername}/'
            f'{filtername.lower()}_{module}{visitid_}{vgroupid_}{exposure_}{desat}{epsf_}{blur_}{group}_iter3_daophot_iterative.fits'
        )
        if not os.path.exists(inferred_iter3_catalog):
            raise ValueError(
                f"iteration_label='iter4resbgrefit' requires the per-frame "
                f"iter3 catalog {inferred_iter3_catalog} to exist; run iter3 "
                f"photometry first."
            )
        seed_catalog = inferred_iter3_catalog
    if is_iter3 and seed_catalog is None:
        raise ValueError(
            "iteration_label='iter3' requires an explicit seed_catalog "
            "pointing at the cross-band union seed file "
            "(build_union_seed_catalog.py output); no fallback is allowed."
        )

    is_second_iteration = seed_catalog is not None
    # iter3 position bound: ±1 SW NIRCam pixel (0.031"), expressed in the
    # current frame's pixel units.  On LW this is ~0.5 pix, on SW it is
    # 1 pix.  Kept None for iter1/iter2 so their behavior is unchanged.
    iter3_xy_bounds_pix = None
    if is_iter3 or is_resbg_refit:
        pixscale_arcsec = float(pixscale.to(u.arcsec).value)
        sw_pix_arcsec = 0.031
        iter3_xy_bounds_pix = float(sw_pix_arcsec / pixscale_arcsec)
        _xyb_label = 'iter4resbgrefit' if is_resbg_refit else 'iter3'
        print(f"{_xyb_label}: pixscale={pixscale_arcsec:.4f}\"/pix -> "
              f"xy_bounds=±{iter3_xy_bounds_pix:.3f} pix per source",
              flush=True)
    if is_second_iteration:
        # iter2 / iter3 tuned finder settings: lower local-SNR cut plus tighter
        # roundness/sharpness to suppress diffuse-background false detections.
        # iter3 is seed-dominated (the union catalog already knows where the
        # sources are), so its post-seed DAOStarFinder augmentation threshold
        # is raised to reduce the flood of low-SNR "discoveries" that add
        # little beyond what the union catalog provides.
        iter2_local_snr_threshold = 6.0 if is_iter3 else 3.0
        iter2_roundlo = -0.3
        iter2_roundhi = 0.3
        iter2_sharplo = 0.50
        iter2_sharphi = 1.00

        # Local-noise-map DAO thresholding for second-iteration residual search.
        local_noise_map = compute_local_noise_map(nan_replaced_data, smooth_sigma_pix=3.0)
        finite_noise = np.isfinite(local_noise_map) & (local_noise_map > 0)
        if not np.any(finite_noise):
            raise ValueError('Local noise map has no positive finite values')
        daofind_threshold = float(np.nanmin(local_noise_map[finite_noise]))
        daofind_tuned = DAOStarFinder(threshold=daofind_threshold,
                                      fwhm=fwhm_pix, roundhi=iter2_roundhi, roundlo=iter2_roundlo,
                                      sharplo=iter2_sharplo, sharphi=iter2_sharphi)
        print(
            f'DAO iter2 local-noise threshold={daofind_threshold}; '
            f'local_snr_threshold={iter2_local_snr_threshold}; '
            f'roundlo={iter2_roundlo}; roundhi={iter2_roundhi}; '
            f'sharplo={iter2_sharplo}; sharphi={iter2_sharphi}',
            flush=True,
        )
    else:
        # Keep original first-pass starfinding behavior unchanged
        # (factored into _first_pass_daofinder).
        daofind_tuned, daofind_threshold = _first_pass_daofinder(
            data, err, nsigma, fwhm_pix, daofind_roundlo, daofind_roundhi)
        print(
            f'DAO first-pass threshold={daofind_threshold}; '
            f'roundlo={daofind_roundlo}; roundhi={daofind_roundhi}',
            flush=True,
        )

    print("Finding stars with daofind_tuned", flush=True)

    satstar_table = None
    # Holds the (NaN-replaced) satstar model image after it has been
    # subtracted from ``nan_replaced_data``.  Re-applied to the residual
    # written to disk so the saved per-frame residual matches what the
    # fitter actually saw (i.e. data minus satstar wings minus phot model).
    satstar_model_subtracted = None
    # Satstar fitting + subtraction runs for EVERY photometry pass --
    # iter1, iter2, iter3 -- before any daofind/daophot.  The previous
    # gate ``and seed_catalog is not None`` left iter1 with bright
    # un-subtracted saturated stars.  Mosaic-mode photometry was
    # deprecated 2026-05-25 (see main() guard), so reaching this point
    # implies each-exposure mode and the per-frame DQ_SATURATED gate the
    # satstar fitter relies on is available.
    if True:
        # Optionally fit saturated stars whose centres lie OUTSIDE this frame's
        # FOV (their wings still bleed into the field).  Default: ON for
        # full-frame runs, OFF for cutout runs (a small cutout rarely benefits
        # and the off-FOV forced fits are wasteful there).  --fit-satstar-
        # outside-fov / --no-fit-satstar-outside-fov force it.  In-FOV satstar
        # fitting below is unaffected either way.
        _fit_outside = getattr(options, 'fit_satstar_outside_fov', None)
        if _fit_outside is None:
            _fit_outside = not _cutout_active
        if _fit_outside:
            # Cut at 32" — matches the radius of the large PSF grid used for forced
            # fits (fovp2048 SW × 0.031"/pix = 31.7"; fovp1024 LW × 0.063"/pix
            # = 32.3").  Anything farther falls outside PSF support so the
            # cutout would contain zero usable pixels and the fit would raise.
            outside_star_pixels, outside_locked = load_outside_fov_satstar_pixels(
                basepath, ww, data_shape=nan_replaced_data.shape, max_offset_arcsec=32.0)
        else:
            outside_star_pixels, outside_locked = [], False
        # When seeds came from the verified ``_locked.reg`` file, skip the
        # ±5 px grid search (radius=0 → single-point flux-only fit at the
        # locked position).  Default radius=5 otherwise.
        forced_grid_search_radius = 0 if outside_locked else 5
        # Namespace the satstar outputs by bgsub/iteration_label so that
        # the non-bgsub and bgsub iter2 array jobs (which can run concurrently
        # on the same frame) don't race each other on a shared filename.
        # The prior shared name (`<frame>_satstar_residual.fits`) caused
        # FileNotFoundError from astropy's writeto(overwrite=True) when a
        # sibling job deleted the file between the existence check and
        # the os.remove call.
        iter_tag = _iteration_token(iteration_label)
        satstar_file_suffix = f'{bgsub}{iter_tag}'
        # MIRI: feed the DEEP coadded data_i2d to the satstar seed gate so the
        # extended-emission phantom rejection (prominence + faint-core) is
        # measured on the noise-averaged coadd, not this single frame.  A
        # per-frame measurement lets a phantom escape via one frame's noise
        # spike (the cross-frame satstar merge then keeps it).  Frame-invariant
        # + matches what the user sees in the final i2d.  NIRCam: not loaded
        # (the gate is MIRI-only inside get_saturated_stars anyway).
        _seed_gate_image = _seed_gate_wcs = None
        if module == 'mirimage':
            _di2d_path = (f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-'
                          f'o{field}_t001_{_inst_token(filtername)}_{pupil}-'
                          f'{filtername.lower()}-{module}_data_i2d.fits')
            if os.path.exists(_di2d_path):
                try:
                    with fits.open(_di2d_path) as _dih:
                        _ext = 'SCI' if 'SCI' in [h.name for h in _dih] else 0
                        _seed_gate_image = _dih[_ext].data.astype(float)
                        _seed_gate_wcs = wcs.WCS(_dih[_ext].header)
                except Exception as _gex:
                    print(f"satstar seed gate: could not load coadd "
                          f"{_di2d_path}: {_gex}", flush=True)
            else:
                print(f"satstar seed gate: coadd data_i2d not found "
                      f"({_di2d_path}); gate falls back to per-frame data",
                      flush=True)
        satstar_table = load_or_make_satstar_catalog(
            filename,
            path_prefix=f'{basepath}/psfs',
            use_merged_psf_for_merged=(module == 'merged'),
            overwrite=bool(outside_star_pixels),
            outside_star_pixels=outside_star_pixels,
            outside_star_fit_box=512,
            forced_grid_search_radius=forced_grid_search_radius,
            file_suffix=satstar_file_suffix,
            seed_gate_image=_seed_gate_image,
            seed_gate_wcs=_seed_gate_wcs,
        )

        # Pipeline-plumbing fix (2026-04-21):
        # The satstar finder fits the bright/saturated stars and writes a
        # satstar_model.fits, but historically `phot_basic`/`phot_iter` ran on
        # ``nan_replaced_data`` (i.e. bgsub-only -- the satstar model was NOT
        # subtracted).  That left the wings of saturated stars fully visible
        # to the regular fitter, which then placed inflated fits at the
        # "stuck-low" central pixel and produced ~-15000-count holes in the
        # final residual image.  Subtract the satstar model here so the
        # downstream photometry sees the satstar-cleaned data.
        #
        # Filenames mirror those produced by remove_saturated_stars()
        # (saturated_star_finding.py) and load_or_make_satstar_catalog().
        # Prefer the extended model image (force-fit union of per-frame
        # satstar positions across the filter) when present.
        extended_model_path = filename.replace(
            '.fits', f'{satstar_file_suffix}_extended_satstar_model.fits')
        satstar_model_path = filename.replace(
            '.fits', f'{satstar_file_suffix}_satstar_model.fits')
        if os.path.exists(extended_model_path):
            satstar_model_path = extended_model_path
        if os.path.exists(satstar_model_path):
            try:
                satstar_model_image = fits.getdata(satstar_model_path).astype(float)
            except (OSError, ValueError) as exc:
                print(f"Could not read satstar_model {satstar_model_path}: {exc}; "
                      f"skipping satstar subtraction", flush=True)
            else:
                if satstar_model_image.shape != nan_replaced_data.shape:
                    print(f"satstar_model shape {satstar_model_image.shape} does not "
                          f"match data shape {nan_replaced_data.shape}; skipping "
                          f"satstar subtraction", flush=True)
                else:
                    # sat-pixel replacement + subtraction; see _subtract_satstar_model.
                    saturated_mask = (
                        ((im1['DQ'].data & dqflags.pixel['SATURATED']) != 0)
                        if 'DQ' in im1 else None)
                    nan_replaced_data, finite_model = _subtract_satstar_model(
                        nan_replaced_data, satstar_model_image, saturated_mask)
                    satstar_model_subtracted = finite_model
                    n_pos = int(np.sum(finite_model > 0))
                    total = float(np.nansum(finite_model))
                    n_sat = int(saturated_mask.sum()) if saturated_mask is not None else 0
                    print(f"Subtracted satstar_model ({satstar_model_path}) "
                          f"from nan_replaced_data: {n_pos} positive pixels, "
                          f"sum={total:.3e} counts; "
                          f"replaced {n_sat} SATURATED-DQ pixels with model "
                          f"before subtract (residual=0 there)",
                          flush=True)
        else:
            print(f"No satstar_model file at {satstar_model_path}; "
                  f"phot_basic/phot_iter will see the satstar wings unmodified",
                  flush=True)

    seeded_init_params = None
    if seed_catalog is not None:
        preferred_seed_skycoord_col = f'skycoord_{filtername.lower()}'
        merged_seed_table = _as_table(seed_catalog)

        # Snap seed positions to this filter's own iter2 astrometry where
        # available.  build_union_seed_catalog.py records only a single
        # ``skycoord_ref`` column taken from the SHORTEST-WAVELENGTH
        # filter that detected each cluster.  When fitting a long-
        # wavelength filter like F480M, the SW position can be offset
        # several pixels from the LW position (filter-dependent
        # astrometry + saturated-core centroid bias on SW, or simply there is no short-wavelength detection of this exact star).  With
        # iter3's tight xy_bounds (=1 SW pix, ~0.5 LW pix), fits cannot
        # move far enough to reach the true LW star: they end up at the
        # boundary (flags=48 = near_bound + no_covariance) with low
        # flux, and the unmodelled star reappears as a large positive
        # residual.  Diagnosed 2026-06-03 on sickle F480M union seed
        # source_id_union=7288 (flux_f480m=52333, skycoord_ref taken
        # from F187N detection ~5 LW pix away from the true F480M
        # position).
        #
        # Solution: load THIS filter's cross-exposure-merged iter2
        # daoiterative catalog (produced by merge_catalogs.py at
        # {basepath}/catalogs/{filt}_merged_indivexp_merged_iter2_
        # daoiterative_iterative.fits) and add a
        # ``skycoord_<filter>`` mixin column on the seed table whose
        # value is the iter2 skycoord of the nearest in-filter match
        # within ``match_radius`` of each seed.  ``SeededFinder``'s
        # ``preferred_skycoord_col`` already prefers
        # ``skycoord_{filter}`` (line above), so the snapped position
        # gets used automatically when it exists.  Seeds with no
        # nearby per-filter detection fall back to ``skycoord_ref``
        # (existing behaviour).
        try:
            _iter2_cat_path = os.path.join(
                basepath, 'catalogs',
                f'{filtername.lower()}_merged_indivexp_merged_iter2_'
                f'daoiterative_iterative.fits',
            )
            if os.path.exists(_iter2_cat_path):
                _it2 = Table.read(_iter2_cat_path)
                if 'skycoord' in _it2.colnames and len(_it2) > 0:
                    _it2_sk = _it2['skycoord']
                    if not isinstance(_it2_sk, SkyCoord):
                        _it2_sk = SkyCoord(_it2_sk)
                    _seed_table_resolved = _resolve_seed_skycoords(
                        Table(merged_seed_table, copy=True), ww=ww,
                        preferred_skycoord_col=preferred_seed_skycoord_col,
                    )
                    _seed_sk = _seed_table_resolved['skycoord']
                    if not isinstance(_seed_sk, SkyCoord):
                        _seed_sk = SkyCoord(_seed_sk)
                    _seed_sk = _seed_sk.unmasked if hasattr(_seed_sk, 'unmasked') else _seed_sk
                    _it2_sk = _it2_sk.unmasked if hasattr(_it2_sk, 'unmasked') else _it2_sk
                    # Bulk nearest-neighbor match.  Use 3 LW pix ~ 0.2"
                    # for LW filters; SW filters get 1.5 SW pix ~ 0.05".
                    _pixscale_arcsec = (ww.proj_plane_pixel_area()**0.5).to(u.arcsec).value if hasattr(ww, 'proj_plane_pixel_area') else 0.063
                    _match_radius_arcsec = max(0.15, 3.0 * _pixscale_arcsec)
                    _idx, _sep2d, _ = _seed_sk.match_to_catalog_sky(_it2_sk)
                    _good = (_sep2d.arcsec < _match_radius_arcsec)
                    if np.any(_good):
                        _new_ra = np.asarray(_seed_sk.ra.deg, dtype=float).copy()
                        _new_dec = np.asarray(_seed_sk.dec.deg, dtype=float).copy()
                        _new_ra[_good] = np.asarray(_it2_sk.ra.deg)[_idx[_good]]
                        _new_dec[_good] = np.asarray(_it2_sk.dec.deg)[_idx[_good]]
                        merged_seed_table[preferred_seed_skycoord_col] = SkyCoord(
                            ra=_new_ra * u.deg, dec=_new_dec * u.deg, frame='icrs')
                        seed_catalog = merged_seed_table
                        print(f"Snapped {int(np.sum(_good))} of {len(merged_seed_table)} "
                              f"seed positions to per-filter iter2 catalog "
                              f"({_iter2_cat_path.split('/')[-1]}) within "
                              f"{_match_radius_arcsec:.2f}\"; "
                              f"populated {preferred_seed_skycoord_col} column.",
                              flush=True)
                    else:
                        print(f"No seeds within {_match_radius_arcsec:.2f}\" of any "
                              f"per-filter iter2 source; leaving seed positions as-is",
                              flush=True)
                else:
                    print(f"Per-filter iter2 catalog {_iter2_cat_path} has no "
                          f"'skycoord' column or is empty; skipping snap",
                          flush=True)
            else:
                print(f"No per-filter iter2 cat at {_iter2_cat_path}; "
                      f"using cross-band union positions unchanged",
                      flush=True)
        except Exception as _snap_exc:
            print(f"Per-filter iter2 snap failed ({_snap_exc!r}); "
                  f"continuing with cross-band union positions",
                  flush=True)

        # V12: inject per-filter iter2 sources as NEW seed rows.
        #
        # V11 snaps existing union rows to nearby iter2 positions but does
        # not fix two failure modes diagnosed 2026-06-04 on the
        # f480_toinvestgiate_june4 reg list:
        #
        # Mode A (stars 1, 2, 12, 13): iter2 detects a faint target ~0.21"
        # from a bright neighbor.  Union has 2+ SW fragments in the area;
        # V11 snaps each to its closest iter2 source, so faint AND bright
        # iter2 positions each get a snapped union row.  But the snapped
        # union rows inherit flux_init=1 (det_f480m=False, flux_f480m
        # masked).  The pre-fit deduplication (line ~4129, 1.0 * FWHM ~
        # 2.5 LW pix for iter3) keeps the BRIGHTER seed in each cluster;
        # in this case the bright iter2 source has nearby union rows with
        # detected_f480m=True / flux_f480m=5787 carried in -- they win,
        # the faint target's seed (flux_init=1) is dropped along with all
        # other "flux=1" near-coincident union fragments.  Empirically
        # verified: for star 1, V11 snapped u[14153] to iter2[5304] at
        # pix (344.21, 121.83) but basic iter3 has zero inits within 2
        # px of that position.
        #
        # Mode B (stars 7, 11, 15): the nearest union seed is 0.197-
        # 0.253" from its iter2 source - just outside V11's 0.19" snap
        # radius (3.0 * 0.063"/pix).  No snap fires, faint iter2 target
        # has no representative in iter3 seeds.
        #
        # Solution: append every per-filter iter2 source as a fresh seed
        # row, carrying its iter2 flux as flux_init.  Then dedup picks
        # the iter2 row (high flux) over nearby low-flux_init union
        # fragments.  Union rows with detected_f480m=True still have
        # their own flux populated and survive on their own merit.
        try:
            _iter2_cat_path = os.path.join(
                basepath, 'catalogs',
                f'{filtername.lower()}_merged_indivexp_merged_iter2_'
                f'daoiterative_iterative.fits',
            )
            if os.path.exists(_iter2_cat_path):
                _it2 = Table.read(_iter2_cat_path)
                if 'skycoord' in _it2.colnames and len(_it2) > 0:
                    _it2_sk = _it2['skycoord']
                    if not isinstance(_it2_sk, SkyCoord):
                        _it2_sk = SkyCoord(_it2_sk)
                    _it2_sk = _it2_sk.unmasked if hasattr(_it2_sk, 'unmasked') else _it2_sk

                    _n_iter2 = len(_it2)
                    _flux_col_lower = f'flux_{filtername.lower()}'
                    _it2_flux = (np.asarray(_it2['flux'], dtype=float)
                                 if 'flux' in _it2.colnames
                                 else np.ones(_n_iter2, dtype=float))

                    _injected = Table()
                    # Ensure injected rows carry iter2 flux as 'flux'
                    # (SeededFinder reads this as flux_init).  This is
                    # the load-bearing column for dedup brightest-wins.
                    _injected['flux'] = _it2_flux
                    _injected['flux_fit'] = _it2_flux
                    for _col in merged_seed_table.colnames:
                        if _col in ('flux', 'flux_fit'):
                            continue
                        _src = merged_seed_table[_col]
                        if isinstance(_src, SkyCoord):
                            _injected[_col] = SkyCoord(
                                ra=_it2_sk.ra, dec=_it2_sk.dec, frame='icrs')
                        elif _col == 'ra':
                            _injected[_col] = np.asarray(_it2_sk.ra.deg, dtype=float)
                        elif _col == 'dec':
                            _injected[_col] = np.asarray(_it2_sk.dec.deg, dtype=float)
                        elif _col == 'source_id_union':
                            _injected[_col] = np.ma.masked_array(
                                np.full(_n_iter2, -1, dtype=np.int64),
                                mask=np.ones(_n_iter2, dtype=bool))
                        elif _col == 'seed_filter_origin':
                            _injected[_col] = np.array(
                                [f'{filtername.upper()}_ITER2'] * _n_iter2)
                        elif _col == 'is_saturated':
                            _injected[_col] = np.zeros(_n_iter2, dtype=bool)
                        elif _col == 'n_filters':
                            _injected[_col] = np.ones(_n_iter2, dtype=np.int32)
                        elif _col == _flux_col_lower:
                            _injected[_col] = _it2_flux
                        elif _col == f'detected_{filtername.lower()}':
                            _injected[_col] = np.ones(_n_iter2, dtype=bool)
                        elif _col == 'flux_fit':
                            _injected[_col] = _it2_flux
                        elif _col.startswith('detected_'):
                            _injected[_col] = np.zeros(_n_iter2, dtype=bool)
                        else:
                            # Fill numeric columns with NaN, others with
                            # the column's default; use a masked array to
                            # preserve dtype across vstack.
                            _dt = _src.dtype if hasattr(_src, 'dtype') else None
                            if _dt is not None and np.issubdtype(_dt, np.floating):
                                _injected[_col] = np.full(_n_iter2, np.nan, dtype=_dt)
                            elif _dt is not None and np.issubdtype(_dt, np.integer):
                                _injected[_col] = np.ma.masked_array(
                                    np.zeros(_n_iter2, dtype=_dt),
                                    mask=np.ones(_n_iter2, dtype=bool))
                            elif _dt is not None and np.issubdtype(_dt, np.bool_):
                                _injected[_col] = np.zeros(_n_iter2, dtype=bool)
                            else:
                                _injected[_col] = np.ma.masked_array(
                                    np.zeros(_n_iter2, dtype=object),
                                    mask=np.ones(_n_iter2, dtype=bool))

                    from astropy.table import vstack as _vstack
                    merged_seed_table = _vstack(
                        [merged_seed_table, _injected],
                        join_type='outer', metadata_conflicts='silent')
                    seed_catalog = merged_seed_table
                    print(f"Injected {_n_iter2} per-filter iter2 sources as "
                          f"new seed rows (flux_init carried from iter2 "
                          f"'flux' column); dedup will collapse against "
                          f"nearby union fragments. Seed table now "
                          f"{len(merged_seed_table)} rows.", flush=True)
        except Exception as _inject_exc:
            print(f"Per-filter iter2 seed injection failed ({_inject_exc!r}); "
                  f"continuing with snap-only union catalog",
                  flush=True)

        # Also snap satstar_table positions to per-filter iter2 where matched.
        # The force-union satstar table (project_force_union_satstar 2026-05-18)
        # adds entries at cross-frame union (skycoord_ref) positions for stars
        # saturated in OTHER filters' frames -- these land on this frame at
        # the SW astrometric position, which is several LW pixels off from
        # the true F480M position.  Pre-fit dedup then merges the snapped
        # union seed with the unsnapped satstar entry; the satstar entry
        # (which carries is_saturated=True and a valid flux) wins and the
        # snapped position is lost.  Detected 2026-06-03 on sickle F480M
        # region (266.57045,-28.80021): union row 13911 was correctly snapped
        # to pix (311.80,127.07), but iter3 source 55 init landed at
        # (310.26,126.62) -- the unsnapped union/satstar position.  Snapping
        # satstar_table's (x_fit,y_fit) to per-filter iter2 positions
        # closes this gap.
        try:
            if (satstar_table is not None and len(satstar_table) > 0
                    and 'x_fit' in satstar_table.colnames
                    and 'y_fit' in satstar_table.colnames):
                _iter2_cat_path = os.path.join(
                    basepath, 'catalogs',
                    f'{filtername.lower()}_merged_indivexp_merged_iter2_'
                    f'daoiterative_iterative.fits',
                )
                if os.path.exists(_iter2_cat_path):
                    _it2 = Table.read(_iter2_cat_path)
                    if 'skycoord' in _it2.colnames and len(_it2) > 0:
                        _it2_sk = _it2['skycoord']
                        if not isinstance(_it2_sk, SkyCoord):
                            _it2_sk = SkyCoord(_it2_sk)
                        _it2_sk = _it2_sk.unmasked if hasattr(_it2_sk, 'unmasked') else _it2_sk
                        _sat_x = np.asarray(satstar_table['x_fit'], dtype=float)
                        _sat_y = np.asarray(satstar_table['y_fit'], dtype=float)
                        _sat_finite = np.isfinite(_sat_x) & np.isfinite(_sat_y)
                        if np.any(_sat_finite):
                            _sat_sk = ww.pixel_to_world(_sat_x[_sat_finite],
                                                        _sat_y[_sat_finite])
                            _pixscale_arcsec = (ww.proj_plane_pixel_area()**0.5).to(u.arcsec).value if hasattr(ww, 'proj_plane_pixel_area') else 0.063
                            _match_radius_arcsec = max(0.15, 3.0 * _pixscale_arcsec)
                            _idx, _sep2d, _ = _sat_sk.match_to_catalog_sky(_it2_sk)
                            _good = _sep2d.arcsec < _match_radius_arcsec
                            if np.any(_good):
                                _it2_xy = ww.world_to_pixel(_it2_sk[_idx[_good]])
                                _new_x = _sat_x.copy()
                                _new_y = _sat_y.copy()
                                _finite_idx = np.where(_sat_finite)[0]
                                _move_idx = _finite_idx[_good]
                                _new_x[_move_idx] = np.asarray(_it2_xy[0], dtype=float)
                                _new_y[_move_idx] = np.asarray(_it2_xy[1], dtype=float)
                                satstar_table['x_fit'] = _new_x
                                satstar_table['y_fit'] = _new_y
                                if 'x_init' in satstar_table.colnames:
                                    _init_x = np.asarray(satstar_table['x_init'], dtype=float).copy()
                                    _init_y = np.asarray(satstar_table['y_init'], dtype=float).copy()
                                    _init_x[_move_idx] = np.asarray(_it2_xy[0], dtype=float)
                                    _init_y[_move_idx] = np.asarray(_it2_xy[1], dtype=float)
                                    satstar_table['x_init'] = _init_x
                                    satstar_table['y_init'] = _init_y
                                print(f"Snapped {int(np.sum(_good))} of "
                                      f"{len(satstar_table)} satstar_table "
                                      f"entries to per-filter iter2 positions "
                                      f"within {_match_radius_arcsec:.2f}\".",
                                      flush=True)
        except Exception as _snap_sat_exc:
            print(f"Per-filter iter2 satstar snap failed ({_snap_sat_exc!r}); "
                  f"continuing with original satstar positions",
                  flush=True)

        # Optional spatial chunking: split the seed catalog into N image-pixel
        # tiles and fit only the regular seeds whose pixel position lies in
        # this chunk's tile.  Each chunk runs as its own SLURM array job and
        # writes a _chunkXXofYY-tagged per-frame catalog; merge_catalogs.py
        # vstacks them back into one per-frame table.  Used to stay under
        # the 96 h walltime for brick LW iter3 (433 k seeds per frame).
        # satstar_table is NOT subset here -- it stays full-frame so the
        # postprocess-residual masking continues to see every satstar; the
        # merge-time dedup catches the resulting cross-chunk satstar
        # duplicates.
        _n_seed_chunks = int(getattr(options, 'n_seed_chunks', 1) or 1)
        _seed_chunk_index = int(getattr(options, 'seed_chunk_index', 0) or 0)
        if _n_seed_chunks > 1:
            merged_seed_table = _seed_table_chunk_subset(
                merged_seed_table, ww=ww, image_shape=data.shape,
                chunk_index=_seed_chunk_index,
                n_seed_chunks=_n_seed_chunks,
            )
            seed_catalog = merged_seed_table

        # Populate flux_fit on is_saturated union-catalog rows from the
        # per-filter flux column so dedup can compare them against current-
        # frame satstar entries.  Previously these rows were stripped en
        # bloc, which removed bright stars that are saturated in some
        # filters but not in THIS one (e.g. seed flagged saturated from
        # F187N but well within linear range in F480M).  Such stars then
        # had no satstar fit (no DQ-saturated pixels in this filter) AND
        # no daophot fit (stripped) — they remained fully un-subtracted
        # in the residual mosaic.  Diagnosed 2026-05-16 on Sickle F480M:
        # user reg position (266.56408,-28.80118) corresponded to
        # seed[11992] is_saturated=True flux_f480m=4.03e5; nearby NaN-
        # flux duplicate seeds (sep~0.3") were used instead, fits hit
        # xy_bounds=±0.5px and gave flag=48 with flux~1500.  Filling
        # flux_fit from per-filter flux lets _dedup_close_sources keep
        # the correct (brightest, on-target) seed.
        if satstar_table is not None:
            st = _as_table(seed_catalog)
            if 'is_saturated' in st.colnames:
                is_sat_mask = np.asarray(st['is_saturated'], dtype=bool)
                n_sat_in_union = int(np.sum(is_sat_mask))
                if n_sat_in_union > 0:
                    # Find the per-filter flux column for the current filter.
                    _flux_col = f'flux_{filtername.lower()}'
                    if _flux_col in st.colnames:
                        if 'flux_fit' not in st.colnames:
                            st['flux_fit'] = np.full(len(st), np.nan, dtype=float)
                        _f = np.asarray(st[_flux_col], dtype=float)
                        # Only fill where flux_fit is currently NaN AND
                        # per-filter flux is finite.
                        _need = is_sat_mask & np.isnan(np.asarray(st['flux_fit'], dtype=float)) & np.isfinite(_f)
                        if np.any(_need):
                            st['flux_fit'] = np.where(_need, _f, st['flux_fit'])
                            print(f"Filled flux_fit from {_flux_col} on "
                                  f"{int(np.sum(_need))} is_saturated union-catalog "
                                  f"seeds (so dedup can compare them against "
                                  f"current-frame satstar entries)",
                                  flush=True)
                    seed_catalog = st
                    merged_seed_table = _as_table(seed_catalog)
                    print(f"Kept {n_sat_in_union} is_saturated=True rows in union seed catalog "
                          f"(flux_fit populated; dedup handles overlap with current-frame satstar)",
                          flush=True)

        seed_catalog = _combine_seed_and_satstars(seed_catalog, satstar_table)
        seed_after_sat_table = _as_table(seed_catalog)
        sat_seed_count = int(np.sum(np.asarray(seed_after_sat_table['is_saturated'], dtype=bool)))
        nonsat_seed_count = int(len(seed_after_sat_table) - sat_seed_count)
        detection_image = nan_replaced_data
        assert not np.any(np.isnan(nan_replaced_data))
        if postprocess_residuals:
            detection_image = postprocess_residual_image(
                nan_replaced_data,
                fwhm_pix,
                negative_threshold=residual_negative_threshold,
                satstar_table=satstar_table,
            )
        if postprocess_residuals:
            extra_noise_map = compute_local_noise_map(detection_image, smooth_sigma_pix=3.0)
            finite_extra_noise = np.isfinite(extra_noise_map) & (extra_noise_map > 0)
            if not np.any(finite_extra_noise):
                raise ValueError('Postprocessed local noise map has no positive finite values')
            extra_noise_floor = float(np.nanmin(extra_noise_map[finite_extra_noise]))
            extra_finder = DAOStarFinder(threshold=extra_noise_floor,
                                         fwhm=fwhm_pix, roundhi=iter2_roundhi, roundlo=iter2_roundlo,
                                         sharplo=iter2_sharplo, sharphi=iter2_sharphi)
            extra_detections = extra_finder(detection_image, mask=mask)
            extra_noise_for_snr = extra_noise_map
            print(f'Postprocessed DAO local-noise threshold: {extra_noise_floor}', flush=True)
        else:
            extra_detections = daofind_tuned(detection_image, mask=mask)
            extra_noise_for_snr = local_noise_map

        if extra_detections is None:
            extra_detections = Table()
        extra_detections, extra_snr_stats = annotate_and_filter_by_local_snr(
            extra_detections,
            extra_noise_for_snr,
            snr_threshold=iter2_local_snr_threshold,
        )
        print(
            'Extra DAO detections local-SNR filter: '
            f'in={extra_snr_stats["input_count"]} '
            f'kept={extra_snr_stats["kept_count"]} '
            f'dropped={extra_snr_stats["dropped_count"]}',
            flush=True,
        )
        seed_catalog, seed_aug_stats = _augment_seed_catalog_with_detections_sky(
            seed_catalog,
            extra_detections,
            ww=ww,
            match_radius_pix=max(1.0, 0.5 * fwhm_pix),
            preferred_seed_skycoord_col=preferred_seed_skycoord_col,
            return_stats=True,
        )
        print(
            'Seed composition: '
            f'merged_seed_rows={len(merged_seed_table)} '
            f'sat_seed_rows={sat_seed_count} '
            f'nonsat_seed_rows={nonsat_seed_count} '
            f'dao_detect_total={seed_aug_stats["detection_input"]} '
            f'dao_detect_finite_xy={seed_aug_stats["detection_finite_xy"]} '
            f'dao_added={seed_aug_stats["detection_added"]} '
            f'dao_rejected_duplicates={seed_aug_stats["detection_rejected_match"]} '
            f'seed_rows_final={len(_as_table(seed_catalog))}'
        )
        _mem_report("before SeededFinder")
        finstars = SeededFinder(seed_catalog, ww=ww,
                                preferred_skycoord_col=preferred_seed_skycoord_col)(nan_replaced_data, mask=mask)
        _mem_report("after SeededFinder call", deep=True)
        seeded_init_params = Table()
        seeded_init_params['x_init'] = np.asarray(finstars['x_init'], dtype=float)
        seeded_init_params['y_init'] = np.asarray(finstars['y_init'], dtype=float)
        seeded_init_params['flux_init'] = np.asarray(finstars['flux_init'], dtype=float)
        # Carry is_saturated through so dedup can preferentially keep
        # known bright/saturated seeds over nearby NaN-flux duplicates.
        if 'is_saturated' in finstars.colnames:
            seeded_init_params['is_saturated'] = np.asarray(
                finstars['is_saturated'], dtype=bool)

        # Deduplicate seeds: remove entries within 0.5 FWHM of a brighter seed.
        # Merged catalogs can contain sub-pixel duplicate entries from multiple
        # per-exposure fits landing at slightly different positions for the same
        # star.  Two seeds at the same position each receive the full star flux,
        # doubling the model and producing large negative residuals.  No
        # quality metric exists at the seed stage, so the brightest init flux
        # wins any flux-disagreement tie.
        #
        # For iter3 the union seed catalog contains sources from all filters,
        # including SW-only detections that are unresolved at LW wavelengths.
        # Fitting many seeds within one PSF FWHM of each other produces an
        # ill-conditioned LSQ system with wildly oscillating positive/negative
        # fluxes that contaminate the residual image.  Tighten the dedup to
        # 1.0×FWHM for iter3 to collapse seeds within one resolution element
        # to a single fit position before entering the solver.
        min_sep_pix = 1.0 * fwhm_pix if is_iter3 else 0.5 * fwhm_pix
        n_before = len(seeded_init_params)
        if n_before > 1:
            keep, n_disagree = _dedup_close_sources(
                xy=np.column_stack([
                    np.asarray(seeded_init_params['x_init'], dtype=float),
                    np.asarray(seeded_init_params['y_init'], dtype=float),
                ]),
                flux=np.asarray(seeded_init_params['flux_init'], dtype=float),
                min_sep_pix=min_sep_pix,
                quality=None,
                is_saturated=(np.asarray(seeded_init_params['is_saturated'], dtype=bool)
                              if 'is_saturated' in seeded_init_params.colnames
                              else None),
            )
            n_removed = int(n_before - np.sum(keep))
            if n_removed > 0:
                seeded_init_params = seeded_init_params[keep]
                print(f"Pre-fit deduplication removed {n_removed} seeds within "
                      f"{min_sep_pix:.2f} pix ({n_before} -> {len(seeded_init_params)}); "
                      f"{n_disagree} clusters had disagreeing init fluxes",
                      flush=True)
        _mem_report("after seed dedup")

        finding_label = 'seeded'
    else:
        finstars = daofind_tuned(nan_replaced_data,
                                 mask=mask)
        if finstars is None:
            finstars = Table()
        finding_label = 'daofind'

    print(f"Found {len(finstars)} with daofind_tuned", flush=True)
    # for diagnostic plotting convenience
    # photutils >=3.0 emits x_centroid/y_centroid; 2.x emits xcentroid/ycentroid.
    if 'xcentroid' in finstars.colnames:
        finstars['x'] = finstars['xcentroid']
        finstars['y'] = finstars['ycentroid']
    elif 'x_centroid' in finstars.colnames:
        finstars['x'] = finstars['x_centroid']
        finstars['y'] = finstars['y_centroid']
    finstars['skycoord'] = ww.pixel_to_world(finstars['x'], finstars['y'])

    # All basepath-based INPUT reads (resbg, seed, satstar, union catalogs)
    # are done by this point; redirect every subsequent OUTPUT to the cutout
    # directory so cutout products never overwrite full-frame photometry.
    if _cutout_active:
        os.makedirs(f'{out_basepath}/{filtername}/pipeline', exist_ok=True)
        basepath = out_basepath

    result = save_photutils_results(finstars, ww, filename,
                                    im1=im1, detector=detector,
                                    basepath=basepath,
                                    filtername=filtername, module=module,
                                    desat=desat, bgsub=bgsub,
                                    blur="",
                                    exposure_=exposure_,
                                    visitid_=visitid_, vgroupid_=vgroupid_,
                                    basic_or_iterative=finding_label,
                                    options=options,
                                    epsf_="",
                                    fpsf="",
                                    group=group,
                                    psf=None,
                                    background_map=background_map,
                                    iteration_label=iteration_label)

    stars = finstars # because I'm copy-pasting code...

    # Set up visualization
    reg = regions.RectangleSkyRegion(center=cen, width=1.5*u.arcmin, height=1.5*u.arcmin)
    preg = reg.to_pixel(ww)
    #mask = preg.to_mask()
    #cutout = mask.cutout(im1[1].data)
    #err = mask.cutout(im1[2].data)
    # Zoom regions live in a shared regions_/ directory across fields (brick,
    # sgrb2, cloudc, sickle).  Two guards keep cross-field regions out:
    # (1) sky-separation gate — if the region center sits more than
    #     max_field_sep from this detector's pointing, it belongs to a
    #     different field and projects to extreme pixel coords.
    # (2) bbox sanity check — `to_mask()` allocates a (ny,nx) float64 array
    #     sized to the bounding box (regions/shapes/rectangle.py:165).  A
    #     half-degree-misplaced rectangle yields a TB-scale allocation that
    #     drove sgrb2 nrca1 photometry to 564 GB peak.  Refuse anything
    #     larger than 4× the detector.
    region_list = [y for x in glob.glob(str(REGIONS_DIR / '*zoom*.reg')) for y in
                   regions.Regions.read(x)]
    ny, nx = data.shape
    data_center_sky = ww.pixel_to_world(nx / 2.0, ny / 2.0)
    max_field_sep = 6 * u.arcmin
    zoomcut_list = {}
    for _reg in region_list:
        reg_center = getattr(_reg, 'center', None)
        if isinstance(reg_center, SkyCoord):
            if reg_center.separation(data_center_sky) > max_field_sep:
                continue
        _pix = _reg.to_pixel(ww)
        bb = _pix.bounding_box
        if bb.ixmax < 0 or bb.iymax < 0 or bb.ixmin >= nx or bb.iymin >= ny:
            continue
        bb_w = bb.ixmax - bb.ixmin
        bb_h = bb.iymax - bb.iymin
        if bb_w * bb_h > 4 * nx * ny:
            print(f"Skipping oversized zoom region '{_reg.meta.get('text', '?')}': "
                  f"bbox {bb_w}x{bb_h} exceeds 4x detector ({nx}x{ny})",
                  flush=True)
            continue
        _slc_tuple = _pix.to_mask().get_overlap_slices(data.shape)
        if _slc_tuple is None or _slc_tuple[0] is None:
            continue
        _slc = _slc_tuple[0]
        if (_slc[0].start > 0 and _slc[1].start > 0
                and _slc[0].stop < data.shape[0] and _slc[1].stop < data.shape[1]):
            zoomcut_list[_reg.meta['text']] = _slc

    _mem_report("after region zoomcut loop", deep=True)
    zoomcut = slice(128, 256), slice(128, 256)
    modsky = data*0 # no model for daofind
    nullslice = (slice(None), slice(None))

    _mem_report("before daofind catalog_zoom_diagnostic block")
    try:
        catalog_zoom_diagnostic(data, modsky, nullslice, stars)
        pl.suptitle(f"daofind Catalog Diagnostics zoomed {filtername} {module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}")
        pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_{inst_token}_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}_catalog_diagnostics_daofind.png',
                bbox_inches='tight')

        catalog_zoom_diagnostic(data, modsky, zoomcut, stars)
        pl.suptitle(f"daofind Catalog Diagnostics {filtername} {module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}")
        pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_{inst_token}_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}_catalog_diagnostics_zoom_daofind.png',
                bbox_inches='tight')

        for name, zoomcut in zoomcut_list.items():
            catalog_zoom_diagnostic(data, modsky, zoomcut, stars)
            pl.suptitle(f"daofind Catalog Diagnostics {filtername} {module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub} zoom {name}")
            pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_{inst_token}_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}_catalog_diagnostics_zoom{name.replace(" ","_")}_daofind.png',
                    bbox_inches='tight')
    except Exception as ex:
        print(f'FAILURE to produce catalog zoom diagnostics for module {module} and filter {filtername} for basic daofinder: {ex}')
    _mem_report("after daofind catalog_zoom_diagnostic block", deep=True)

    if not options.nocrowdsource:

        t0 = time.time()

        if False: # why do the unweighted version?
            print()
            print("starting crowdsource unweighted", flush=True)
            results_unweighted = fit_im(nan_replaced_data, psf_model,
                                        weight=np.ones_like(data)*np.nanmedian(weight)*(~mask),
                                        # psfderiv=np.gradient(-psf_initial[0].data),
                                        dq=dq,
                                        nskyx=0, nskyy=0, refit_psf=False, verbose=True,
                                        **crowdsource_default_kwargs,
                                        )
            print(f"Done with unweighted crowdsource. dt={time.time() - t0}")
            stars, modsky, skymsky, psf = results_unweighted
            stars = save_crowdsource_results(results_unweighted, ww, filename,
                                             im1=im1, detector=detector,
                                             basepath=basepath,
                                             filtername=filtername, module=module,
                                             desat=desat, bgsub=bgsub,
                                             blur=options.blur,
                                             exposure_=exposure_,
                                             visitid_=visitid_,
                                             vgroupid_=vgroupid_,
                                             options=options,
                                             suffix="unweighted", psf=None,
                                             iteration_label=iteration_label)

            zoomcut = slice(128, 256), slice(128, 256)

            try:
                catalog_zoom_diagnostic(data, modsky, nullslice, stars)
                pl.suptitle(f"Crowdsource nsky=0 unweighted Catalog Diagnostics zoomed {filtername} {module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{blur_}")
                pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_{inst_token}_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{blur_}_catalog_diagnostics_unweighted.png',
                        bbox_inches='tight')

                catalog_zoom_diagnostic(data, modsky, zoomcut, stars)
                pl.suptitle(f"Crowdsource nsky=0 unweighted Catalog Diagnostics {filtername} {module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{blur_}")
                pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_{inst_token}_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{blur_}_catalog_diagnostics_zoom_unweighted.png',
                        bbox_inches='tight')
                for name, zoomcut in zoomcut_list.items():
                    catalog_zoom_diagnostic(data, modsky, zoomcut, stars)
                    pl.suptitle(f"Crowdsource nsky=0 Catalog Diagnostics {filtername} {module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{blur_} zoom {name}")
                    pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_{inst_token}_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{blur_}_catalog_diagnostics_zoom{name.replace(" ","_")}_unweighted.png',
                            bbox_inches='tight')
            except Exception as ex:
                print(f'FAILURE to produce catalog zoom diagnostics for module {module} and filter {filtername} for unweighted crowdsource: {ex}')
                exc_tb = sys.exc_info()[2]
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(f"Exception {ex} was in {fname} line {exc_tb.tb_lineno}")

            fig = pl.figure(0, figsize=(10,10))
            fig.clf()
            ax = fig.gca()
            im = ax.imshow(weight, norm=simple_norm(weight, stretch='log'))
            pl.colorbar(mappable=im)
            pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_{inst_token}_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}_weights.png',
                    bbox_inches='tight')

        #for refit_psf, fpsf in zip((False, True), ('', '_fitpsf',)):
        for refit_psf, fpsf in zip((False, ), ('', )):
            for nsky in (0, ): #1, ):
                t0 = time.time()
                print()
                print(f"Running crowdsource fit_im with weights & nskyx=nskyy={nsky} & fpsf={fpsf} & blur={blur_}")
                print(f"data.shape={data.shape} weight_shape={weight.shape}", flush=True)
                _mem_report("before crowdsource fit_im")
                results = fit_im(nan_replaced_data, psf_model, weight=weight * (~mask),
                                 nskyx=nsky, nskyy=nsky, refit_psf=refit_psf, verbose=True,
                                 dq=dq,
                                 **crowdsource_default_kwargs
                                 )
                _mem_report("after crowdsource fit_im", deep=True)
                print(f"Done with weighted, refit={fpsf}, nsky={nsky} crowdsource. dt={time.time() - t0}")
                stars, modsky, skymsky, psf = results
                stars = save_crowdsource_results(results, ww, filename,
                                                 im1=im1, detector=detector,
                                                 basepath=basepath,
                                                 filtername=filtername,
                                                 module=module, desat=desat,
                                                 bgsub=bgsub, fpsf=fpsf,
                                                 blur=options.blur,
                                                 exposure_=exposure_,
                                                 visitid_=visitid_,
                                                 vgroupid_=vgroupid_,
                                                 psf=psf if refit_psf else None,
                                                 options=options,
                                                 suffix=f"nsky{nsky}",
                                                 iteration_label=iteration_label)
                _mem_report("after save_crowdsource_results")

                zoomcut = slice(128, 256), slice(128, 256)

                try:
                    catalog_zoom_diagnostic(data, modsky, nullslice, stars)
                    pl.suptitle(f"Catalog Diagnostics {filtername} {module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{fpsf}{blur_} nsky={nsky} weighted")
                    pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_{inst_token}_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{fpsf}{blur_}_nsky{nsky}_weighted_catalog_diagnostics.png',
                            bbox_inches='tight')

                    catalog_zoom_diagnostic(data, modsky, zoomcut, stars)
                    pl.suptitle(f"Catalog Diagnostics zoomed {filtername} {module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{fpsf}{blur_} nsky={nsky} weighted")
                    pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_{inst_token}_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{fpsf}{blur_}_nsky{nsky}_weighted_catalog_diagnostics_zoom.png',
                            bbox_inches='tight')

                    for name, zoomcut in zoomcut_list.items():
                        catalog_zoom_diagnostic(data, modsky, zoomcut, stars)
                        pl.suptitle(f"Crowdsource nsky={nsky} weighted Catalog Diagnostics {filtername} {module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{fpsf}{blur_} zoom {name}")
                        pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_{inst_token}_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{fpsf}{blur_}_nsky{nsky}_weighted_catalog_diagnostics_zoom{name.replace(" ","_")}.png',
                                bbox_inches='tight')
                except Exception as ex:
                    print(f'FAILURE to produce catalog zoom diagnostics for module {module} and filter {filtername} for crowdsource nsky={nsky} refitpsf={refit_psf} blur={options.blur}: {ex}')
                    exc_tb = sys.exc_info()[2]
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(f"Exception {ex} was in {fname} line {exc_tb.tb_lineno}")
                _mem_report("after crowdsource diag plotting", deep=True)

    if options.daophot:
        t0 = time.time()
        print("Starting basic PSF photometry", flush=True)
        _mem_report("before phot_basic setup")

        basic_finder = None if seeded_init_params is not None else daofind_tuned
        _phot_basic_extra = {}
        if iter3_xy_bounds_pix is not None:
            _phot_basic_extra['xy_bounds'] = (iter3_xy_bounds_pix,
                                              iter3_xy_bounds_pix)

        _parallel_workers = int(getattr(options, 'parallel_workers', 1) or 1)
        if _parallel_workers > 1:
            # EXPERIMENTAL parallel path.  Run the finder serially (cheap
            # relative to fitting), then chunk-parallel the fit.  See
            # _parallel_psfphotometry / _FakePhot for details.  Off
            # unless --parallel-workers > 1.
            if seeded_init_params is not None:
                _basic_init = seeded_init_params
            else:
                print("Running basic finder (serial, pre-chunking)", flush=True)
                _basic_sources = daofind_tuned(nan_replaced_data, mask=mask)
                _basic_init = Table()
                _xcol = ('x_centroid' if _basic_sources is not None
                         and 'x_centroid' in _basic_sources.colnames
                         else 'xcentroid')
                _ycol = ('y_centroid' if _basic_sources is not None
                         and 'y_centroid' in _basic_sources.colnames
                         else 'ycentroid')
                if _basic_sources is None or len(_basic_sources) == 0:
                    _basic_init['x_init'] = np.zeros(0)
                    _basic_init['y_init'] = np.zeros(0)
                    _basic_init['flux_init'] = np.zeros(0)
                else:
                    _basic_init['x_init'] = _basic_sources[_xcol]
                    _basic_init['y_init'] = _basic_sources[_ycol]
                    _basic_init['flux_init'] = _basic_sources['flux']

            _phot_basic_kwargs = dict(
                psf_model=dao_psf_model,
                fitter=LevMarLSQFitter(),
                fit_shape=(5, 5),
                aperture_radius=aperture_radius_pix,
                progress_bar=False,
                grouper=grouper if options.group else None,
                finder=None,
            )
            _phot_basic_kwargs['localbkg_estimator'] = LocalBackground(
                localbkg_inner, localbkg_outer)
            _phot_basic_kwargs.update(_phot_basic_extra)
            _chunk_size = int(getattr(options, 'parallel_chunk_size', 100))
            print(f"About to do BASIC photometry (PARALLEL, "
                  f"n_workers={_parallel_workers}, chunk={_chunk_size}, "
                  f"n_sources={len(_basic_init)})....", flush=True)
            _mem_report("before phot_basic call")
            result, _ = _parallel_psfphotometry(
                nan_replaced_data,
                photometry_kwargs=_phot_basic_kwargs,
                init_params=_basic_init,
                error=np.where(bad, 1e10, err),
                mask=mask,
                n_workers=_parallel_workers,
                chunk_size=_chunk_size,
                group_min_separation=2 * fwhm_pix,
                return_model=False,
            )
            phot_basic = _FakePhot(results=result,
                                   psf_model=dao_psf_model,
                                   init_params=_basic_init)
        else:
            phot_basic = _make_psfphotometry(
                                       finder=basic_finder,
                                       # filter-scaled: inner clears aperture
                                       # (2*FWHM) plus first Airy sidelobe;
                                       # see header comment near aperture_radius_pix.
                                       localbkg_estimator=LocalBackground(localbkg_inner, localbkg_outer),
                                       grouper=grouper if options.group else None,
                                       psf_model=dao_psf_model,
                                       fitter=LevMarLSQFitter(),
                                       fit_shape=(5, 5),
                                       aperture_radius=aperture_radius_pix,
                                       progress_bar=True,
                                       **_phot_basic_extra,
                                      )

            print("About to do BASIC photometry....")
            _mem_report("before phot_basic call")
            if seeded_init_params is not None:
                result = phot_basic(nan_replaced_data, mask=mask, init_params=seeded_init_params, error=np.where(bad, 1e10, err))
            else:
                result = phot_basic(nan_replaced_data, mask=mask, error=np.where(bad, 1e10, err))
        print(f"Done with BASIC photometry. len(result)={len(result)}  dt={time.time() - t0}")
        _mem_report("after phot_basic")

        # Post-fit deduplication: the unseeded DAO finder can detect multiple
        # local maxima near a single bright star; each is fit independently
        # without a grouper, and they can converge to the same (x_fit, y_fit).
        # Summing those PSFs in make_model_image() produces 2x-4x overfits.
        # When a cluster of fits converges to the same position, the same
        # physical source should yield matching fluxes; if fluxes disagree
        # the fits reached different minima and we keep the one with the
        # best qfit (smallest chi-squared/pixel).  Filter the phot_basic
        # object's own state so the saved catalog and the rendered model
        # image are both built from the deduplicated set.
        # 2026-04-24: threshold loosened from 0.5 * fwhm_pix to
        # 1.5 * fwhm_pix.  The tighter threshold left seeded-duplicate
        # fits at 2.5-3.2 LW pix apart uncaught, which drove
        # progressively deeper negative residuals in iter2 (p0.1=-96)
        # and iter3 (p0.1=-172).
        # 2026-06-04 (V13): tighten to 1.0 px to preserve real
        # adjacent stars from V12 iter2 inject (e.g. sickle F480M
        # star 1 faint target 3.32 px from bright neighbor was being
        # qfit-merged into bright at 1.5*FWHM=3.86 px).  Risk: may
        # re-introduce 2026-04-24 deep negative residuals if drift-
        # together duplicates beyond 1.0 px exist; will be measured
        # on V13 mosaic.
        min_sep_pix = 1.0
        xfit_arr = np.asarray(result['x_fit'], dtype=float)
        yfit_arr = np.asarray(result['y_fit'], dtype=float)
        flux_arr = np.asarray(result['flux_fit'], dtype=float)
        qfit_arr = (np.asarray(result['qfit'], dtype=float)
                    if 'qfit' in result.colnames else None)
        keep_full, n_disagree = _dedup_close_sources(
            xy=np.column_stack([xfit_arr, yfit_arr]),
            flux=flux_arr,
            min_sep_pix=min_sep_pix,
            quality=qfit_arr,
        )
        n_removed = int(len(keep_full) - np.sum(keep_full))
        if n_removed > 0:
            tiebreak = "qfit" if qfit_arr is not None else "brightest"
            print(f"Post-fit deduplication: dropping {n_removed} drift-together "
                  f"fits within {min_sep_pix:.2f} pix "
                  f"({len(result)} -> {int(np.sum(keep_full))}); "
                  f"{n_disagree} clusters had disagreeing fitted fluxes "
                  f"(resolved by {tiebreak})", flush=True)
            # Filter the PSFPhotometry object's own state so the saved catalog
            # AND the model image (via make_model_image) are both built from
            # the deduplicated set.
            phot_basic.results = phot_basic.results[keep_full]
            if (phot_basic.init_params is not None
                    and len(phot_basic.init_params) == len(keep_full)):
                phot_basic.init_params = phot_basic.init_params[keep_full]
            # Invalidate the @lazyproperty cache so _model_image_params
            # regenerates from the filtered results on next access.
            phot_basic.__dict__.pop('_model_image_params', None)
            # Keep the local `result` name in sync with the filtered table.
            result = phot_basic.results

        # Saturation-proximity filter: regular phot_basic fits placed on
        # or right next to a saturated DQ pixel are unreliable (the central
        # data value is "stuck" while the wings drive the flux up), so drop
        # them before the catalog and the model image are written.  The
        # dedicated satstar catalog lives in a separate file and is not
        # touched.
        _dqarr_for_satfilter = im1['DQ'].data if 'DQ' in im1 else None
        _filter_near_saturation(phot_basic, _dqarr_for_satfilter,
                                max_sat_dist_pix=1.0,  # changed from 5.0 -> 1.0 (2026-05-28): the
                                # original intent is to drop fits whose CENTER
                                # is on a saturated pixel ("stuck-low"
                                # central data drives wing-only fit to bogus
                                # flux).  5.0 also killed legitimate stars
                                # within 5 px of a saturated neighbour --
                                # detected on Star B (17:46:14.175, 3500
                                # MJy/sr) sitting 2.24 px from the donut
                                # neighbour's nearest saturated edge pixel.
                                # 1.0 keeps the strict "center on or
                                # immediately adjacent to a sat pixel" guard
                                # while letting nearby real stars survive.
                                label='basic')
        _filter_satstar_artifacts(phot_basic, satstar_model_subtracted, err,
                                  sig_K=float(options.satstar_artifact_sigK),
                                  ratio_cut=float(options.satstar_artifact_ratio),
                                  label='basic')
        result = phot_basic.results

        result = save_photutils_results(result, ww, filename,
                                        im1=im1, detector=detector,
                                        basepath=basepath,
                                        filtername=filtername, module=module,
                                        desat=desat, bgsub=bgsub,
                                        blur=options.blur,
                                        exposure_=exposure_,
                                        visitid_=visitid_,
                                        vgroupid_=vgroupid_,
                                        basic_or_iterative='basic',
                                        options=options,
                                        epsf_=epsf_,
                                        group=group,
                                        psf=None,
                                        background_map=background_map,
                                        iteration_label=iteration_label)

        stars = result
        stars['x'] = stars['x_fit']
        stars['y'] = stars['y_fit']
        print("Creating BASIC residual image, using 21x21 patches")
        _mem_report("before basic model image")
        modsky = _make_model_image(phot_basic, data.shape, psf_shape=(21, 21), include_local_bkg=False)
        _mem_report("after basic model image")
        # The fitter saw ``data - satstar_model`` (when a satstar model
        # exists), so the saved residual must subtract the satstar model
        # too -- otherwise the bright-star wings reappear and dominate
        # the residual mosaic.  See pipeline-plumbing block above.
        # iter4resbgrefit: build the residual against the ORIGINAL (pre-bg-
        # subtraction) data so the saved residual = original - star models
        # (background retained).  All other iterations use ``data`` as before.
        _resid_base = (original_data if (is_resbg_refit and original_data is not None)
                       else data)
        data_for_residual = (_resid_base if satstar_model_subtracted is None
                             else _resid_base - satstar_model_subtracted)
        residual = data_for_residual - modsky
        print("Done creating BASIC residual image, using 21x21 patches")
        save_residual_datamodel(
            filename,
            f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_{inst_token}_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}{iter_}_daophot_basic_residual.fits',
            residual,
        )
        save_residual_datamodel(
            filename,
            f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_{inst_token}_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}{iter_}_daophot_basic_model.fits',
            modsky,
        )
        print("Saved BASIC residual image, now making diagnostics.")
        catalog_zoom_diagnostic(data_for_residual, modsky, nullslice, stars)
        pl.suptitle(f"daophot basic Catalog Diagnostics zoomed {filtername} {module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}")
        pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_{inst_token}_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}_catalog_diagnostics_daophot_basic.png',
                bbox_inches='tight')

        catalog_zoom_diagnostic(data_for_residual, modsky, zoomcut, stars)
        pl.suptitle(f"daophot basic Catalog Diagnostics {filtername} {module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}")
        pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_{inst_token}_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}_catalog_diagnostics_zoom_daophot_basic.png',
                bbox_inches='tight')

        for name, zoomcut in zoomcut_list.items():
            catalog_zoom_diagnostic(data_for_residual, modsky, zoomcut, stars)
            pl.suptitle(f"daophot basic Catalog Diagnostics {filtername} {module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group} zoom {name}")
            pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_{inst_token}_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}__catalog_diagnostics_zoom_daophot_basic{name.replace(" ","_")}.png',
                    bbox_inches='tight')

        print(f"Done with diagnostics for BASIC photometry.  dt={time.time() - t0}")
        pl.close('all')

        if not options.basic_only:
            t0 = time.time()
            print("Iterative PSF photometry")
            if options.epsf:
                print("Building EPSF")
                epsf_builder = EPSFBuilder(oversampling=3, maxiters=10,
                                           smoothing_kernel='quadratic',
                                           progress_bar=True)

                epsfsel = ((finstars['peak'] > 200) &
                           (finstars['roundness1'] > -0.25) &
                           (finstars['roundness1'] < 0.25) &
                           (finstars['roundness2'] > -0.25) &
                           (finstars['roundness2'] < 0.25) &
                           (finstars['sharpness'] > 0.4) &
                           (finstars['sharpness'] < 0.8))

                print(f"Extracting {epsfsel.sum()} stars")
                stars = extract_stars(NDData(data=nan_replaced_data), finstars[epsfsel], size=35)

                for star in stars:
                    background = np.nanpercentile(star.data, 5)
                    star.data[:] -= background

                epsf, fitted_stars = epsf_builder(stars)
                epsf._data = epsf.data[2:-2, 2:-2]

                norm = simple_norm(epsf.data, 'log', percent=99.0)
                pl.figure(1).clf()
                pl.imshow(epsf.data, norm=norm, origin='lower', cmap='viridis')
                pl.colorbar()
                pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_{inst_token}_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}_daophot_epsf.png',
                           bbox_inches='tight')
                dao_psf_model = epsf

            _phot_iter_extra = {}
            if iter3_xy_bounds_pix is not None:
                _phot_iter_extra['xy_bounds'] = (iter3_xy_bounds_pix,
                                                 iter3_xy_bounds_pix)

            _parallel_workers = int(getattr(options, 'parallel_workers', 1) or 1)
            if _parallel_workers > 1:
                # EXPERIMENTAL parallel path.  Reimplements
                # IterativePSFPhotometry mode='new' with chunked
                # PSFPhotometry calls.  Returns a `_FakePhot` stand-in
                # whose `.results`, `.make_model_image()` etc. satisfy
                # the downstream dedup / sat-filter / model-rendering
                # code paths without depending on photutils internal
                # _fit_models state (which is per-worker and not
                # reconstructable).  Off unless --parallel-workers > 1.
                _phot_iter_kwargs = dict(
                    psf_model=dao_psf_model,
                    fitter=LevMarLSQFitter(),
                    fit_shape=(5, 5),
                    aperture_radius=aperture_radius_pix,
                    progress_bar=False,
                    grouper=grouper if options.group else None,
                )
                _phot_iter_kwargs['localbkg_estimator'] = LocalBackground(
                    localbkg_inner, localbkg_outer)
                _phot_iter_kwargs.update(_phot_iter_extra)
                _chunk_size = int(getattr(options, 'parallel_chunk_size', 100))
                print(f"About to do ITERATIVE photometry (PARALLEL, "
                      f"n_workers={_parallel_workers}, chunk={_chunk_size})....",
                      flush=True)
                _mem_report("before phot_iter call")
                result2 = _parallel_iterative_psfphotometry(
                    nan_replaced_data,
                    photometry_kwargs=_phot_iter_kwargs,
                    finder=daofind_tuned,
                    init_params=seeded_init_params,
                    error=np.where(bad, 1e10, err),
                    mask=mask,
                    maxiters=5,
                    sub_shape=(15, 15),
                    psf_model=dao_psf_model,
                    n_workers=_parallel_workers,
                    chunk_size=_chunk_size,
                    group_min_separation=2 * fwhm_pix,
                )
                phot_iter = _FakePhot(results=result2,
                                      psf_model=dao_psf_model,
                                      init_params=seeded_init_params)
            else:
                phot_iter = _make_iterative_psfphotometry(
                                                   finder=daofind_tuned,
                                                   localbkg_estimator=LocalBackground(localbkg_inner, localbkg_outer),
                                                   grouper=grouper if options.group else None,
                                                   psf_model=dao_psf_model,
                                                   fitter=LevMarLSQFitter(),
                                                   maxiters=5,
                                                   fit_shape=(5, 5),
                                                   sub_shape=(15, 15),
                                                   aperture_radius=aperture_radius_pix,
                                                   progress_bar=True,
                                                   **_phot_iter_extra,
                                                  )

                print("About to do ITERATIVE photometry....")
                _mem_report("before phot_iter call")
                if seeded_init_params is not None:
                    result2 = phot_iter(nan_replaced_data, mask=mask, init_params=seeded_init_params, error=np.where(bad, 1e10, err))
                else:
                    result2 = phot_iter(nan_replaced_data, mask=mask, error=np.where(bad, 1e10, err))
            print(f"Done with ITERATIVE photometry. len(result2)={len(result2)}  dt={time.time() - t0}")
            _mem_report("after phot_iter")

            # Apply the same post-fit deduplication to the iterative results.
            # IterativePSFPhotometry can produce drift-together duplicates across
            # iterations (a source detected in the residual of iter N matches the
            # same star fit in iter N-1).  Left in place, these duplicates
            # (i) double the flux in the rendered model image, and
            # (ii) trigger a photutils bug in make_model_image where the
            #      composite model parameters become array-valued and overlap_slices
            #      raises ValueError("The truth value of an array ... is ambiguous").
            # 2026-04-24: threshold loosened to 1.5 * fwhm_pix, same
            # rationale as in phot_basic above.
            # 2026-06-04 (V13): tighten to 1.0 px to preserve real
            # adjacent stars (see phot_basic note above).
            min_sep_pix = 1.0
            xfit_arr = np.asarray(result2['x_fit'], dtype=float)
            yfit_arr = np.asarray(result2['y_fit'], dtype=float)
            flux_arr = np.asarray(result2['flux_fit'], dtype=float)
            qfit_arr = (np.asarray(result2['qfit'], dtype=float)
                        if 'qfit' in result2.colnames else None)
            iter_keep, iter_n_disagree = _dedup_close_sources(
                xy=np.column_stack([xfit_arr, yfit_arr]),
                flux=flux_arr,
                min_sep_pix=min_sep_pix,
                quality=qfit_arr,
            )
            iter_n_removed = int(len(iter_keep) - np.sum(iter_keep))
            if iter_n_removed > 0:
                iter_tiebreak = "qfit" if qfit_arr is not None else "brightest"
                print(f"Post-fit deduplication (iterative): dropping {iter_n_removed} "
                      f"drift-together fits within {min_sep_pix:.2f} pix "
                      f"({len(result2)} -> {int(np.sum(iter_keep))}); "
                      f"{iter_n_disagree} clusters had disagreeing fitted fluxes "
                      f"(resolved by {iter_tiebreak})", flush=True)
                phot_iter.results = phot_iter.results[iter_keep]
                # IterativePSFPhotometry has no init_params attribute of its
                # own, but its internal PSFPhotometry (self._psfphot) does.
                inner_phot = getattr(phot_iter, '_psfphot', None)
                if (inner_phot is not None
                        and inner_phot.init_params is not None
                        and len(inner_phot.init_params) == len(iter_keep)):
                    inner_phot.init_params = inner_phot.init_params[iter_keep]
                phot_iter.__dict__.pop('_model_image_params', None)
                result2 = phot_iter.results

            # photutils.datasets.images.make_model_image uses a per-row
            # 'model_shape' column in the params table when present; if that
            # column has been populated with array-valued entries during the
            # iterative fit it triggers the ndarray-vs-tuple comparison inside
            # astropy.nddata.utils.overlap_slices.  Drop that column so the
            # caller's psf_shape=(21, 21) argument governs stamp size for all
            # sources uniformly.
            if 'model_shape' in phot_iter.results.colnames:
                phot_iter.results.remove_column('model_shape')
                phot_iter.__dict__.pop('_model_image_params', None)

            # Saturation-proximity filter for the iterative path -- same
            # rationale as the basic path: drop fits placed within
            # max_sat_dist_pix of any saturated DQ pixel.  Iterative
            # photometry is more aggressive and produces more of these
            # spurious near-saturation fits than basic, so the filter has
            # bigger impact here.
            _dqarr_for_satfilter_iter = im1['DQ'].data if 'DQ' in im1 else None
            _filter_near_saturation(phot_iter, _dqarr_for_satfilter_iter,
                                    max_sat_dist_pix=1.0,  # changed from 5.0 -> 1.0 (2026-05-28): the
                                # original intent is to drop fits whose CENTER
                                # is on a saturated pixel ("stuck-low"
                                # central data drives wing-only fit to bogus
                                # flux).  5.0 also killed legitimate stars
                                # within 5 px of a saturated neighbour --
                                # detected on Star B (17:46:14.175, 3500
                                # MJy/sr) sitting 2.24 px from the donut
                                # neighbour's nearest saturated edge pixel.
                                # 1.0 keeps the strict "center on or
                                # immediately adjacent to a sat pixel" guard
                                # while letting nearby real stars survive.
                                    label='iterative')
            _filter_satstar_artifacts(phot_iter, satstar_model_subtracted, err,
                                      sig_K=float(options.satstar_artifact_sigK),
                                      ratio_cut=float(options.satstar_artifact_ratio),
                                      label='iterative')
            result2 = phot_iter.results

            result2 = save_photutils_results(result2, ww, filename,
                                             im1=im1, detector=detector,
                                             basepath=basepath,
                                             filtername=filtername, module=module,
                                             desat=desat, bgsub=bgsub,
                                             blur=options.blur,
                                             exposure_=exposure_,
                                             visitid_=visitid_,
                                             vgroupid_=vgroupid_,
                                             basic_or_iterative='iterative',
                                             options=options,
                                             epsf_=epsf_,
                                             group=group,
                                             psf=None,
                                             background_map=background_map,
                                             iteration_label=iteration_label)

            stars = result2
            stars['x'] = stars['x_fit']
            stars['y'] = stars['y_fit']

            print("Creating iterative residual")
            _mem_report("before iter model image")
            modsky = _make_model_image(phot_iter, data.shape, psf_shape=(21, 21), include_local_bkg=False)
            _mem_report("after iter model image")
            # iter4resbgrefit: residual against the ORIGINAL (pre-bg-subtraction)
            # data; all other iterations use ``data`` (see basic block above).
            _resid_base = (original_data if (is_resbg_refit and original_data is not None)
                           else data)
            data_for_residual = (_resid_base if satstar_model_subtracted is None
                                 else _resid_base - satstar_model_subtracted)
            residual = data_for_residual - modsky
            print("finished iterative residual")
            save_residual_datamodel(
                filename,
                f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_{inst_token}_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}{iter_}_daophot_iterative_residual.fits',
                residual,
            )
            save_residual_datamodel(
                filename,
                f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_{inst_token}_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}{iter_}_daophot_iterative_model.fits',
                modsky,
            )
            print("Saved iterative residual")
            catalog_zoom_diagnostic(data_for_residual, modsky, nullslice, stars)
            pl.suptitle(f"daophot iterative Catalog Diagnostics zoomed {filtername} {module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}")
            pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_{inst_token}_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}_catalog_diagnostics_daophot_iterative.png',
                    bbox_inches='tight')

            catalog_zoom_diagnostic(data_for_residual, modsky, zoomcut, stars)
            pl.suptitle(f"daophot iterative Catalog Diagnostics {filtername} {module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}")
            pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_{inst_token}_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}_catalog_diagnostics_zoom_daophot_iterative.png',
                    bbox_inches='tight')

            for name, zoomcut in zoomcut_list.items():
                catalog_zoom_diagnostic(data_for_residual, modsky, zoomcut, stars)
                pl.suptitle(f"daophot iterative Catalog Diagnostics {filtername} {module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group} zoom {name}")
                pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_{inst_token}_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}__catalog_diagnostics_zoom_daophot_iterative{name.replace(" ","_")}.png',
                        bbox_inches='tight')

            print(f"Done with diagnostics for ITERATIVE photometry.  dt={time.time() - t0}")
            pl.close('all')
        else:
            print("Skipping ITERATIVE photometry because --basic-only was requested")

