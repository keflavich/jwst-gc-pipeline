"""
Driver: JWST-epoch x JWST-epoch proper motions via flystar, for a field whose
only long-baseline counterpart is another JWST program rather than a ground
catalog (VIRAC2/GNS) -- e.g. Sgr B2 (program 5365) x GC Treasury (program
10678), or Cloud e/f (program 2092) x GC Treasury, once GC Treasury source
photometry exists over that footprint.

Unlike build_multiepoch_pm.py (JWST x GNS x VIRAC2, VIRAC2 defines the
reference frame and supplies a pm PRIOR for matching), neither epoch here has
an independent proper-motion catalog. The frame tie is therefore a full
6-parameter affine fit (multiepoch_pm.affine_tie), not a pm-propagated match
-- a translation-only tie leaves any relative rotation, plate-scale, or shear
difference between the two epochs' astrometric solutions in the residual,
which reads as a spurious COHERENT proper motion across the field.

CAVEAT this shares with affine_tie (see its docstring): a real bulk/rotation/
shear velocity field is ALSO linear to first order across a few arcmin, so
the tie cannot distinguish real coherent motion from a plate-scale/shear
astrometric error and removes both by construction. The output pm_ra/pm_dec
here are RELATIVE to the mean motion + mean shear of the tie's own bright/
compact matching sample, not an absolute frame -- e.g. a rotation curve or
bulk streaming measurement from this catalog will read artificially close to
zero. The fitted per-observation tie coefficients (A, B) are stored in
pm.meta['tie_diag_json'] precisely so this can be checked/undone later.

Example:
    python -m jwst_gc_pipeline.astrometry.build_treasury_pm \\
        --src /orange/.../sgrb2/catalogs/f212n_..._vetted.fits --src-epoch 2024.684 \\
        --ref /orange/.../gc-treasury/catalogs/..._o127.fits \\
              /orange/.../gc-treasury/catalogs/..._o129.fits \\
        --ref-epoch 2026.696 --filter f212n \\
        --out /orange/.../sgrb2/astrometry_diag/pm_flystar/pm_sgrb2_treasury_f212n.fits
"""
import argparse
import json
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord, concatenate, search_around_sky
import astropy.units as u

from . import multiepoch_pm as M


def _col(t, base, filt):
    """Return column name: bare `base` if present, else `base_{filt}`."""
    if base in t.colnames:
        return base
    c = f'{base}_{filt}'
    if c in t.colnames:
        return c
    raise KeyError(f'neither {base!r} nor {c!r} in {t.colnames}')


# merge_catalogs.py computes std_ra/std_dec as the scatter of avg_ra/avg_dec,
# which it builds as SkyCoord(..., unit=(u.deg, u.deg)) -- so these columns
# are in DEGREES, not arcsec (no FITS unit is set on them either, so this has
# to be read out of merge_catalogs.py, not the table). Every JWST-native PM
# script this session assumed arcsec -- including the existing, VIRAC-
# validated cloudef_pm_flystar.py, where it went unnoticed only because
# VIRAC2's much larger position error dominates that error budget and the
# JWST-side term is negligible either way. It is NOT negligible here: both
# epochs are JWST-native, so getting this wrong shrinks the combined
# pm_ra_err/pm_dec_err by ~3600x and makes the "trustworthy" error cut pass
# almost everything regardless of real quality.
DEG_TO_ARCSEC = 3600.0
# merge_catalogs.py's degenerate-position guard sets std_ra/std_dec to exactly
# 0.0 for seeded/copied per-frame positions with no real multi-frame scatter
# to measure; treat that the same as NaN/missing rather than a real zero.
_DEGENERATE_DEG = 0.0


def _load_one(path, filt, epoch, sn_cut, obs_index):
    t = Table.read(path)
    sc_col = _col(t, 'skycoord', filt)
    fl_col = _col(t, 'flux', filt)
    fe_col = _col(t, 'flux_err', filt)
    ex_col = _col(t, 'std_ra', filt)
    ey_col = _col(t, 'std_dec', filt)
    sc = SkyCoord(t[sc_col])
    flux = np.asarray(t[fl_col], float)
    flux_err = np.asarray(t[fe_col], float)
    ex = np.asarray(t[ex_col], float) * DEG_TO_ARCSEC
    ey = np.asarray(t[ey_col], float) * DEG_TO_ARCSEC
    ok = (np.isfinite(sc.ra.deg) & np.isfinite(flux) & (flux > 0) &
          np.isfinite(flux_err) & (flux_err > 0) & (flux / flux_err > sn_cut))
    return dict(sc=sc[ok],
                ex=np.where(np.isfinite(ex[ok]) & (ex[ok] > _DEGENERATE_DEG), ex[ok], 0.005),
                ey=np.where(np.isfinite(ey[ok]) & (ey[ok] > _DEGENERATE_DEG), ey[ok], 0.005),
                flux=flux[ok], mag=-2.5 * np.log10(flux[ok]),
                epoch=epoch, n=int(ok.sum()), obs_index=obs_index)


def load_filter_catalog(paths, filt, epoch, sn_cut=5.0, dedup_radius=0.05):
    """Load one or more per-obs/per-field catalogs, keep one filter's usable
    (S/N > sn_cut) sources, concatenate. Works for both the bare-column
    single-filter vetted catalogs (skycoord/flux/...) and the suffixed
    multi-filter merged catalogs (skycoord_f212n/flux_f212n/...).

    This is only ever called for ``src`` (a single-epoch field), which
    normally means a single path -- but the CLI accepts more than one, and
    if those ever cover overlapping sky (as opposed to ``ref``'s deliberate
    multi-observation concatenation, which is untangled downstream by
    build_pm_catalog_2epoch's same_star_radius), the same star would get one
    row per path here with nothing to merge them back down, inflating
    matched/trustworthy counts with duplicate PM rows for one physical star.
    Source-association dedup here, not a correction -- no reduce derived
    from the match, just which duplicate rows are kept.
    """
    cats = [_load_one(p, filt, epoch, sn_cut, i) for i, p in enumerate(paths)]
    sc = concatenate([c['sc'] for c in cats]) if len(cats) > 1 else cats[0]['sc']
    obs_index = np.concatenate([np.full(c['n'], c['obs_index']) for c in cats])
    ex = np.concatenate([c['ex'] for c in cats])
    ey = np.concatenate([c['ey'] for c in cats])
    flux = np.concatenate([c['flux'] for c in cats])
    mag = np.concatenate([c['mag'] for c in cats])
    if len(cats) > 1:
        keep = _dedup_mask(sc, dedup_radius)
        sc, ex, ey, flux, mag, obs_index = (
            sc[keep], ex[keep], ey[keep], flux[keep], mag[keep], obs_index[keep])
    return dict(sc=sc, ex=ex, ey=ey, flux=flux, mag=mag,
               epoch=epoch, n=int(len(sc)), obs_index=obs_index)


def _dedup_mask(sc, radius):
    """Boolean mask keeping one row per CONNECTED COMPONENT of positions in
    ``sc`` within ``radius`` (arcsec) of each other -- the lowest-indexed
    row of each component.

    A single-nearest-neighbour version of this (each row paired only with
    its own closest other row) can leave two rows of the same 3+-row group
    both marked "not a duplicate of the other": on a near-collinear chain
    A-B-C with ~30 mas steps, A's nearest is B and B's nearest is A (a
    mutual pair, so B is dropped), but C's nearest is B, an ALREADY-DROPPED
    row that never gets checked against A -- C survives even though it is
    well within radius of A transitively through B. Using
    search_around_sky (returns every pair within radius, not just each
    row's single closest) plus connected components -- every row within one
    component's radius chain collapses to a single kept row regardless of
    chain length, not just direct pairs.
    """
    n = len(sc)
    i, j, _, _ = search_around_sky(sc, sc, radius * u.arcsec)
    edges = i != j
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    graph = coo_matrix((np.ones(edges.sum()), (i[edges], j[edges])), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    comp_min_idx = np.full(labels.max() + 1, n, dtype=int)
    np.minimum.at(comp_min_idx, labels, np.arange(n))
    return np.arange(n) == comp_min_idx[labels]


def load_and_tie_ref_catalogs(ref_paths, filt, ref_epoch, src, sn_cut=5.0,
                              tie_magcut=None, tie_bright_percentile=20.0,
                              verbose=True):
    """Load each ref path SEPARATELY and affine-tie it onto src's frame
    BEFORE combining.

    Each JWST "observation" is its own independent visit with its own guide-
    star lock, hence its own small absolute pointing (and possibly
    rotation/scale) offset -- even when, as for GC Treasury o127/o129/o132/
    o135, several observations cover essentially the SAME sky footprint
    (revisits, not separate tiles). Concatenating them first and fitting one
    global tie only removes the AVERAGE of 4 different offsets, leaving each
    observation's individual residual in the data -- which then reads as a
    spatially-incoherent (by whichever observation happened to be the
    nearest-neighbor match at a given position) spurious proper motion.
    Tying each observation independently removes this before it can leak
    into the PM fit.

    ``tie_magcut=None`` (the default) derives an ABSOLUTE cutoff from
    ``src``'s own magnitude distribution (its ``tie_bright_percentile``,
    i.e. the brightest 20% by default) rather than using a fixed literal
    value. affine_tie's own default (magcut=15) assumes a calibrated
    Vega/AB-like scale, as VIRAC/GNS use -- but mag_src/mag here is
    -2.5*log10(flux) of this pipeline's raw instrumental flux, which runs
    roughly -21 to -3 for GC Treasury (found by checking why a coherent
    ~6 mas/yr DEC bias in Cloud c's trustworthy PMs grew monotonically with
    faintness: ``mag < 15`` was true for EVERY star, so the tie's supposedly
    "bright, well-measured" calibration sample was silently the ENTIRE
    matched population, uncurated, for every field measured this way so
    far). An explicit ``tie_magcut`` still overrides this when given.
    """
    if tie_magcut is None:
        tie_magcut = float(np.percentile(src['mag'], tie_bright_percentile))
        if verbose:
            n_cut = int((src['mag'] < tie_magcut).sum())
            print(f'  tie_magcut (bright {tie_bright_percentile:g}% of src): '
                  f'{tie_magcut:.2f} -- {n_cut:,}/{src["n"]:,} src stars pass '
                  f'({100 * n_cut / src["n"]:.1f}%, vs {tie_bright_percentile:g}% '
                  f'intended -- ties are matched to REF separately per observation, '
                  f'so this is src-side selectivity only, not the eventual tie sample size)')
    tied, diags = [], []
    for i, p in enumerate(ref_paths):
        cat = _load_one(p, filt, ref_epoch, sn_cut, i)
        n_ref_cut = int((cat['mag'] < tie_magcut).sum())
        sc_tied, diag = M.affine_tie(cat['sc'], cat['mag'], src['sc'], src['mag'],
                                     magcut=tie_magcut, match_radius=0.3)
        if verbose:
            print(f'  ref[{i}] {p.split("/")[-1]}: {n_ref_cut:,}/{cat["n"]:,} '
                  f'pass the magcut ({100 * n_ref_cut / cat["n"]:.1f}%) -- '
                  f'diag["n_match"]={diag["n_match"]:,} is the count AFTER '
                  f'this cut and the 0.3" match radius both apply, so it is '
                  f'not directly comparable to an uncut n_match without '
                  f'rerunning affine_tie with magcut=inf.')
        cat['sc'] = sc_tied
        tied.append(cat)
        diags.append(diag)
        if verbose:
            print(f'  ref[{i}] {p.split("/")[-1]}: affine tie '
                  f'{diag["n_kept"]}/{diag["n_match"]} kept, '
                  f'median ({diag["med_dx_mas"]:+.1f},{diag["med_dy_mas"]:+.1f}) mas, '
                  f'resid rms {diag["rms_resid_mas"]:.1f} mas')
    sc = concatenate([c['sc'] for c in tied]) if len(tied) > 1 else tied[0]['sc']
    obs_index = np.concatenate([np.full(c['n'], c['obs_index']) for c in tied])
    ref = dict(sc=sc,
              ex=np.concatenate([c['ex'] for c in tied]),
              ey=np.concatenate([c['ey'] for c in tied]),
              flux=np.concatenate([c['flux'] for c in tied]),
              mag=np.concatenate([c['mag'] for c in tied]),
              epoch=ref_epoch, n=int(sum(c['n'] for c in tied)), obs_index=obs_index)
    return ref, diags


def build(src_paths, ref_paths, filt, src_epoch, ref_epoch, out_path,
         match_radius=0.15, tie_magcut=None, isolation_radius=None, verbose=True):
    src = load_filter_catalog(src_paths, filt, src_epoch)
    if verbose:
        print(f'  src ({filt}, epoch {src_epoch:.3f}) usable: {src["n"]:,}')
    ref, diags = load_and_tie_ref_catalogs(ref_paths, filt, ref_epoch, src,
                                           tie_magcut=tie_magcut, verbose=verbose)
    if verbose:
        print(f'  ref ({filt}, epoch {ref_epoch:.3f}) usable: {ref["n"]:,} '
              f'(from {len(ref_paths)} independently-tied observations)')
        # same_star_radius (build_pm_catalog_2epoch's default 0.05" = 50 mas)
        # assumes a tied observation's own residual scatter is well below
        # that, so a re-detection of the SAME star lands within it. A tie
        # whose resid_mas approaches 50 mas can't reliably tell "same star,
        # noisy" from "different, close star" at that radius either way.
        for i, d in enumerate(diags):
            if d['rms_resid_mas'] > 25.0:
                print(f'  WARNING: ref[{i}] tie resid {d["rms_resid_mas"]:.1f} mas is '
                      f'within 2x the same_star_radius (50 mas) build_pm_catalog_2epoch '
                      f'uses to tell a re-detection of the same star from a real close '
                      f'neighbor -- isolation-cut results here are less trustworthy than '
                      f'the trustworthy flag alone suggests.')
    pm = M.build_pm_catalog_2epoch(src, ref, match_radius=match_radius,
                                   isolation_radius=isolation_radius)
    pm.meta['filter'] = filt
    pm.meta['frame'] = ('relative: mean motion + mean shear of the affine tie\'s own '
                        'bright/compact matching sample subtracted per observation; '
                        'NOT an absolute frame -- see build_treasury_pm module docstring')
    # Per-observation fitted tie coefficients (A, B: dx/dy = A0/B0 + A1/B1*x +
    # A2/B2*y, arcsec) plus match/residual diagnostics -- numerically usable
    # (json.loads this), not just a log string, so a later reader can see
    # exactly how much linear motion/shear each tie removed and add it back
    # if needed.
    pm.meta['tie_diag_json'] = json.dumps(diags)
    pm.write(out_path, overwrite=True)
    ntrust = int(pm['trustworthy'].sum())
    if verbose:
        g = pm['trustworthy']
        print(f'  matched {len(pm):,}  trustworthy {ntrust:,}  '
              f'dt {pm.meta["baseline_yr"]:.3f} yr')
        if ntrust > 0:
            print(f'  trustworthy pm_tot median {np.median(pm["pm_tot"][g]):.2f} mas/yr '
                  f'(systematic-tie sanity: should be a few mas/yr for CMZ stars, '
                  f'not tens -- if it is, suspect a frame-tie residual, not real motion)')
    print(f'  wrote {out_path}')
    return pm


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', required=True, nargs='+', help='epoch-0 catalog path(s)')
    ap.add_argument('--src-epoch', type=float, required=True)
    ap.add_argument('--ref', required=True, nargs='+', help='epoch-1 catalog path(s)')
    ap.add_argument('--ref-epoch', type=float, required=True)
    ap.add_argument('--filter', required=True, help='filter, e.g. f212n')
    ap.add_argument('--out', required=True)
    ap.add_argument('--match-radius', type=float, default=0.15)
    ap.add_argument('--isolation-radius', type=float, default=None,
                    help='beam (x3) for the trustworthy isolation check; '
                         'defaults to --match-radius. Widen this for a src '
                         'field with real sub-arcsec structure (e.g. a dense '
                         'cluster core) where match_radius alone is too small '
                         'to catch a blended-but-undetected companion.')
    args = ap.parse_args()
    build(args.src, args.ref, args.filter, args.src_epoch, args.ref_epoch, args.out,
         match_radius=args.match_radius, isolation_radius=args.isolation_radius)


if __name__ == '__main__':
    main()
