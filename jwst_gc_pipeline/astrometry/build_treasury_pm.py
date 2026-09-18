"""
Driver: JWST-epoch x JWST-epoch proper motions via flystar, for a field whose
only long-baseline counterpart is another JWST program rather than a ground
catalog (VIRAC2/GNS) -- e.g. Sgr B2 (program 5365) x GC Treasury (program
10678), or Cloud e/f (program 2092) x GC Treasury, once GC Treasury source
photometry exists over that footprint.

Unlike build_multiepoch_pm.py (JWST x GNS x VIRAC2, VIRAC2 defines the
reference frame and supplies a pm PRIOR for matching), neither epoch here has
an independent proper-motion catalog. The frame tie is therefore an affine
fit (offset + rotation/plate-scale; multiepoch_pm.affine_tie), not a
pm-propagated match -- a translation-only tie leaves any relative rotation or
plate-scale difference between the two epochs' astrometric solutions in the
residual, which reads as a spurious COHERENT proper motion across the field.

Example:
    python -m jwst_gc_pipeline.astrometry.build_treasury_pm \\
        --src /orange/.../sgrb2/catalogs/f212n_..._vetted.fits --src-epoch 2024.684 \\
        --ref /orange/.../gc-treasury/catalogs/..._o127.fits \\
              /orange/.../gc-treasury/catalogs/..._o129.fits \\
        --ref-epoch 2026.696 --filter f212n \\
        --out /orange/.../sgrb2/astrometry_diag/pm_flystar/pm_sgrb2_treasury_f212n.fits
"""
import argparse
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord, concatenate
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


def load_filter_catalog(paths, filt, epoch, sn_cut=5.0):
    """Load one or more per-obs/per-field catalogs, keep one filter's usable
    (S/N > sn_cut) sources, concatenate. Works for both the bare-column
    single-filter vetted catalogs (skycoord/flux/...) and the suffixed
    multi-filter merged catalogs (skycoord_f212n/flux_f212n/...)."""
    cats = [_load_one(p, filt, epoch, sn_cut, i) for i, p in enumerate(paths)]
    sc = concatenate([c['sc'] for c in cats]) if len(cats) > 1 else cats[0]['sc']
    obs_index = np.concatenate([np.full(c['n'], c['obs_index']) for c in cats])
    return dict(sc=sc,
                ex=np.concatenate([c['ex'] for c in cats]),
                ey=np.concatenate([c['ey'] for c in cats]),
                flux=np.concatenate([c['flux'] for c in cats]),
                mag=np.concatenate([c['mag'] for c in cats]),
                epoch=epoch, n=int(sum(c['n'] for c in cats)), obs_index=obs_index)


def load_and_tie_ref_catalogs(ref_paths, filt, ref_epoch, src, sn_cut=5.0,
                              tie_magcut=15.0, verbose=True):
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
    """
    tied, diags = [], []
    for i, p in enumerate(ref_paths):
        cat = _load_one(p, filt, ref_epoch, sn_cut, i)
        sc_tied, diag = M.affine_tie(cat['sc'], cat['mag'], src['sc'], src['mag'],
                                     magcut=tie_magcut, match_radius=0.3)
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
         match_radius=0.15, tie_magcut=15.0, verbose=True):
    src = load_filter_catalog(src_paths, filt, src_epoch)
    if verbose:
        print(f'  src ({filt}, epoch {src_epoch:.3f}) usable: {src["n"]:,}')
    ref, diags = load_and_tie_ref_catalogs(ref_paths, filt, ref_epoch, src,
                                           tie_magcut=tie_magcut, verbose=verbose)
    if verbose:
        print(f'  ref ({filt}, epoch {ref_epoch:.3f}) usable: {ref["n"]:,} '
              f'(from {len(ref_paths)} independently-tied observations)')
    pm = M.build_pm_catalog_2epoch(src, ref, match_radius=match_radius)
    pm.meta['filter'] = filt
    pm.meta['tie_diag'] = str(diags)
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
    args = ap.parse_args()
    build(args.src, args.ref, args.filter, args.src_epoch, args.ref_epoch, args.out,
         match_radius=args.match_radius)


if __name__ == '__main__':
    main()
