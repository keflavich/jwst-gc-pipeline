"""
Driver: build a JWST x GNS x VIRAC2 proper-motion catalog for one JWST m7 field.

Example:
    python -m jwst_gc_pipeline.astrometry.build_multiepoch_pm \
        --m7 /orange/.../catalogs/basic_..._m7_o023.fits \
        --virac /orange/.../refcat/virac2_gc2211.fits \
        --gns   /orange/.../refcat/gns_gc2211.fits \
        --out   /orange/.../catalogs/pm_o023_jwst_gns_virac.fits
"""
import argparse
import numpy as np
from . import multiepoch_pm as M


def build_for_field(m7_path, virac_path, gns_path, out_path,
                    ref_filter='f200w', match_radius=0.12, require_jwst=True,
                    verbose=True):
    jwst = M.load_jwst_m7(m7_path, ref_filter=ref_filter)
    virac = M.load_ref(virac_path, 'virac')
    gns = M.load_ref(gns_path, 'gns')
    # restrict the (large, region-wide) reference catalogs to this field's footprint
    virac = M.restrict(virac, jwst['sc'])
    gns = M.restrict(gns, jwst['sc'])
    if verbose:
        print(f"  JWST n={jwst['n']} (epoch {jwst['epoch']:.2f})  "
              f"VIRAC2 n={virac['n']}  GNS n={gns['n']}", flush=True)
    # put GNS + JWST on the VIRAC2/Gaia frame
    gsc, gd = M.shift_to_virac_frame(gns, virac, M.EPOCH_GNS)
    gns['sc'] = gsc
    jsc, jd = M.shift_to_virac_frame(jwst, virac, jwst['epoch'])
    jwst['sc'] = jsc
    if verbose:
        print(f"  GNS->VIRAC: {gd['n_kept']} stars, dDec={gd['med_dy_mas']:.1f} mas, "
              f"resid={gd['rms_resid_mas']:.1f} mas", flush=True)
        print(f"  JWST->VIRAC: {jd['n_kept']} stars, dDec={jd['med_dy_mas']:.1f} mas, "
              f"resid={jd['rms_resid_mas']:.1f} mas", flush=True)
    pm = M.build_pm_catalog(jwst, virac, gns, match_radius=match_radius,
                            require_jwst=require_jwst)
    pm.meta['m7_catalog'] = m7_path
    pm.meta['ref_filter'] = ref_filter
    pm.write(out_path, overwrite=True)
    g = np.isfinite(pm['pm_ra']) & np.isfinite(pm['pm_dec'])
    if verbose:
        n3 = int(((pm['n_epoch'] == 3) & g).sum())
        print(f"  PM: {int(g.sum())} stars ({n3} 3-epoch); "
              f"pm_tot med={np.nanmedian(pm['pm_tot'][g]):.1f} mas/yr -> {out_path}",
              flush=True)
    return pm


def _validate_vs_virac(pm, virac_path):
    """Quick sanity: our PM vs VIRAC2's own pm for tightly-matched stars."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    vir = M.load_ref(virac_path, 'virac')
    g = np.isfinite(pm['pm_ra'])
    idx, sep, _ = SkyCoord(pm['ra0'] * u.deg, pm['dec0'] * u.deg).match_to_catalog_sky(vir['sc'])
    ok = (sep < 0.08 * u.arcsec) & g
    vra, vde = vir['pmra'][idx][ok], vir['pmde'][idx][ok]
    m = np.isfinite(vra)
    dra = pm['pm_ra'][ok][m] - vra[m]
    dde = pm['pm_dec'][ok][m] - vde[m]
    rstd = lambda d: float(np.percentile(np.abs(d - np.median(d)), 68) * 1.48)
    return dict(n=int(m.sum()), med_dra=float(np.median(dra)), std_dra=rstd(dra),
                med_dde=float(np.median(dde)), std_dde=rstd(dde))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--m7', required=True, help='JWST per-obs m7 cross-band catalog')
    ap.add_argument('--virac', required=True)
    ap.add_argument('--gns', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--ref-filter', default='f200w')
    ap.add_argument('--match-radius', type=float, default=0.12)
    ap.add_argument('--validate', action='store_true',
                    help='print PM-vs-VIRAC2 sanity stats')
    args = ap.parse_args()
    pm = build_for_field(args.m7, args.virac, args.gns, args.out,
                         ref_filter=args.ref_filter, match_radius=args.match_radius)
    if args.validate:
        print("  validate vs VIRAC2:", _validate_vs_virac(pm, args.virac), flush=True)


if __name__ == '__main__':
    main()
