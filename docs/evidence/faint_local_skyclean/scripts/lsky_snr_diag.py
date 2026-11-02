"""What are the bright #1019 (local sky-clean) additions?

For each per-frame S/N bin of the sources #1019 adds over the #1015 base:
  * realness (compare.realness) and the nearest-reference separation
    distribution against base-kept stars of the same flux;
  * the nearest BRIGHTER base-kept star: distance in pixels and flux ratio
    (a companion fit in the PSF-mismatch ring of a bright star sits at
    1.5-4.5 px with a small flux ratio);
  * the star-like reason the base vetting would assign (compare.reason);
  * realness split by 'brighter kept star within R px' vs not.

usage: python lsky_snr_diag.py <field> [variant=lsky]
"""
import json
import os
import sys

import numpy as np
import astropy.units as u
from astropy.table import Table
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from compare import FIELDS, realness, sky, reason  # noqa: E402
sys.path.insert(0, os.path.join(HERE, '..', '..', 'faint_m7_seed_union', 'scripts'))
from realness import in_footprint  # noqa: E402

SNR_EDGES = [0, 5, 7, 10, 20, np.inf]
SEP_EDGES = [0, 60, 150, 300, 600, np.inf]   # mas
RING_PX = 5.0


def main(field, v='lsky'):
    cfg = FIELDS[field]
    band = cfg['band']
    base = Table.read(f'{HERE}/out/{field}_{band}_seed.fits')
    var = Table.read(f'{HERE}/out/{field}_{band}_{v}.fits')
    assert np.array_equal(np.asarray(var['rowid']), np.asarray(base['rowid']))
    ref = sky(Table.read(cfg['ref']))
    sc = sky(base)
    kb = np.asarray(base['kept'], bool)
    kv = np.asarray(var['kept'], bool)
    add = kv & ~kb
    infp = in_footprint(sc, ref)
    flux = np.asarray(base['flux'], float)
    snr = flux / np.asarray(base['flux_err'], float)
    qf = np.asarray(base['qfit'], float)
    nm = np.asarray(base['nmatch'], float)
    rs = reason(base)
    _, sep, _ = sc.match_to_catalog_sky(ref)
    sepmas = sep.to_value(u.mas)
    # pixel frame: tangent-plane offsets in units of the band's pixel (0.031" SW, 0.063" LW)
    pix = 0.0311 if band in ('f182m', 'f187n', 'f212n', 'f200w', 'f150w') else 0.0629
    c0 = sc[0]
    off = sc.transform_to(c0.skyoffset_frame())
    xy = np.c_[off.lon.to_value(u.arcsec), off.lat.to_value(u.arcsec)] / pix
    ik = np.flatnonzero(kb)
    tree = cKDTree(xy[ik])
    # nearest brighter kept star within 20 px
    d_b = np.full(len(base), np.inf)
    r_b = np.full(len(base), np.nan)
    # added sources and base-kept sources (control: are real close companions
    # in the reference catalog at all?)
    ia = np.flatnonzero(add | kb)
    dd, jj = tree.query(xy[ia], k=16, distance_upper_bound=20)
    for n, i in enumerate(ia):
        for d, j in zip(dd[n], jj[n]):
            if not np.isfinite(d):
                break
            k = ik[j]
            if k != i and flux[k] > flux[i]:
                d_b[i], r_b[i] = d, flux[i] / flux[k]
                break
    bsel = kb & infp
    out = {}
    print(f'== {field} {v}: {int(add.sum())} added, {int((add & infp).sum())} in footprint')
    for lo, hi in zip(SNR_EDGES[:-1], SNR_EDGES[1:]):
        k = add & infp & (snr >= lo) & (snr < hi)
        if not k.any():
            continue
        r = realness(sc[k], flux[k], ref, sc[bsel], flux[bsel])
        ring = k & (d_b <= RING_PX)
        far = k & ~(d_b <= RING_PX)
        rr = realness(sc[ring], flux[ring], ref, sc[bsel], flux[bsel]) if ring.any() else dict(n=0)
        rf = realness(sc[far], flux[far], ref, sc[bsel], flux[bsel]) if far.any() else dict(n=0)
        # control: base-kept stars of the same S/N with a brighter kept star within RING_PX
        kc = bsel & (snr >= lo) & (snr < hi) & (d_b <= RING_PX)
        rc = realness(sc[kc], flux[kc], ref, sc[bsel], flux[bsel]) if kc.any() else dict(n=0)
        kcf = bsel & (snr >= lo) & (snr < hi) & ~(d_b <= RING_PX)
        rcf = realness(sc[kcf], flux[kcf], ref, sc[bsel], flux[bsel]) if kcf.any() else dict(n=0)
        # flux-matched base-kept separation distribution (same log-flux bins, 3000 per bin max)
        hs = np.histogram(sepmas[k], SEP_EDGES)[0] / k.sum()
        lf = np.log10(np.clip(flux, 1e-30, None))
        fe = np.arange(0, 7.01, 0.25)
        idx = []
        rng = np.random.default_rng(0)
        for a, b in zip(fe[:-1], fe[1:]):
            na = int((k & (lf >= a) & (lf < b)).sum())
            if na == 0:
                continue
            pool = np.flatnonzero(bsel & (lf >= a) & (lf < b))
            if pool.size:
                idx.append(rng.choice(pool, na * 5, replace=True))
        idx = np.concatenate(idx) if idx else np.array([], int)
        hb = np.histogram(sepmas[idx], SEP_EDGES)[0] / max(idx.size, 1)
        reasons = {s: int((k & (rs == s)).sum()) for s in np.unique(rs[k])}
        row = dict(lo=lo, hi=hi, n=int(k.sum()), rel=r.get('rel'),
                   sep_frac_added=hs.tolist(), sep_frac_basekept=hb.tolist(),
                   frac_ring=float(ring.sum() / k.sum()), n_ring=int(ring.sum()), rel_ring=rr.get('rel'),
                   n_far=int(far.sum()), rel_far=rf.get('rel'),
                   n_basekept_ring=int(kc.sum()), rel_basekept_ring=rc.get('rel'),
                   n_basekept_far=int(kcf.sum()), rel_basekept_far=rcf.get('rel'),
                   ring_dist_px_median=float(np.median(d_b[ring])) if ring.any() else None,
                   ring_flux_ratio_median=float(np.nanmedian(r_b[ring])) if ring.any() else None,
                   qfit_median=float(np.nanmedian(qf[k])), nmatch_median=float(np.median(nm[k])),
                   reasons=reasons)
        out[f'{lo:g}-{hi:g}'] = row
        print(f'S/N [{lo:g},{hi:g}) n {row["n"]:5d} rel {r.get("rel", np.nan):.2f} | '
              f'brighter kept star <= {RING_PX:g} px: {row["frac_ring"]:.0%} (n {row["n_ring"]}, rel {rr.get("rel", np.nan):.2f}, '
              f'median {row["ring_dist_px_median"] or np.nan:.1f} px, flux ratio {row["ring_flux_ratio_median"] or np.nan:.3f}) '
              f'| beyond: n {row["n_far"]} rel {rf.get("rel", np.nan):.2f} | qfit {row["qfit_median"]:.2f} nmatch {row["nmatch_median"]:.0f}')
        print('   sep to nearest ref (mas) ' + ' '.join(f'[{a:g},{b:g}) {x:.2f}/{y:.2f}' for a, b, x, y in
                                                     zip(SEP_EDGES[:-1], SEP_EDGES[1:], hs, hb)) + '  (added/base-kept)')
        print(f'   control base-kept same S/N: ring n {row["n_basekept_ring"]} rel {rc.get("rel", np.nan):.2f} | '
              f'beyond n {row["n_basekept_far"]} rel {rcf.get("rel", np.nan):.2f}')
        print(f'   reasons {reasons}')
    # realness by distance to the nearest brighter kept star, added vs
    # base-kept of the same S/N range: does the S/N gradient follow the
    # environment (additions sit nearer bright stars) or the added sources?
    DIST_EDGES = [0, 3, 5, 8, 12, 20, np.inf]
    out['by_neighbour_distance'] = {}
    for lo, hi in ((0, 5), (5, 10), (10, np.inf)):
        rows = []
        print(f'-- S/N [{lo:g},{hi:g}) by distance (px) to nearest brighter kept star: added | base-kept')
        for a, b in zip(DIST_EDGES[:-1], DIST_EDGES[1:]):
            # last bin: no brighter kept star among the 16 nearest within 20 px (d_b = inf)
            sel = infp & (snr >= lo) & (snr < hi) & (d_b >= a) & ((d_b < b) | ~np.isfinite(b))
            ka, kc = sel & add, sel & kb
            ra = realness(sc[ka], flux[ka], ref, sc[bsel], flux[bsel]) if ka.sum() > 10 else dict(n=int(ka.sum()))
            rc = realness(sc[kc], flux[kc], ref, sc[bsel], flux[bsel]) if kc.sum() > 10 else dict(n=int(kc.sum()))
            rows.append(dict(d_lo=a, d_hi=b, n_added=int(ka.sum()), rel_added=ra.get('rel'),
                             n_basekept=int(kc.sum()), rel_basekept=rc.get('rel')))
            print(f'   [{a:g},{b:g}) added n {int(ka.sum()):5d} rel {ra.get("rel", np.nan):5.2f} | '
                  f'base-kept n {int(kc.sum()):6d} rel {rc.get("rel", np.nan):5.2f}')
        out['by_neighbour_distance'][f'{lo:g}-{hi:g}'] = rows
    with open(f'{HERE}/lsky_snr_diag_{field}.json', 'w') as fh:
        json.dump(out, fh, indent=1, default=float)


if __name__ == '__main__':
    main(*sys.argv[1:])
