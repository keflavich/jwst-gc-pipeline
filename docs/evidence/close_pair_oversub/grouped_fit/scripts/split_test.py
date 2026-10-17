"""Do grouped m7 fits split single stars into two catalogue entries?

Injected stars are single by construction.  For every injected star inside the
inner box of injection runs 1-10 (primary band), counts the m7 vetted sources
within R_SPLIT FWHM of the injected position in that run, minus the sources
within the same radius in the clean run (seed 0) of the same variant: the
field's own stars there.  A recovered single star gives +1; +2 or more means
the injected star is carried by two or more entries (split), or its fit moved
or added a neighbour.  Restricting to "isolated" injections (no clean-run
source within ISO FWHM) removes the field-star bookkeeping.

usage: python split_test.py <variantA> <variantB> <out-prefix>
"""
import json
import sys

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from scipy.spatial import cKDTree

from jwst_gc_pipeline.photometry import reference_fields as RF
from jwst_gc_pipeline.photometry.reference_fields import evaluate as EV

R_SPLIT = 1.5
ISO = 2.5


def cat_xy(spec, variant, seed, filt):
    prod = EV.find_products(RF.run_dir(spec, variant, seed), filt, allow_missing_bg=True)
    img, _, wcs, _ = EV._image(prod['data'])
    fr = EV.Frame(spec, wcs, img.shape)
    sc, flux, _ = EV.load_catalog(prod['catalog'])
    x, y = fr.xy(sc)
    ok = np.isfinite(x) & np.isfinite(y)
    return fr, np.c_[x[ok], y[ok]], flux[ok]


def run(va, vb, outp):
    _, fields = RF.load_config()
    out = {}
    lines = []
    for name, spec in fields.items():
        prim = spec['filters'][0]
        fw = EV.fwhm_pix(prim)
        for v in (va, vb):
            fr0, xy0, _ = cat_xy(spec, v, 0, prim)
            t0 = cKDTree(xy0)
            rows = []
            for seed in range(1, 11):
                inj = Table.read(RF.injection_table_path(name, seed))
                fr, xy, flux = cat_xy(spec, v, seed, prim)
                isc = SkyCoord(np.asarray(inj['ra']) * u.deg, np.asarray(inj['dec']) * u.deg)
                ix, iy = fr.xy(isc)
                keep = fr.inside(ix, iy)
                p = np.c_[ix[keep], iy[keep]]
                snr = np.asarray(inj[f'snr_true_{prim}'], float)[keep]
                t = cKDTree(xy)
                n_run = np.array([len(l) for l in t.query_ball_point(p, R_SPLIT * fw)])
                n_cln = np.array([len(l) for l in t0.query_ball_point(p, R_SPLIT * fw)])
                iso = np.array([len(l) == 0 for l in t0.query_ball_point(p, ISO * fw)])
                # the two nearest run sources to the injected position
                d, j = t.query(p, k=2)
                for k in range(len(p)):
                    rows.append(dict(seed=seed, snr=float(snr[k]), n_run=int(n_run[k]), n_clean=int(n_cln[k]),
                                     iso=bool(iso[k]), d1=float(d[k, 0] / fw), d2=float(d[k, 1] / fw)))
            ex = np.array([r['n_run'] - r['n_clean'] for r in rows])
            iso = np.array([r['iso'] for r in rows])
            snr = np.array([r['snr'] for r in rows])
            n1 = np.array([r['n_run'] for r in rows])
            res = dict(n=len(rows), n_iso=int(iso.sum()),
                       excess_hist={int(k): int(np.sum(ex == k)) for k in np.unique(ex)},
                       iso_nrun_hist={int(k): int(np.sum(n1[iso] == k)) for k in np.unique(n1[iso])},
                       frac_excess_ge2=float(np.mean(ex >= 2)),
                       iso_frac_split=float(np.mean(n1[iso] >= 2)) if iso.any() else None,
                       bright_iso_frac_split=float(np.mean(n1[iso & (snr >= 40)] >= 2))
                       if (iso & (snr >= 40)).any() else None,
                       n_bright_iso=int(np.sum(iso & (snr >= 40))),
                       rows=rows)
            out[f'{name}_{v}'] = res
            line = (f'{name:13s} {prim} {v}: n_inj={res["n"]} excess(run-clean within {R_SPLIT} FWHM) '
                    f'hist={res["excess_hist"]}  >=2: {res["frac_excess_ge2"]:.3f} | isolated n={res["n_iso"]} '
                    f'n_run hist={res["iso_nrun_hist"]} split={res["iso_frac_split"]} '
                    f'(S/N>=40: {res["bright_iso_frac_split"]} of {res["n_bright_iso"]})')
            print(line, flush=True)
            lines.append(line)
    json.dump(out, open(outp + '.json', 'w'), indent=1)
    open(outp + '.txt', 'w').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    run(*sys.argv[1:4])
