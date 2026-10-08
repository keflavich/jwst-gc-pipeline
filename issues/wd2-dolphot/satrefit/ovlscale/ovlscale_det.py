"""Per-detector split of the satstar (and daophot) fluxes against dolphot.
Usage: nice -19 python -u ovlscale_det.py   (reads the ovlscale caches and matched_Q_main2.fits; writes ovlscale_det_results.json,
ovlscale_det_tables.md, ovlscale_det.png in this directory).
dm_row = m_row + c_b - m_dolphot - ZP_b.  ZP_b is analyze.py's ZP for main2 (same function, read-only import).  c_b converts the
AB magnitude computed here from a per-frame flux_fit to the magnitude system of the merged catalog: median over replaced stars of
(our_<band> - m_satstar_median) (includes the constant +0.007/+0.003 and any Vega/AB offset)."""
import sys
import json
import numpy as np
from astropy.table import Table
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
sys.argv = sys.argv[:1]
import analyze as an
import ovlscale as o

OUT = o.OUT
BANDS = ['F150W', 'F162M', 'F182M', 'F200W', 'F250M', 'F277W', 'F300M']
REFDET = {'F': 'nrcb1'}


def mad(v):
    return o.mad(v)


def main():
    an.ZPWIN.update(an.zp_windows())
    A = an.Arm('main2')
    m = A.m
    mra = np.asarray(m['RA'], float)
    mdec = np.asarray(m['DEC'], float)
    res = {}
    L = []
    fig, axs = plt.subplots(2, 4, figsize=(19, 8), sharey=True)
    axs = axs.ravel()
    dcol = {}
    for bi, band in enumerate(BANDS):
        bb = band[1:]
        S, D, pixsr = o.load_band(band)
        ra0, dec0 = np.median(D['ra']), np.median(D['dec'])
        sxy = o.sky_xy(S['ra'], S['dec'], ra0, dec0)
        lab = o.fof(sxy, o.RADIUS)
        nst = lab.max() + 1
        cen = np.array([np.median(sxy[lab == k], axis=0) for k in range(nst)])
        fmed = np.array([np.nanmedian(S['flux'][lab == k]) for k in range(nst)])
        ref = A.ref[bb]
        ok = A.matched & np.isfinite(ref) & np.isfinite(A.our[bb])
        mxy = o.sky_xy(mra, mdec, ra0, dec0)
        tree = cKDTree(mxy[ok])
        okidx = np.where(ok)[0]
        dist, j = tree.query(cen)
        good = dist < 0.1
        star_m = okidx[j]
        # calibration constant c_b on replaced stars
        rep = A.rep[bb][star_m] & good
        mS_star = o.mag(fmed, pixsr)
        cb = float(np.nanmedian(A.our[bb][star_m][rep] - mS_star[rep])) if rep.sum() else np.nan
        cb_sd = float(mad(A.our[bb][star_m][rep] - mS_star[rep])) if rep.sum() else np.nan
        zp = A.zp[bb]
        # per S row
        rowstar = lab
        rgood = good[rowstar]
        rref = ref[star_m[rowstar]]
        out = {}
        for kk, key in (('final', 'flux'), ('precap', 'precap'), ('raw', 'raw')):
            out[kk] = o.mag(S[key], pixsr) + cb - rref - zp
        det = S['det']
        sel = rgood & np.isfinite(out['final']) & np.isfinite(rref)
        dets = sorted(set(det[sel].tolist()))
        r = dict(cb=cb, cb_scatter_MAD=cb_sd, zp=float(zp), zpwin=A and an.ZPWIN.get(bb), n_rows=int(sel.sum()),
                 n_stars=int(len(np.unique(rowstar[sel]))), n_stars_total=int(nst))
        r['overall'] = {kk: (float(np.median(out[kk][sel])), float(mad(out[kk][sel]))) for kk in out}
        # star-level (median over rows) check against analyze
        r['per_det'] = {}
        for d in dets:
            s = sel & (det == d)
            r['per_det'][d] = dict(n_rows=int(s.sum()), n_stars=int(len(np.unique(rowstar[s]))),
                                   **{kk: (float(np.median(out[kk][s])), float(mad(out[kk][s]))) for kk in out},
                                   median_dolphot_mag=float(np.median(rref[s])),
                                   cap_frac=float(np.mean(S['raw'][s] < 0.999 * S['precap'][s])),
                                   cap_dmag_med=float(np.median(o.mag(S['raw'][s], pixsr) - o.mag(S['precap'][s], pixsr))),
                                   wing_med=float(np.median(S['wing'][s])), wing_n1=float(np.mean(S['wing'][s] == 1.0)),
                                   area_med=float(np.median(S['area'][s])),
                                   final_over_raw_mag=float(np.median(o.mag(S['flux'][s], pixsr) - o.mag(S['raw'][s], pixsr))))
        # magnitude bins (1 mag)
        lo, hi = np.floor(np.nanmin(rref[sel])), np.ceil(np.nanmax(rref[sel]))
        edges = np.arange(lo, hi + 1, 1.0)
        r['bins'] = {}
        for d in dets:
            for e0, e1 in zip(edges[:-1], edges[1:]):
                s = sel & (det == d) & (rref >= e0) & (rref < e1)
                if s.sum() >= 5:
                    r['bins'].setdefault(d, []).append([float(e0), float(e1), int(s.sum())] +
                                                      [x for kk in ('final', 'precap', 'raw')
                                                       for x in (float(np.median(out[kk][s])), float(mad(out[kk][s])))])
        # equalised medians
        refd = 'nrcb1' if 'nrcb1' in dets else ('nrcblong' if 'nrcblong' in dets else dets[0])
        offs = {d: r['per_det'][d]['final'][0] for d in dets}
        eq = out['final'][sel] - np.array([offs[d] - offs[refd] for d in det[sel]])
        eq0 = out['final'][sel] - np.array([offs[d] for d in det[sel]])
        r['equalised'] = dict(ref_det=refd, median_to_ref=float(np.median(eq)), median_to_zero=float(np.median(eq0)),
                              overall=float(np.median(out['final'][sel])),
                              spread_det_offsets=float(np.std(list(offs.values()))))
        # rows weights: per-star median dm star-level
        # control: daophot rows vs dolphot for unsaturated stars in the ZP window
        lo_z, hi_z = an.ZPWIN.get(bb, (0, 19))
        dxy = o.sky_xy(D['ra'], D['dec'], ra0, dec0)
        dd, dj = tree.query(dxy)
        dgood = (dd < 0.1)
        sm = okidx[dj]
        dref = ref[sm]
        unsat_star = ~A.rep[bb][sm] & ~A.sat[bb][sm]
        dsel = (dgood & (D['flags'] == 0) & ~D['forced'] & ~D['sat5'] & np.isfinite(D['flux']) & (D['flux'] > 0) & unsat_star
                & (dref >= lo_z) & (dref < hi_z))
        dmD = o.mag(D['flux'], pixsr) + cb - dref - zp
        r['control'] = {}
        for d in sorted(set(D['det'][dsel].tolist())):
            s = dsel & (D['det'] == d)
            r['control'][d] = (int(s.sum()), float(np.median(dmD[s])), float(mad(dmD[s])))
        r['control_all'] = (int(dsel.sum()), float(np.median(dmD[dsel])), float(mad(dmD[dsel])))
        # control restricted to the satstar magnitude range (dolphot mag within the S-row range of this band)
        res[band] = r
        print(band, r['n_rows'], r['n_stars'], 'cb', cb, 'zp', zp, r['overall'], flush=True)
        # figure
        ax = axs[bi]
        cm = plt.get_cmap('tab10')
        for ci, d in enumerate(dets):
            s = sel & (det == d)
            ax.plot(rref[s], out['final'][s], '.', ms=2, color=cm(ci), alpha=0.25)
            for bn in r['bins'].get(d, []):
                pass
            if d in r['bins']:
                xs = [(b_[0] + b_[1]) / 2 for b_ in r['bins'][d]]
                ax.plot(xs, [b_[3] for b_ in r['bins'][d]], '-o', color=cm(ci), ms=5, lw=1.5, label=f"{d} ({r['per_det'][d]['n_rows']})")
        ax.axhline(0, color='gray', lw=0.7)
        ax.set_ylim(-0.4, 0.4)
        ax.set_title(f'{band} (dm_row final, ZP {zp:+.3f})')
        ax.set_xlabel('dolphot mag')
        ax.legend(fontsize=6, ncol=2)
        if bi in (0, 4):
            ax.set_ylabel('m_S,row - m_dolphot - ZP')
    axs[-1].axis('off')
    fig.tight_layout()
    fig.savefig(f'{OUT}/ovlscale_det.png', dpi=110)
    json.dump(res, open(f'{OUT}/ovlscale_det_results.json', 'w'), indent=1, default=lambda x: x.item() if hasattr(x, 'item') else str(x))
    f = lambda x: f'{x:+.3f}'
    for band, r in res.items():
        L.append(f"\n#### {band}: ZP {r['zp']:+.3f}, c_b {r['cb']:+.4f} (MAD {r['cb_scatter_MAD']:.3f}); {r['n_rows']} S rows of {r['n_stars']} dolphot-matched stars ({r['n_stars_total']} S clusters)")
        L.append(f"Overall median dm_row final/precap/raw: " + ' / '.join(f"{f(r['overall'][k][0])} (MAD {r['overall'][k][1]:.3f})" for k in ('final', 'precap', 'raw')))
        e = r['equalised']
        L.append(f"Equalised to {e['ref_det']}: {f(e['median_to_ref'])}; equalised to zero per detector: {f(e['median_to_zero'])}; spread (sd) of detector offsets {e['spread_det_offsets']:.3f}")
        L.append('\n| detector | N rows | N stars | median dolphot mag | final | precap | raw | cap binds (frac) | cap dmag | wing ratio med | final-raw (mag) | sat_area med |')
        L.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
        for d, q in r['per_det'].items():
            L.append(f"| {d} | {q['n_rows']} | {q['n_stars']} | {q['median_dolphot_mag']:.1f} | {f(q['final'][0])} ({q['final'][1]:.3f}) | {f(q['precap'][0])} | {f(q['raw'][0])} | "
                     f"{q['cap_frac']:.2f} | {f(q['cap_dmag_med'])} | {q['wing_med']:.3f} | {f(q['final_over_raw_mag'])} | {q['area_med']:.0f} |")
        L.append('\nBins (dolphot mag, 1 mag): median (MAD) [N] final | precap | raw\n')
        L.append('| detector | bin | N | final | precap | raw |')
        L.append('|---|---|---|---|---|---|')
        for d, bl in r['bins'].items():
            for b_ in bl:
                L.append(f"| {d} | {b_[0]:.0f}-{b_[1]:.0f} | {b_[2]} | {f(b_[3])} ({b_[4]:.3f}) | {f(b_[5])} ({b_[6]:.3f}) | {f(b_[7])} ({b_[8]:.3f}) |")
        L.append(f"\nControl, daophot rows vs dolphot (unreplaced unsaturated stars in the ZP window, dolphot mag {an.ZPWIN.get(band[1:], (0, 19))[0]:.1f}-{an.ZPWIN.get(band[1:], (0, 19))[1]:.1f}; flags 0, no DQ saturation): all {f(r['control_all'][1])} (N {r['control_all'][0]}); " +
                 ', '.join(f"{d} {f(v[1])} [{v[0]}]" for d, v in r['control'].items()))
    open(f'{OUT}/ovlscale_det_tables.md', 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
