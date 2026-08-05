"""Coherent-distortion-floor diagnostic: CRDS vs affine-anchored STDGDC.

The other GDC experiment metrics (per-star consensus scatter, frame-pair
offsets, VIRAC/Gaia bulk, Hosek median separation) are all dominated by the
~1-1.5 mas per-exposure centroid noise of these fields.  A distortion swap
changes a *coherent, position-dependent* term of only ~0.1-0.3 mas, which adds
in quadrature under that noise and is invisible to those metrics
(sqrt(1.02^2 + 0.11^2) = 1.006 vs sqrt(0.99^2 + 0.05^2) = 0.994 -> < 0.04 mas,
exactly the "unchanged" they report).

This diagnostic isolates the coherent term the way the pre-treasury report 09
does: match the same bright, isolated star across the dither set, tie each
frame to the running mean with a per-frame 6-parameter linear transform (which
removes pointing + scale + rotation but NOT distortion), then BIN the residual
by detector position.  Binning averages the centroid noise down while the
static distortion pattern survives, so a flatter distortion solution leaves a
smaller binned floor.  Both solutions are measured on the identical star set:
CRDS = the catalog's ``skycoord_centroid``; GDC = ``GDCSkySolution`` on the same
``x_fit``/``y_fit``.

Reproduces report 09 through this package's own machinery: on arches F212N
NRCA4, CRDS binned floor 0.113 mas (worst cell 0.275) -> GDC 0.051 (worst
0.150), ~2x flatter, while per-detection scatter is unchanged (1.02 -> 0.99).

CLI::

    python -m jwst_gc_pipeline.astrometry_gdc.distortion_floor_diagnostic \
        --catalog-glob '.../f212n_nrca4_*_m1_daophot_basic.fits' \
        --crf-glob    '.../jw*_nrca4_destreak_o001_crf.fits' \
        [--out floor_results.json] [--figure floor.png]

Catalogs and crf frames are paired by the ``NNNNN`` exposure token in the file
name.  Astrometry-rules compliant: same-star matched-pair residuals only, no
dense-NN-median against any reference.
"""
import argparse
import glob
import json
import re

import numpy as np
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
from astropy import units as u
from scipy.spatial import cKDTree

from .gdc_wcs import gdc_sky

__all__ = ['run_diagnostic']

_EXP_RE = re.compile(r'(\d{5})')


def _exp_token(path):
    """The 5-digit exposure token used to pair a catalog with its crf frame."""
    m = _EXP_RE.findall(path.split('/')[-1])
    if not m:
        raise ValueError(f"no 5-digit exposure token in {path}")
    # for jw<...>_<vgroup>_<NNNNN>_<det> the exposure is the last such run in
    # the visit block; the m1 catalog name carries it as exp<NNNNN>.
    if 'exp' in path:
        em = re.search(r'exp(\d{5})', path)
        if em:
            return em.group(1)
    return m[-1]


def _load(catalog_glob, crf_glob, qfit_max=0.1, snr_min=10.0):
    cats = sorted(g for g in glob.glob(catalog_glob) if '_group_' not in g)
    crfs = {_exp_token(c): c for c in glob.glob(crf_glob)}
    if not cats:
        raise ValueError(f"no catalogs matched {catalog_glob}")

    dets = []
    for fi, cf in enumerate(cats):
        tok = _exp_token(cf)
        if tok not in crfs:
            raise ValueError(f"no crf frame for exposure {tok} ({cf})")
        t = Table.read(cf)
        x = np.asarray(t['x_fit'], float)
        y = np.asarray(t['y_fit'], float)
        flux = np.asarray(t['flux_fit'], float)
        ferr = np.asarray(t['flux_err'], float)
        qfit = np.asarray(t['qfit'], float)
        sc = SkyCoord(t['skycoord_centroid'])
        ra_c, dec_c = sc.ra.deg, sc.dec.deg
        with np.errstate(divide='ignore', invalid='ignore'):
            snr = flux / ferr
        good = (np.isfinite(x) & np.isfinite(y) & np.isfinite(flux)
                & (flux > 0) & np.isfinite(ferr) & (ferr > 0)
                & np.isfinite(ra_c) & (qfit <= qfit_max) & (snr >= snr_min))
        x, y, flux = x[good], y[good], flux[good]
        ra_c, dec_c = ra_c[good], dec_c[good]

        sky_gdc, sol = gdc_sky(x, y, crfs[tok])
        d = Table()
        d['frame'] = np.full(x.size, fi)
        d['x'] = x
        d['y'] = y
        d['instr'] = -2.5 * np.log10(flux)
        d['ra_crds'] = ra_c
        d['dec_crds'] = dec_c
        d['ra_gdc'] = sky_gdc.icrs.ra.deg
        d['dec_gdc'] = sky_gdc.icrs.dec.deg
        dets.append(d)
        print(f'  frame {fi} exp{tok}: {x.size} kept  '
              f'affine_rms={sol.affine_rms_mas:.2f} mas')
    return vstack(dets)


def _tangent(cen, ra, dec):
    sc = SkyCoord(np.asarray(ra) * u.deg, np.asarray(dec) * u.deg)
    dlon, dlat = cen.spherical_offsets_to(sc)
    return dlon.to_value(u.arcsec) * 1000.0, dlat.to_value(u.arcsec) * 1000.0  # mas


def _per_frame_nn(D):
    nn = np.full(len(D), np.inf)
    for fi in np.unique(D['frame']):
        idx = np.where(np.asarray(D['frame']) == fi)[0]
        xy = np.column_stack([np.asarray(D['x'])[idx], np.asarray(D['y'])[idx]])
        dd, _ = cKDTree(xy).query(xy, k=2)
        nn[idx] = dd[:, 1]
    return nn


def _clusters(cen, D, link_mas=40.0):
    xi, eta = _tangent(cen, D['ra_crds'], D['dec_crds'])
    pts = np.column_stack([xi, eta])
    pairs = cKDTree(pts).query_pairs(link_mas, output_type='ndarray')
    parent = np.arange(len(D))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in pairs:
        ra_, rb_ = find(a), find(b)
        if ra_ != rb_:
            parent[ra_] = rb_
    roots = np.array([find(i) for i in range(len(D))])
    out = {}
    for i, r in enumerate(roots):
        out.setdefault(r, []).append(i)
    return [np.array(v) for v in out.values()]


def _tie_and_bin(cen, ra, dec, D, frames, tracks, nbin=8, n_iter=4):
    """Per-frame 6-param linear tie to running mean, then bin residual by (x,y).

    Returns (binned_floor_mas, worst_cell_mas, per_det_scatter_mas, n_det,
    bin_centres, U_mas, V_mas).
    """
    xi, eta = _tangent(cen, ra, dec)
    txi, teta = xi.copy(), eta.copy()
    mx = np.zeros(len(txi))
    me = np.zeros(len(txi))
    for _ in range(n_iter):
        mx[:] = 0.0
        me[:] = 0.0
        for tr in tracks:
            mx[tr] = txi[tr].mean()
            me[tr] = teta[tr].mean()
        for fi in np.unique(frames):
            sel = np.where(frames == fi)[0]
            s = sel[mx[sel] != 0]
            if s.size < 6:
                continue
            A = np.column_stack([np.ones(s.size), xi[s], eta[s]])
            cx, *_ = np.linalg.lstsq(A, mx[s], rcond=None)
            cy, *_ = np.linalg.lstsq(A, me[s], rcond=None)
            Aall = np.column_stack([np.ones(sel.size), xi[sel], eta[sel]])
            txi[sel] = Aall @ cx
            teta[sel] = Aall @ cy
    all_idx = np.concatenate(tracks)
    mx[:] = 0.0
    me[:] = 0.0
    for tr in tracks:
        mx[tr] = txi[tr].mean()
        me[tr] = teta[tr].mean()
    rxi = txi[all_idx] - mx[all_idx]
    reta = teta[all_idx] - me[all_idx]
    per_det = float(np.sqrt(np.mean(rxi**2 + reta**2)))

    xd = np.asarray(D['x'])[all_idx]
    yd = np.asarray(D['y'])[all_idx]
    edges = np.linspace(0, 2048, nbin + 1)
    ix = np.clip(np.digitize(xd, edges) - 1, 0, nbin - 1)
    iy = np.clip(np.digitize(yd, edges) - 1, 0, nbin - 1)
    centres = (edges[:-1] + edges[1:]) / 2
    U = np.full((nbin, nbin), np.nan)
    V = np.full((nbin, nbin), np.nan)
    binmean, binw = [], []
    for a in range(nbin):
        for b in range(nbin):
            m = (ix == a) & (iy == b)
            if m.sum() >= 5:
                U[b, a] = rxi[m].mean()
                V[b, a] = reta[m].mean()
                binmean.append(np.hypot(U[b, a], V[b, a]))
                binw.append(m.sum())
    binmean = np.array(binmean)
    binw = np.array(binw)
    floor = float(np.sqrt(np.average(binmean**2, weights=binw)))
    worst = float(binmean.max())
    return floor, worst, per_det, len(all_idx), centres, U, V


def run_diagnostic(catalog_glob, crf_glob, bright_frac=0.40,
                   iso_list=(3.0, 5.0, 8.0), min_frames=6, out=None,
                   figure=None):
    """Run the CRDS-vs-GDC distortion-floor comparison; return a results dict."""
    D = _load(catalog_glob, crf_glob)
    print(f'total detections (quality cut): {len(D)}')
    cen = SkyCoord(np.median(D['ra_crds']) * u.deg,
                   np.median(D['dec_crds']) * u.deg)
    nn_px = _per_frame_nn(D)
    clusters = _clusters(cen, D)
    frames = np.asarray(D['frame'])
    instr = np.asarray(D['instr'])
    bright_thresh = float(np.percentile(instr, bright_frac * 100.0))

    def track_set(iso):
        out_tr = []
        for cl in clusters:
            fr = frames[cl]
            if len(np.unique(fr)) < min_frames:
                continue
            if len(fr) != len(np.unique(fr)):   # duplicate frame => blend
                continue
            if not np.all(instr[cl] <= bright_thresh):
                continue
            if not np.all(nn_px[cl] > iso):
                continue
            out_tr.append(cl)
        return out_tr

    rows = []
    fig_payload = None
    print('\niso_px |   N   | CRDS floor | GDC floor | ratio | '
          'CRDS worst | GDC worst | CRDS perdet | GDC perdet')
    for iso in iso_list:
        tracks = track_set(iso)
        if len(tracks) < 20:
            print(f'{iso:5.0f} | {len(tracks):5d} | too few')
            continue
        fc, wc, pc, n, ctr, Uc, Vc = _tie_and_bin(
            cen, D['ra_crds'], D['dec_crds'], D, frames, tracks)
        fg, wg, pg, _, _, Ug, Vg = _tie_and_bin(
            cen, D['ra_gdc'], D['dec_gdc'], D, frames, tracks)
        print(f'{iso:5.0f} | {len(tracks):5d} | {fc:9.3f}  | {fg:8.3f}  | '
              f'{fg/fc:5.2f} | {wc:9.3f}  | {wg:8.3f}  | {pc:10.3f}  | {pg:8.3f}')
        rows.append(dict(iso_px=iso, n_stars=len(tracks), n_det=n,
                         crds_floor_mas=round(fc, 4), gdc_floor_mas=round(fg, 4),
                         ratio=round(fg / fc, 3), crds_worst_mas=round(wc, 4),
                         gdc_worst_mas=round(wg, 4), crds_perdet_mas=round(pc, 4),
                         gdc_perdet_mas=round(pg, 4)))
        if iso == iso_list[0]:
            fig_payload = (ctr, Uc, Vc, Ug, Vg, len(tracks), fc, wc, fg, wg)

    result = dict(catalog_glob=catalog_glob, n_frames=int(len(np.unique(frames))),
                  n_det_quality=int(len(D)),
                  bright_instr_max=round(bright_thresh, 3), results=rows)
    if out:
        with open(out, 'w') as fh:
            json.dump(result, fh, indent=2)
        print(f'wrote {out}')
    if figure and fig_payload is not None:
        _make_figure(figure, *fig_payload)
        print(f'wrote {figure}')
    return result


def _make_figure(path, ctr, Uc, Vc, Ug, Vg, nstar, fc, wc, fg, wg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    XX, YY = np.meshgrid(ctr, ctr)
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.2), sharex=True, sharey=True)
    # Data-driven arrow scale, SHARED by both panels so CRDS and Jay are directly comparable.
    # With scale_units='xy' the arrow length in pixels is (residual mas / scale), so a hand-set
    # scale=3 made a ~0.4 mas vector 0.13 px long -- invisible on a 2048 px axis.  Instead size the
    # WORST vector across both panels to ~0.9 of the grid spacing so arrows are clearly visible
    # without overrunning neighbouring cells.
    grid = float(ctr[1] - ctr[0]) if len(ctr) > 1 else 256.0
    vmax = max(float(np.nanmax(np.hypot(Uc, Vc))), float(np.nanmax(np.hypot(Ug, Vg))), 1e-6)
    scale = vmax / (0.9 * grid)                        # mas per pixel
    q = None
    for ax, U, V, ttl, fl, wo in [
            (axs[0], Uc, Vc, 'CRDS / SIAF', fc, wc),
            (axs[1], Ug, Vg, 'Jay STDGDC', fg, wg)]:
        mag = np.hypot(U, V)
        q = ax.quiver(XX, YY, U, V, mag, scale=scale, scale_units='xy',
                      angles='xy', cmap='viridis', clim=(0, 0.35),
                      width=0.006, headwidth=4, headlength=5)
        ax.set_title(f'{ttl}\nbinned floor {fl:.3f} mas, worst {wo:.3f} mas')
        ax.set_xlabel('detector x (px)')
        ax.set_aspect('equal')
        ax.set_xlim(0, 2048)
        ax.set_ylim(0, 2048)
    # labelled reference arrow (a round value near the worst vector), below the panels so it does
    # not collide with the two-line titles
    key = round(vmax, 2) if vmax >= 0.1 else round(vmax, 3)
    axs[0].quiverkey(q, 0.0, -0.16, key, f'reference: {key} mas', labelpos='E',
                     coordinates='axes', fontproperties={'size': 8})
    axs[0].set_ylabel('detector y (px)')
    cb = fig.colorbar(q, ax=axs, fraction=0.04, pad=0.02)
    cb.set_label('|binned same-star residual| (mas)')
    fig.suptitle('Coherent distortion floor after per-frame linear tie '
                 f'(isolated bright stars, N={nstar})')
    fig.savefig(path, dpi=130, bbox_inches='tight')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--catalog-glob', required=True,
                   help='glob of per-exposure m1 catalogs (one detector, one '
                        'filter; skycoord_centroid + x_fit/y_fit + qfit/flux)')
    p.add_argument('--crf-glob', required=True,
                   help='glob of the matching crf/cal frames (paired by the '
                        '5-digit exposure token)')
    p.add_argument('--bright-frac', type=float, default=0.40,
                   help='keep the brightest this fraction of detections')
    p.add_argument('--out', default=None, help='write results JSON here')
    p.add_argument('--figure', default=None, help='write the quiver PNG here')
    args = p.parse_args(argv)
    run_diagnostic(args.catalog_glob, args.crf_glob, bright_frac=args.bright_frac,
                   out=args.out, figure=args.figure)


if __name__ == '__main__':
    main()
