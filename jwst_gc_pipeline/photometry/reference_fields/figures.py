"""Before/after cutout figures of a reference field: catalogs and residuals.

::

    # one field, the 'main' run against a variant (seed 0 = clean run)
    python -m jwst_gc_pipeline.photometry.reference_fields.figures \\
        --field dark --variant qfitsnr --out docs/evidence/qfitsnr/

    # every field
    python -m jwst_gc_pipeline.photometry.reference_fields.figures \\
        --variant qfitsnr --out docs/evidence/qfitsnr/

    # any two cutout runs of the same region
    python -m jwst_gc_pipeline.photometry.reference_fields.figures \\
        --base-dir RUN_A --variant-dir RUN_B --filter F182M --out cmp.png

Each figure has an overview row (the whole evaluated box: data with the zoom
boxes, both residuals, their difference) and one row per zoom of
``--zoom-arcsec`` (default 1.2") on the places where the two catalogs differ
most.  Columns: data + current catalog | current residual | data + proposed
catalog | proposed residual | current - proposed residual.  Residuals have
the smoothed background removed and share one linear stretch per row
(+/-``--stretch`` sigma of the current residual), so a star subtracted only
by the proposed run is a dark point in column 2, absent in column 4 and
bright in column 5.  Markers: white = in both catalogs, green = proposed
only, red = current only, yellow + = injected truth (seed > 0), magenta x =
hand-labelled emission structure (``emission_labels`` in fields.yaml).
"""
import argparse
import os
import sys

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy.table import Table
from scipy.spatial import cKDTree

from jwst_gc_pipeline.photometry import reference_fields as RF
from jwst_gc_pipeline.photometry.reference_fields import evaluate as EV

C_BOTH, C_NEW, C_DROP, C_TRUE, C_EMIS = 'white', 'lime', 'red', 'yellow', 'magenta'


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load_run(prod):
    """Images and catalog of one run's products (``EV.find_products`` dict)."""
    data, _, wcs, _ = EV._image(prod['data'])
    res, _, rwcs, _ = EV._image(prod['residual'])
    if prod.get('smoothed_bg'):
        res = res - EV._image(prod['smoothed_bg'])[0]
    sc, flux, ferr = EV.load_catalog(prod['catalog'])
    return dict(data=data, wcs=wcs, res=res, rwcs=rwcs, sc=sc,
                snr=flux / ferr, prod=prod)


def match_catalogs(sc_a, sc_b, radius_as):
    """Boolean ``in_b`` per source of ``a`` and ``in_a`` per source of ``b``
    (nearest neighbour within ``radius_as``, tangent-plane distances)."""
    ref = sc_a[0] if len(sc_a) else sc_b[0]
    off = ref.skyoffset_frame()

    def xy(sc):
        s = sc.transform_to(off)
        return np.c_[s.lon.to_value(u.arcsec), s.lat.to_value(u.arcsec)]

    if len(sc_a) == 0 or len(sc_b) == 0:
        return np.zeros(len(sc_a), bool), np.zeros(len(sc_b), bool)
    pa, pb = xy(sc_a), xy(sc_b)
    da = cKDTree(pb).query(pa)[0]
    db = cKDTree(pa).query(pb)[0]
    return da <= radius_as, db <= radius_as


def pick_zooms(x_new, y_new, x_drop, y_drop, box, zoom_pix, n):
    """Up to ``n`` non-overlapping zoom centres (pixels) in ``box`` =
    (x0, x1, y0, y1), ranked by the number of catalog differences they hold.
    Each zoom is centred on the mean position of the differences it holds (a
    zoom near the box edge may extend past it).  The box centre fills in when
    there are no differences."""
    x0, x1, y0, y1 = box
    h = zoom_pix / 2
    step = max(zoom_pix / 4, 1)
    gx = np.arange(x0, x1 + 1, step)
    gy = np.arange(y0, y1 + 1, step)
    px = np.r_[x_new, x_drop]
    py = np.r_[y_new, y_drop]
    cands = []
    for cx in gx:
        for cy in gy:
            sel = (np.abs(px - cx) < h) & (np.abs(py - cy) < h)
            if sel.any():
                mx, my = px[sel].mean(), py[sel].mean()
                cands.append((int(sel.sum()), np.hypot(mx - cx, my - cy), mx, my))
    # most differences first; among equals, the window best centred on them
    cands.sort(key=lambda c: (-c[0], c[1]))
    chosen = []
    for k, _, cx, cy in cands:
        if len(chosen) >= n:
            break
        if all(max(abs(cx - a), abs(cy - b)) >= zoom_pix for a, b in chosen):
            chosen.append((cx, cy))
    return chosen or [((x0 + x1) / 2, (y0 + y1) / 2)]


def robust_sigma(a):
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 1.0
    s = 1.4826 * np.median(np.abs(a - np.median(a)))
    return s if s > 0 else (np.std(a) or 1.0)


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

def _show(ax, img, vmin, vmax, stretch='linear'):
    from astropy.visualization import ImageNormalize, AsinhStretch, LinearStretch
    norm = ImageNormalize(vmin=vmin, vmax=vmax,
                          stretch=AsinhStretch(0.05) if stretch == 'asinh' else LinearStretch())
    ax.imshow(img, origin='lower', cmap='gray', norm=norm, interpolation='nearest')
    ax.set_xlim(-0.5, img.shape[1] - 0.5)
    ax.set_ylim(-0.5, img.shape[0] - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])


def _mark(ax, wcs, shape, sc, color, r, kind='o', lw=0.8, alpha=1.0):
    """Markers of radius ``r`` pixels at ``sc`` inside a cutout of ``shape``."""
    from matplotlib.patches import Circle, Rectangle
    if len(sc) == 0:
        return
    x, y = wcs.world_to_pixel(sc)
    x, y = np.atleast_1d(x), np.atleast_1d(y)
    ok = (x > -r) & (x < shape[1] - 1 + r) & (y > -r) & (y < shape[0] - 1 + r)
    for xi, yi in zip(x[ok], y[ok]):
        if kind == '+':
            ax.plot([xi - r, xi + r], [yi, yi], color=color, lw=lw, alpha=alpha)
            ax.plot([xi, xi], [yi - r, yi + r], color=color, lw=lw, alpha=alpha)
        elif kind == 'x':
            ax.plot([xi - r, xi + r], [yi - r, yi + r], color=color, lw=lw, alpha=alpha)
            ax.plot([xi - r, xi + r], [yi + r, yi - r], color=color, lw=lw, alpha=alpha)
        elif kind == 's':
            ax.add_patch(Rectangle((xi - r, yi - r), 2 * r, 2 * r, fill=False,
                                   ec=color, lw=lw, alpha=alpha))
        else:
            ax.add_patch(Circle((xi, yi), r, fill=False, ec=color, lw=lw, alpha=alpha))


def _cut(img, wcs, center, size_pix):
    c = Cutout2D(img, center, (size_pix, size_pix), wcs=wcs, mode='partial',
                 fill_value=np.nan)
    return c.data, c.wcs


def compare_figure(base, prop, out, *, title='', filt='', zoom_arcsec=1.2,
                   n_zoom=4, stretch=5.0, match_radius_as=None, box=None,
                   truth=None, emission=None, base_label='current',
                   prop_label='proposed', extra_text=''):
    """Write the before/after figure of two runs (``load_run`` dicts).

    ``box`` = (x0, x1, y0, y1) on the base data mosaic's pixel grid (default:
    the whole mosaic less 10 px); ``truth`` = injected positions (SkyCoord);
    ``emission`` = hand-labelled emission structures (SkyCoord).
    Returns a dict of the source counts inside the box.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    wcs = base['wcs']
    ny, nx = base['data'].shape
    pixas = float(np.sqrt(abs(np.linalg.det(wcs.pixel_scale_matrix))) * 3600)
    fw = EV.fwhm_pix(filt) if filt else 2.0
    if box is None:
        box = (10, nx - 10, 10, ny - 10)
    if match_radius_as is None:
        match_radius_as = max(1.0, 0.5 * fw) * pixas

    in_p, in_b = match_catalogs(base['sc'], prop['sc'], match_radius_as)
    xb, yb = wcs.world_to_pixel(base['sc'])
    xp, yp = wcs.world_to_pixel(prop['sc'])

    def inbox(x, y):
        return (x >= box[0]) & (x <= box[1]) & (y >= box[2]) & (y <= box[3])

    ib, ip = inbox(xb, yb), inbox(xp, yp)
    counts = dict(n_base=int(ib.sum()), n_prop=int(ip.sum()),
                  n_new=int((ip & ~in_b).sum()), n_dropped=int((ib & ~in_p).sum()))

    zoom_pix = int(round(zoom_arcsec / pixas))
    zooms = pick_zooms(xp[ip & ~in_b], yp[ip & ~in_b], xb[ib & ~in_p], yb[ib & ~in_p],
                       box, zoom_pix, n_zoom)
    r_mark = 1.6 * fw          # marker radius, pixels

    ncol = 5
    nrow = 1 + len(zooms)
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.4, nrow * 2.5),
                             squeeze=False)
    heads = [f'data + {base_label} cat', f'{base_label} residual',
             f'data + {prop_label} cat', f'{prop_label} residual',
             f'{base_label} - {prop_label} residual']

    # overview: the evaluated box
    cx, cy = (box[0] + box[1]) / 2, (box[2] + box[3]) / 2
    side = int(max(box[1] - box[0], box[3] - box[2])) + 4
    center = wcs.pixel_to_world(cx, cy)
    rows = [(center, side, 'overview')] + [
        (wcs.pixel_to_world(zx, zy), zoom_pix, chr(ord('A') + i))
        for i, (zx, zy) in enumerate(zooms)]
    sb, sp = base['sc'], prop['sc']
    new_sc, drop_sc = sp[~in_b], sb[~in_p]
    for r, (cen, size, tag) in enumerate(rows):
        d, dw = _cut(base['data'], wcs, cen, size)
        rb, rbw = _cut(base['res'], base['rwcs'], cen, size)
        rp, rpw = _cut(prop['res'], prop['rwcs'], cen, size)
        rd = rb - rp if rb.shape == rp.shape else np.full(rb.shape, np.nan)
        # data stretch from the zoom's sky: median - 3 sigma to median + 40
        # sigma (robust), so the noise is grey and faint stars visible; a
        # bright star in one corner of the zoom saturates instead of setting
        # a stretch under which the faint stars are black
        if np.isfinite(d).any():
            _med, _sd = np.nanmedian(d), robust_sigma(d)
            lo = _med - 3 * _sd
            hi = min(np.nanpercentile(d, 99.7), _med + 40 * _sd)
            hi = hi if hi > lo else lo + 1
        else:
            lo, hi = 0, 1
        sig = robust_sigma(rb)
        ax = axes[r]
        _show(ax[0], d, lo, hi, 'asinh')
        _show(ax[1], rb, -stretch * sig, stretch * sig)
        _show(ax[2], d, lo, hi, 'asinh')
        _show(ax[3], rp, -stretch * sig, stretch * sig)
        _show(ax[4], rd, -stretch * sig, stretch * sig)
        sh = d.shape
        if tag == 'overview':
            for i, (zx, zy) in enumerate(zooms):
                zc = wcs.pixel_to_world(zx, zy)
                ox, oy = dw.world_to_pixel(zc)
                ax[0].add_patch(Rectangle((ox - zoom_pix / 2, oy - zoom_pix / 2),
                                          zoom_pix, zoom_pix, fill=False,
                                          ec='cyan', lw=0.8))
                ax[0].text(ox - zoom_pix / 2 + 1, oy + zoom_pix / 2 - 1,
                           chr(ord('A') + i), color='cyan', fontsize=8, va='top')
            _mark(ax[0], dw, sh, drop_sc, C_DROP, r_mark, 's', 0.6)
            _mark(ax[2], dw, sh, new_sc, C_NEW, r_mark, 'o', 0.6)
            if emission is not None:
                for a in (ax[0], ax[2]):
                    _mark(a, dw, sh, emission, C_EMIS, r_mark * 0.5, 'x', 0.8)
        else:
            _mark(ax[0], dw, sh, sb[in_p], C_BOTH, r_mark, 'o', 0.5, 0.6)
            _mark(ax[0], dw, sh, drop_sc, C_DROP, r_mark, 's', 1.0)
            _mark(ax[1], rbw, rb.shape, sb[in_p], C_BOTH, r_mark, 'o', 0.4, 0.5)
            _mark(ax[1], rbw, rb.shape, drop_sc, C_DROP, r_mark, 's', 1.0)
            _mark(ax[2], dw, sh, sp[in_b], C_BOTH, r_mark, 'o', 0.5, 0.6)
            _mark(ax[2], dw, sh, new_sc, C_NEW, r_mark, 'o', 1.0)
            _mark(ax[3], rpw, rp.shape, sp[in_b], C_BOTH, r_mark, 'o', 0.4, 0.5)
            _mark(ax[3], rpw, rp.shape, new_sc, C_NEW, r_mark, 'o', 1.0)
            _mark(ax[4], rbw, rb.shape, new_sc, C_NEW, r_mark, 'o', 1.0)
            _mark(ax[4], rbw, rb.shape, drop_sc, C_DROP, r_mark, 's', 1.0)
            if truth is not None:
                for a, w_, shp in ((ax[0], dw, sh), (ax[1], rbw, rb.shape),
                                   (ax[2], dw, sh), (ax[3], rpw, rp.shape)):
                    _mark(a, w_, shp, truth, C_TRUE, r_mark * 0.6, '+', 0.8)
            if emission is not None:
                for a, w_, shp in ((ax[0], dw, sh), (ax[2], dw, sh)):
                    _mark(a, w_, shp, emission, C_EMIS, r_mark * 0.5, 'x', 1.0)
        ax[0].set_ylabel(f'{tag}  ({size * pixas:.1f}")', fontsize=8)
        if r == 0:
            for a, h in zip(ax, heads):
                a.set_title(h, fontsize=8)
    summary = (f"{title}\n{base_label}: {counts['n_base']} sources   "
               f"{prop_label}: {counts['n_prop']}   "
               f"proposed-only (green): {counts['n_new']}   "
               f"current-only (red): {counts['n_dropped']}   "
               f"{'injected (yellow +)   ' if truth is not None else ''}"
               f"{'emission label (magenta x)   ' if emission is not None else ''}"
               f"(match {match_radius_as * 1000:.0f} mas; residual stretch "
               f"+/-{stretch:g} sigma of the current residual per row)")
    if extra_text:
        summary += '\n' + extra_text
    fig.suptitle(summary, fontsize=8, y=0.995, va='top')
    fig.tight_layout(rect=(0, 0, 1, (0.975 if not extra_text else 0.96) ** (1 / max(nrow / 5, 1))),
                     h_pad=0.3, w_pad=0.2)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return counts


# ---------------------------------------------------------------------------
# reference-field driver
# ---------------------------------------------------------------------------

def _metrics_text(spec, variant, phase=None):
    try_keys = ('n_sources', 'residual_excess', 'ring_ratio', 'r_on', 'r_off',
                'labels_recovered', 'emission_labels_cataloged')
    try:
        m = EV.evaluate_clean(spec, variant, phase=phase)
    except FileNotFoundError:
        return ''
    return ', '.join(f'{k}={m[k]:.3g}' if isinstance(m[k], float) else f'{k}={m[k]}'
                     for k in try_keys if k in m)


def _completeness_text(spec, variant, seed, phase=None):
    """Recovered/injected per S/N_true bin of injection run ``seed``."""
    try:
        m = EV.evaluate_injected(spec, variant, seed, phase=phase)
    except FileNotFoundError:
        return ''
    comp = EV.completeness_by_bin(np.asarray(m['snr_true']), np.asarray(m['recovered']),
                                  spec['snr_bins'])
    return 'recovered by S/N_true ' + ' '.join(f'{b}:{k}/{n}' for b, (n, k, _) in comp.items())


def field_figure(name, variant, out_dir, *, base='main', seed=0, filt=None,
                 phase=None, **kw):
    """Figure of one reference field: run ``base`` vs run ``variant`` at
    ``phase`` (default: the last phase of the ``base`` run)."""
    _, fields = RF.load_config()
    spec = fields[name]
    filt = filt or spec['filters'][0]
    bdir = RF.run_dir(spec, base, seed)
    vdir = RF.run_dir(spec, variant, seed)
    bprod = EV.find_products(bdir, filt, phase)
    vprod = EV.find_products(vdir, filt, phase=bprod['phase'])
    b, v = load_run(bprod), load_run(vprod)
    fr = EV.Frame(spec, b['wcs'], b['data'].shape)
    box = (fr.cx - fr.half, fr.cx + fr.half, fr.cy - fr.half, fr.cy + fr.half)
    truth = None
    if int(seed):
        t = Table.read(RF.injection_table_path(name, seed))
        truth = SkyCoord(t['ra'] * u.deg, t['dec'] * u.deg)
    emission = None
    if spec.get('emission_labels'):
        ek = np.asarray(spec['emission_labels'], float)
        emission = SkyCoord(ek[:, 0] * u.deg, ek[:, 1] * u.deg)
    text = ''
    if int(seed):
        mb = _completeness_text(spec, base, seed, bprod['phase'])
        mv = _completeness_text(spec, variant, seed, bprod['phase'])
        if mb or mv:
            text = f'{base}: {mb}\n{variant}: {mv}'
    else:
        mb = _metrics_text(spec, base, bprod['phase'])
        mv = _metrics_text(spec, variant, bprod['phase'])
        if mb or mv:
            text = f'{base}: {mb}\n{variant}: {mv}'
    out = os.path.join(out_dir, f'{name}_{filt.lower()}_{bprod["phase"]}_s{seed}.png')
    title = (f"{name} [{spec['environment']}] {filt} {bprod['phase']}, "
             f"seed {seed}: {base} vs {variant}")
    counts = compare_figure(b, v, out, title=title, filt=filt, box=box, truth=truth,
                            emission=emission, base_label=base, prop_label=variant, extra_text=text, **kw)
    print(f'{out}: {counts}')
    return out, counts


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--variant', help='proposed run label')
    p.add_argument('--base', default='main', help='current run label (default main)')
    p.add_argument('--fields', default='', help='comma list (default: all)')
    p.add_argument('--seeds', default='0', help='comma list (default 0 = clean)')
    p.add_argument('--filter', default=None, help="default: each field's primary band")
    p.add_argument('--base-dir', help='explicit current run dir (with --variant-dir)')
    p.add_argument('--variant-dir', help='explicit proposed run dir')
    p.add_argument('--phase', default=None)
    p.add_argument('--zoom-arcsec', type=float, default=1.2)
    p.add_argument('--n-zoom', type=int, default=4)
    p.add_argument('--stretch', type=float, default=5.0)
    p.add_argument('--out', required=True,
                   help='output directory (reference fields) or .png (explicit dirs)')
    a = p.parse_args(argv)
    kw = dict(zoom_arcsec=a.zoom_arcsec, n_zoom=a.n_zoom, stretch=a.stretch)

    if a.base_dir or a.variant_dir:
        if not (a.base_dir and a.variant_dir and a.filter):
            p.error('--base-dir needs --variant-dir and --filter')
        bprod = EV.find_products(a.base_dir, a.filter, a.phase)
        vprod = EV.find_products(a.variant_dir, a.filter, a.phase or bprod['phase'])
        counts = compare_figure(load_run(bprod), load_run(vprod), a.out, filt=a.filter,
                                title=f'{a.filter} {bprod["phase"]}: '
                                      f'{os.path.basename(a.base_dir.rstrip("/"))} vs '
                                      f'{os.path.basename(a.variant_dir.rstrip("/"))}',
                                **kw)
        print(f'{a.out}: {counts}')
        return 0

    if not a.variant:
        p.error('--variant (or --base-dir/--variant-dir) is required')
    _, fields = RF.load_config()
    names = [n for n in a.fields.split(',') if n] or list(fields)
    for name in names:
        for seed in [int(s) for s in a.seeds.split(',') if s]:
            try:
                field_figure(name, a.variant, a.out, base=a.base, seed=seed,
                             filt=a.filter, phase=a.phase, **kw)
            except FileNotFoundError as exc:
                print(f'{name} s{seed}: skipped ({exc})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
