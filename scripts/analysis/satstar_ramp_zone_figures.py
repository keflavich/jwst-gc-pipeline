"""Figures for the zone-based saturated-star ramp recovery subsection.

Produces, from a real brick F200W nrca1 frame + its Detector1 ``_ramp.fits``:
  1. ``satstar_ramp_zones_cutouts_<date>.png`` -- cutouts of a few saturated
     stars with the four recovery ZONES overlaid (unrecoverable deep core,
     ramp-slope recoverable, group-0 recoverable, charge-migration shoulder).
     DO_NOT_USE pixels (bad pixels + cosmic-ray/outlier rejections; NaN in the
     cal frame) that are NOT part of any recovery zone are marked with red x's,
     so the "white" background/holes are explicitly explained rather than
     ambiguous.
  2. ``satstar_ramp_linearity_<date>.png`` -- the per-zone linearity model:
     (a) DN-vs-group ramps for representative pixels in each zone with the
     leading-group slope fit, the pile-up ceiling, and the group-0 read;
     (b) cal (MJy/sr) vs slope (DN/group) over unsaturated pixels showing
     cal = K*slope (the flat calibration) with the K-band clamp, and cal vs
     group-0 showing the brightness-drifting R(g0) that motivates the slope.

Density-immune method: no nearest-neighbour median anywhere.  Uses the public
recovery code (``ramp_slope_map`` / ``ramp_recover_saturated``); zones are
reproduced with the same ceiling + n_good logic for display only.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from astropy.io import fits
from astropy.visualization import AsinhStretch, ImageNormalize, PercentileInterval
from scipy import ndimage

from jwst_gc_pipeline.reduction.saturated_star_finding import (
    ramp_slope_map, ramp_recover_saturated)
try:
    from jwst.datamodels import dqflags
    PSAT = dqflags.pixel['SATURATED']
    PDNU = dqflags.pixel['DO_NOT_USE']
    POUT = dqflags.pixel['OUTLIER']       # resample/median outlier rejection
    PJMP = dqflags.pixel['JUMP_DET']      # ramp jump-detection (pixel-level summary)
    PJUMP = dqflags.group['JUMP_DET']     # ramp jump-detection (group-level)
except (ImportError, KeyError):
    PSAT, PDNU, POUT, PJMP, PJUMP = 2, 1, 2**21, 4, 4

DATE = '2026-07-21'
BASE = '/orange/adamginsburg/jwst/brick'
CRF = f'{BASE}/F200W/pipeline/jw01182004001_04101_00001_nrca1_destreak_o004_crf.fits'
RAMP = f'{BASE}/F200W/pipeline/jw01182004001_04101_00001_nrca1_ramp.fits'
OUT = f'{BASE}/astrometry_paper/figures/evidence'
SLOPE_MIN = 20.0
CAL_BLOCK = 64

# zone colours (colour-blind-safe-ish)
C_DEEP = '#d62728'   # unrecoverable deep core - red
C_RAMP = '#2ca02c'   # ramp-slope recoverable - green
C_G0 = '#1f77b4'     # group-0 recoverable - blue
C_CM = '#ff7f0e'     # charge-migration shoulder - orange
C_REJ = '#00e5ff'    # jump/outlier-rejected REAL bright-star signal - cyan
C_BAD = '#ff0000'    # genuine bad pixel (dead/hot/RC) - red


def load():
    with fits.open(CRF) as h:
        cal = h['SCI'].data.astype(float)
        dq = h['DQ'].data.astype(int)
    with fits.open(RAMP) as h:
        rsci = h['SCI'].data[0].astype(float)      # (ng, ny, nx)
        rgdq = h['GROUPDQ'].data[0].astype(int)
    return cal, dq, rsci, rgdq


def zones(cal, dq, rsci, rgdq):
    """Reproduce the recovery's decision zones for display."""
    ng = rsci.shape[0]
    bright = rsci[np.isfinite(rsci) & (rsci > 1000)]
    ceiling = 0.9 * np.nanpercentile(bright, 99.9) if bright.size >= 100 else 48000.
    slope, n_good = ramp_slope_map(rsci, rgdq, ceiling=ceiling)
    recovered, rim, deep, K = ramp_recover_saturated(
        cal, dq, rsci, rgdq, slope_min=SLOPE_MIN, cal_block=CAL_BLOCK, ceiling=ceiling)
    sat = (dq & PSAT) != 0
    # Z2 ramp-slope recoverable: DQ-sat rim rewritten from a >=2-group slope fit
    z_ramp = sat & rim
    # Z1 unrecoverable deep core: sat, group-0 already at/over the ceiling (n_good==0)
    z_deep = sat & (n_good == 0)
    # Z3 group-0 recoverable: sat, exactly ONE usable leading read (n_good==1) --
    # no slope, but the calibrated first read anchors it (the zeroframe fallback)
    z_g0 = sat & (n_good == 1)
    # Z4 charge-migration shoulder: NOT DQ-sat, but the buffer pixels the recovery
    # de-inflates (rewritten rim outside the SAT mask)
    z_cm = rim & ~sat
    # DO_NOT_USE pixels (NaN in cal) NOT in any recovery zone are the "white"
    # pixels in the grayscale.  They split into two physically-different classes
    # that must NOT share a label:
    #   REJECTED SIGNAL -- flagged OUTLIER and/or JUMP_DET: overwhelmingly REAL
    #     bright-star PSF-wing / diffraction-spike pixels whose steep but smooth
    #     ramps trip the jump step and whose sharp undersampled structure fails
    #     the resample-median outlier test, so genuine signal is set to NaN.
    #     These trace the diffraction spikes of bright stars -- they are neither
    #     cosmic rays nor bad detector pixels (verified: smooth monotonic ramps).
    #   BAD PIXEL -- the remainder (DEAD/HOT/RC/reference/no-linearity etc.):
    #     genuine detector defects or non-science pixels.
    any_zone = z_deep | z_ramp | z_g0 | z_cm
    dnu = ((dq & PDNU) != 0) & ~any_zone
    z_reject = dnu & (((dq & POUT) != 0) | ((dq & PJMP) != 0))
    z_bad = dnu & ~z_reject
    # inflation actually removed at those pixels: original data / de-inflated value
    with np.errstate(invalid='ignore', divide='ignore'):
        data_over_recov_cm = np.where(recovered > 0, cal / recovered, np.nan)[z_cm]
    return dict(slope=slope, n_good=n_good, recovered=recovered, K=K,
                ceiling=ceiling, sat=sat, rim=rim, z_reject=z_reject, z_bad=z_bad,
                any_zone=any_zone,
                z_deep=z_deep, z_ramp=z_ramp, z_g0=z_g0, z_cm=z_cm, ng=ng,
                data_over_recov_cm=data_over_recov_cm)


def pick_stars(z, cal, half=26):
    """Label saturated clusters and choose three illustrative examples:
      A - the biggest/brightest (deep core + group-0 ring + shoulder),
      B - the cluster whose stamp holds the most RAMP-slope-recoverable pixels
          (a green recoverable rim -- the whole point of the method),
      C - a mildly saturated one (group-0 + shoulder, little/no deep core).
    Stamps that are mostly non-finite (large saturated-bleed / no-data blobs)
    are rejected so every panel is a clean single star.  For C we additionally
    require a genuine bright PSF peak on the saturated cluster (peak within a few
    px of its centre and near the top of the frame brightness distribution), so
    the panel is a real mildly-saturated star and not an off-centre cosmic-ray /
    warm-pixel cluster that merely tripped DQ-SATURATED."""
    lab, nlab = ndimage.label(z['sat'])
    sizes = ndimage.sum(np.ones_like(lab), lab, range(1, nlab + 1))
    coms = ndimage.center_of_mass(z['sat'], lab, range(1, nlab + 1))
    ny, nx = cal.shape
    rows = []
    for sz, (cy, cx) in zip(sizes, coms):
        cy, cx = int(round(cy)), int(round(cx))
        if not (half < cy < ny - half and half < cx < nx - half):
            continue
        sl = (slice(cy - half, cy + half), slice(cx - half, cx + half))
        stamp = cal[sl]
        nan_frac = np.mean(~np.isfinite(stamp))
        if nan_frac > 0.35:                 # skip huge bleed / no-data stamps
            continue
        satn = max(1, z['sat'][sl].sum())
        # brightest finite pixel in the stamp and its offset from the centre
        fin = np.where(np.isfinite(stamp), stamp, -np.inf)
        py, px = np.unravel_index(np.argmax(fin), stamp.shape)
        peakval = float(stamp[py, px])
        peakoff = float(np.hypot(py - half, px - half))
        rows.append(dict(sz=sz, cy=cy, cx=cx,
                         deep=int(z['z_deep'][sl].sum()),
                         ramp=int(z['z_ramp'][sl].sum()),
                         g0=int(z['z_g0'][sl].sum()),
                         cm=int(z['z_cm'][sl].sum()),
                         peakval=peakval, peakoff=peakoff,
                         deepfrac=z['z_deep'][sl].sum() / satn))
    A = max(rows, key=lambda r: r['sz'])
    # B: the fullest zone MIX -- a moderately-saturated star that shows the ramp
    # ring together with a core + group-0 ring + shoulder (all four zones)
    bpool = [r for r in rows if r['deep'] >= 3 and r['g0'] >= 3 and r['ramp'] >= 2
             and 15 <= r['sz'] <= 120]
    B = (max(bpool, key=lambda r: 10 * r['ramp'] + r['g0'] + r['cm']) if bpool
         else max(rows, key=lambda r: r['ramp']))
    # C: MILD -- a clean, peak-CENTRED lightly-saturated star: a small deep core
    # with a group-0-recoverable rim (blue) + charge-migration shoulder (orange).
    # We require the saturated cluster to sit ON a bright PSF peak (peakoff small)
    # so C is a real star, NOT an off-centre cosmic-ray/warm-pixel cluster that
    # merely tripped DQ-SATURATED (the earlier pick at (631,1300) had its bright
    # peak 26 px away -- a neighbour -- and was spurious).  A genuinely mild star's
    # tiny core is group-0-recoverable, not ramp-slope; the green ramp-slope zone
    # lives in the RIM of deeper stars (panel B), so C shows no green.
    cpool = [r for r in rows if r['sz'] >= 6 and r['deep'] <= 10
             and r['peakoff'] <= 4.0 and (r['g0'] + r['ramp']) >= 5]
    if cpool:
        C = min(cpool, key=lambda r: (r['deep'], -(3 * r['ramp'] + r['g0'] + r['cm'])))
    else:
        C = min(rows, key=lambda r: r['peakoff'])
    picks, seen = [], set()
    for r in (A, B, C):
        key = (r['cy'], r['cx'])
        if key not in seen:
            seen.add(key); picks.append((r['cy'], r['cx']))
    return picks[:3]


def fig_cutouts(cal, z, stars, half=26):
    n = len(stars)
    fig, axes = plt.subplots(1, n, figsize=(4.1 * n, 4.6))
    if n == 1:
        axes = [axes]
    # NaN (DO_NOT_USE) pixels render as dark grey, not jarring white -- and the
    # unclassified DO_NOT_USE pixels are separately marked with red x's below.
    gray = matplotlib.colormaps['gray'].copy()
    gray.set_bad('0.12')
    for ax, (cy, cx) in zip(axes, stars):
        sl = (slice(cy - half, cy + half), slice(cx - half, cx + half))
        stamp = cal[sl]
        norm = ImageNormalize(stamp, interval=PercentileInterval(99.5),
                              stretch=AsinhStretch(a=0.02))
        ax.imshow(stamp, origin='lower', cmap=gray, norm=norm)
        # zone masks as semi-transparent single-colour overlays
        for zk, col in ((z['z_deep'], C_DEEP), (z['z_g0'], C_G0),
                        (z['z_ramp'], C_RAMP), (z['z_cm'], C_CM)):
            m = zk[sl]
            over = np.zeros((*m.shape, 4))
            rgba = matplotlib.colors.to_rgba(col)
            over[m] = (rgba[0], rgba[1], rgba[2], 0.55)
            ax.imshow(over, origin='lower')
        # cyan + on jump/outlier-REJECTED real bright-star signal; red x on
        # genuine bad pixels -- the two "white" classes labelled separately
        rys, rxs = np.where(z['z_reject'][sl])
        if len(rys):
            ax.plot(rxs, rys, '+', color=C_REJ, ms=5.0, mew=0.9, ls='none')
        bys, bxs = np.where(z['z_bad'][sl])
        if len(bys):
            ax.plot(bxs, bys, 'x', color=C_BAD, ms=4.0, mew=0.9, ls='none')
        ndeep = z['z_deep'][sl].sum(); nramp = z['z_ramp'][sl].sum()
        ng0 = z['z_g0'][sl].sum(); ncm = z['z_cm'][sl].sum()
        ax.set_title(f'({cx},{cy})  deep={ndeep} ramp={nramp}\n'
                     f'g0={ng0} shoulder={ncm}', fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(-0.5, 2 * half - 0.5); ax.set_ylim(-0.5, 2 * half - 0.5)
    handles = [Patch(color=C_DEEP, label='unrecoverable deep core (rails at $g_0$)'),
               Patch(color=C_RAMP, label='ramp-slope recoverable ($\\geq2$ pre-sat reads)'),
               Patch(color=C_G0, label='group-0 recoverable (1 usable read)'),
               Patch(color=C_CM, label='charge-migration shoulder (de-inflated)'),
               Line2D([0], [0], color=C_REJ, marker='+', ls='none', mew=1.4,
                      label='jump/outlier-rejected (real bright-star PSF signal)'),
                      Line2D([0], [0], color=C_BAD, marker='x', ls='none', mew=1.4,
                      label='bad pixel (dead/hot/RC)')]
    fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle('Saturated-star recovery zones — brick F200W nrca1 '
                 '(exp 00001, 7-group ramp)', fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    p = f'{OUT}/satstar_ramp_zones_cutouts_{DATE}.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return p


def fig_linearity(cal, z, rsci, rgdq, star_radial):
    ng = z['ng']
    t = np.arange(ng)
    ceiling = z['ceiling']
    slope, n_good = z['slope'], z['n_good']
    sat = z['sat']
    rng = np.random.default_rng(0)

    def global_sample(zonemask, kmax=3, near_bright=True, ngood_window=None):
        """A few representative pixels of a zone, taken near the brightest
        cluster of that zone so the traces are clean and comparable.
        ``ngood_window`` optionally restricts to pixels with that many leading
        good reads (e.g. a partially-saturated ramp that rises then rails)."""
        m = zonemask.copy()
        if ngood_window is not None:
            lo, hi = ngood_window
            m = m & (n_good >= lo) & (n_good <= hi) & (rsci.max(axis=0) >= ceiling)
        ys, xs = np.where(m)
        if len(ys) == 0:
            ys, xs = np.where(zonemask)
        if len(ys) == 0:
            return []
        if near_bright:
            g0 = rsci[0, ys, xs]
            order = np.argsort(-g0)
            ys, xs = ys[order], xs[order]
        idx = np.linspace(0, min(len(ys), 400) - 1, kmax).astype(int)
        return list(zip(ys[idx], xs[idx]))

    fig, (axr, axk) = plt.subplots(1, 2, figsize=(10.5, 4.7))

    # ---- panel (a): per-zone ramps + leading-group slope fit ----
    axr.axhline(ceiling, color='k', ls=':', lw=1.3,
                label=f'pile-up ceiling ({ceiling:.0f} DN)')
    for zk, col in (('z_ramp', C_RAMP), ('z_g0', C_G0), ('z_deep', C_DEEP)):
        win = (2, 5) if zk == 'z_ramp' else None
        for (py, px) in global_sample(z[zk], kmax=2, ngood_window=win):
            ramp = rsci[:, py, px]; gdq = rgdq[:, py, px]
            below = np.isfinite(ramp) & (ramp < ceiling) & ((gdq & PJUMP) == 0)
            lead = np.cumprod(below).astype(bool)
            axr.plot(t, ramp, 'o-', color=col, ms=4, lw=1, alpha=0.85)
            if lead.sum() >= 2:
                m, b = np.polyfit(t[lead], ramp[lead], 1)
                xx = np.array([0, ng - 1])
                axr.plot(xx, m * xx + b, '--', color=col, lw=1.4, alpha=0.9)
            axr.plot(0, ramp[0], 's', color=col, ms=8, mfc='none', mew=1.6)
    axr.legend(handles=[
        Line2D([0], [0], color=C_RAMP, marker='o', label='ramp-slope zone ($\\geq2$ reads)'),
        Line2D([0], [0], color=C_G0, marker='o', label='group-0 zone (1 read)'),
        Line2D([0], [0], color=C_DEEP, marker='o', label='deep-core zone (rails at $g_0$)'),
        Line2D([0], [0], color='k', ls='--', label='leading-group slope fit'),
        Line2D([0], [0], color='k', ls=':', label='pile-up ceiling'),
        Line2D([0], [0], color='k', marker='s', mfc='none', ls='', label='group-0 read')],
        fontsize=8, loc='center right')
    axr.set_xlabel('ramp group'); axr.set_ylabel('signal (DN)')
    axr.set_title('(a) per-zone ramps: slope fit vs railed reads')

    # ---- clean bright-wing calibrator sample: unsaturated pixels in the
    # shoulders of saturated stars (what K_local is actually measured on) ----
    near = ndimage.binary_dilation(sat, iterations=12)
    calib = (near & ~sat & np.isfinite(cal) & np.isfinite(slope)
             & (slope > 200) & (cal > 0))
    yy, xx = np.where(calib)
    if len(yy) > 40000:
        p = rng.choice(len(yy), 40000, replace=False); yy, xx = yy[p], xx[p]
    sv = slope[yy, xx]; cv = cal[yy, xx]; g0 = rsci[0, yy, xx]
    ratio = cv / sv
    K = np.median(ratio)
    mad = np.median(np.abs(ratio - K)) * 1.4826
    robust_pct = 100 * mad / K

    # ---- panel (b): cal vs slope, flat LOCAL proportionality + clamp band ----
    axk.plot(sv, cv, '.', ms=1.5, alpha=0.25, color='0.45')
    hi = np.percentile(sv, 99.5)
    xs = np.array([200, hi])
    axk.plot(xs, K * xs, '-', color=C_RAMP, lw=2,
             label=f'cal $=K\\,$slope, $K={K:.4g}$')
    axk.plot(xs, K * 4 * xs, '--', color=C_CM, lw=1.3, label='K-band clamp $\\pm4\\times$')
    axk.plot(xs, K / 4 * xs, '--', color=C_CM, lw=1.3)
    axk.set_xlim(0, hi); axk.set_ylim(0, K * hi * 1.6)
    axk.set_xlabel('ramp slope (DN/group)'); axk.set_ylabel('cal (MJy/sr)')
    axk.set_title(f'(b) cal $\\propto$ slope  (robust scatter {robust_pct:.0f}%)')
    axk.legend(fontsize=8, loc='upper left')

    fig.suptitle('Per-zone linearity model — brick F200W nrca1 '
                 '(calibration is block-LOCAL)', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = f'{OUT}/satstar_ramp_linearity_{DATE}.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return p, K, robust_pct


def main():
    os.makedirs(OUT, exist_ok=True)
    cal, dq, rsci, rgdq = load()
    z = zones(cal, dq, rsci, rgdq)
    tot = dict(deep=int(z['z_deep'].sum()), ramp=int(z['z_ramp'].sum()),
               g0=int(z['z_g0'].sum()), cm=int(z['z_cm'].sum()))
    print('frame-wide zone pixel counts:', tot, 'K=%.4g' % z['K'],
          'ceiling=%.0f' % z['ceiling'])
    stars = pick_stars(z, cal)
    print('example stars (cx,cy):', [(x, y) for y, x in stars])
    p1 = fig_cutouts(cal, z, stars)
    p2, K, rp = fig_linearity(cal, z, rsci, rgdq, None)
    print('linearity K=%.4g robust_scatter=%.0f%%' % (K, rp))
    print('wrote', p1)
    print('wrote', p2)


if __name__ == '__main__':
    main()
