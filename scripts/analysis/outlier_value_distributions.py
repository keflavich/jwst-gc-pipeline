"""Investigate WHAT outlier_detection computes and rejects (keflavich/jwst-gc-pipeline#161).

1. Re-runs OutlierDetectionStep on the brick o004 F200W nrca association with
   ``save_intermediate_results=True`` to dump the blotted-median image the step
   compares against.
2. For the analyzed frame, reconstructs the per-pixel decision quantities
   (`diff = |sci - blot - backg|`, `blot_deriv`, `err`, and the default vs raised
   thresholds `scale*blot_deriv + snr*err`) and plots:
     (a) SCI-value distributions of the rejected / recovered-by-tweak / kept pops,
     (b) diff vs blot_deriv with both threshold lines (why a pixel flips),
     (c) diff / threshold histograms,
     (d) rejection fraction vs radius from the bright star.
Uses the actual `_crf`/`outlierdetectionstep` DQ from the earlier validation run
for the ground-truth OUTLIER masks (default + tweaked).
"""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy import ndimage

BASE = '/orange/adamginsburg/jwst/brick/F200W/pipeline'
SRC_ASN = f'{BASE}/jw01182-o004_20251215t004215_image3_00002_nrca_asn.json'
VALID = '/blue/adamginsburg/adamginsburg/tmp/outlier_validate'   # earlier run's crf DQ
OUT = os.environ.get('DIST_OUTDIR', '/blue/adamginsburg/adamginsburg/tmp/outlier_dist')
FRAME = 'jw01182004001_04101_00001_nrca1'
PRE = f'{BASE}/{FRAME}_destreak.fits'
STAR = (1362, 84)      # cx, cy of the bright star
# stcal defaults / proposed (snr1 snr2 / scale1 scale2)
DEF = dict(snr=(5.0, 4.0), scale=(1.2, 0.7))
TWK = dict(snr=(7.0, 5.0), scale=(2.0, 1.5))


def build_asn(dst):
    a = json.load(open(SRC_ASN))
    for m in a['products'][0]['members']:
        tw = m['expname'].replace('_destreak.fits', '_destreak_tweakreg.fits')
        p = tw if os.path.exists(os.path.join(BASE, tw)) else m['expname']
        m['expname'] = os.path.join(BASE, p)
    a['products'][0]['name'] = a['products'][0]['name'] + '_dist'
    json.dump(a, open(dst, 'w'), indent=2)
    return dst


def run_blot():
    from jwst.outlier_detection import OutlierDetectionStep
    os.makedirs(OUT, exist_ok=True)
    asn = build_asn(f'{OUT}/asn.json')
    print('[dist] running OutlierDetectionStep save_intermediate_results=True', flush=True)
    OutlierDetectionStep.call(
        asn, save_results=True, save_intermediate_results=True,
        in_memory=True, output_dir=OUT,
        snr=f"{DEF['snr'][0]} {DEF['snr'][1]}",
        scale=f"{DEF['scale'][0]} {DEF['scale'][1]}")
    print('[dist] intermediate products:', flush=True)
    for pat in ('*median*', '*blot*', '*i2d*'):
        for f in sorted(glob.glob(f'{OUT}/{pat}'))[:4]:
            print('   ', os.path.basename(f), flush=True)


def _abs_deriv(a):
    """max abs difference to the 4 neighbours (stcal _abs_deriv, simplified)."""
    out = np.zeros_like(a)
    for ax in (0, 1):
        d = np.abs(np.diff(a, axis=ax))
        sl0 = [slice(None)] * 2; sl0[ax] = slice(0, -1)
        sl1 = [slice(None)] * 2; sl1[ax] = slice(1, None)
        out[tuple(sl0)] = np.maximum(out[tuple(sl0)], d)
        out[tuple(sl1)] = np.maximum(out[tuple(sl1)], d)
    return out


def _dq(outdir):
    from jwst.datamodels import dqflags
    f = sorted(x for x in glob.glob(f'{outdir}/*.fits')
               if FRAME + '_' in os.path.basename(x)
               and ('crf' in x or 'outlierdetectionstep' in x))[0]
    return fits.getdata(f, 'DQ').astype(np.uint32), dqflags.pixel


def analyze():
    from jwst.datamodels import dqflags
    P = dqflags.pixel
    # blot for this frame
    bl = sorted(x for x in glob.glob(f'{OUT}/*blot*') if FRAME in os.path.basename(x))
    if not bl:
        raise SystemExit(f'[dist] no blot for {FRAME}; found {os.listdir(OUT)[:8]}')
    blot = fits.getdata(bl[0], 'SCI').astype(float) if 'SCI' in [h.name for h in fits.open(bl[0])] \
        else fits.getdata(bl[0]).astype(float)
    with fits.open(PRE) as h:
        sci = h['SCI'].data.astype(float)
        err = h['ERR'].data.astype(float)
    dqd, _ = _dq(f'{VALID}/default')
    dqt, _ = _dq(f'{VALID}/tweaked')
    sat = (dqd & P['SATURATED']) != 0
    od = ((dqd & P['OUTLIER']) != 0) & ~sat
    ot = ((dqt & P['OUTLIER']) != 0) & ~sat
    recovered = od & ~ot          # rejected by default, kept by tweaked
    kept = ~od & ~sat             # never rejected

    diff = np.abs(sci - blot)
    bd = _abs_deriv(blot)
    th_def1 = DEF['scale'][0] * bd + DEF['snr'][0] * err
    th_twk1 = TWK['scale'][0] * bd + TWK['snr'][0] * err

    cx, cy, h = STAR[0], STAR[1], 150
    box = (slice(max(0, cy - h), cy + h), slice(max(0, cx - h), cx + h))
    def bx(a): return a[box]

    fig, ax = plt.subplots(2, 2, figsize=(15, 12))

    # (a) SCI value distributions
    a0 = ax[0, 0]
    bins = np.linspace(-20, 300, 120)
    for m, lab, c in [(od, f'default-rejected ({int(od.sum())})', 'tab:red'),
                      (recovered, f'recovered by tweak ({int(recovered.sum())})', 'tab:green'),
                      (ot, f'still rejected ({int(ot.sum())})', 'tab:orange'),
                      (kept, 'kept (never rej)', '0.6')]:
        v = sci[m]; v = v[np.isfinite(v)]
        a0.hist(v, bins=bins, histtype='step', color=c, density=True, label=lab)
    a0.set_xlabel('destreak SCI value (MJy/sr)'); a0.set_ylabel('density')
    a0.set_title('(a) pixel VALUE distribution by rejection class (frame)'); a0.legend(fontsize=8)
    a0.set_yscale('log')

    # (b) diff vs blot_deriv in the bright-star box, threshold lines
    a1 = ax[0, 1]
    mk = bx(~sat) & np.isfinite(bx(diff)) & np.isfinite(bx(bd))
    for m, c, lab in [(bx(kept), '0.7', 'kept'), (bx(recovered), 'tab:green', 'recovered'),
                      (bx(ot), 'tab:orange', 'still rejected')]:
        mm = m & mk
        a1.scatter(bx(bd)[mm], bx(diff)[mm], s=3, c=c, alpha=0.4, label=lab, rasterized=True)
    xx = np.linspace(0, np.nanpercentile(bx(bd)[mk], 99), 100)
    emed = np.nanmedian(bx(err)[mk])
    a1.plot(xx, DEF['scale'][0] * xx + DEF['snr'][0] * emed, 'r-',
            label=f'default thresh (err~{emed:.1f})')
    a1.plot(xx, TWK['scale'][0] * xx + TWK['snr'][0] * emed, 'b-', label='tweaked thresh')
    a1.set_xlabel('blot_deriv (structure term)'); a1.set_ylabel('|sci - blot| (diff)')
    a1.set_title('(b) what is computed: diff vs threshold (bright-star box)')
    a1.legend(fontsize=8); a1.set_xlim(0, xx.max()); a1.set_ylim(0, np.nanpercentile(bx(diff)[mk], 99.5))

    # (c) diff and threshold histograms (box)
    a2 = ax[1, 0]
    a2.hist(bx(diff)[mk], bins=100, histtype='step', color='k', label='diff |sci-blot|', density=True)
    a2.hist(bx(th_def1)[mk], bins=100, histtype='step', color='r', label='threshold default', density=True)
    a2.hist(bx(th_twk1)[mk], bins=100, histtype='step', color='b', label='threshold tweaked', density=True)
    a2.set_xlabel('MJy/sr'); a2.set_ylabel('density'); a2.set_xlim(0, np.nanpercentile(bx(diff)[mk], 99))
    a2.set_title('(c) diff vs default/tweaked threshold (box)'); a2.legend(fontsize=8)

    # (d) rejection fraction vs radius from star
    a3 = ax[1, 1]
    yy, xx2 = np.mgrid[box[0], box[1]]
    r = np.hypot(yy - cy, xx2 - cx).ravel()
    for m, c, lab in [(bx(od).ravel(), 'r', 'default'), (bx(ot).ravel(), 'b', 'tweaked')]:
        rb = np.linspace(0, h, 30); frac = []
        notsat = ~bx(sat).ravel()
        for i in range(len(rb) - 1):
            sel = (r >= rb[i]) & (r < rb[i + 1]) & notsat
            frac.append(m[sel].mean() if sel.sum() else np.nan)
        a3.plot(0.5 * (rb[:-1] + rb[1:]), frac, '-', color=c, label=lab)
    a3.set_xlabel('radius from star (px)'); a3.set_ylabel('OUTLIER fraction')
    a3.set_title('(d) rejection vs radius (bright star)'); a3.legend(fontsize=8)

    fig.suptitle(f'outlier_detection: what is computed & rejected — {FRAME} @ star {STAR}', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p = f'{OUT}/outlier_value_distributions.png'
    fig.savefig(p, dpi=120); print('[dist] wrote', p, flush=True)


def main():
    run_blot()
    analyze()


if __name__ == '__main__':
    main()
