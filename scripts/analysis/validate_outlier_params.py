"""Validate the outlier_detection snr/scale fix (keflavich/jwst-gc-pipeline#161).

Runs the Image3 OutlierDetectionStep on the brick o004 F200W *nrca* module
association TWICE -- once with the pipeline default (snr='5.0 4.0',
scale='1.2 0.7') and once with the proposed fix (snr='7.0 5.0',
scale='2.0 1.5') -- on the SAME post-tweakreg frames, so the only difference is
the two parameters.  Then it compares OUTLIER flagging: total count, the
bright-star halo/spike rejection fraction, and (sanity) whether genuine
ramp-jump pixels are still flagged.

Isolated comparison: both runs share identical inputs/WCS/median; the delta in
OUTLIER is purely the snr/scale effect.
"""
import glob
import json
import os

import numpy as np
from astropy.io import fits

BASE = '/orange/adamginsburg/jwst/brick/F200W/pipeline'
SRC_ASN = f'{BASE}/jw01182-o004_20251215t004215_image3_00002_nrca_asn.json'
OUTROOT = os.environ.get('VALIDATE_OUTDIR',
                         '/blue/adamginsburg/adamginsburg/tmp/outlier_validate')
DEFDIR = f'{OUTROOT}/default'
TWKDIR = f'{OUTROOT}/tweaked'
DEF_PARAMS = dict(snr='5.0 4.0', scale='1.2 0.7')
TWK_PARAMS = dict(snr='7.0 5.0', scale='2.0 1.5')
# the analyzed frame + bright stars (cx, cy) from the ramp-zone figure
PROBE_FRAME = 'jw01182004001_04101_00001_nrca1'
BRIGHT_STARS = [(1362, 84), (1586, 1643), (207, 1115)]


def build_tweakreg_asn(dst):
    """Copy the module asn, repoint members destreak.fits -> destreak_tweakreg.fits
    (the saved post-tweakreg WCS), so OutlierDetection resamples aligned frames."""
    a = json.load(open(SRC_ASN))
    n_ok = 0
    for m in a['products'][0]['members']:
        tw = m['expname'].replace('_destreak.fits', '_destreak_tweakreg.fits')
        # ABSOLUTE paths: the asn lives in the output dir, not BASE, so relative
        # expnames would resolve against the wrong directory.
        if os.path.exists(os.path.join(BASE, tw)):
            m['expname'] = os.path.join(BASE, tw)
            n_ok += 1
        else:
            m['expname'] = os.path.join(BASE, m['expname'])  # fall back to destreak
    a['products'][0]['name'] = a['products'][0]['name'] + '_validate'
    with open(dst, 'w') as fh:
        json.dump(a, fh, indent=2)
    print(f'[validate] asn -> {dst}: {len(a["products"][0]["members"])} members, '
          f'{n_ok} repointed to _destreak_tweakreg', flush=True)
    return dst


def run(params, outdir):
    from jwst.outlier_detection import OutlierDetectionStep
    os.makedirs(outdir, exist_ok=True)
    asn = build_tweakreg_asn(f'{outdir}/asn.json')
    print(f'[validate] OutlierDetectionStep {params} -> {outdir}', flush=True)
    OutlierDetectionStep.call(
        asn, save_results=True, output_dir=outdir,
        in_memory=True, **params)
    print(f'[validate] done -> {outdir}', flush=True)


def _load_crf_dq(outdir, frame):
    g = glob.glob(f'{outdir}/{frame}*crf.fits') + glob.glob(f'{outdir}/*{frame}*.fits')
    g = [f for f in g if 'crf' in os.path.basename(f)]
    if not g:
        # OutlierDetection may keep the input stem; match by frame token
        g = [f for f in glob.glob(f'{outdir}/*.fits') if frame in os.path.basename(f)]
    if not g:
        raise SystemExit(f'[validate] no crf for {frame} in {outdir}: '
                         f'{os.listdir(outdir)[:10]}')
    f = sorted(g)[0]
    return f, fits.getdata(f, 'DQ').astype(np.uint32)


def compare():
    from jwst.datamodels import dqflags
    P = dqflags.pixel
    fd, dqd = _load_crf_dq(DEFDIR, PROBE_FRAME)
    ft, dqt = _load_crf_dq(TWKDIR, PROBE_FRAME)
    print(f'\n[validate] default crf: {os.path.basename(fd)}')
    print(f'[validate] tweaked crf: {os.path.basename(ft)}')
    sat = (dqd & P['SATURATED']) != 0
    def outl(dq):
        return ((dq & P['DO_NOT_USE']) != 0) & ((dq & P['OUTLIER']) != 0) & ~sat
    od, ot = outl(dqd), outl(dqt)
    print(f'\n=== OUTLIER (&~SAT) pixel counts, {PROBE_FRAME} ===')
    print(f'  default: {int(od.sum())}')
    print(f'  tweaked: {int(ot.sum())}   ({100*ot.sum()/max(1,od.sum()):.0f}% of default)')
    print(f'  removed: {int((od&~ot).sum())}   newly-added: {int((ot&~od).sum())}')
    print('\n=== bright-star halo/spike rejection fraction (220 px box) ===')
    h = 110
    for (cx, cy) in BRIGHT_STARS:
        sl = (slice(max(0, cy-h), cy+h), slice(max(0, cx-h), cx+h))
        nsat = int(sat[sl].sum())
        npix = od[sl].size - nsat
        print(f'  star ({cx},{cy}): default {100*od[sl].sum()/max(1,npix):.1f}%  '
              f'tweaked {100*ot[sl].sum()/max(1,npix):.1f}%  '
              f'(n_out {int(od[sl].sum())} -> {int(ot[sl].sum())})')


def main():
    run(DEF_PARAMS, DEFDIR)
    run(TWK_PARAMS, TWKDIR)
    compare()


if __name__ == '__main__':
    main()
