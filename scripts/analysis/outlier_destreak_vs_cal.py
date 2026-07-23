"""Does the outlier over-rejection depend on destreaking? (keflavich/jwst-gc-pipeline#161)

Runs OutlierDetectionStep (default params) on the brick o004 F200W nrca frames in
their NON-destreaked `_cal` form, but grafted with the SAME post-tweakreg WCS as
the destreak run, so the ONLY difference from the earlier destreak validation is
the 1/f-stripe removal. Compares the bright-star halo OUTLIER fraction against the
destreak run to see if destreaking causes/worsens the over-rejection.
"""
import glob, json, os
import numpy as np
from astropy.io import fits
import jwst.datamodels as dm
from jwst.datamodels import dqflags

BASE = '/orange/adamginsburg/jwst/brick/F200W/pipeline'
SRC_ASN = f'{BASE}/jw01182-o004_20251215t004215_image3_00002_nrca_asn.json'
OUT = os.environ.get('CALTEST_OUTDIR', '/blue/adamginsburg/adamginsburg/tmp/outlier_caltest')
GRAFT = f'{OUT}/graft'          # cal SCI + tweakreg WCS
DEST_DEFAULT = '/blue/adamginsburg/adamginsburg/tmp/outlier_validate/default'  # destreak baseline
FRAME = 'jw01182004001_04101_00001_nrca1'
STAR = (1362, 84)
P = dqflags.pixel


def graft_cal_onto_tweakreg():
    """For each member, write a model with the _cal (non-destreak) science arrays
    but the _destreak_tweakreg GWCS + meta, so alignment is identical."""
    os.makedirs(GRAFT, exist_ok=True)
    a = json.load(open(SRC_ASN))
    members = a['products'][0]['members']
    out_members = []
    for m in members:
        stem = m['expname'].replace('_destreak.fits', '')
        tw = f'{BASE}/{stem}_destreak_tweakreg.fits'
        cal = f'{BASE}/{stem}_cal.fits'
        if not (os.path.exists(tw) and os.path.exists(cal)):
            print(f'  skip {stem} (missing tw/cal)', flush=True)
            continue
        mt = dm.open(tw)          # has tweakreg WCS
        mc = dm.open(cal)         # non-destreak science
        mt.data = mc.data
        mt.err = mc.err
        mt.dq = mc.dq
        outp = f'{GRAFT}/{os.path.basename(stem)}_calwcs.fits'
        mt.save(outp)
        out_members.append({'expname': outp, 'exptype': 'science'})
        mt.close(); mc.close()
    a['products'][0]['members'] = out_members
    a['products'][0]['name'] = a['products'][0]['name'] + '_caltest'
    asn = f'{OUT}/asn.json'
    json.dump(a, open(asn, 'w'), indent=2)
    print(f'[caltest] grafted {len(out_members)} cal-with-tweakreg-WCS frames', flush=True)
    return asn


def run(asn):
    from jwst.outlier_detection import OutlierDetectionStep
    OutlierDetectionStep.call(asn, save_results=True, in_memory=True, output_dir=OUT,
                              snr='5.0 4.0', scale='1.2 0.7')


def compare():
    def dq_of(d):
        f = sorted(x for x in glob.glob(f'{d}/*.fits')
                   if FRAME + '_' in os.path.basename(x)
                   and ('crf' in x or 'outlierdetectionstep' in x))[0]
        return fits.getdata(f, 'DQ').astype(np.uint32)
    dq_cal = dq_of(OUT)
    dq_dest = dq_of(DEST_DEFAULT)
    sat = (dq_dest & P['SATURATED']) != 0
    o_cal = ((dq_cal & P['OUTLIER']) != 0) & ~sat
    o_dest = ((dq_dest & P['OUTLIER']) != 0) & ~sat
    print(f'\n=== OUTLIER(&~SAT) {FRAME}: destreak vs cal (default params) ===')
    print(f'  destreak: {int(o_dest.sum())}')
    print(f'  cal     : {int(o_cal.sum())}   ({100*o_cal.sum()/max(1,o_dest.sum()):.0f}% of destreak)')
    cx, cy, h = STAR[0], STAR[1], 110
    sl = (slice(max(0, cy-h), cy+h), slice(max(0, cx-h), cx+h))
    npix = o_dest[sl].size - int(sat[sl].sum())
    print(f'  bright-star halo (220px box): destreak {100*o_dest[sl].sum()/npix:.1f}%  '
          f'cal {100*o_cal[sl].sum()/npix:.1f}%')


def main():
    os.makedirs(OUT, exist_ok=True)
    asn = graft_cal_onto_tweakreg()
    run(asn)
    compare()


if __name__ == '__main__':
    main()
