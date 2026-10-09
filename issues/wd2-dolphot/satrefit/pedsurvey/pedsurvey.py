"""Pedestal between the crf and group 0: does the crf carry a level that group 0 lacks?

For each frame of survey_fields.json (brick, sgrb2, sgrc exposure 1) and the wd2 exposure-1 frames:
  off      = median(cal - crf) over unsaturated pixels [MJy/sr], i.e. the level removed after cal (destreak, skymatch);
  off_dn   = off / R_header  [group-0 DN];
  fit      = crf = a g0 + b on far-field bin medians (edt >= 25, g0 500-4000), reported as a / R_header and -b / R_header;
  fitcal   = the same fit on the cal SCI;
  g0rim    = median g0 of SATURATED pixels touching an unsaturated pixel (g0 < 60000);
  rimbias  = off_dn / g0rim, the fraction by which R_header g0 overshoots the crf level at the rim.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit')
from satrefit_core import S, fits, ndimage  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
SF = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit/survey_fields/survey_fields.json'
WD2 = '/orange/adamginsburg/jwst/wd2'
VG = {'F150W': '10101', 'F200W': '12101', 'F250M': '04101', 'F300M': '12101'}
BINS = [(500, 1000), (1000, 1500), (1500, 2000), (2000, 3000), (3000, 4000)]


def frames():
    L = [(r['field'], r['band'], r['det'], r['file']) for r in json.load(open(SF))]
    for band in ('F150W', 'F200W'):
        for det in ('nrca1', 'nrca2', 'nrca3', 'nrca4', 'nrcb1', 'nrcb2', 'nrcb3', 'nrcb4'):
            L.append(('wd2', band, det, f'{WD2}/{band}/pipeline/jw03523005001_{VG[band]}_00001_{det}_align_o005_crf.fits'))
    for band in ('F250M', 'F300M'):
        for det in ('nrcalong', 'nrcblong'):
            L.append(('wd2', band, det, f'{WD2}/{band}/pipeline/jw03523005001_{VG[band]}_00001_{det}_align_o005_crf.fits'))
    return L


def calfile(crf):
    d = os.path.dirname(crf)
    stem = os.path.basename(crf).split('_')[:4]
    f = os.path.join(d, '_'.join(stem) + '_cal.fits')
    return f if os.path.exists(f) else None


def binfit(g0, y, m):
    gx, yx = [], []
    for lo, hi in BINS:
        s = m & (g0 >= lo) & (g0 < hi)
        if s.sum() >= 30:
            gx.append(np.median(g0[s]))
            yx.append(np.median(y[s]))
    if len(gx) < 3:
        return float('nan'), float('nan'), len(gx)
    a, b = np.polyfit(gx, yx, 1)
    return float(a), float(b), len(gx)


def one(field, band, det, fn):
    with fits.open(fn, memmap=False) as fh:
        hdr = fh[0].header
        crf = np.array(fh['SCI'].data, float)
        dq = np.array(fh['DQ'].data)
        photmjsr = fh['SCI'].header.get('PHOTMJSR', hdr.get('PHOTMJSR'))
    Rh = S.zeroframe_header_R(hdr, photmjsr)
    with fits.open(S._find_ramp_for(fn), memmap=True) as r:
        g0 = np.array(r['SCI'].data[0, 0], float)
    sat = (dq & S.dqflags.pixel['SATURATED']) != 0
    dnu = (dq & 1) != 0
    edt = ndimage.distance_transform_edt(~sat)
    out = dict(field=field, band=band, det=det, file=os.path.basename(fn), Rh=float(Rh),
               steps={k: hdr.get(k) for k in ('S_SKYMAT', 'S_REFPIX', 'READPATT')})
    cf = calfile(fn)
    cal = None
    if cf is not None:
        with fits.open(cf, memmap=False) as fh:
            cal = np.array(fh['SCI'].data, float)
        ok = np.isfinite(cal) & np.isfinite(crf) & ~sat & ~dnu
        dd = cal - crf
        out['off'] = float(np.median(dd[ok]))
        blk = []
        for y0 in range(0, 2048, 256):
            for x0 in range(0, 2048, 256):
                s = ok[y0:y0 + 256, x0:x0 + 256]
                if s.sum() > 1000:
                    blk.append(np.median(dd[y0:y0 + 256, x0:x0 + 256][s]))
        out['off_blk16'], out['off_blk84'] = [float(x) for x in np.percentile(blk, [16, 84])]
    else:
        out['off'] = out['off_blk16'] = out['off_blk84'] = float('nan')
    out['off_dn'] = out['off'] / Rh
    far = np.isfinite(crf) & np.isfinite(g0) & ~sat & ~dnu & (edt >= 25)
    a, b, nb = binfit(g0, crf, far)
    out['slope'], out['ped_dn'], out['nbins'] = a / Rh, -b / Rh, nb
    if cal is not None:
        a, b, _ = binfit(g0, cal, far & np.isfinite(cal))
        out['slope_cal'], out['ped_cal_dn'] = a / Rh, -b / Rh
    else:
        out['slope_cal'] = out['ped_cal_dn'] = float('nan')
    rim = sat & ndimage.binary_dilation(~sat) & np.isfinite(g0) & (g0 < 60000)
    out['nrim'] = int(rim.sum())
    out['g0rim'] = float(np.median(g0[rim])) if rim.any() else float('nan')
    out['rimbias'] = out['off_dn'] / out['g0rim']
    out['rimbias_fit'] = out['ped_dn'] / out['g0rim']
    return out


if __name__ == '__main__':
    res = []
    for field, band, det, fn in frames():
        if not os.path.exists(fn) or S._find_ramp_for(fn) is None:
            print('SKIP missing', field, band, det, fn, flush=True)
            continue
        r = one(field, band, det, fn)
        res.append(r)
        print(f"{field:6s} {band} {det:9s} off={r['off']:+.3f} ({r['off_blk16']:+.2f}..{r['off_blk84']:+.2f}) "
              f"off_dn={r['off_dn']:+.0f} slope={r['slope']:.3f} ped={r['ped_dn']:+.0f} "
              f"slope_cal={r['slope_cal']:.3f} ped_cal={r['ped_cal_dn']:+.0f} g0rim={r['g0rim']:.0f} "
              f"rimbias={r['rimbias']:+.4f} {r['steps']}", flush=True)
        json.dump(res, open(OUT + '/pedsurvey.json', 'w'), default=str)
    print('ALLDONE', len(res), flush=True)
