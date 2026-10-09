"""#1148: the cal - crf level D (per row and amplifier, ``zeroframe_cal_offset``) under the header rate.

Per frame (w51 exposure 1 and the surveyed frames of dryall.py):
* D statistics, and the far-field (>= 25 px from SATURATED) crf / (R_header g0) and (crf + D) / (R_header g0)
  in group-0 bins, with the pedestal B before and after adding D;
* dry passes of ``zeroframe_fit_anchor`` with the measured curve (SATSTAR_ZF_R_HEADER=0), the header rate
  without D (353ac7df) and the header rate with D: SATURATED-rim and buffer counts, and the median rewritten
  SAT-rim value of each header variant over the curve's, binned by group 0.
PR branch code (jwst-gc-pipeline-rcexcl), cap on, read-only.  ``python dcorr.py i n`` runs frames i, i + n, ...
into dcorr_<i>.json (run_dcorr.sh).
"""
import contextlib
import io
import json
import os
import re
import sys

import numpy as np

os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
os.environ['NIRCAM_SATSTAR_RECOVERED_CAP'] = '1'
os.environ.pop('SATSTAR_ZF_KEEP_FINITE', None)
sys.path.insert(0, '/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline-rcexcl')
from astropy.io import fits  # noqa: E402
from scipy import ndimage  # noqa: E402
from jwst_gc_pipeline.reduction import saturated_star_finding as S  # noqa: E402
from stdatamodels.jwst.datamodels import dqflags  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dryall  # noqa: E402
import pedsurvey as P  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
GB = [(0, 1000), (1000, 2000), (2000, 5000), (5000, 15000), (15000, 70000)]
FB = [(150, 300), (300, 500), (500, 1000), (1000, 2000), (2000, 3000), (3000, 4000)]
CB = [(300, 500), (500, 1000), (1000, 2000), (2000, 3000), (3000, 4000), (4000, 6000), (6000, 15000)]
RE_HDR = re.compile(r'measured bright-end R=([0-9.eE+-]+) is')


def anchor(data, dq, zf, g0s, ff, Rh, flag, off):
    os.environ['SATSTAR_ZF_R_HEADER'] = flag
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out, deep, rim, _ = S.zeroframe_fit_anchor(data.copy(), dq, zf, group0_saturated=g0s, first_frame=ff,
                                                  R_header=Rh, cal_offset=off)
    return out, rim, buf.getvalue()


def one(field, band, det, fn):
    with fits.open(fn, memmap=False) as fh:
        header = fh[0].header
        raw = np.array(fh['SCI'].data, float)
        dq0 = np.array(fh['DQ'].data)
        dq = S.correct_dq_first_group_saturation(dq0.copy(), fn, header.get('INSTRUME', ''))
        data = raw.copy()
        data[np.isnan(fh['VAR_POISSON'].data)] = 0
        photmjsr = fh['SCI'].header.get('PHOTMJSR', header.get('PHOTMJSR'))
    zf = S._find_zeroframe_for(fn)
    if zf is None:
        return dict(field=field, band=band, det=det, skip='no zeroframe')
    ff = S._find_first_frame_for(fn)
    g0s = S._find_group0_saturation_for(fn, do_not_use=(ff is not None))
    Rh = S.zeroframe_header_R(header, photmjsr)
    off = S.zeroframe_cal_offset(fn, raw, dq0, photmjsr=photmjsr)
    D = np.zeros_like(raw) if off is None else off
    sat = (dq0 & dqflags.pixel['SATURATED']) != 0
    dnu = (dq0 & dqflags.pixel['DO_NOT_USE']) != 0
    edt = ndimage.distance_transform_edt(~sat)
    far = np.isfinite(raw) & np.isfinite(zf) & ~sat & ~dnu & (edt >= 25) & (raw != 0)
    s = far & (zf >= 500) & (zf < 2000)
    res = dict(field=field, band=band, det=det, file=os.path.basename(fn), Rh=float(Rh), destrkmd=header.get('DESTRKMD'),
               cal=S._find_cal_for(fn) is not None, has_D=off is not None,
               D_p=[float(x) for x in np.percentile(D, [1, 16, 50, 84, 99])],
               D_far=float(np.median(D[s])) if s.sum() else float('nan'),
               B=float(np.median(raw[s] / Rh - zf[s])) if s.sum() >= 50 else float('nan'),
               B_D=float(np.median((raw[s] + D[s]) / Rh - zf[s])) if s.sum() >= 50 else float('nan'), nB=int(s.sum()))
    fr = []
    for lo, hi in FB:
        m = far & (zf >= lo) & (zf < hi)
        if m.sum() >= 30:
            fr.append([float(np.median(zf[m])), float(np.median(raw[m] / (Rh * zf[m]))),
                       float(np.median((raw[m] + D[m]) / (Rh * zf[m]))), int(m.sum())])
    res['far'] = fr
    c_out, c_rim, c_log = anchor(data, dq, zf, g0s, ff, Rh, '0', None)
    h_out, h_rim, h_log = anchor(data, dq, zf, g0s, ff, Rh, '1', None)
    d_out, d_rim, d_log = anchor(data, dq, zf, g0s, ff, Rh, '1', off)
    mh = RE_HDR.search(h_log)
    res['hdr_ratio'] = float(mh.group(1)) / float(Rh) if mh else float('nan')
    res['d_logged'] = 'minus the cal - crf level' in d_log
    for tag, rim in (('curve', c_rim), ('hdr', h_rim), ('hdrD', d_rim)):
        res[f'sat_{tag}'] = int((rim & sat).sum())
        res[f'buf_{tag}'] = int((rim & ~sat).sum())
    both = c_rim & h_rim & d_rim & sat & (c_out > 0)
    res['g0rim'] = float(np.median(zf[both])) if both.any() else float('nan')
    for tag, o in (('hdr', h_out), ('hdrD', d_out)):
        bins = []
        for lo, hi in GB:
            mm = both & (zf >= lo) & (zf < hi)
            bins.append([float(np.median(o[mm] / c_out[mm])) if mm.any() else float('nan'), int(mm.sum())])
        res[f'ratio_{tag}'] = bins
        res[f'ratio_{tag}_all'] = float(np.median(o[both] / c_out[both])) if both.any() else float('nan')
    # per group-0 bin: rewritten SAT rim / (R_header g0) for each variant, against the far-field crf / (R_header g0)
    # and the SATURATED pixels the anchor kept (finite crf, KEEP_FINITE), whose crf is a ramp-fit rate
    kept = sat & np.isfinite(raw) & (raw != 0) & ~dnu & ~c_rim & np.isfinite(zf) & (zf > 0)
    cmp = []
    for lo, hi in CB:
        mm = both & (zf >= lo) & (zf < hi)
        mf = far & (zf >= lo) & (zf < hi)
        mk = kept & (zf >= lo) & (zf < hi)
        row = dict(lo=lo, hi=hi, n_rim=int(mm.sum()), n_far=int(mf.sum()), n_kept=int(mk.sum()))
        if mm.any():
            for tag, o in (('curve', c_out), ('hdr', h_out), ('hdrD', d_out)):
                row[tag] = float(np.median(o[mm] / (Rh * zf[mm])))
            row['D_rim'] = float(np.median(D[mm] / (Rh * zf[mm])))
        if mf.sum() >= 30:
            row['far'] = float(np.median(raw[mf] / (Rh * zf[mf])))
            row['farD'] = float(np.median((raw[mf] + D[mf]) / (Rh * zf[mf])))
        if mk.sum() >= 10:
            row['kept'] = float(np.median(raw[mk] / (Rh * zf[mk])))
            row['keptD'] = float(np.median((raw[mk] + D[mk]) / (Rh * zf[mk])))
        cmp.append(row)
    res['cmp'] = cmp
    if fr and np.isfinite(res['g0rim']):
        g = np.array([x[0] for x in fr])
        res['far_at_rim'] = float(np.interp(res['g0rim'], g, [x[1] for x in fr]))
        res['farD_at_rim'] = float(np.interp(res['g0rim'], g, [x[2] for x in fr]))
    return res


if __name__ == '__main__':
    todo = dryall.w51_frames() + P.frames()
    tag = ''
    if len(sys.argv) == 3:
        i, n = int(sys.argv[1]), int(sys.argv[2])
        todo, tag = todo[i::n], f'_{i}'
    res = []
    for field, band, det, fn in todo:
        if not os.path.exists(fn) or S._find_ramp_for(fn) is None:
            print('SKIP missing', field, band, det, fn, flush=True)
            continue
        r = one(field, band, det, fn)
        res.append(r)
        if 'skip' in r:
            print('SKIP', field, band, det, r['skip'], flush=True)
        else:
            print(f"ROW {field} {band} {det} D50={r['D_p'][2]:+.3g} Dfar={r['D_far']:+.3g} B={r['B']:+.0f} "
                  f"B_D={r['B_D']:+.0f} sat c/h/hD {r['sat_curve']}/{r['sat_hdr']}/{r['sat_hdrD']} "
                  f"buf {r['buf_curve']}/{r['buf_hdr']}/{r['buf_hdrD']} h/c={r['ratio_hdr_all']:.3f} "
                  f"hD/c={r['ratio_hdrD_all']:.3f} g0rim={r['g0rim']:.0f}", flush=True)
        with open(OUT + f'/dcorr{tag}.json', 'w') as fh:
            json.dump(res, fh)
    print('ALLDONE', len(res), flush=True)
