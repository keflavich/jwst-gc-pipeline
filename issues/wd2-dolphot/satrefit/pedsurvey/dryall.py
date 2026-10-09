"""#1148 review request 2: dry pass of the ZEROFRAME rim rewrite, measured curve (SATSTAR_ZF_R_HEADER=0)
against the header rate (=1), on every surveyed exposure-1 frame plus w51 (the reviewer's candidate).

Per frame: the logged bright-end R used by the curve and its faint-end R, both / R_header; the crf/group-0
pedestal B (as in pedpred.py); SATURATED rim and dilation-buffer pixels rewritten under each rate; and the
median on/off ratio of the rewritten SAT rim value in group-0 bins.  PR branch code (jwst-gc-pipeline-rcexcl,
353ac7df), cap on, read-only.  Output: dryall.json (one row per frame, rewritten as rows land);
``python dryall.py i n`` runs frames i, i + n, ... into dryall_<i>.json (run_dryall.sh).
"""
import contextlib
import glob
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
import pedsurvey as P  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
W51 = '/orange/adamginsburg/jwst/w51'
W51_BANDS = ['F140M', 'F162M', 'F182M', 'F187N', 'F210M', 'F335M', 'F360M', 'F405N', 'F410M', 'F480M']
SW = ['nrca1', 'nrca2', 'nrca3', 'nrca4', 'nrcb1', 'nrcb2', 'nrcb3', 'nrcb4']
GB = [(0, 2000), (2000, 5000), (5000, 15000), (15000, 70000)]
RE_CURVE = re.compile(r'R_faint=([0-9.eE+-]+) \(g0~(\d+)\)')
RE_USED = re.compile(r'R used at bright end=([0-9.eE+-]+)')
RE_HDR = re.compile(r'measured bright-end R=([0-9.eE+-]+) is')


def w51_frames():
    L = []
    for band in W51_BANDS:
        dets = SW if int(band[1:4]) < 250 else ['nrcalong', 'nrcblong']
        for det in dets:
            c = sorted(glob.glob(f'{W51}/{band}/pipeline/jw*_00001_{det}_destreak_*_crf.fits'))
            if c:
                L.append(('w51', band, det, c[0]))
    return L


def run(fn, flag):
    os.environ['SATSTAR_ZF_R_HEADER'] = flag
    with fits.open(fn, memmap=False) as fh:
        header = fh[0].header
        dq = S.correct_dq_first_group_saturation(np.array(fh['DQ'].data), fn, header.get('INSTRUME', ''))
        data = np.array(fh['SCI'].data, float)
        data[np.isnan(fh['VAR_POISSON'].data)] = 0
        photmjsr = fh['SCI'].header.get('PHOTMJSR', header.get('PHOTMJSR'))
    sw = S.satstar_fit_switches()
    zf = S._find_zeroframe_for(fn) if S._zeroframe_fit_enabled() else None
    if zf is None:
        return None
    ff = S._find_first_frame_for(fn) if sw['first_frame'] else None
    g0s = None
    if sw['g0_groupdq'] or ff is not None:
        g0s = S._find_group0_saturation_for(fn, do_not_use=(ff is not None))
    Rh = S.zeroframe_header_R(header, photmjsr)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out, deep, rim, _ = S.zeroframe_fit_anchor(data.copy(), dq, zf, group0_saturated=g0s,
                                                  first_frame=ff, R_header=Rh)
    log = buf.getvalue()
    sys.stdout.write(log)
    m = RE_CURVE.search(log)
    used = RE_USED.findall(log)          # the satcheck rebuild logs a second, final value
    return dict(data=data, out=out, rim=rim, dq=dq, zf=zf, Rh=Rh, log=log,
                R_faint=float(m.group(1)) if m else float('nan'), g_faint=int(m.group(2)) if m else -1,
                R_used=float(used[-1]) if used else float('nan'), rebuilt='curve rebuilt' in log,
                destrkmd=header.get('DESTRKMD'))


def one(field, band, det, fn):
    a = run(fn, '0')
    if a is None:
        return dict(field=field, band=band, det=det, file=os.path.basename(fn), skip='no zeroframe')
    h = run(fn, '1')
    zf, Rh, crf, dq = a['zf'], a['Rh'], a['data'], a['dq']
    sat = (dq & dqflags.pixel['SATURATED']) != 0
    dnu = (dq & dqflags.pixel['DO_NOT_USE']) != 0
    edt = ndimage.distance_transform_edt(~sat)
    far = np.isfinite(crf) & np.isfinite(zf) & ~sat & ~dnu & (edt >= 25) & (crf != 0)
    s = far & (zf >= 500) & (zf < 2000)
    B = float(np.median(crf[s] / Rh - zf[s])) if s.sum() >= 50 else float('nan')
    s0, s1 = a['rim'] & sat, h['rim'] & sat
    b0, b1 = a['rim'] & ~sat, h['rim'] & ~sat
    both = s0 & s1 & (a['out'] > 0)
    gbins = []
    for lo, hi in GB:
        mm = both & (zf >= lo) & (zf < hi)
        gbins.append([float(np.median(h['out'][mm] / a['out'][mm])) if mm.any() else float('nan'), int(mm.sum())])
    hdr_line = 'R_header=' in h['log']
    mh = RE_HDR.search(h['log'])
    return dict(field=field, band=band, det=det, file=os.path.basename(fn), Rh=float(Rh) if Rh else float('nan'),
                R_used=a['R_used'], R_faint=a['R_faint'], g_faint=a['g_faint'], rebuilt=a['rebuilt'],
                hdr_logged=bool(hdr_line), hdr_ratio=float(mh.group(1)) / float(Rh) if (mh and Rh) else float('nan'),
                B=B, nB=int(s.sum()), destrkmd=a['destrkmd'],
                g0rim=float(np.median(zf[both])) if both.any() else float('nan'),
                sat_off=int(s0.sum()), sat_on=int(s1.sum()), buf_off=int(b0.sum()), buf_on=int(b1.sum()),
                buf_only_off=int((b0 & ~b1).sum()), buf_only_on=int((b1 & ~b0).sum()), sat_ratio_bins=gbins,
                sat_ratio_all=float(np.median(h['out'][both] / a['out'][both])) if both.any() else float('nan'))


if __name__ == '__main__':
    todo = w51_frames() + P.frames()
    tag = ''
    if len(sys.argv) == 3:               # worker i of n: frames i, i + n, ... -> dryall_<i>.json
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
            print(f"ROW {field} {band} {det} R_used/Rh={r['R_used'] / r['Rh']:.4f} R_faint/Rh={r['R_faint'] / r['Rh']:.4f} "
                  f"B={r['B']:+.0f} sat {r['sat_off']}/{r['sat_on']} buf {r['buf_off']}/{r['buf_on']} "
                  f"on/off={r['sat_ratio_all']:.4f} g0rim={r['g0rim']:.0f}", flush=True)
        with open(OUT + f'/dryall{tag}.json', 'w') as fh:
            json.dump(res, fh)
    print('ALLDONE', len(res), flush=True)
