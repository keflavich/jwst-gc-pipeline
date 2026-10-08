"""Per-visit refit of the corrected-pair linear term (SIGN=-1 pairs from pairsm_*.pkl)."""
import glob, os, pickle, re, sys
from collections import defaultdict
import numpy as np
from stdatamodels.jwst import datamodels

DETS = ['nrca1', 'nrca2', 'nrca3', 'nrca4', 'nrcb1', 'nrcb2', 'nrcb3', 'nrcb4']
CASES = {  # tag: (field root, band, prog-tag)
    'brick_f200w_v_f187n': ('brick', 'F200W'), 'brick_f182m_v_f187n': ('brick', 'F182M'),
    **{f'brickL_f200w_v_f182m_{d}': ('brick', 'F200W') for d in DETS},
    'ngc_f200w6778_v_f187n': ('ngc6334', 'F200W'), 'ngc_f200w7213_v_f182m': ('ngc6334', 'F200W'),
}


def fitJ(px, py, rx, ry):
    A = np.vstack([np.ones_like(px), px, py]).T
    m = np.ones(len(px), bool)
    for _ in range(5):
        cx, *_ = np.linalg.lstsq(A[m], rx[m], rcond=None)
        cy, *_ = np.linalg.lstsq(A[m], ry[m], rcond=None)
        ex, ey = rx - A @ cx, ry - A @ cy
        s = 1.4826 * np.median(np.abs(np.r_[ex[m], ey[m]]))
        m = (np.abs(ex) < 3 * s) & (np.abs(ey) < 3 * s)
    return np.array([[cx[1], cx[2]], [cy[1], cy[2]]])


def Cmat(w):
    r0, d0 = w(1023.5, 1023.5)
    rx1, dx1 = w(1024.5, 1023.5)
    ry1, dy1 = w(1023.5, 1024.5)
    c = np.cos(np.deg2rad(d0))
    return np.array([[(rx1 - r0) * c, (ry1 - r0) * c], [dx1 - d0, dy1 - d0]]) * 3.6e6


out = []
for tag, (field, band) in CASES.items():
    f = f'pairsm_{tag}.pkl'
    if not os.path.exists(f):
        continue
    R = f'/orange/adamginsburg/jwst/{field}'
    P = pickle.load(open(f, 'rb'))
    res = {}
    for (b, det), lst in P.items() if False else [(k, v) for k, v in P.items() if k != 'anchor']:
        groups = defaultdict(list)
        for e in lst:
            m = re.search(r'visit(\d+)_vgroup([0-9a-z]+)_exp(\d+)_', e['name'])
            pr = re.search(r'_j(\d+)_', e['name'])
            groups[(pr.group(1).zfill(5) if pr else '*', m.group(1), m.group(2))].append(e)
        for (prog, vis, vg), es in sorted(groups.items()):
            m0 = re.search(r'exp(\d+)_', es[0]['name'])
            crf = glob.glob(f'{R}/{band}/pipeline/jw{prog}*{vis}_{vg}_{m0.group(1)}_{det}_*_o00?_crf.fits')[0]
            with datamodels.open(crf) as dm:
                C = Cmat(dm.meta.wcs)
            x = np.concatenate([e['x'] for e in es]); y = np.concatenate([e['y'] for e in es])
            rx = np.concatenate([e['rx'] for e in es]); ry = np.concatenate([e['ry'] for e in es])
            M = np.linalg.solve(C, fitJ(x, y, rx, ry))
            line = f'{tag} {det} visit{vis} frames={len(es)} pairs={len(x)} scale={(M[0,0]+M[1,1])/2*1e6:+7.1f} ppm rot={(M[1,0]-M[0,1])/2*206265:+6.2f}"'
            print(line, flush=True)
            out.append(line)
open('pervisit.txt', 'w').write('\n'.join(out) + '\n')
