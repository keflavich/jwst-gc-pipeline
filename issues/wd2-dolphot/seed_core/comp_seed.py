"""Seeds of _refine_coms_by_data's NaN-VAR_POISSON ("genuine core") branch against the dolphot
position, with two candidate gates.  Per SATURATED component (scipy label of DQ & 2) with >= 3
NaN-variance pixels:
  cur  : current rule, COM of the largest NaN-variance sub-cluster (first on ties);
  ero  : the eroded-core COM fallback the function uses otherwise;
  A    : as cur, with OUTLIER-flagged (DQ 16) pixels left out of the core;
  B    : as cur, only when the largest sub-cluster has >= 3 px;
  A2   : as cur, with OUTLIER and static bad-pixel pixels (DEAD, HOT, WARM, LOW_QE, RC, TELEGRAPH,
         NO_LIN_CORR, NO_SAT_CHECK, NO_GAIN_VALUE, NO_FLAT_FIELD, UNRELIABLE_BIAS, OTHER_BAD_PIXEL,
         REFERENCE_PIXEL) left out of the core.
Truth: the brightest dolphot star (any magnitude) inside the component dilated by 1 px.
usage: python comp_seed.py"""
import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy import ndimage
import score_rc as S
from stdatamodels.jwst.datamodels import dqflags
EXCL = 0
for k in ('OUTLIER', 'DEAD', 'HOT', 'WARM', 'LOW_QE', 'RC', 'TELEGRAPH', 'NO_LIN_CORR', 'NO_SAT_CHECK',
          'NO_GAIN_VALUE', 'NO_FLAT_FIELD', 'UNRELIABLE_BIAS', 'OTHER_BAD_PIXEL', 'REFERENCE_PIXEL'):
    EXCL |= dqflags.pixel[k]

def core_seed(core, min_big=1):
    cl, n = ndimage.label(core)
    if n == 0:
        return None, 0
    sz = ndimage.sum_labels(core, cl, np.arange(1, n + 1))
    k = int(np.argmax(sz))
    if sz[k] < min_big:
        return None, int(sz[k])
    return ndimage.center_of_mass(cl == k + 1), int(sz[k])

def ero_seed(clm):
    ys, xs = np.where(clm)
    e = ndimage.binary_erosion(clm, iterations=2)
    if not e.any():
        return ((ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2)
    el, en = ndimage.label(e)
    if en > 1:
        sz = ndimage.sum_labels(e, el, np.arange(1, en + 1))
        e = el == int(np.argmax(sz)) + 1
    return ndimage.center_of_mass(e)

rows = []
ALLCHG = []
for line in open(f'{S.H}/frames.txt'):
    band, fr = line.split()
    F = S.frame_data(band, fr)
    f = fits.open(f'{S.R}/{band}/pipeline/{fr}_align_o005_crf.fits')
    dq = f['DQ'].data
    sat = (dq & 2) != 0
    unrec = np.isnan(f['VAR_POISSON'].data)
    outl = (dq & 16) != 0
    badp = (dq.astype(np.int64) & EXCL) != 0
    nchg = [0, 0]
    lab, n = ndimage.label(sat)
    rx, ry = F['w'].world_to_pixel(F['rsc'])
    ok = np.isfinite(rx) & (rx > 0) & (ry > 0) & (rx < sat.shape[1] - 1) & (ry < sat.shape[0] - 1)
    rl = np.zeros(len(rx), int)
    labd = ndimage.grey_dilation(lab, size=3)
    rl[ok] = labd[np.round(ry[ok]).astype(int), np.round(rx[ok]).astype(int)]
    objs = ndimage.find_objects(lab)
    nun = ndimage.sum_labels(unrec & sat, lab, np.arange(1, n + 1))
    for l in np.where(nun >= 3)[0] + 1:
        sl = objs[l - 1]
        sl = (slice(max(sl[0].start - 3, 0), sl[0].stop + 3), slice(max(sl[1].start - 3, 0), sl[1].stop + 3))
        clm = lab[sl] == l
        oy, ox = sl[0].start, sl[1].start
        cur, big = core_seed(clm & unrec[sl])
        ero = ero_seed(clm)
        a, _ = core_seed(clm & unrec[sl] & ~outl[sl])
        b, _ = core_seed(clm & unrec[sl], min_big=3)
        a2, _ = core_seed(clm & unrec[sl] & ~badp[sl])
        a = a or ero; b = b or ero; a2 = a2 or ero
        sh = float(np.hypot(a2[0] - cur[0], a2[1] - cur[1]))
        nchg[0] += 1; nchg[1] += sh > 0.5
        st = np.where(rl == l)[0]
        if len(st) == 0:
            ALLCHG.append((band, int(clm.sum()), sh))
            continue
        ALLCHG.append((band, int(clm.sum()), sh))
        i = st[np.argmin(F['rm'][st])]
        tx, ty = rx[i] - ox, ry[i] - oy
        d = lambda s: float(np.hypot(s[1] - tx, s[0] - ty))
        rows.append((band, fr.split('_', 1)[1], float(rx[i]), float(ry[i]), F['rm'][i], int(clm.sum()),
                     int(nun[l - 1]), int((clm & unrec[sl] & outl[sl]).sum()), big, d(cur), d(ero), d(a), d(b), d(a2)))
T = Table(rows=rows, names=('band', 'frame', 'x', 'y', 'mag', 'sat_area', 'n_nanvar', 'n_nanvar_outlier',
                            'big', 'd_cur', 'd_ero', 'd_A', 'd_B', 'd_A2'))
for c in ('x', 'y', 'mag', 'd_cur', 'd_ero', 'd_A', 'd_B', 'd_A2'):
    T[c].format = '.2f'
T.write(f'{S.H}/comp_seed.ecsv', overwrite=True)
print(f'{len(T)} components with >= 3 NaN-variance px and a dolphot star')
for c in ('d_cur', 'd_ero', 'd_A', 'd_B', 'd_A2'):
    v = T[c]
    print(f'{c}: median {np.median(v):.2f}  >1.5 px: {(v > 1.5).sum()}  >2 px: {(v > 2).sum()}  >3 px: {(v > 3).sum()}')
print('NaN-variance pixels that are OUTLIER-flagged:', T['n_nanvar_outlier'].sum(), 'of', T['n_nanvar'].sum())
C = np.array([c[2] for c in ALLCHG]); A_ = np.array([c[1] for c in ALLCHG])
print(f'all components with >= 3 NaN-variance px: {len(C)}; A2 moves the seed > 0.5 px on {(C > 0.5).sum()} '
      f'({((C > 0.5) & (A_ < 120)).sum()} with sat_area < 120)')
bad = T[((T['d_cur'] > 1.5) | (T['d_A2'] > 1.5)) & (T['sat_area'] < 400)]
bad.sort('d_cur')
bad.pprint(max_lines=-1, max_width=-1)
