"""Load the per-frame star npz files of one detector into Stars objects and a table."""
import glob
import os

import numpy as np

from epsf import Stars

PER_FRAME = ['fname', 'detector', 'filter', 'expstart', 'date_obs', 'obs', 'visit', 'expnum',
             'photmjsr', 'tgroup', 'ngroups', 'nints']
PER_STAR = ['x0', 'y0', 'ix', 'iy', 'amp', 'bkg0', 'snr', 'satfrac', 'maskfrac', 'ra0', 'dec0', 'jac']


def load(stampdir, wing=False, max_files=None):
    files = sorted(glob.glob(os.path.join(stampdir, '*_stars.npz')))[:max_files]
    tab = {k: [] for k in PER_STAR + PER_FRAME + ['frame']}
    S, E, M, W = [], [], [], []
    frames = []
    for fi, fn in enumerate(files):
        d = np.load(fn)
        n = len(d['x0'])
        if wing:
            sel = d['wing_idx']
            R = int(d['r_wing'])
            s, e, m = d['wing_s'], d['wing_e'], d['wing_m']
        else:
            sel = np.arange(n)
            R = int(d['r_core'])
            s, e, m = d['core_s'], d['core_e'], d['core_m']
        if len(sel) == 0:
            continue
        S.append(s); E.append(e)
        M.append(np.unpackbits(m, axis=1)[:, :(2 * R + 1) ** 2].reshape(-1, 2 * R + 1, 2 * R + 1).astype(bool))
        for k in PER_STAR:
            tab[k].append(d[k][sel])
        for k in PER_FRAME:
            tab[k].append(np.repeat(d[k][()], len(sel)))
        tab['frame'].append(np.full(len(sel), fi))
        frames.append(os.path.basename(fn))
    tab = {k: np.concatenate(v) for k, v in tab.items()}
    stars = Stars(np.concatenate(S), np.concatenate(E), np.concatenate(M),
                  tab['x0'], tab['y0'], tab['ix'], tab['iy'], R)
    return stars, tab, frames
