"""Cross-exposure consistency of satstar fits, arm s0 (seed-core filter off) vs s1 (on), gc-treasury F480M o040.
A star's sky position and flux are the same in all 6 dithered exposures; static bad pixels move relative to the
star between dithers, so a seed pulled onto them shows up as scatter.  usage: cross_exp.py [armA armB]  (default s0 s1)"""
import glob, os, sys, warnings
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
warnings.filterwarnings('ignore')
S = os.path.dirname(os.path.abspath(__file__))
FR = [f'jw10678040001_02101_0000{e}_nrcalong' for e in range(1, 7)]
TOL = 0.25 * u.arcsec


def load(arm, fr):
    p = f'{S}/tree_{arm}/F480M/pipeline/{fr}_destreak_o040_crf_rctest_satstar_catalog.fits'
    if not os.path.exists(p):
        return None
    t = Table.read(p)
    if 'reject_reason' in t.colnames:
        rr = np.array([str(v) if v is not np.ma.masked else '' for v in t['reject_reason']])
        load.nrej[arm] = load.nrej.get(arm, 0) + int(np.sum((rr != '') & (rr != '--')))
        t = t[(rr == '') | (rr == '--')]
    t['sx'] = t['x_init'] + (t['x_0'] - t['x_fit'])      # seed in frame pixels
    t['sy'] = t['y_init'] + (t['y_0'] - t['y_fit'])
    return t


load.nrej = {}


def stars(arm):
    """Group rows of all exposures by sky position (to exposure-1 anchors, then any unmatched as new anchors)."""
    tabs = [load(arm, f) for f in FR]
    anchors = None; rows = []
    for k, t in enumerate(tabs):
        if t is None:
            continue
        sc = SkyCoord(t['skycoord_fit'])
        if anchors is None:
            anchors = sc; ids = np.arange(len(sc))
        else:
            i, d, _ = sc.match_to_catalog_sky(anchors)
            ids = np.where(d < TOL, i, -1)
            new = ids < 0
            ids[new] = len(anchors) + np.arange(new.sum())
            anchors = SkyCoord(np.concatenate([anchors.ra.deg, sc.ra.deg[new]]) * u.deg,
                               np.concatenate([anchors.dec.deg, sc.dec.deg[new]]) * u.deg)
        for j in range(len(t)):
            r = t[j]
            rows.append((ids[j], k, sc[j].ra.deg, sc[j].dec.deg, float(r['flux_fit']), int(r['flags']),
                         float(r['qfit']), float(r['sx']), float(r['sy'])))
    return anchors, np.array(rows)


def per_star(rows, nmin=4):
    out = {}
    for sid in np.unique(rows[:, 0]):
        r = rows[rows[:, 0] == sid]
        if len(np.unique(r[:, 1])) < nmin or len(r) != len(np.unique(r[:, 1])):
            continue
        dec0 = np.median(r[:, 3])
        dx = (r[:, 2] - np.median(r[:, 2])) * np.cos(np.deg2rad(dec0)) * 3.6e6
        dy = (r[:, 3] - dec0) * 3.6e6
        f = r[:, 4]
        mag = -2.5 * np.log10(np.where(f > 0, f, np.nan))
        out[sid] = dict(ra=np.median(r[:, 2]), dec=dec0, n=len(r), pos_rms=np.sqrt(np.mean(dx ** 2 + dy ** 2)),
                        mag_rms=np.nanstd(mag), mag=np.nanmedian(mag), f16=int(np.sum((r[:, 5].astype(int) & 16) > 0)),
                        qfit=np.median(r[:, 6]), exps=set(r[:, 1].astype(int)))
    return out


if __name__ == '__main__':
    a0, a1 = sys.argv[1:3] if len(sys.argv) >= 3 else ('s0', 's1')
    A = {}
    for arm in (a0, a1):
        anc, rows = stars(arm)
        A[arm] = per_star(rows)
        print(f'{arm}: rows {len(rows)}, stars with >= 4 exposures (one row each) {len(A[arm])}, '
              f'rows with flags&16 {int(np.sum((rows[:, 5].astype(int) & 16) > 0))}, rejected rows {load.nrej.get(arm, 0)}')
    # pair stars across arms by sky position
    k0 = list(A[a0]); k1 = list(A[a1])
    c0 = SkyCoord([A[a0][k]['ra'] for k in k0] * u.deg, [A[a0][k]['dec'] for k in k0] * u.deg)
    c1 = SkyCoord([A[a1][k]['ra'] for k in k1] * u.deg, [A[a1][k]['dec'] for k in k1] * u.deg)
    i, d, _ = c0.match_to_catalog_sky(c1)
    pairs = [(k0[a], k1[i[a]]) for a in range(len(k0)) if d[a] < TOL]
    P = np.array([(A[a0][a]['pos_rms'], A[a1][b]['pos_rms'], A[a0][a]['mag_rms'], A[a1][b]['mag_rms'],
                   A[a0][a]['f16'], A[a1][b]['f16'], A[a0][a]['mag'], A[a0][a]['n'], A[a1][b]['n'])
                  for a, b in pairs])
    print(f'stars in both arms: {len(P)}')
    ch = np.abs(P[:, 0] - P[:, 1]) > 1.0          # position scatter changed by > 1 mas
    print(f'pos-rms changed by > 1 mas: {ch.sum()}  (better in {a1}: {(P[ch, 1] < P[ch, 0]).sum()}, '
          f'worse: {(P[ch, 1] > P[ch, 0]).sum()})')
    for lab, m in (('all', np.ones(len(P), bool)), ('changed', ch)):
        if m.sum():
            print(f'  {lab:8s} N={m.sum():4d} median pos_rms {a0} {np.median(P[m, 0]):6.1f} {a1} {np.median(P[m, 1]):6.1f} mas; '
                  f'median mag_rms {a0} {np.nanmedian(P[m, 2]):.3f} {a1} {np.nanmedian(P[m, 3]):.3f}; '
                  f'flags16 rows {a0} {int(P[m, 4].sum())} {a1} {int(P[m, 5].sum())}')
    big = ch & ((P[:, 0] > 30) | (P[:, 1] > 30))
    print(f'  changed with pos_rms > 30 mas in either arm: {big.sum()}  better {(P[big, 1] < P[big, 0]).sum()} '
          f'worse {(P[big, 1] > P[big, 0]).sum()}')
    np.save(f'{S}/pairs_{a0}_{a1}.npy', P)
