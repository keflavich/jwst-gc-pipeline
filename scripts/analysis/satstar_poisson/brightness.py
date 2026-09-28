"""Relative brightness of the saturated stars used in the brightness sequence (README §7).

F/F_target is the ratio of the star's halo radial profile to the target's, in off-spike
annuli (> 8 deg from every diffraction spike) at r = 40-85 px, background-subtracted
(off-spike median at r = 350-450 px); each profile is the median over that star's
dithers, and F/F_target is the median of the ratio over the four annuli 40-50-60-70-85.
This inner-halo range is used because there the star dominates the crowded GC scene: at
r >~ 100 px the fainter stars' profiles are biased by neighbours and by the background
estimate.  All 10678 visits share one roll, so the spike angles measured on the target
apply to every star.  Also reported: the ratio on the spikes (a second, noisier proxy),
and the sizes of the SATURATED (DQ bit 2) and lost (NaN / DO_NOT_USE) regions connected
to the star centre (median over dithers).

    python brightness.py tgt trn_116 trn_069 ...     (in the directory holding <prefix>_e*.npz)
writes brightness.json."""
import sys, glob, json
import numpy as np
from scipy import ndimage

EDGES = [40, 50, 60, 70, 85, 100, 120, 150, 200]
NUSE = 4                     # annuli used for F/F_target: 40-85 px


def geom(z):
    n = z['d'].shape[0]
    yy, xx = np.mgrid[:n, :n]
    r = np.hypot(xx-z['xt'], yy-z['yt'])
    th = np.degrees(np.arctan2(yy-z['yt'], xx-z['xt'])) % 360
    return r, th


def spike_angles(fn):
    z = np.load(fn); d = z['d']; r, th = geom(z)
    m = (r > 70) & (r < 130) & np.isfinite(d)
    bins = np.arange(0, 360.5, 0.5); idx = np.digitize(th[m], bins)
    ang = np.array([np.nanmedian(d[m][idx == i]) for i in range(1, len(bins))])
    sm = ndimage.maximum_filter1d(ang, 21, mode='wrap')
    pk = np.nonzero((ang == sm) & (ang > np.percentile(ang, 90)))[0]
    return bins[pk]+0.25


def region_at_centre(mask, z):
    lab, _ = ndimage.label(mask)
    c = lab[int(round(float(z['yt']))), int(round(float(z['xt'])))]
    return int((lab == c).sum()) if c > 0 else 0


def profile(fn, sp):
    z = np.load(fn); d = z['d']; dq = z['dq']; r, th = geom(z)
    dth = np.min(np.abs(((th[..., None]-sp[None, None, :])+180) % 360-180), -1)
    ok = np.isfinite(d) & ((dq & 1) == 0)
    bk = np.median(d[ok & (r > 350) & (r < 450) & (dth > 8)])
    ann = list(zip(EDGES[:-1], EDGES[1:]))
    halo = [np.median(d[ok & (r >= a) & (r < b) & (dth > 8)])-bk for a, b in ann]
    spike = [np.median(d[ok & (r >= a) & (r < b) & (dth < 1)])-bk for a, b in ann]
    return np.array(halo), np.array(spike), region_at_centre((dq & 2) > 0, z), region_at_centre(~ok, z), bk


if __name__ == '__main__':
    pres = sys.argv[1:]
    if 'tgt' not in pres:
        pres = ['tgt']+pres
    sp = spike_angles('tgt_e1.npz')
    print('spike angles (deg):', np.round(sp, 1))
    P = {}
    for p in pres:
        R = [profile(f, sp) for f in sorted(glob.glob(f'{p}_e[1-6].npz'))]
        P[p] = dict(halo=np.median([x[0] for x in R], 0), spike=np.median([x[1] for x in R], 0),
                    nsat=int(np.median([x[2] for x in R])), nlost=int(np.median([x[3] for x in R])),
                    bkg=float(np.median([x[4] for x in R])), ndith=len(R))
    T = P['tgt']
    res = {}
    for p in pres:
        hr = P[p]['halo']/T['halo']; sr = P[p]['spike']/T['spike']
        res[p] = dict(F_rel=float(np.median(hr[:NUSE])), F_rel_spread=float(np.ptp(hr[:NUSE])),
                      F_rel_spike=float(np.median(sr[:NUSE])), halo_ratio=hr.tolist(), halo=P[p]['halo'].tolist(),
                      nsat_core=P[p]['nsat'], nlost_core=P[p]['nlost'], bkg=P[p]['bkg'], ndith=P[p]['ndith'])
        print(f"{p:8s} ndith={P[p]['ndith']} F/F_tgt={res[p]['F_rel']:.3f} (range {res[p]['F_rel_spread']:.3f}, spikes "
              f"{res[p]['F_rel_spike']:.3f})  halo ratio by annulus {np.round(hr, 2)}  SAT core={P[p]['nsat']:5d}"
              f"  lost core={P[p]['nlost']:4d}  bkg={P[p]['bkg']:.1f}")
    json.dump(res, open('brightness.json', 'w'), indent=1)
