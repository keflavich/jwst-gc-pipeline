"""Does the saturated core feed charge to its unsaturated neighbours?  (README §8)

For every unsaturated pixel of a raw ramp (4 groups, BRIGHT2, both integrations),
R = (g4 - g3)/(g2 - g1) measures how the pixel's gain of signal changes through the
ramp.  Classical nonlinearity makes R a function of the pixel's own fill level only.
Charge from saturated pixels (overflow, lateral diffusion of charge generated in debiased
pixels, brighter-fatter at the core edge) makes R depend on the distance to the nearest
saturated pixel at fixed fill.  IPC from a pixel that stops integrating lowers R next to
it.  So R is tabulated on (fill, distance) over the whole detector: at fixed fill, a
distance dependence is a saturated-core effect; its sign says which.

fill   = g4 - bias, bias = ZEROFRAME - (g2 - g1)/2 (the zero frame is read half a group
         after reset in BRIGHT2).  Pixels with g4 > SATFRAC * plateau, or a (g2-g1)
         below MINRATE, are excluded.
dist   = distance to the nearest pixel saturated before group 2 (g2 - g1 < 100 DN and
         g1 > 0.8 plateau), from a Euclidean distance transform.

    python ramp_satedge.py <out.npz> <uncal> [<uncal> ...]
"""
import sys
import numpy as np
from astropy.io import fits
from scipy import ndimage

FILLB = [2000, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000]
DISTB = [1, 2, 3, 5, 8, 12, 20, 35, 60, 1e9]
SATFRAC = 0.9
MINRATE = 300


def one(fn, acc):
    with fits.open(fn) as f:
        d = f['SCI'].data.astype(np.float32)          # (nint, ngroup, y, x)
        z = f['ZEROFRAME'].data.astype(np.float32)
    for i in range(d.shape[0]):
        g = d[i]
        dg1 = g[1]-g[0]; dg3 = g[3]-g[2]
        sat1 = (dg1 < 100) & (g[0] > 40000)
        plateau = np.median(g[0][sat1]) if sat1.sum() > 100 else 60000.
        sat1 &= g[0] > 0.8*plateau
        dist = ndimage.distance_transform_edt(~sat1)
        bias = z[i]-dg1/2
        fill = g[3]-bias
        ok = (~sat1) & (g[3] < SATFRAC*plateau) & (dg1 > MINRATE)
        ok[:4] = ok[-4:] = False; ok[:, :4] = ok[:, -4:] = False       # reference pixels
        R = dg3/np.maximum(dg1, 1)
        fi = np.digitize(fill[ok], FILLB)-1; di = np.digitize(dist[ok], DISTB)-1
        k = (fi >= 0) & (fi < len(FILLB)-1) & (di >= 0) & (di < len(DISTB)-1)
        cell = fi[k]*(len(DISTB)-1)+di[k]
        nc = (len(FILLB)-1)*(len(DISTB)-1)
        Rk = R[ok][k].astype(float)
        acc['n'] += np.bincount(cell, minlength=nc).reshape(acc['n'].shape)
        acc['s'] += np.bincount(cell, weights=Rk, minlength=nc).reshape(acc['n'].shape)
        acc['s2'] += np.bincount(cell, weights=Rk**2, minlength=nc).reshape(acc['n'].shape)
        # medians need the samples: keep a capped random subset per cell
        rng = np.random.default_rng(i)
        for c in np.unique(cell):
            v = Rk[cell == c]
            if len(v) > 4000:
                v = rng.choice(v, 4000, replace=False)
            acc['samp'].setdefault(int(c), []).append(v.astype(np.float32))
        print(fn.split('/')[-1], 'int', i+1, 'plateau', plateau, 'nsat1', int(sat1.sum()), flush=True)


def main():
    out = sys.argv[1]
    shape = (len(FILLB)-1, len(DISTB)-1)
    acc = dict(n=np.zeros(shape), s=np.zeros(shape), s2=np.zeros(shape), samp={})
    for fn in sys.argv[2:]:
        one(fn, acc)
    med = np.full(shape, np.nan); err = np.full(shape, np.nan)
    for c, lst in acc['samp'].items():
        v = np.concatenate(lst)
        if len(v) >= 50:
            med.flat[c] = np.median(v); err.flat[c] = 1.2533*np.std(v)/np.sqrt(len(v))
    print('median R = (g4-g3)/(g2-g1); rows = fill [DN], cols = distance to nearest group-1-saturated pixel [px]')
    print('fill \\ dist ' + ' '.join(f'{DISTB[j]:>5.0f}-' for j in range(shape[1])))
    for i in range(shape[0]):
        print(f'{FILLB[i]:5d}-{FILLB[i+1]:5d} ' + ' '.join('   -  ' if not np.isfinite(med[i, j]) else f'{med[i, j]:6.3f}' for j in range(shape[1])))
    print('pixel counts')
    for i in range(shape[0]):
        print(f'{FILLB[i]:5d}-{FILLB[i+1]:5d} ' + ' '.join(f'{int(acc["n"][i, j]):6d}' for j in range(shape[1])))
    np.savez(out, med=med, err=err, n=acc['n'], FILLB=FILLB, DISTB=DISTB)


if __name__ == '__main__':
    main()
