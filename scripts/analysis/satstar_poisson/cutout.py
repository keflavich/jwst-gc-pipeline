"""Cut out a star-centred region from a NIRCam _cal and attach per-pixel V2V3 coordinates
from the GWCS (never the SIP header -- CLAUDE.md astrometry rule #2)."""
import numpy as np, warnings, sys
import stdatamodels.jwst.datamodels as dm
warnings.simplefilter('ignore')

def prep(calfile, ra, dec, hw, out):
    m = dm.open(calfile)
    w = m.meta.wcs
    xt, yt = w.invert(ra, dec)
    xc, yc = int(round(float(xt))), int(round(float(yt)))
    n = 2*hw
    X0, Y0 = xc-hw, yc-hw
    def cut(a, fill):
        o = np.full((n, n), fill, dtype=a.dtype)
        y0, y1 = max(0, Y0), min(a.shape[0], Y0+n); x0, x1 = max(0, X0), min(a.shape[1], X0+n)
        o[y0-Y0:y1-Y0, x0-X0:x1-X0] = a[y0:y1, x0:x1]
        return o
    d = cut(m.data.astype(np.float64), np.nan)
    dq = cut(m.dq.astype(np.uint32), np.uint32(1))
    var = cut((m.var_poisson+m.var_rnoise).astype(np.float64), np.nan)
    vf = cut(m.var_flat.astype(np.float64), np.nan)
    vr = cut(m.var_rnoise.astype(np.float64), np.nan)
    t = w.get_transform('detector', 'v2v3')
    yy, xx = np.mgrid[0:n, 0:n]
    v2, v3 = t((xx+X0).ravel().astype(float), (yy+Y0).ravel().astype(float))
    v2 = v2.reshape(n, n); v3 = v3.reshape(n, n)
    vt = np.array(t(float(xt), float(yt)))
    # local jacobian d(v2,v3)/d(x,y) at target
    h = 0.5
    J = np.array([[(t(xt+h, yt)[0]-t(xt-h, yt)[0])/(2*h), (t(xt, yt+h)[0]-t(xt, yt-h)[0])/(2*h)],
                  [(t(xt+h, yt)[1]-t(xt-h, yt)[1])/(2*h), (t(xt, yt+h)[1]-t(xt, yt-h)[1])/(2*h)]])
    np.savez(out, d=d, dq=dq, var=var, vf=vf, vr=vr, v2=v2, v3=v3, X0=X0, Y0=Y0,
             xt=float(xt)-X0, yt=float(yt)-Y0, vt=vt, J=J,
             expstart=m.meta.exposure.start_time, fn=calfile)
    print(out, xc, yc, float(xt)-X0, float(yt)-Y0)

if __name__ == '__main__':
    # python cutout.py RA DEC HALFWIDTH OUTPREFIX cal1.fits [cal2.fits ...]
    # -> OUTPREFIX_e1.npz, ... (the target of the demo: 266.5090306896523 -28.95658817641266,
    #    jw10678061001_02101_0000[1-6]_nrcblong_cal.fits, HALFWIDTH 512, prefix tgt)
    ra, dec, hw, pre = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    for i, f in enumerate(sys.argv[5:], 1):
        prep(f, ra, dec, hw, f'{pre}_e{i}.npz')
