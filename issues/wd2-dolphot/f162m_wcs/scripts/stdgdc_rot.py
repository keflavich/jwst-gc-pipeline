"""In-detector scale and rotation of Jay Anderson's STDGDC solutions for each
SW filter relative to F212N, per detector: similarity fit of the two forward
maps (raw pixel -> corrected pixel) on a grid.  Compare with refscale.txt
(CRDS distortion references)."""
import numpy as np
from jwst_gc_pipeline.astrometry_gdc.stdgdc import STDGDC, GDCFileNotFoundError

DETS = ['NRCA1', 'NRCA2', 'NRCA3', 'NRCA4', 'NRCB1', 'NRCB2', 'NRCB3', 'NRCB4']
BANDS = ['F115W', 'F150W', 'F200W', 'F182M', 'F210M']
g = np.linspace(100, 1947, 15)
GX, GY = np.meshgrid(g, g)
X, Y = GX.ravel(), GY.ravel()


def sim(a, b):
    """(scale ppm, rotation arcsec, rms resid pix) of b relative to a."""
    za = a[0] + 1j * a[1]
    zb = b[0] + 1j * b[1]
    ok = np.isfinite(za) & np.isfinite(zb)
    za, zb = za[ok] - za[ok].mean(), zb[ok] - zb[ok].mean()
    k = np.vdot(za, zb) / np.vdot(za, za)
    res = zb - k * za
    return (abs(k) - 1) * 1e6, np.rad2deg(np.angle(k)) * 3600, np.sqrt(np.mean(abs(res) ** 2))


def lin(m):
    """Jacobian at the detector centre of a forward map (corrected pix / raw pix)."""
    h = 5.0
    x0 = y0 = 1023.5
    fx1, fy1 = m.forward(np.array([x0 + h]), np.array([y0]))
    fx0, fy0 = m.forward(np.array([x0 - h]), np.array([y0]))
    gx1, gy1 = m.forward(np.array([x0]), np.array([y0 + h]))
    gx0, gy0 = m.forward(np.array([x0]), np.array([y0 - h]))
    return np.array([[fx1 - fx0, gx1 - gx0], [fy1 - fy0, gy1 - gy0]])[:, :, 0] / (2 * h)


print('STDGDC: band - F212N similarity per detector: scale(ppm) rot(arcsec) resid(pix)')
for det in DETS:
    a = STDGDC.load(det, 'F212N')
    fa = a.forward(X, Y)
    Ja = lin(a)
    print(f'{det}  F212N centre J = [{Ja[0,0]:.6f} {Ja[0,1]:+.6f}; {Ja[1,0]:+.6f} {Ja[1,1]:.6f}]  file {a}')
    for band in BANDS:
        try:
            b = STDGDC.load(det, band)
        except (GDCFileNotFoundError, FileNotFoundError) as exc:
            print(f'  {band}: missing ({exc})')
            continue
        s, r, rr = sim(fa, b.forward(X, Y))
        Jb = lin(b)
        print(f'  {band}  scale {s:+8.1f}  rot {r:+7.2f}  resid {rr:.4f}   centre J = [{Jb[0,0]:.6f} {Jb[0,1]:+.6f}; {Jb[1,0]:+.6f} {Jb[1,1]:.6f}]  {b}')
