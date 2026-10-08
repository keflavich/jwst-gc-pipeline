"""In-detector rotation residual vs the distortion references' rotation.

For each (band - anchor) per-detector linear fit J (mas/pix, sky frame,
datascale_field.py / datascale_dol.py), a similarity error (scale a, rotation
b) in the pixel frame appears, under the pixel->sky parity flip, only in the
invariant pair (tr-, curl+) = (J00 - J11, J10 + J01), with magnitude
2 * 31 * sqrt(a^2 + b^2).  m = |(tr-, curl+)| / (2 * 31) is printed in arcsec
equivalent, next to |rot| of the band's reference relative to F212N's
(refscale.txt), and the angle of (tr-, curl+) in degrees."""
import re
import numpy as np

PIX = 31.0
RAD = 206264.806


def jfits(path):
    out = {}
    band = None
    for line in open(path):
        m = re.match(r'### (\w+) - (\w+)', line)
        if m:
            band = (m.group(1).upper(), m.group(2))
            continue
        m = re.match(r'(nrc[ab][1-4])\s.*J=\[([^;]+);([^\]]+)\]', line)
        if m and band:
            a = [float(x) for x in m.group(2).split()] + [float(x) for x in m.group(3).split()]
            out[(band, m.group(1))] = np.array(a).reshape(2, 2)
    return out


def refrot():
    out = {}
    band = None
    for line in open('refscale.txt'):
        m = re.match(r'### (\w+) - F212N', line)
        if m:
            band = m.group(1)
            continue
        p = line.split()
        if band and p and re.fullmatch(r'nrc[ab][1-4]', p[0]):
            out[(band, p[0])] = (float(p[1]), float(p[2]))
    return out


R = refrot()
J = {}
for f in ['datascale_f162m.txt', 'datascale_other.txt', 'datascale_dol_m02_f162m.txt',
          'datascale_dol_m02_f164n.txt', 'datascale_dol_m02_f182m.txt', 'datascale_dol_m02_f212n.txt']:
    J.update(jfits(f))
DETS = [f'nrc{m}{i}' for m in 'ab' for i in range(1, 5)]
print('m = in-detector similarity residual, arcsec-equivalent; ref = |rot(band ref) - rot(F212N ref)| arcsec; ang = deg')
for anchor in ['f212n', 'dolphot']:
    bands = sorted({ba[0] for (ba, _d) in J if ba[1] == anchor})
    print(f'\n### anchor {anchor}')
    print('band   ' + ' '.join(f'{d:>17s}' for d in DETS))
    for b in bands:
        cells = []
        for d in DETS:
            j = J.get(((b, anchor), d))
            if j is None:
                cells.append(f'{"":>17s}')
                continue
            t, c = j[0, 0] - j[1, 1], j[1, 0] + j[0, 1]
            m = np.hypot(t, c) / (2 * PIX) * RAD
            ang = np.degrees(np.arctan2(c, t))
            ref = abs(R[(b, d)][1]) if (b, d) in R else np.nan
            cells.append(f'{m:5.1f}/{ref:4.1f} {ang:+5.0f}')
        print(f'{b:6s} ' + ' '.join(cells))
