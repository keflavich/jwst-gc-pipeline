"""In-detector rotation residual (arcsec-equivalent similarity term of the
per-detector linear fit) against the distortion references' rotation
difference.  (a) band - F212N on wd2 vs |rot(band ref) - rot(F212N ref)|;
(b) band - dolphot per detector."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rot_vs_ref as rv

DETS = rv.DETS
COL = {'F115W': '#1f77b4', 'F150W': '#2ca02c', 'F162M': '#d62728', 'F164N': '#ff7f0e',
       'F200W': '#9467bd', 'F182M': '#7f7f7f', 'F187N': '#17becf', 'F212N': '#000000'}


def m_of(j):
    t, c = j[0, 0] - j[1, 1], j[1, 0] + j[0, 1]
    return np.hypot(t, c) / (2 * rv.PIX) * rv.RAD


fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
for b in ['F115W', 'F150W', 'F162M', 'F164N', 'F200W', 'F182M', 'F187N']:
    xs, ys = [], []
    for d in DETS:
        j = rv.J.get(((b, 'f212n'), d))
        if j is None:
            continue
        xs.append(abs(rv.R[(b, d)][1]))
        ys.append(m_of(j))
    a1.scatter(xs, ys, s=36, color=COL[b], label=b, edgecolor='white', lw=0.6, zorder=3)
a1.plot([0, 33], [0, 33], color='0.6', lw=1, ls='--', zorder=1)
a1.set_xlabel('|rotation of band reference - F212N reference| (arcsec)')
a1.set_ylabel('measured in-detector similarity term (arcsec equiv.)')
a1.set_title('(a) band - F212N, wd2 m6, 8 detectors each', fontsize=10)
a1.legend(fontsize=8, frameon=False)
a1.set_xlim(0, 33)
a1.set_ylim(0, 33)
a1.set_aspect('equal')

w = 0.2
x = np.arange(len(DETS))
for k, b in enumerate(['F162M', 'F164N', 'F182M', 'F212N']):
    ys = [m_of(rv.J[((b, 'dolphot'), d)]) for d in DETS]
    a2.bar(x + (k - 1.5) * w, ys, w * 0.9, color=COL[b], label=b)
a2.set_xticks(x)
a2.set_xticklabels([d[3:].upper() for d in DETS])
a2.set_ylabel('in-detector similarity term vs dolphot (arcsec equiv.)')
a2.set_title('(b) band - dolphot, wd2 m6, exposures 1-2', fontsize=10)
a2.legend(fontsize=8, frameon=False)
a2.text(0.98, 0.95, '25" over the 32" detector half-width = 3.9 mas;\n5.5 mas at the corners',
        transform=a2.transAxes, ha='right', va='top', fontsize=8, color='0.3')
fig.tight_layout()
fig.savefig('fig_rot.png', dpi=130)
print('wrote fig_rot.png')
