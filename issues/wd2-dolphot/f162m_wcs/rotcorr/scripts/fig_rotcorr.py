"""Figure for the #1135 rotation-correction validation: pixel-frame rotation
and scale of (band - F212N) per detector, before and after the reference
rotation correction, from rotcorr_proto.txt (wd2 m6, exposures 1-2)."""
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DETS = ['nrca1', 'nrca2', 'nrca3', 'nrca4', 'nrcb1', 'nrcb2', 'nrcb3', 'nrcb4']
BANDS = ['f115w', 'f150w', 'f162m', 'f164n', 'f200w']
COL = dict(zip(BANDS, ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#D55E00']))
MRK = dict(zip(BANDS, ['o', 's', '^', 'D', 'v']))

pix = re.compile(r'pixel frame: scale uncorr\s+([+-][\d.]+) ppm rot\s+([+-][\d.]+)"\s+->\s+'
                 r'corrected scale\s+([+-][\d.]+) ppm rot\s+([+-][\d.]+)"')
det = re.compile(r'^(nrc[ab]\d)\s+\d+\s+([+-][\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)')
res, band, last = {}, None, None
for line in open('rotcorr_proto.txt'):
    m = re.match(r'### (f\d+[wmn]) - f212n', line)
    if m:
        band = m.group(1)
        continue
    m = pix.search(line)
    if m:
        last = [float(v) for v in m.groups()]
        continue
    m = det.match(line)
    if m and band and last is not None:
        d, ref, m0, mm, mp = m.groups()
        res[(band, d)] = dict(s0=last[0], r0=last[1], s1=last[2], r1=last[3],
                              ref=float(ref), m0=float(m0), m1=float(mm), mflip=float(mp))
        last = None

x = np.arange(len(DETS))
fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True, constrained_layout=True)
off = np.linspace(-0.28, 0.28, len(BANDS))
for i, b in enumerate(BANDS):
    r0 = [res[(b, d)]['r0'] for d in DETS]
    r1 = [res[(b, d)]['r1'] for d in DETS]
    s1 = [res[(b, d)]['s1'] for d in DETS]
    m0 = [res[(b, d)]['m0'] for d in DETS]
    m1 = [res[(b, d)]['m1'] for d in DETS]
    kw = dict(color=COL[b], marker=MRK[b], ms=8, ls='none')
    axes[0].plot(x + off[i], r0, mfc='none', mew=1.5, **kw)
    axes[0].plot(x + off[i], r1, label=b.upper(), **kw)
    axes[1].plot(x + off[i], m0, mfc='none', mew=1.5, **kw)
    axes[1].plot(x + off[i], m1, label=b.upper(), **kw)
    axes[2].plot(x + off[i], s1, label=b.upper(), **kw)
for ax in axes:
    ax.axhline(0, color='0.6', lw=0.8, zorder=0)
    ax.grid(axis='y', color='0.9', lw=0.6)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel('pixel-frame rotation of\n(band − F212N)  [″]')
axes[0].set_title('wd2 m6 per-frame catalogs vs F212N, exposures 1–2.  '
                  'Open: CRDS references.  Filled: rotation-corrected.', fontsize=10, loc='left')
axes[1].legend(ncol=5, frameon=False, loc='upper right', fontsize=9)
axes[1].set_ylabel('in-detector term m  [″]')
axes[1].set_ylim(-1, 38)
axes[2].set_ylabel('pixel-frame scale of\n(band − F212N), corrected  [ppm]')
axes[2].set_xticks(x, [d.upper() for d in DETS])
fig.savefig('fig_rotcorr.png', dpi=130)

print('band   det     rot0    rot1     m0    m1  scale1')
for b in BANDS:
    for d in DETS:
        r = res[(b, d)]
        print(f"{b:6s} {d:6s} {r['r0']:+7.2f} {r['r1']:+7.2f} {r['m0']:6.1f} {r['m1']:5.1f} {r['s1']:+7.1f}")
