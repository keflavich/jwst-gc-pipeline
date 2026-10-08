"""F200W - F150W corrected pixel-frame scale per detector, wd2 vs wd1, from
tables_AC.txt (section A).  A field-level scale common to both bands cancels."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

rows = {}
sec = None
for line in open('tables_AC.txt'):
    if line.startswith('A:'):
        sec = 'A'
        continue
    if line.startswith('C:'):
        break
    p = line.split()
    if sec == 'A' and p and p[0].startswith('nrc'):
        rows[p[0]] = [float(v.split('/')[0]) for v in p[1:5]]
dets = sorted(rows)
w2 = np.array([rows[d][0] - rows[d][1] for d in dets])
w1 = np.array([rows[d][2] - rows[d][3] for d in dets])
x = np.arange(len(dets))
fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
ax.axhline(0, color='0.6', lw=0.8, zorder=0)
ax.grid(axis='y', color='0.9', lw=0.6)
ax.plot(x - 0.1, w2, 'o', color='#0072B2', ms=8, label='wd2')
ax.plot(x + 0.1, w1, 's', color='#D55E00', ms=8, mfc='none', mew=1.5, label='wd1')
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
ax.set_xticks(x, [d.upper() for d in dets])
ax.set_ylabel('(F200W − F150W) pixel-frame scale\nvs F212N, rotation-corrected  [ppm]')
ax.set_title('F200W relative to F150W, per detector, two fields', loc='left', fontsize=10)
ax.legend(frameon=False)
fig.savefig('fig_f200w_minus_f150w.png', dpi=130)
print('det    wd2(F200W-F150W) wd1(F200W-F150W)  diff  mean')
for d, a, b in zip(dets, w2, w1):
    print(f'{d}  {a:+7.1f}  {b:+7.1f}  {a - b:+6.1f}  {(a + b) / 2:+6.1f}')
print(f'rms(wd2-wd1) = {np.sqrt(np.mean((w2 - w1) ** 2)):.1f} ppm')
