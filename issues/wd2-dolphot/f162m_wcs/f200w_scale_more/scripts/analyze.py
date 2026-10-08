import glob, os, re, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DETS = ['nrca1', 'nrca2', 'nrca3', 'nrca4', 'nrcb1', 'nrcb2', 'nrcb3', 'nrcb4']
WD2 = np.array([42.1, 20.2, 22.2, 15.5, 1.9, -17.0, -18.0, -17.3])
WD1 = np.array([37.5, 25.4, 44.4, 44.7, 22.2, -4.4, -21.8, -10.5])
WDDIFF = np.array([34.0, 19.9, 21.4, 13.5, -8.2, -12.2, -10.4, -13.5])


def parse(files):
    """-> {det: (scale_unc, rot_unc, scale_cor, rot_cor, nframes)}"""
    out = {}
    for f in files:
        L = open(f).read().splitlines()
        for i, l in enumerate(L):
            m = re.match(r'\s+pixel frame: scale uncorr\s+([-+\d.]+) ppm rot\s+([-+\d.]+)"\s+-> corrected scale\s+([-+\d.]+) ppm rot\s+([-+\d.]+)"', l)
            if m and i + 1 < len(L):
                d = L[i + 1].split()
                out[d[0]] = tuple(float(x) for x in m.groups()) + (int(d[1]),)
    return out


def npairs(tag, band):
    f = f'pairsm_{tag}.pkl'
    if not os.path.exists(f):
        return {}
    P = pickle.load(open(f, 'rb'))
    return {k[1]: sum(len(e['x']) for e in v) for k, v in P.items() if k != 'anchor'}


def collect(spec):
    """spec: (glob of out files, tag glob prefix, band)"""
    files = sorted(glob.glob(spec[0]))
    res = parse(files)
    npr = {}
    for f in files:
        tag = os.path.basename(f)[:-4]
        tag = tag[4:] if tag.startswith('rcm_') else tag
        npr.update(npairs(tag, spec[2]))
    return res, npr


runs = {
 'brick F200W-F187N': ('rcm_brick_f200w_v_f187n.txt', 'f200w'),
 'brick F200W-F182M': ('rcm_brickL_f200w_v_f182m_*.txt', 'f200w'),
 'brick F182M-F187N (ctl)': ('rcm_brick_f182m_v_f187n.txt', 'f182m'),
 'brick F187N-F182M (ctl)': ('rcm_brick_f187n_v_f182m.txt', 'f187n'),
 'ngc F200W(6778)-F187N': ('rcm_ngc_f200w6778_v_f187n.txt', 'f200w'),
 'ngc F200W(6778)-F182M': ('rcm_ngc_f200w6778_v_f182m.txt', 'f200w'),
 'ngc F200W(7213)-F182M': ('rcm_ngc_f200w7213_v_f182m.txt', 'f200w'),
 'ngc F182M-F187N (ctl)': ('rcm_ngc_f182m_v_f187n.txt', 'f182m'),
 'ngc F187N-F182M (ctl)': ('rcm_ngc_f187n_v_f182m.txt', 'f187n'),
}
T = {}
lines = []
for name, (g, band) in runs.items():
    res, npr = collect((g, None, band))
    sc = np.array([res[d][2] if d in res else np.nan for d in DETS])
    rot = np.array([res[d][3] if d in res else np.nan for d in DETS])
    scu = np.array([res[d][0] if d in res else np.nan for d in DETS])
    nf = [res[d][4] if d in res else 0 for d in DETS]
    npp = [npr.get(d, 0) for d in DETS]
    T[name] = (sc, rot, scu, nf, npp)
    lines.append(f'### {name}\n| det | scale corr (ppm) | rot corr (") | scale uncorr (ppm) | frames | pairs |\n|---|---|---|---|---|---|')
    for i, d in enumerate(DETS):
        lines.append(f'| {d.upper()} | {sc[i]:+.1f} | {rot[i]:+.2f} | {scu[i]:+.1f} | {nf[i]} | {npp[i]} |')
    lines.append('')
open('tables_per_run.txt', 'w').write('\n'.join(lines))

# comparison
def rms(a):
    return np.sqrt(np.nanmean(a**2))
comp = ['| series | mean offset (ppm) | rms vs wd (F200W-F150W) raw | after mean removal | rms vs wd2 F200W-F212N raw | after mean removal | n det |', '|---|---|---|---|---|---|---|']
for name in runs:
    if 'F200W' not in name:
        continue
    sc = T[name][0]
    k = np.isfinite(sc)
    if k.sum() < 3:
        continue
    row = [name]
    mo = np.mean(sc[k] - WDDIFF[k])
    mo2 = np.mean(sc[k] - WD2[k])
    row.append(f'{mo:+.1f} (vs diff), {mo2:+.1f} (vs wd2)')
    row.append(f'{rms((sc - WDDIFF)[k]):.1f}')
    row.append(f'{rms((sc - WDDIFF)[k] - mo):.1f}')
    row.append(f'{rms((sc - WD2)[k]):.1f}')
    row.append(f'{rms((sc - WD2)[k] - mo2):.1f}')
    row.append(str(int(k.sum())))
    comp.append('| ' + ' | '.join(row) + ' |')
# wd1 reference rows
for nm, ref in (('wd1 F200W-F212N', WD1),):
    mo = np.mean(ref - WDDIFF); mo2 = np.mean(ref - WD2)
    comp.append(f'| {nm} | {mo:+.1f} (vs diff), {mo2:+.1f} (vs wd2) | {rms(ref-WDDIFF):.1f} | {rms(ref-WDDIFF-mo):.1f} | {rms(ref-WD2):.1f} | {rms(ref-WD2-mo2):.1f} | 8 |')
open('table_compare.txt', 'w').write('\n'.join(comp))

summ = ['| det | wd2 F200W-F212N | wd1 F200W-F212N | wd mean F200W-F150W | brick F200W-F187N | brick F200W-F182M | ngc F200W(6778)-F187N | ngc F200W(6778)-F182M | ngc F200W(7213)-F182M |', '|' + '---|' * 9]
for i, d in enumerate(DETS):
    c = [T[n][0][i] for n in ('brick F200W-F187N', 'brick F200W-F182M', 'ngc F200W(6778)-F187N', 'ngc F200W(6778)-F182M', 'ngc F200W(7213)-F182M')]
    summ.append(f'| {d.upper()} | {WD2[i]:+.1f} | {WD1[i]:+.1f} | {WDDIFF[i]:+.1f} | ' + ' | '.join(f'{x:+.1f}' for x in c) + ' |')
open('table_summary.txt', 'w').write('\n'.join(summ))

# figure
x = np.arange(8)
fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw=dict(height_ratios=[3, 2]))
a = ax[0]
a.axhline(0, c='0.7', lw=0.8)
a.plot(x - .3, WD2, 'o-', c='k', label='wd2 F200W-F212N')
a.plot(x - .2, WD1, 's-', c='0.4', label='wd1 F200W-F212N')
a.plot(x - .1, WDDIFF, 'D--', c='0.6', label='wd mean (F200W-F150W)')
a.plot(x + 0.0, T['brick F200W-F187N'][0], '^-', c='C3', label='brick F200W-F187N')
a.plot(x + 0.1, T['brick F200W-F182M'][0], '^:', c='C1', label='brick F200W-F182M')
a.plot(x + 0.2, T['ngc F200W(6778)-F187N'][0], 'v-', c='C0', label='NGC6334 F200W(6778)-F187N')
a.plot(x + 0.3, T['ngc F200W(6778)-F182M'][0], 'v:', c='C2', label='NGC6334 F200W(6778)-F182M')
a.plot(x + 0.4, T['ngc F200W(7213)-F182M'][0], 'v:', c='C4', label='NGC6334 F200W(7213)-F182M')
a.set_ylabel('pixel-frame scale (ppm), rotation-corrected'); a.legend(fontsize=8, ncol=2); a.set_title('F200W per-detector scale residual')
b = ax[1]
b.axhline(0, c='0.7', lw=0.8)
for n, mk, c in (('brick F182M-F187N (ctl)', 'o', 'C3'), ('brick F187N-F182M (ctl)', 'x', 'C3'), ('ngc F182M-F187N (ctl)', 'o', 'C0'), ('ngc F187N-F182M (ctl)', 'x', 'C0')):
    b.plot(x, T[n][0], mk, c=c, label=n, ms=7)
b.plot(x, [0, 0, 0, 0, 0, 0, 0, 0], alpha=0)
b.set_ylabel('control scale (ppm)'); b.legend(fontsize=8, ncol=2)
b.set_xticks(x); b.set_xticklabels([d.upper() for d in DETS])
fig.tight_layout(); fig.savefig('fig_f200w_scale_more.png', dpi=130)
print('\n'.join(lines)); print(); print('\n'.join(summ)); print(); print('\n'.join(comp))
