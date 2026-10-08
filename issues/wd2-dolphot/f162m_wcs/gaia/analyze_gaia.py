"""Summaries and figure from gaia_fit_wd2.pkl (cluster-bootstrap over Gaia sources, shared across bands)."""
import pickle, re, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PIX, RAD = 31.0, 206264.806
K = 2 * PIX
TAG = sys.argv[1] if len(sys.argv) > 1 else 'wd2'
P = pickle.load(open(f'gaia_fit_{TAG}.pkl', 'rb'))
res, boot = P['res'], P['boot']
DETS = [f'nrc{m}{i}' for m in 'ab' for i in range(1, 5)]
BANDS = ['f212n', 'f150w', 'f162m']
PARENT = '..'


def vec(J):
    return np.array([J[..., 0, 0] - J[..., 1, 1], J[..., 1, 0] + J[..., 0, 1]])


def jfits(path):
    out, band = {}, None
    for line in open(path):
        m = re.match(r'### (\w+) - (\w+)', line)
        if m:
            band = m.group(1)
            continue
        m = re.match(r'(nrc[ab][1-4])\s.*J=\[([^;]+);([^\]]+)\]', line)
        if m and band:
            a = [float(x) for x in m.group(2).split()] + [float(x) for x in m.group(3).split()]
            out[(band, m.group(1))] = np.array(a).reshape(2, 2)
    return out


# (band - F212N) J from the m6 pair fits against the F212N catalog (same stars/frames)
E = {}
for f in ['datascale_other.txt', 'datascale_f162m.txt']:
    E.update(jfits(f'{PARENT}/{f}'))
mm = lambda v: np.hypot(*v) / K * RAD
out = []
P_ = lambda s='': out.append(s)


def robust(x):
    lo, md, hi = np.nanpercentile(x, [16, 50, 84])
    return md, 0.5 * (hi - lo), lo, hi


S = {}   # (band, det) -> dict
for b in BANDS:
    for d in DETS:
        if (b, d) not in res:
            continue
        v = vec(res[(b, d)]['J'])
        Bv = vec(boot[(b, d)])            # (2, nb)
        ok = np.isfinite(Bv[0])
        sig = np.array([robust(Bv[0][ok])[1], robust(Bv[1][ok])[1]])
        mb = np.hypot(*Bv[:, ok]) / K * RAD
        S[(b, d)] = dict(v=v, Bv=Bv, ok=ok, sig=sig, m=mm(v), mb=mb, n=res[(b, d)]['nstar'], npair=res[(b, d)]['npair'],
                         s=res[(b, d)]['sigma'], J=res[(b, d)]['J'])

P_(f'## {TAG}: per-detector Gaia-anchored similarity term')
P_('m = |(tr-, curl+)|/(2*31)*206265 [arcsec]; sig = robust (16-84 half-width) bootstrap sigma of each component [mas/pix]; '
   'm_lo-m_hi = bootstrap 16-84 interval of m.  1 arcsec of m = 3.0e-4 mas/pix in (tr-, curl+).')
for b in BANDS:
    P_(f'\n### {b} - Gaia')
    P_('det    Nstar Npair rms(mas) |   tr-   +-sig    curl+  +-sig  |  m    m_lo-m_hi  | angle(deg)')
    for d in DETS:
        if (b, d) not in S:
            P_(f'{d}   no data'); continue
        s = S[(b, d)]
        ang = np.degrees(np.arctan2(s['v'][1], s['v'][0]))
        P_(f'{d}  {s["n"]:4d} {s["npair"]:5d}  {s["s"]:5.1f}   | {s["v"][0]:+.4f} {s["sig"][0]:.4f}  {s["v"][1]:+.4f} {s["sig"][1]:.4f}  | '
           f'{s["m"]:5.1f}  {np.percentile(s["mb"],16):4.1f}-{np.percentile(s["mb"],84):4.1f} | {ang:+5.0f}')

P_('\n### J_F150W(gaia) - J_F212N(gaia) vs the (F150W - F212N) pair-fit J from the F212N catalog (E)')
P_('det    Delta(tr-,curl+)  +-sig          m_gaia_diff  | E(tr-,curl+)   m_E | Delta - E (tr-,curl+)  in sigma')
for d in DETS:
    if ('f212n', d) not in S or ('f150w', d) not in S:
        continue
    a, c = S[('f150w', d)], S[('f212n', d)]
    D = a['v'] - c['v']
    BD = a['Bv'] - c['Bv']
    ok = np.isfinite(BD[0])
    sg = np.array([robust(BD[0][ok])[1], robust(BD[1][ok])[1]])
    e = vec(E[('f150w', d)])
    mb = np.hypot(*BD[:, ok]) / K * RAD
    S[('diff', d)] = dict(v=D, sig=sg, m=mm(D), mb=mb, e=e)
    P_(f'{d}  ({D[0]:+.4f},{D[1]:+.4f}) +-({sg[0]:.4f},{sg[1]:.4f})  {mm(D):5.1f} [{np.percentile(mb,16):.1f}-{np.percentile(mb,84):.1f}] | '
       f'({e[0]:+.4f},{e[1]:+.4f}) {mm(e):5.1f} | ({D[0]-e[0]:+.4f},{D[1]-e[1]:+.4f}) ({(D[0]-e[0])/sg[0]:+.1f},{(D[1]-e[1])/sg[1]:+.1f})')

# pooled projection onto the reference-difference vector E = (F150W-group minus F212N-group) per detector
P_('\n### Pooled projection beta onto E (E from F150W-F212N pair fits).')
P_('beta = sum_d w_d (v_d . E_d) / sum_d w_d |E_d|^2, w_d = 1/(sig_d^2) with sig_d the robust bootstrap sigma of v_d . Ehat_d.')
P_('F150W-group-right hypothesis:  beta(F150W)=0, beta(F212N)=-1.   F212N-group-right hypothesis:  beta(F150W)=+1, beta(F212N)=0.')
GROUPS = {'A2+A3': ['nrca2', 'nrca3'], 'A1-A4': ['nrca1', 'nrca2', 'nrca3', 'nrca4'], 'A2': ['nrca2'], 'A3': ['nrca3']}
BETA = {}
for b in ['f212n', 'f150w', 'f162m']:
    for gname, dl in GROUPS.items():
        dl = [d for d in dl if (b, d) in S]
        if not dl:
            continue
        nb = S[(b, dl[0])]['Bv'].shape[1]
        num = np.zeros(nb); den = 0.0; num0 = 0.0
        # weights from the per-detector bootstrap sigma of the projection
        w = {}
        for d in dl:
            e = vec(E[(b, d)]) if (b, d) in E else vec(E[('f150w', d)])
            eh = e / np.hypot(*e)
            proj = eh @ S[(b, d)]['Bv']
            ok = np.isfinite(proj)
            sg = robust(proj[ok])[1]
            w[d] = (1 / sg**2, e)
        for d in dl:
            wd, e = w[d]
            num += wd * (e @ S[(b, d)]['Bv'])
            num0 += wd * (e @ S[(b, d)]['v'])
            den += wd * (e @ e)
        beta = num0 / den
        bb = num / den
        ok = np.isfinite(bb)
        md, sg, lo, hi = robust(bb[ok])
        BETA[(b, gname)] = (beta, sg, lo, hi)
        P_(f'{b:6s} {gname:6s} beta = {beta:+.2f} +- {sg:.2f}  (boot 16-84: {lo:+.2f} to {hi:+.2f})   |E| m={[round(mm(w[d][1]),1) for d in dl]}')

P_('\n## J elements (mas/pix) with robust bootstrap sigma (16-84 half-width)')
for b in BANDS:
    P_(f'\n### {b} - Gaia')
    for d in DETS:
        if (b, d) not in S:
            continue
        B = boot[(b, d)].reshape(-1, 4)
        B = B[np.isfinite(B[:, 0])]
        sg = [robust(B[:, k])[1] for k in range(4)]
        J = S[(b, d)]['J']
        P_(f'{d}  J00={J[0,0]:+.4f}+-{sg[0]:.4f}  J01={J[0,1]:+.4f}+-{sg[1]:.4f}  J10={J[1,0]:+.4f}+-{sg[2]:.4f}  J11={J[1,1]:+.4f}+-{sg[3]:.4f}')

open(f'results_{TAG}.txt', 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))

# figure
fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw=dict(width_ratios=[3, 1.3]))
ax = axs[0]
cols = {'f212n': '#1f77b4', 'f150w': '#d95f02', 'f162m': '#7570b3'}
wd = 0.27
x0 = np.arange(len(DETS))
for k, b in enumerate(BANDS):
    for i, d in enumerate(DETS):
        if (b, d) not in S:
            continue
        s = S[(b, d)]
        lo, hi = np.percentile(s['mb'], [16, 84])
        ax.bar(i + (k - 1) * wd, s['m'], wd, color=cols[b], alpha=0.85, label=b.upper() if i == 0 else None)
        ax.errorbar(i + (k - 1) * wd, s['m'], yerr=[[max(s['m'] - lo, 0)], [max(hi - s['m'], 0)]], color='k', lw=1, capsize=2)
for i, d in enumerate(DETS):
    if ('f150w', d) in E:
        ax.hlines(mm(vec(E[('f150w', d)])), i - 0.45, i + 0.45, color='k', ls='--', lw=1.5, label='expected |ref(F150W) - ref(F212N)| (pair fits)' if i == 0 else None)
ax.set_xticks(x0); ax.set_xticklabels([d.upper() for d in DETS])
ax.set_ylim(0, 70); ax.set_ylabel('m [arcsec] of band - Gaia in-detector similarity term')
ax.set_title(f'{TAG}: m per detector (bars: nominal; whiskers: bootstrap 16-84%)')
ax.legend(fontsize=8, loc='upper right')
ax = axs[1]
labs, ys, es = [], [], []
for b in ['f212n', 'f150w']:
    for g in ['A2+A3', 'A1-A4']:
        if (b, g) in BETA:
            labs.append(f'{b.upper()}\n{g}'); ys.append(BETA[(b, g)][0]); es.append(BETA[(b, g)][1])
ax.errorbar(range(len(ys)), ys, yerr=es, fmt='o', color='k', capsize=3)
ax.axhline(0, color='gray', lw=0.8); ax.axhline(1, color='#d95f02', ls=':', lw=1); ax.axhline(-1, color='#1f77b4', ls=':', lw=1)
ax.text(len(ys) - 0.5, 1.03, 'beta=+1 (F150W carries E)', color='#d95f02', fontsize=7, ha='right')
ax.text(len(ys) - 0.5, -0.95, 'beta=-1 (F212N carries -E)', color='#1f77b4', fontsize=7, ha='right')
ax.set_xticks(range(len(ys))); ax.set_xticklabels(labs, fontsize=8); ax.set_xlim(-0.5, len(ys) - 0.5)
ax.set_ylabel('beta: projection of (band - Gaia) onto E'); ax.set_title('Module A pooled projection')
fig.tight_layout(); fig.savefig('fig_gaia_rot.png', dpi=130)
