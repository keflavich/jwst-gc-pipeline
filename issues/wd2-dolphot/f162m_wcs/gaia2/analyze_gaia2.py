"""Summaries from gaia_fit_<TAG>.pkl (datascale_gaia2.py).  Usage: python analyze_gaia2.py TAG FIELD(wd2|wd1)
E (reference-pair J of band - F212N) is read from the parent pair-fit text files of that field."""
import pickle, re, sys
import numpy as np

PIX, RAD = 31.0, 206264.806
K = 2 * PIX
TAG = sys.argv[1]
FIELD = sys.argv[2] if len(sys.argv) > 2 else 'wd2'
P = pickle.load(open(f'gaia_fit_{TAG}.pkl', 'rb'))
res, boot = P['res'], P['boot']
DETS = [f'nrc{m}{i}' for m in 'ab' for i in range(1, 5)]
F212G = ('f212n', 'f187n', 'f182m')   # F182M/F187N/F212N reference group
PARENT = '..'
EFILES = (['datascale_other.txt', 'datascale_f162m.txt'] if FIELD == 'wd2' else
          ['datascale_wd1_other.txt', 'datascale_wd1_f115w.txt', 'datascale_wd1_f164n.txt', 'datascale_wd1_f187n.txt', 'datascale_wd1_f200w.txt'])


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


E = {}
for f in EFILES:
    E.update(jfits(f'{PARENT}/{f}'))
mm = lambda v: np.hypot(*v) / K * RAD
out = []
P_ = lambda s='': out.append(s)


def robust(x):
    lo, md, hi = np.nanpercentile(x, [16, 50, 84])
    return md, 0.5 * (hi - lo), lo, hi


BANDS = [b for b in ['f212n', 'f187n', 'f182m', 'f150w', 'f164n', 'f115w', 'f200w', 'f162m'] if any((b, d) in res for d in DETS)]
S = {}
for b in BANDS:
    for d in DETS:
        if (b, d) not in res:
            continue
        v = vec(res[(b, d)]['J'])
        Bv = vec(boot[(b, d)])
        ok = np.isfinite(Bv[0])
        if ok.sum() < 100:
            continue   # too few valid bootstrap draws (few Gaia stars)
        sig = np.array([robust(Bv[0][ok])[1], robust(Bv[1][ok])[1]])
        mb = np.hypot(*Bv[:, ok]) / K * RAD
        S[(b, d)] = dict(v=v, Bv=Bv, ok=ok, sig=sig, m=mm(v), mb=mb, n=res[(b, d)]['nstar'], npair=res[(b, d)]['npair'],
                         s=res[(b, d)]['sigma'], J=res[(b, d)]['J'])


def Edir(b, d):
    """Expected (band - F212N-group-ref) vector: own pair fit for F150W-group bands, F150W's for F212N-group bands."""
    if b in F212G or (b, d) not in E:
        return vec(E[('f150w', d)]) if ('f150w', d) in E else None
    return vec(E[(b, d)])


P_(f'## {TAG} ({FIELD}): per-detector Gaia-anchored similarity term')
P_('m = |(tr-, curl+)|/(2*31)*206265 [arcsec]; sig = robust (16-84 half-width) bootstrap sigma of each component [mas/pix]; m_lo-m_hi = bootstrap 16-84 interval of m.')
for b in BANDS:
    P_(f'\n### {b} - Gaia')
    P_('det    Nstar Npair rms(mas) |   tr-   +-sig    curl+  +-sig  |  m    m_lo-m_hi  | angle(deg) | |E_expected| m')
    for d in DETS:
        if (b, d) not in S:
            P_(f'{d}   no data / too few Gaia stars'); continue
        s = S[(b, d)]
        ang = np.degrees(np.arctan2(s['v'][1], s['v'][0]))
        e = Edir(b, d)
        P_(f'{d}  {s["n"]:4d} {s["npair"]:5d}  {s["s"]:5.1f}   | {s["v"][0]:+.4f} {s["sig"][0]:.4f}  {s["v"][1]:+.4f} {s["sig"][1]:.4f}  | '
           f'{s["m"]:5.1f}  {np.percentile(s["mb"],16):4.1f}-{np.percentile(s["mb"],84):4.1f} | {ang:+5.0f} | {mm(e) if e is not None else float("nan"):5.1f}')

P_('\n### Pooled projection beta onto E.  beta = sum_d w_d (v_d . E_d) / sum_d w_d |E_d|^2, w_d = 1/sig_d^2 (sig_d = robust bootstrap sigma of v_d . Ehat_d).')
P_('F212N-group band: beta = 0 if F212N-group refs right, -1 if F150W-group refs right.  F150W-group band: beta = +1 if F212N-group right, 0 if F150W-group right.')
GROUPS = {'A2+A3': ['nrca2', 'nrca3'], 'A1-A4': ['nrca1', 'nrca2', 'nrca3', 'nrca4'], 'A2': ['nrca2'], 'A3': ['nrca3']}
BETA = {}
for b in BANDS:
    for gname, dl in GROUPS.items():
        dl = [d for d in dl if (b, d) in S and Edir(b, d) is not None]
        if not dl:
            continue
        nb = S[(b, dl[0])]['Bv'].shape[1]
        num = np.zeros(nb); den = 0.0; num0 = 0.0
        w = {}
        for d in dl:
            e = Edir(b, d)
            eh = e / np.hypot(*e)
            proj = eh @ S[(b, d)]['Bv']
            sg = robust(proj[np.isfinite(proj)])[1]
            w[d] = (1 / sg**2, e)
        for d in dl:
            wd, e = w[d]
            num += wd * (e @ S[(b, d)]['Bv'])
            num0 += wd * (e @ S[(b, d)]['v'])
            den += wd * (e @ e)
        beta = num0 / den
        bb = num / den
        md, sg, lo, hi = robust(bb[np.isfinite(bb)])
        BETA[(b, gname)] = (beta, sg, lo, hi)
        P_(f'{b:6s} {gname:6s} beta = {beta:+.2f} +- {sg:.2f}  (boot 16-84: {lo:+.2f} to {hi:+.2f})   |E| m={[round(mm(w[d][1]),1) for d in dl]}')

P_('\n## Module-B m (arcsec, expected <= ~3 from E): ' + ', '.join(
    f'{b}:' + '/'.join(f'{S[(b,d)]["m"]:.0f}' if (b, d) in S else '-' for d in DETS[4:]) for b in BANDS))
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
pickle.dump(dict(S={k: {kk: vv for kk, vv in v.items() if kk != 'Bv'} for k, v in S.items()}, BETA=BETA, E={k: vec(v) for k, v in E.items()}), open(f'summary_{TAG}.pkl', 'wb'))
print('\n'.join(out))
