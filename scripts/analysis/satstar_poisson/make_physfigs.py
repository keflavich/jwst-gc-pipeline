"""Figures for the physical-optics PSF test (README section 7).

    python make_physfigs.py <static.npz> <outdir> [loo log files ...]

fig11: the static pattern Q against the physical-optics (geometric pupil) PSF: images,
      the vertical-spike cross-section, and the between-spike / on-spike radial profiles.
fig12: robust sigma(chi) of the held-out dithers for each per-exposure physical mode family
      (parsed from physloo.py logs).
"""
import sys, re, glob
import numpy as np
from scipy import ndimage
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from physpsf import PhysPSF, f480m_band

N = 1152
SPIKES = (0, 30, 90, 150, 180, 210, 270, 330)


def static_figure(stat, out):
    J1 = np.load('tgt_e1.npz')['J']
    z = np.load(stat, allow_pickle=True)
    lams, wl = f480m_band(12)
    ps = PhysPSF(N, J1, lams, wl, geom=z['geom'].item()); ps.det.update(z['det'].item())
    xs, ys = z['src'][0]
    M = ps.detect(ps.intensity(np.zeros(ps.G**2)), shift=(xs, ys))
    Q = np.load('Q_q3.npy'); cnt = np.load('wq.npz')['cnt']
    yy, xx = np.mgrid[:N, :N]-N//2
    r = np.hypot(xx-xs, yy-ys); th = np.degrees(np.arctan2(yy-ys, xx-xs)) % 360
    dth = np.min([np.abs((th-a+180) % 360-180) for a in SPIKES], 0)
    bg = np.median(Q[(r > 300) & (r < 450)])
    # flux from the on-spike high-pass structure (spikes are geometric diffraction)
    hp = lambda a: a-ndimage.uniform_filter(a, 15)
    sel = (cnt >= 3) & (r > 60) & (r < 280) & (dth < 3)
    a = np.sum(hp(Q)[sel]*hp(M)[sel])/np.sum(hp(M)[sel]**2)
    fig, ax = plt.subplots(2, 3, figsize=(16, 10.5))
    h = 150; sl = slice(N//2-h, N//2+h); ext = [-h, h, -h, h]
    kw = dict(origin='lower', cmap='magma', vmin=0.5, vmax=3.5, extent=ext)
    ax[0, 0].imshow(np.log10(np.clip(np.where(cnt >= 3, Q, np.nan)[sl, sl]-bg, 3, None)), **kw)
    ax[0, 0].set_title('static pattern Q - sky (log), 6 dithers')
    ax[0, 1].imshow(np.log10(np.clip(a*M[sl, sl], 3, None)), **kw)
    ax[0, 1].set_title('physical-optics PSF, geometric pupil (log)')
    c1 = np.corrcoef(np.clip(hp(Q)[sel], *np.percentile(hp(Q)[sel], [2, 98])), hp(M)[sel])[0, 1]
    s2 = (cnt >= 3) & (r > 50) & (r < 120) & (dth > 6)
    c2 = np.corrcoef(np.clip(hp(Q)[s2], *np.percentile(hp(Q)[s2], [2, 98])), hp(M)[s2])[0, 1]
    ax[0, 2].imshow(np.where(cnt >= 3, hp(Q), np.nan)[sl, sl], origin='lower', cmap='RdBu_r', vmin=-60, vmax=60, extent=ext)
    ax[0, 2].set_title(f'Q high-pass (15 px)\ncorr. with model: spikes {c1:.2f}, between spikes {c2:.2f}')
    for y0, col in ((60, 'C0'), (-160, 'C1'), (120, 'C2')):
        xcut = np.arange(-12, 13)
        ax[1, 0].plot(xcut+0.0, Q[N//2+y0, N//2+xcut]-bg, '-', color=col, label=f'Q, y={y0}')
        ax[1, 0].plot(xcut, a*M[N//2+y0, N//2+xcut], '--', color=col, label=f'model, y={y0}')
    ax[1, 0].set_xlabel('x [px]'); ax[1, 0].set_title('cross-section of the vertical spike'); ax[1, 0].legend(fontsize=8)
    rb = np.geomspace(30, 400, 25); rc = np.sqrt(rb[1:]*rb[:-1])
    for nm, m, ls in (('between spikes', dth > 8, '-'), ('on spikes', dth < 1, ':')):
        pq = [np.median(Q[(cnt >= 3) & m & (r >= lo) & (r < hi)]-bg) if ((cnt >= 3) & m & (r >= lo) & (r < hi)).sum() > 20 else np.nan
              for lo, hi in zip(rb[:-1], rb[1:])]
        pm = [np.median(a*M[m & (r >= lo) & (r < hi)]) for lo, hi in zip(rb[:-1], rb[1:])]
        ax[1, 1].loglog(rc, pq, 'k'+ls, label=f'Q {nm}'); ax[1, 1].loglog(rc, pm, 'C3'+ls, label=f'model {nm}')
    ax[1, 1].set_xlabel('r [px]'); ax[1, 1].set_ylabel('MJy/sr above sky'); ax[1, 1].legend(fontsize=8)
    ax[1, 1].set_title('between the spikes the halo is 3-10x the pupil diffraction')
    rat = [np.median((Q-bg)[(cnt >= 3) & (dth > 8) & (r >= lo) & (r < hi)])/np.median(a*M[(dth > 8) & (r >= lo) & (r < hi)])
           for lo, hi in zip(rb[:-1], rb[1:])]
    rat = np.where(rc < 200, rat, np.nan)
    ax[1, 2].loglog(rc, rat, 'k.-'); ax[1, 2].set_xlabel('r [px]'); ax[1, 2].set_ylabel('(Q - sky) / diffraction, between spikes')
    ax[1, 2].set_title('excess (non-diffraction) halo')
    plt.tight_layout(); plt.savefig(f'{out}/fig11_physpsf_static.png', dpi=60); plt.close()
    print('flux from spikes', a, 'corr', c1, c2)


def parse(logs):
    tab = {}
    for fn in logs:
        for line in open(fn):
            m = re.match(r'e(\d) (baseline|\+\S+)\s*(?:\(\s*(\d+)\))?[^:]*:\s+([\d. ]+)$', line.strip())
            if m:
                k, fam = int(m.group(1)), m.group(2).lstrip('+')
                tab.setdefault(fn, {})[(k, fam)] = [float(v) for v in m.group(4).split()]
    return tab


def results_figure(logs, out, name='fig12_physpsf_modes.png'):
    t = {}
    for v in parse(logs).values():
        t.update(v)
    labels = ['30-50', '50-80', '80-120', '120-200', '200-300', 'spike\n80-120', 'spike\n120-200', 'spike\n200-300']
    fams = sorted({f for _, f in t if f != 'baseline'})
    ks = sorted({k for k, _ in t})
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for k in ks:
        ax[0].plot(range(len(labels)), t[(k, 'baseline')], 'o-', label=f'dither {k}')
    ax[0].set_ylabel('robust sigma(chi), held-out dither'); ax[0].set_title('LOO baseline (static pattern + nuisance)')
    for f in fams:
        rat = np.array([np.array(t[(k, f)])/np.array(t[(k, 'baseline')]) for k in ks if (k, f) in t])
        ax[1].plot(range(len(labels)), rat.mean(0), 'o-', label=f)
    ax[1].axhline(1, color='k', lw=0.6)
    ax[1].set_ylabel('sigma(chi) with modes / baseline (mean of dithers)')
    ax[1].set_title('per-exposure physical-optics modes fitted on the held-out dither')
    for a in ax:
        a.set_xticks(range(len(labels))); a.set_xticklabels(labels, fontsize=8); a.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(f'{out}/{name}', dpi=60); plt.close()


if __name__ == '__main__':
    static_figure(sys.argv[1], sys.argv[2])
    if len(sys.argv) > 3:
        results_figure(sys.argv[3:], sys.argv[2])
