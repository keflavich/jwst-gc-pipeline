"""Figures for halocal_samepos.py: the same detector move, seen by two different stars.

Row per quadruple: D_A, D_B, D_A - D_B and a control D_C (a third star making a
move of the same length far away), each as a fractional change of the local PSF
intensity, D / <I>, smoothed 1.5 px, r <= RSHOW.  Last panel: corr/ceiling vs
separation from the table.

    python halocal_samepos_fig.py <meas.npz> <samepos_NRCBLONG.npz> <out.png> <patdir> ...
"""
import os, sys, glob, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from halocal_samepos import load_star, RC, RR, ANN, RHOB, RBINS
from halocal_psfcal import amplitudes

RSHOW = int(os.environ.get('RSHOW', 50))
NQ = int(os.environ.get('NQ', 3))


def frac(s, k, i):
    D = s['m'][k]-s['m'][i]; V = s['v'][k]+s['v'][i]
    I = s['I']/np.nanmean(s['a'])
    f = np.where((RR <= RSHOW) & (RR >= 12) & (I > 0), D/I, np.nan)
    g = np.nan_to_num(f); c = ndimage.gaussian_filter(np.isfinite(f)*1., 1.5)
    return np.where(np.isfinite(f), ndimage.gaussian_filter(g, 1.5)/np.maximum(c, 1e-3), np.nan)


def main():
    measfn, resfn, out = sys.argv[1:4]
    patdirs = sys.argv[4:]
    det = 'NRCBLONG' if 'NRCB' in resfn else 'NRCALONG'
    z = np.load(measfn); R = np.load(resfn)
    rows = json.loads(str(R['rows'])); sids = list(R['sids'])
    Fs, _, _ = amplitudes(z, det)
    files = {}
    for pd in patdirs:
        for fn in glob.glob(os.path.join(pd, f'{det}_s*.npz')):
            files.setdefault(int(os.path.basename(fn).split('_s')[1].split('.')[0]), fn)
    acc = np.zeros((3, len(RBINS)-1, len(RHOB)-1, 4))
    cache = {}

    def star(n):
        if n not in cache:
            cache[n] = load_star(files[sids[n]], z, Fs[sids[n]], acc); cache[n]['sid'] = sids[n]
        return cache[n]
    # best quadruples: small separation, high ceiling at 30-50 px halo
    cand = [r for r in rows if r['ann'] == 1 and r['part'] == 'halo' and r['sep'] < 6 and not r['anchor'] and not r['same']]
    cand = [r for r in cand if r['n'] > 3000]
    cand.sort(key=lambda r: -r['ceil'])
    pick = []; used = set()
    for r in cand:
        if (r['A'], r['B']) in used:
            continue
        pick.append(r); used.add((r['A'], r['B']))
        if len(pick) == NQ:
            break
    far = [r for r in rows if r['ann'] == 1 and r['part'] == 'halo' and r['sep'] > 100 and not r['anchor'] and not r['same']]
    fig, ax = plt.subplots(NQ, 4, figsize=(15, 3.7*NQ+0.6), squeeze=False)
    for q, r in enumerate(pick):
        A, B = star(r['A']), star(r['B'])
        dA = frac(A, r['k'], r['i']); dB = frac(B, r['l'], r['j'])
        # control: star A's same two exposures paired with a star B' 100-300 px away that
        # made the same dither move (so D_B' is the same move, elsewhere on the detector)
        ctl = [c for c in far if c['A'] == r['A'] and c['i'] == r['i'] and c['k'] == r['k']]
        ctl = max(ctl, key=lambda c: c['ceil']) if ctl else None
        if ctl is not None:
            C = star(ctl['B']); dC = frac(C, ctl['l'], ctl['j'])
        else:
            C = None; dC = np.full_like(dA, np.nan)
        vm = np.nanpercentile(np.abs(np.concatenate([dA[np.isfinite(dA)], dB[np.isfinite(dB)]])), 97)
        panels = [(dA, f'star A s{A["sid"]} (obs {A["obs"]}, F={A["F"]:.2f})\nexp {A["ex"][r["k"]]} - exp {A["ex"][r["i"]]}'),
                  (dB, f'star B s{B["sid"]} (obs {B["obs"]}, F={B["F"]:.2f})\nexp {B["ex"][r["l"]]} - exp {B["ex"][r["j"]]}'),
                  (dA-dB, f'A - B   (positions agree to {r["sep"]:.1f} px)\ncorr={r["corr"]:+.2f}, noise ceiling {r["ceil"]:.2f}'),
                  (dC, 'control: no far star' if C is None else f'control: s{C["sid"]} (obs {C["obs"]}), SAME move {ctl["sep"]:.0f} px away\ncorr with A={ctl["corr"]:+.2f}, ceiling {ctl["ceil"]:.2f}')]
        for p, (img, title) in enumerate(panels):
            a = ax[q, p]
            a.imshow(img[RC-RSHOW:RC+RSHOW+1, RC-RSHOW:RC+RSHOW+1], origin='lower', cmap='RdBu_r', vmin=-vm, vmax=vm,
                     extent=[-RSHOW-.5, RSHOW+.5, -RSHOW-.5, RSHOW+.5])
            a.set_title(title, fontsize=9); a.set_xticks([]); a.set_yticks([])
        pA = A['pos'][[r['i'], r['k']]]
        ax[q, 0].set_ylabel(f'detector ({pA[0,0]:.0f},{pA[0,1]:.0f}) -> ({pA[1,0]:.0f},{pA[1,1]:.0f})\n'
                            f'colour: fractional PSF change, +/-{100*vm:.0f}%', fontsize=9)
    fig.suptitle(f'{det}: the SAME detector move made by two DIFFERENT stars (in-sample exposure-difference residuals, r <= {RSHOW} px)', fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=80)
    print('wrote', out, [(p['A'], p['B'], p['sep'], p['corr'], p['ceil']) for p in pick])


if __name__ == '__main__':
    main()
