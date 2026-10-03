"""#1017: realness of the sources each step of k admits (k_eff bands, from
kband_qsnr.py) with binomial errors, and the cumulative count against k.  The
dotted line marks realness 0.5, where an admitted source is as likely
spurious as real; the red line marks the default k = 5.

KBAND_VARIANT picks the kband_qsnr.py selection: promq0b-promqeb (default;
current base, the sources whose keep depends on the bound) or prom2 (the
#1018-era study: #1018's additions over the #1015 base).

usage: [KBAND_VARIANT=...] python kband_fig.py <out.png> [field ...]
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = {'brick': 'Brick F182M (ref: F200W, other visit)',
         'sgrb2': 'Sgr B2 F187N (ref: F182M, same visit)',
         'w51': 'W51 F187N (ref: F182M, same visit)'}
VARIANT = os.environ.get('KBAND_VARIANT', 'promq0b-promqeb')
WHAT = {'promq0b-promqeb': 'current base (pr/faint-reference-fields ef01f404): sources whose keep depends on the bound',
        'prom2': '#1018-era base: #1018 additions over the #1015 base'}
MIN_N = 10     # bands with fewer sources in the reference footprint: open marker, no error bar


def _lab(a, b):
    if np.isnan(a):
        return 'no k_eff'
    if a == 0:
        return f'≤{b:g}'
    if not np.isfinite(b):
        return f'>{a:g}'
    return f'{a:g}–{b:g}'


def rows(field, v=VARIANT):
    with open(f'{HERE}/kband_qsnr_{field}_{v.replace("-", "_minus_")}.json') as fh:
        d = json.load(fh)
    out = []
    for r in d:
        n, m = r.get('r_n', 0), r.get('r_match', np.nan)
        den = r.get('r_match_exp', np.nan) - r.get('r_chance_exp', np.nan)
        err = np.sqrt(m * (1 - m) / n) / den if n >= MIN_N and den > 0 else np.nan
        out.append((r['k_lo'], r['k_hi'], r['n_added'], r.get('r_rel', np.nan), err, n))
    return out


def main(out, *fields):
    fields = fields or ('brick', 'sgrb2', 'w51')
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4), layout='constrained',
                                 gridspec_kw=dict(width_ratios=(1.5, 1)))
    for j, (c, f) in enumerate(zip(('C0', 'C1', 'C2'), fields)):
        rr = rows(f)
        x = np.arange(len(rr)) + (j - 1) * 0.12
        rel = np.array([r[3] for r in rr], float)
        err = np.array([r[4] for r in rr], float)
        small = np.array([r[5] for r in rr]) < MIN_N
        a1.plot(x, rel, color=c, ls='-')
        a1.errorbar(x[~small], rel[~small], yerr=err[~small], color=c, marker='o', capsize=3, ls='none',
                    label=NAMES.get(f, f))
        a1.plot(x[small], rel[small], 'o', mfc='w', mec=c, ls='none')
        fin = [r for r in rr if np.isfinite(r[0]) and np.isfinite(r[1])]
        kk = np.array([0] + [b for _, b, *_ in fin])
        cum = np.cumsum([0] + [r[2] for r in fin])
        a2.plot(kk, cum, color=c, marker='o', label=NAMES.get(f, f))
        a2.axhline(sum(r[2] for r in rr), color=c, ls='--', lw=0.8)
        print(f, ' '.join(f'[{_lab(a, b)}] n {n} rel {r:.2f}+/-{e:.2f}' for a, b, n, r, e, _ in rr))
    a1.axhline(0.5, color='k', ls=':', lw=1)
    a1.axhline(1.0, color='0.6', lw=0.5)
    i5 = [_lab(a, b) for a, b, *_ in rows(fields[0])].index('4.5–5')
    a1.axvline(i5 + 0.5, color='r', lw=1, alpha=0.5)
    a1.set_xticks(np.arange(len(rr)), [_lab(a, b) for a, b, *_ in rr], fontsize=8)
    a1.set_xlabel('k_eff = S/N·sqrt(qfit² − 0.2²) band (the prominence-keep additions admitted when k crosses it)')
    a1.set_ylabel('realness of the band\'s additions')
    a1.set_title('marginal realness per k step (prominence ≥ 7)\nred line: default k = 5 (bands right of it are refused); '
                 'open marker: < 10 sources',
                 fontsize=9)
    a1.legend(fontsize=7, loc='upper right')
    a2.axvline(5.0, color='r', lw=1, alpha=0.5)
    a2.set_xlabel('k in the bound qfit ≤ sqrt(0.2² + (k / S/N)²)')
    a2.set_ylabel('cumulative additions (prominence ≥ 7)')
    a2.set_title('additions admitted at each k\n(dashed: total without the bound; qfit_snr_k = 0 turns it off)',
                 fontsize=9)
    a2.legend(fontsize=7, loc='lower right')
    fig.suptitle(WHAT.get(VARIANT, VARIANT), fontsize=10)
    fig.savefig(out, dpi=110)
    print(out)


if __name__ == '__main__':
    main(*sys.argv[1:])
