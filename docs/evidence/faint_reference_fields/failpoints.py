"""True-value shifts at which each reference-field threshold fails with
probability 0.5 and 0.9 (same noise model as evaluate.pass_probability).

usage: python failpoints.py jwst_gc_pipeline/photometry/reference_fields/fields.yaml
"""
import math
import sys
import yaml
from scipy.stats import binom, norm
from scipy.optimize import brentq

cfg = yaml.safe_load(open(sys.argv[1]))


def p_pass_bin(p, n, f):
    kmin = math.ceil(round(f * n, 9))
    return binom.sf(kmin - 1, n, p)


def p_at(n, f, pfail):
    return brentq(lambda p: p_pass_bin(p, n, f) - (1 - pfail), 1e-9, 1 - 1e-9)


for name, spec in cfg['fields'].items():
    thr, cal = spec['thresholds'], spec['calibration']
    print(f'## {name}')
    for b, f in thr.get('completeness_min', {}).items():
        k, n = cal['completeness'][b]
        p5, p9 = p_at(n, f, 0.5), p_at(n, f, 0.9)
        print(f'  completeness {b}: cal {k}/{n}={k/n:.2f} floor {f}: fail50 at {p5:.2f} ({p5*n:.1f}/{n}, -{(k/n-p5)*n:.1f} stars)'
              f' fail90 at {p9:.2f} ({p9*n:.1f}/{n}, -{(k/n-p9)*n:.1f} stars)')
    if 'labels_recovered_min' in thr:
        k, n = cal['labels_recovered']
        f = thr['labels_recovered_min']
        p5, p9 = p_at(n, f, 0.5), p_at(n, f, 0.9)
        print(f'  labels: cal {k}/{n} floor {f}: fail50 at {p5*n:.1f}/{n} fail90 at {p9*n:.1f}/{n}')
    for key in ('residual_excess', 'oversubtracted'):
        if f'{key}_max' in thr:
            c = cal[key]; T = thr[f'{key}_max']
            print(f'  {key}: cal {c["clean"]} max {T} sd {c["seed_std"]}: fail50 at {T:.2f} (+{T-c["clean"]:.2f}) fail90 at {T+norm.ppf(0.9)*c["seed_std"]:.2f} (+{T+norm.ppf(0.9)*c["seed_std"]-c["clean"]:.2f})')
    if 'flux_bias_max_mag' in thr:
        c = cal['flux_bias_mag']; T = thr['flux_bias_max_mag']
        # fail50: |N(mu,e)|>T w.p. 0.5 -> mu ~ T (same-sign side)
        s = 1 if c['median'] >= 0 else -1
        f = lambda m, q: (norm.cdf(T, s*m, c['err']) - norm.cdf(-T, s*m, c['err'])) - (1 - q)
        m5 = brentq(lambda m: f(m, 0.5), 0, T + 1); m9 = brentq(lambda m: f(m, 0.9), 0, T + 1)
        print(f'  bias: cal {c["median"]:+.3f}+-{c["err"]} max {T}: fail50 at {s*m5:+.3f} fail90 at {s*m9:+.3f}')
