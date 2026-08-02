"""Aggregate the multi-field injection-recovery runs into bias-vs-saturation-regime
curves + a summary table. Reads all inj_recovery_*.ecsv under the fleet roots
(one per field x filter x f_bf) and produces:

  - a per-field panel figure: recovered-injected mag vs saturation depth, with
    f_bf=0 (no charge migration) and f_bf>0 overlaid;
  - a summary ecsv: field, filter, detector, NGROUPS, f_bf, per-regime median
    bias / scatter / N;
  - the #210 test: does charge migration (f_bf) shift the bias more for the
    NGROUPS=2 field (gc2211, charge_migration OFF) than NGROUPS>=3 fields?
"""
import glob
import os
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.table import Table, vstack

import json as _json
# ROOTS maps an f_bf label -> fleet output dir. Override via SYN_ROOTS env
# (JSON, e.g. '{"2.0": "/.../final"}').
ROOTS = {float(k): v for k, v in _json.loads(os.environ.get('SYN_ROOTS', _json.dumps(
    {'0.0': '/blue/adamginsburg/adamginsburg/tmp/satinject_fleet/fbf0.0',
     '0.3': '/blue/adamginsburg/adamginsburg/tmp/satinject_fleet/fbf0.3'}))).items()}
OUT = os.environ.get('SYN_OUT', '/blue/adamginsburg/adamginsburg/tmp/satinject_fleet/synthesis')
REGIMES = [(1, 30, 'mild'), (30, 150, 'moderate'), (150, 500, 'deep'), (500, 1e9, 'hard')]


def _ngroups(field):
    return 2 if field == 'gc2211' else 7   # gc2211 wide readout = NGROUPS 2 (cm OFF)


def collect():
    rows = []
    per_run = {}
    for fbf, root in ROOTS.items():
        for f in sorted(glob.glob(f'{root}/*/inj_recovery_*.ecsv')):
            base = os.path.basename(f).replace('inj_recovery_', '').replace('.ecsv', '')
            m = re.match(r'(.+)_(F\d+[A-Z])_(NRC\w+)', base)
            if not m:
                continue
            field, filt, det = m.groups()
            t = Table.read(f)
            t = t[t['matched'] & np.isfinite(t['dmag'])]
            per_run[(field, filt, det, fbf)] = t
            for lo, hi, lab in REGIMES:
                s = (t['sat_area'] >= lo) & (t['sat_area'] < hi)
                if s.sum() >= 3:
                    d = np.asarray(t['dmag'][s]) * 1000
                    rows.append((field, filt, det, _ngroups(field), fbf, lab,
                                 int(s.sum()), float(np.median(d)),
                                 float(np.std(d)),
                                 float(np.median(d) / max(np.std(d) / np.sqrt(s.sum()), 1e-6))))
    summ = Table(rows=rows, names=('field', 'filter', 'detector', 'ngroups', 'f_bf',
                                   'regime', 'N', 'bias_mmag', 'scatter_mmag', 'signif'))
    return summ, per_run


def main():
    os.makedirs(OUT, exist_ok=True)
    summ, per_run = collect()
    if not len(summ):
        raise SystemExit('[syn] no matched injection results found (fleet not done?)')
    summ.write(f'{OUT}/injection_bias_summary.ecsv', overwrite=True)

    print('=' * 92)
    print('SATURATED-STAR RECOVERY BIAS (recovered - injected, mmag; +ve = too faint)')
    print('=' * 92)
    print(f'  {"field":9} {"filt":6} {"NG":>2} {"f_bf":>4} {"regime":9} {"N":>4} '
          f'{"bias":>7} {"scatter":>8} {"signif":>6}')
    for r in summ:
        print(f'  {r["field"]:9} {r["filter"]:6} {r["ngroups"]:2d} {r["f_bf"]:4.1f} '
              f'{r["regime"]:9} {r["N"]:4d} {r["bias_mmag"]:+7.0f} {r["scatter_mmag"]:8.0f} '
              f'{r["signif"]:+6.1f}')

    # #210 test: mean |bias shift from f_bf 0->0.3| for NGROUPS=2 vs 7
    print('\n' + '=' * 60)
    print('CHARGE-MIGRATION (f_bf) EFFECT vs NGROUPS  (#210 test)')
    print('=' * 60)
    for ng in (2, 7):
        shifts = []
        for (field, filt, det, fbf), t in per_run.items():
            if fbf != 0.0 or _ngroups(field) != ng:
                continue
            t3 = per_run.get((field, filt, det, 0.3))
            if t3 is None:
                continue
            b0 = np.median(t['dmag']) * 1000
            b3 = np.median(t3['dmag']) * 1000
            shifts.append(b3 - b0)
        if shifts:
            print(f'  NGROUPS={ng}: median |bias shift 0->0.3| = {np.median(np.abs(shifts)):.0f} mmag '
                  f'(N_runs={len(shifts)})  charge_migration '
                  f'{"OFF (excess kept)" if ng==2 else "ON (excess flagged)"}')

    # per-field panels
    fields = sorted(set((r['field'], r['filter'], r['detector']) for r in summ))
    ncol = 3; nrow = int(np.ceil(len(fields) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4 * nrow), squeeze=False)
    for k, (field, filt, det) in enumerate(fields):
        a = axes[k // ncol][k % ncol]
        for fbf, c in [(0.0, 'tab:blue'), (0.3, 'tab:red')]:
            t = per_run.get((field, filt, det, fbf))
            if t is None or not len(t):
                continue
            a.scatter(t['sat_area'], np.asarray(t['dmag']) * 1000, s=10, alpha=0.4,
                      color=c, label=f'f_bf={fbf}')
            # binned median
            for lo, hi, _ in REGIMES:
                s = (t['sat_area'] >= lo) & (t['sat_area'] < hi)
                if s.sum() >= 3:
                    a.plot(np.median(t['sat_area'][s]), np.median(t['dmag'][s]) * 1000,
                           'o', color=c, ms=9, mec='k')
        a.axhline(0, color='0.5'); a.axhline(20, color='0.7', ls=':'); a.axhline(-20, color='0.7', ls=':')
        a.set_xscale('symlog'); a.set_ylim(-400, 400)
        a.set_title(f'{field} {filt} (NG={_ngroups(field)})', fontsize=10)
        a.set_xlabel('saturated-pixel area'); a.set_ylabel('rec - inj (mmag)')
        a.legend(fontsize=8)
    for k in range(len(fields), nrow * ncol):
        axes[k // ncol][k % ncol].axis('off')
    fig.suptitle('Saturated-star recovery bias vs saturation depth, per field '
                 '(+ve = recovered too faint)', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p = f'{OUT}/injection_bias_by_field.png'
    fig.savefig(p, dpi=120)
    print(f'\n[syn] wrote {p}')
    print(f'[syn] wrote {OUT}/injection_bias_summary.ecsv')


if __name__ == '__main__':
    main()
