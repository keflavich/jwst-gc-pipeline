#!/usr/bin/env python
"""Summarise the baseline-vs-deblend satstar comparison once the slurm jobs finish.

Reads out_compare/satcmp_{baseline,deblend}_*.log: prints accepted-row counts +
runtime, and a histogram of per-source rejection reasons parsed from the
'Skipping source N: snr=..., qfit=..., ssr_ratio=...' lines.
"""
import os, glob, re
from collections import Counter

LOG = os.path.join(os.path.dirname(__file__), 'out_compare')
SKIP = re.compile(r'Skipping source \d+: snr=([^,]+), fluxerr=([^,]+), '
                  r'qfit=([^,]+), sidelobe_resid_sigma=([^,]+), ssr_ratio=([^,\s]+)')

for mode in ('baseline', 'deblend'):
    logs = sorted(glob.glob(f'{LOG}/satcmp_{mode}_*.log'))
    if not logs:
        print(f'[{mode}] no log yet'); continue
    txt = open(logs[-1]).read()
    res = re.search(r'COMPARE_RESULT mode=\S+ accepted=(\d+) seconds=(\d+)', txt)
    done = 'COMPARE_DONE' in txt
    comp = re.search(r'(\d+) components -> (\d+) star seeds', txt)
    reasons = Counter()
    for m in SKIP.finditer(txt):
        snr, ferr, qfit, sl, ssr = m.groups()
        try:
            if ferr.strip() == 'nan' or snr.strip() == 'nan':
                reasons['nan_fluxerr/snr'] += 1
            elif float(qfit) > 5.0:
                reasons['qfit>5'] += 1
            elif float(ssr) > 1.0:
                reasons['ssr_ratio>1'] += 1
            elif float(sl) < -10.0:
                reasons['sidelobe<-10'] += 1
            else:
                reasons['other'] += 1
        except ValueError:
            reasons['parse_fail'] += 1
    nskip = sum(reasons.values())
    seeds = comp.group(2) if comp else '?'
    acc = res.group(1) if res else '?'
    secs = res.group(2) if res else '?'
    print(f'[{mode}] done={done} seeds={seeds} accepted={acc} '
          f'skipped={nskip} runtime={secs}s')
    for k, v in reasons.most_common():
        print(f'    {k}: {v}')
