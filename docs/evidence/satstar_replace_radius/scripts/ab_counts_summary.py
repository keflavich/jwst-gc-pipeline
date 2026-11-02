"""Sum the per-run counts of ab_gallery_m7.py (n=0 runs) per field / band.

usage: python ab_counts_summary.py <counts.json> [<counts.json> ...]
"""
import json
import sys
from collections import defaultdict

KEYS = ('n_b_only', 'n_left_in_a_snr5', 'n_left_in_b_snr5', 'n_near_satstar',
        'n_near_satstar_left_in_a_snr5', 'n_injected', 'n_brighter_nb')

for p in sys.argv[1:]:
    d = json.load(open(p))
    tot = defaultdict(lambda: defaultdict(int))
    runs = defaultdict(int)
    for s in d['summary']:
        k = (s['field'], s['filt'])
        runs[k] += 1
        for key in KEYS:
            tot[k][key] += int(s[key])
    print(f"B = {d['var_b']} m7 vetted sources with no {d['var_a']} (A) m7 vetted source within 1 px")
    print('| field / band | runs | ' + ' | '.join(KEYS) + ' |')
    print('|---' * (len(KEYS) + 2) + '|')
    for k in tot:
        print(f'| {k[0]} {k[1]} | {runs[k]} | ' + ' | '.join(str(tot[k][key]) for key in KEYS) + ' |')
    print('| all | ' + str(sum(runs.values())) + ' | '
          + ' | '.join(str(sum(tot[k][key] for k in tot)) for key in KEYS) + ' |')
    print()
