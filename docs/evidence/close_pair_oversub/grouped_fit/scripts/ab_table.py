"""Markdown table of ab_gallery_m7.py counts, seeds pooled per (field, band).

usage: python ab_table.py <ab_counts_A_B.json> <ab_counts_B_A.json> > table.md
"""
import collections
import json
import sys

COLS = ('n_inner_b', 'n_b_only', 'n_left_in_a_snr5', 'n_left_in_b_snr5', 'n_near_satstar', 'n_injected')

print('| B-only = in B, no A source within 1 px | n B (inner) | B-only | B-only with S/N > 5 left in A residual '
      '| B-only with S/N > 5 left in B residual | B-only near a satstar | B-only injected |')
print('|---|---|---|---|---|---|---|')
for path in sys.argv[1:]:
    d = json.load(open(path))
    agg = collections.OrderedDict()
    for r in d['summary']:
        a = agg.setdefault((r['field'], r['filt']), collections.Counter())
        for c in COLS:
            a[c] += r[c]
    for (field, filt), a in agg.items():
        print(f'| {field} {filt}, A={d["var_a"]} B={d["var_b"]} | ' + ' | '.join(str(a[c]) for c in COLS) + ' |')
