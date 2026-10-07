"""Moved stars split by forced_filled status in A and B.  usage: ffsplit.py A B"""
import sys
import numpy as np
from astropy.table import Table
A_, B_ = sys.argv[1:3]
Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/faint_sw'
S = Table.read(f'{Q}/stars_{A_}_{B_}.ecsv'); F = Table.read(f'{Q}/frames_{A_}_{B_}.ecsv')
L = ['| band | forced_filled (A, B) | N | median B-A | frac B fainter |', '|---|---|---|---|---|']
for b in ['150W', '182M', '187N', '162M']:
    s = S[(S['band'] == b) & S['moved']]
    a = np.asarray(s['forced_filled_A'], float) > 0; bb = np.asarray(s['forced_filled_B'], float) > 0
    for nm, m in (('ffA & ffB', a & bb), ('neither', ~a & ~bb), ('only B', ~a & bb), ('only A', a & ~bb)):
        if m.sum():
            L.append(f'| F{b} | {nm} | {m.sum()} | {np.median(s["dBA"][m]):+.2f} | {np.mean(s["dBA"][m] > 0):.2f} |')
    f = F[F['band'] == b]
    L.append(f'| F{b} | moved stars with >= 1 A per-frame daophot row within 0.1" | {len(set(f["i"]) & set(s["i"]))} of {len(s)} | | |')
open(f'{Q}/ffsplit_{A_}_{B_}.md', 'w').write('\n'.join(L)); print('\n'.join(L))
