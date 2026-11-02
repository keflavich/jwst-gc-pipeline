"""Sum the phase-loss summaries over seeds, per field / band / variant.

usage: python aggregate_ref.py <variant[,variant...]> <outdir of run_ref.py>
writes <outdir>/ref_lost_<variants>.fits (every lost row, all runs)
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
from astropy.table import Table, vstack

variants, OUT = sys.argv[1].split(','), sys.argv[2]
rows = []
for p in sorted(glob.glob(f'{OUT}/*_summary.json')):
    m = re.match(r'(.+)_([a-z0-9]+)_s(\d+)_(f\w+)_summary.json', os.path.basename(p))
    field, variant, seed, filt = m.group(1), m.group(2), int(m.group(3)), m.group(4)
    if variant not in variants:
        continue
    s = json.load(open(p))
    lost = Table.read(p.replace('_summary.json', '_lost.fits'))
    if len(lost) == 0:
        continue
    lost['field'], lost['variant'], lost['seed'], lost['filt'] = field, variant, seed, filt
    rows.append(lost)
T = vstack(rows, metadata_conflicts='silent')
T.write(f'{OUT}/ref_lost_{"_".join(variants)}.fits', overwrite=True)
sl = np.asarray(T['starlike_left'], bool)
for (field, filt) in sorted(set(zip(T['field'], T['filt']))):
    print(f'== {field} {filt}')
    for v in variants:
        m = (T['field'] == field) & (T['filt'] == filt) & (T['variant'] == v)
        nseed = len(set(T['seed'][m]))
        mech = defaultdict(lambda: [0, 0])
        for r in np.flatnonzero(m):
            ph = str(T['last_phase'][r]).replace('resbgsub_', '')
            key = f"{ph}->{T['category'][r]}"
            if T['seed_reason'][r]:
                key += f":{T['seed_reason'][r]}"
            mech[key][0] += 1
            mech[key][1] += int(sl[r])
        tot = int(m.sum()); tsl = int((m & sl).sum())
        print(f'  {v:8s} seeds={nseed:2d} lost={tot:4d} starlike_left={tsl:4d} ({tsl / max(nseed, 1):.1f}/run)  ' +
              '  '.join(f'{k}={a}/{b}' for k, (a, b) in sorted(mech.items(), key=lambda kv: -kv[1][1])))
