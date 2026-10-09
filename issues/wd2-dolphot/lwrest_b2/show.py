import sys, numpy as np
from astropy.table import Table
b=sys.argv[1]; cat=sys.argv[2]
S=Table.read(f'stars_{b}.ecsv'); R=Table.read(f'rows_{b}.ecsv')
for s in S[S['cat']==cat]:
    r=R[R['dolphot_idx']==s['dolphot_idx']]
    print(f"=== {b} idx={s['dolphot_idx']} ref={s['ref_mag']:.2f} nfr1px={s['n_frames_row_1px']}")
    for k in ('frame','pre','final','satstar'):
        q=r[r['kind']==k]; q.sort('sep')
        for x in q[:7 if k!='frame' else 12]:
            ex=f" shift_to_final={x['x']:.3f} satsep={x['y']:.3f}" if k=='pre' else (f" satsep={x['y']:.3f}" if k=='final' else '')
            print(f"  {k:7s} {x['where']:>16s} sep={x['sep']:.3f} mag={x['mag']:.2f} n={x['nmatch']} fl={x['flags']:.2f} q={x['qfit']:.2f} rep={x['rep']}{ex}" + (f" frc={x['forced']} gs={x['gsize']}" if k=='frame' else ''))
