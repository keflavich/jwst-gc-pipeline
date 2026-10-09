import numpy as np
from astropy.table import Table
cfgs=['cur','a10','a1','a0','e1.0','e0.5','e0','a10e0.5','a1e0.5','a1e0']
out=['| band | config | rule | good stars reached (of N good) | lost stars reached (of N lost) |','|---|---|---|---|---|']
for b in ('277W','250M','300M'):
    s=Table.read(f'class_{b}.ecsv'); r=Table.read(f'reach_{b}.ecsv')
    good=dict(zip(s['dolphot_idx'],s['good']))
    # per-star: reached if any frame hand-off within RAD AND (restore: pixel restored | handoff only)
    ids=np.unique(r['dolphot_idx'])
    gi=np.array([good[i] for i in ids])
    def reach(col):
        return np.array([r[col][r['dolphot_idx']==i].any() for i in ids])
    for c in cfgs:
        for rule in ('H','R0','R1'):
            if rule=='H': v=reach('h_'+c)
            else:
                v=np.array([ (r['h_'+c]&r[f'r_{c}|{rule}'])[r['dolphot_idx']==i].any() for i in ids])
            out.append(f'| {b} | {c} | {rule} | {(v&gi).sum()} / {gi.sum()} | {(v&~gi).sum()} / {(~gi).sum()} |')
    # comp_has_acc stats for b1
    b1=s['dolphot_idx'][[c.startswith('b1') for c in s['cat']]]
    g=r[np.isin(r['dolphot_idx'],b1)&r['h_cur']]
    out.append(f'| {b} | b1 star-frames handed off (cur): {len(g)}; component holds accepted centre: {g["comp_has_acc"].sum()}; pixel restored R0: {g["r_cur|R0"].sum()}; R1: {g["r_cur|R1"].sum()}; bad_pix {g["bad_pix"].sum()} | | | |')
open('reach.md','w').write('\n'.join(out)+'\n'); print('\n'.join(out))
