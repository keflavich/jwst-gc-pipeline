"""Evaluate replace_saturated second-pass veto rules with dolphot as truth. wrong = dolphot star at the old (pre-replacement) position AND a distinct dolphot star at the satstar position (a real neighbour is overwritten);
right = dolphot star only at the satstar position (the old row was a junk/clipped row). b2_fixed = b2 R2 stars whose old row is kept and has |oldmag-ref|<0.3."""
import numpy as np, warnings
from astropy.table import Table
warnings.filterwarnings('ignore')
out = []
M = Table.read('mech_all.ecsv')
rules = {'dmag<1.5': lambda p, dm: dm < 1.5, 'dmag<2.5': lambda p, dm: dm < 2.5, 'dmag<3.5': lambda p, dm: dm < 3.5,
         'qfit<0.10': lambda p, dm: p['qfit'] < 0.10, 'qfit<0.15': lambda p, dm: p['qfit'] < 0.15, 'qfit<0.20': lambda p, dm: p['qfit'] < 0.20,
         'qfit<0.15 & dmag<4': lambda p, dm: (p['qfit'] < 0.15) & (dm < 4), 'qfit<0.15 & satsep>0.25': lambda p, dm: (p['qfit'] < 0.15) & (p['satsep'] > 0.25),
         'qfit<0.10 & satsep>0.25': lambda p, dm: (p['qfit'] < 0.10) & (p['satsep'] > 0.25)}
for b in ('277W', '250M', '300M'):
    t = Table.read(f'second_pass_{b}.ecsv'); p = t[t['pass'] == 2]
    dm = np.asarray(p['oldmag'] - p['satmag'])
    wrong = np.asarray((p['dold'] < 0.08) & (p['dsat'] < 0.08) & ~p['same_dol'])
    right = np.asarray((p['dsat'] < 0.08) & (p['dold'] >= 0.08))
    other = ~wrong & ~right
    m = M[(M['band'] == b) & (M['cat'] == 'b2') & (M['mech'] == 'R2_replace_sat_second_pass')]
    for name, fn in rules.items():
        veto = np.asarray(np.ma.filled(fn(p, dm), False), bool)
        fixed = 0
        for r in m:
            qi = np.where((p['old_dol_idx'] == r['dolphot_idx']) & (p['dold'] < 0.08))[0]
            if len(qi) and veto[qi[0]] and abs(p['oldmag'][qi[0]] - r['ref_mag']) < 0.3: fixed += 1
        out.append(dict(band=b, rule=name, n_second_pass=len(p), vetoed=int(veto.sum()), v_wrong=int((veto & wrong).sum()), v_right=int((veto & right).sum()), v_other=int((veto & other).sum()),
                        wrong_total=int(wrong.sum()), right_total=int(right.sum()), b2_R2=len(m), b2_fixed=fixed))
O = Table(rows=out); O.write('rule_eval.ecsv', overwrite=True)
O.pprint_all()
