"""Round 4: q = cal/(R(g0) g0) vs distance for LW frames.  usage: python q4_collect.py BAND VG EXP DET   -> out4/raw_<band>_<exp>_<det>.pkl
Reuses the read-only wingmig_collect.run (same selections and R(g0) method)."""
import sys, pickle
import numpy as np
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/wingmig')
import wingmig_collect as W
band, vg, exp, det = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
o = W.run(band, vg, exp, det)
pickle.dump(o, open(f'/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit/out4/raw_{band}_{exp}_{det}.pkl', 'wb'))
print(band, exp, det, o['meta'], 'nsat', len(o['sats']), 'nctl', len(o['ctl']), flush=True)
