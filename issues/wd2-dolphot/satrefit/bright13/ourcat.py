import sys, numpy as np
sys.path.insert(0,'/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an
from astropy.table import Table
an.ZPWIN.update(an.zp_windows())
A = an.Arm('main2')
for b in ('250M','300M'):
    c = A.cat
    cols=[x for x in c.colnames if b.lower() in x.lower()]
    print(b, cols[:30])
    break
