import numpy as np, sys, warnings
from astropy.table import Table
warnings.filterwarnings('ignore')
b=sys.argv[1]
T='/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/tree_main2kfpk'
pre=Table.read(f'/blue/adamginsburg/adamginsburg/tmp/claude-3663/pre_{b}.fits')
fin=Table.read(f'{T}/catalogs/f{b.lower()}_merged_indivexp_merged_resbgsub_m7_dao_basic.fits')
print(len(pre),len(fin), fin['replaced_saturated'].sum())
print(pre['qfit_avg'].dtype, fin['qfit'].dtype)
def key(q,c,f,n): return list(zip(np.round(np.asarray(q,float),7),np.round(np.asarray(c,float),7),np.round(np.asarray(f,float),5),np.asarray(n)))
kp=key(pre['qfit_avg'],pre['cfit_avg'],pre['flux_init_avg'],pre['nmatch'])
kf=key(fin['qfit'],fin['cfit'],fin['flux_init'],fin['nmatch'])
from collections import defaultdict
d=defaultdict(list)
for i,k in enumerate(kf): d[k].append(i)
m=np.full(len(pre),-1); amb=0
for i,k in enumerate(kp):
    if k in d and len(d[k])==1: m[i]=d[k][0]
    elif k in d: amb+=1
print('mapped',(m>=0).sum(),'ambiguous',amb)
