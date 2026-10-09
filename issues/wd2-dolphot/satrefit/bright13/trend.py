import pickle, numpy as np
Q='/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit/out7/'
def mad(x): return 1.4826*np.median(np.abs(x-np.median(x)))
for b in ('250M','300M','150W','200W'):
    pk=pickle.load(open(Q+f'score7_{b}.pkl','rb')); ref=pk['ref']; h=pk['have']
    print('\n',b,'N have',h.sum(),'ref range',ref[h].min(),ref[h].max(),'unsat_dm',round(pk['unsat_dm'],3))
    edges=[10,12.5,12.7,12.85,13,13.5,14,15,16,17,18]
    print('bin     N  | final | H+cap | H+bgfree+cap | uncapped   (median, MAD)')
    for lo,hi in zip(edges[:-1],edges[1:]):
        s=h&(ref>=lo)&(ref<hi)
        if s.sum()<3: continue
        print(f'{lo}-{hi} {s.sum():4d} | '+' | '.join(f"{np.median(pk['dm'][k][s]):+.3f} ({mad(pk['dm'][k][s]):.3f})" for k in ('final','H+cap','H+bgfree+cap','uncapped')))
