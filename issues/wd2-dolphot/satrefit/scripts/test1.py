import sys, numpy as np
sys.path.insert(0,'/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit')
import satrefit_core as C
fn=C.Q+'/tree_main2/F150W/pipeline/jw03523005001_10101_00001_nrcb1_align_o005_crf.fits'
hdr=C.fits.getheader(fn)
grid,gf=C.load_grid(C.Q+'/tree_main2/psfs',hdr,False)
C.log('grid',gf)
V={'g4':{'guard':4},'nbr_all':{'nbr':'all'},'nbr_none':{'nbr':'none'},'fill':{'fill':1}}
t=C.run_frame(fn,'150W',V,grid,max_rows=None)
t.write('test1.fits',overwrite=True)
ok=np.isfinite(t['a_base'])
r=t['a_base'][ok]/t['a_cat'][ok]
print('N',ok.sum(),len(t),'median ratio',np.median(r),'16/84',np.percentile(r,[16,84]))
print(t.meta)
