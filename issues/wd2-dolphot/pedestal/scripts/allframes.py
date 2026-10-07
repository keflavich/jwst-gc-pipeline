import numpy as np, warnings, json, glob, sys
from astropy.io import fits
warnings.filterwarnings('ignore')
P='/orange/adamginsburg/jwst/wd2/'
def med(a):
    a=a[np.isfinite(a)&(a!=0)]; return float(np.median(a[::7])) if a.size else np.nan
res={}
for b in sys.argv[1:]:
    bl=b.lower()
    a=json.load(open(glob.glob(f'{P}{b}/pipeline/*{bl}-merged_resbgsub_m6_daophot_basic_mergedcat_residual_asn.json')[0]))
    out=[]
    for m in a['products'][0]['members']:
        n=m['expname'].split('/')[-1]
        det=n.split('-')[-1].split('_')[0]; vg=n.split('vgroup')[1].split('_')[0]; ex=n.split('_exp')[1][:5]
        base=f'{P}{b}/pipeline/jw03523005001_{vg}_{ex}_{det}_'
        r=med(fits.getdata(m['expname'],'SCI'))
        al=med(fits.getdata(base+'align_o005_crf.fits','SCI'))
        try: ds=med(fits.getdata(base+'destreak_o005_crf.fits','SCI'))
        except FileNotFoundError: ds=np.nan
        out.append((det,ex,r,al,ds))
    res[b]=out
    arr=np.array([(o[2],o[3],o[4]) for o in out],float)
    print(b,'n',len(out),'median over frames: m6 resid %.3f  align crf %.3f  destreak crf %.3f'%tuple(np.nanmedian(arr,axis=0)))
    print('  by detector (first exposure):',[(o[0],round(o[2],2),round(o[3],2),round(o[4],2)) for o in out if o[1]=='00001'])
json.dump({k:[list(map(lambda x: x if isinstance(x,str) else float(x),o)) for o in v] for k,v in res.items()},open('allframes_'+'_'.join(sys.argv[1:])+'.json','w'))
