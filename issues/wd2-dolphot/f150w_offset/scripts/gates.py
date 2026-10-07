import re,sys
from astropy.io import fits
L='/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/logs/wd23523-o005-Q-mainfcbg-fanout_%s.out'
T='/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/tree_mainfcbg'
VG={'F115W':'08101','F150W':'10101','F162M':'14101','F182M':'16101','F200W':'12101'}
def nums(s): return [int(x) for x in re.findall(r'\d+',s)]
def parse(logid,mod):
    lines=open(L%logid,errors='replace').read().splitlines()
    # band segments delimited by "Completed basic photometry" lines for that module
    idx=[i for i,l in enumerate(lines) if 'Completed basic photometry' in l and f'_{mod}_visit001' in l]
    out={}; start=0
    bands=[re.search(r'/(F\d+[WMN])/',lines[i]).group(1) for i in idx]
    ends={}
    for i,b in zip(idx,bands): ends[b]=max(ends.get(b,0),i)
    prev=0
    for b in sorted(ends,key=lambda k:ends[k]):
        seg=lines[prev:ends[b]+1]; prev=ends[b]+1
        out[b]=seg
    return out
def chains(seg):
    dao=[];ch=[];cur=None;osh={}
    for l in seg:
        m=re.search(r'daofind: (\d+) -> (\d+) after',l)
        if m: dao.append([int(m[1]),int(m[2]),None,None])
        m=re.search(r'seed dedup: removed (\d+)',l)
        if m and dao and dao[-1][2] is None: dao[-1][2]=int(m[1])
        m=re.search(r'seed footprint subset: (\d+) -> (\d+)',l)
        if m: dao.append([None,None,None,int(m[2])]) if False else None
        m=re.search(r'post-fit dedup: (\d+) -> (\d+)',l)
        if m: cur={'fit_in':int(m[1]),'dedup':int(m[2])}; ch.append(cur)
        if cur is None: continue
        m=re.search(r'dropping (\d+) fits within.*\((\d+) -> (\d+)\)',l)
        if m: cur['nearsat_drop']=int(m[1]); cur['after_nearsat']=int(m[3])
        m=re.search(r'Satstar-artifact filter.*dropping (\d+) fits.*; (\d+) -> (\d+)',l)
        if m: cur['satart_drop']=int(m[1]); cur['after_satart']=int(m[3])
        m=re.search(r'overshoot \(>1.2x\): (\d+)/(\d+) fits',l)
        if m: osh[int(m[2])]=int(m[1])
    ph=[int(re.search(r'dropped (\d+) phantom',l)[1]) for l in seg if 'phantom' in l and 'dropped' in l]
    npf=[nums(re.search(r'dropped (\d+) non-positive',l)[0])[0] for l in seg if 'non-positive-flux' in l]
    fin=[int(re.search(r'len\(result\) = (\d+)',l)[1]) for l in seg if 'len(result)' in l]
    for c in ch: c['overshoot_flagged']=osh.get(c.get('after_satart'))
    return dao,ch,ph,npf,fin
rows=[]
for mod,logid in (('nrca1','44737260_0'),('nrcb3','44737260_6')):
    segs=parse(logid,mod)
    for b in ('F115W','F150W','F162M','F182M','F200W'):
        if b not in segs: print('missing',mod,b); continue
        dao,ch,ph,npf,fin=chains(segs[b])
        target=fits.getheader(f'{T}/{b}/{b.lower()}_{mod}_visit001_vgroup{VG[b]}_exp00001_resbgsub_m7_daophot_basic.fits',1)['NAXIS2']
        # chain whose final == target: after_satart - phantom - nonpos
        got=None
        for c in ch:
            for p in ph:
                for n in npf:
                    if c.get('after_satart',0)-p-n==target: got=(c,p,n)
        # pair daofind by size order with chain order
        dsort=sorted(dao,key=lambda d:d[1]); csort=sorted(ch,key=lambda c:c['fit_in'])
        d=dsort[csort.index(got[0])] if got else None
        rows.append((b,mod,d,got,target))
hdr='band\tframe\tdaofind_raw\tdaofind_kept_localSN3\tkept_frac\tseed_dedup\tfit_in(post-fit dedup in)\tpostdedup\tnearsat_drop\tsatartifact_drop\tovershoot_flagged\tphantom_drop\tnonpos_drop\tfinal_rows\tfile_rows'
with open('gate_table_exp00001.tsv','w') as o:
    o.write(hdr+'\n')
    for b,mod,d,got,t in rows:
        if got is None: o.write(f'{b}\t{mod}\tNOMATCH\t\t\t\t\t\t\t\t\t\t\t\t{t}\n'); continue
        c,p,n=got
        o.write('\t'.join(map(str,[b,mod+'_exp1',d[0],d[1],round(d[1]/d[0],3),d[2],c['fit_in'],c['dedup'],c['nearsat_drop'],c['satart_drop'],c.get('overshoot_flagged'),p,n,c['after_satart']-p-n,t]))+'\n')
print(open('gate_table_exp00001.tsv').read())
