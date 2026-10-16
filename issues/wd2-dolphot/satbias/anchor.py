"""HST WFC3/IR (HSC v3 F125W/F160W) and Ascenso+2007 Ks anchors for satstars. usage: python anchor.py"""
import sys
sys.path.insert(0,'/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import numpy as np, analyze as an
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
a='main2kfpr'
an.PATH.setdefault(a, (f'{an.Q}/tree_{a}/{an.M8}', f'{an.D}/matched_Q_{a}.fits'))
an.ZPWIN.update(an.zp_windows())
A=an.Arm(a)
jw=SkyCoord(np.asarray(A.m['RA'])*u.deg,np.asarray(A.m['DEC'])*u.deg)
H=Table.read('hsc_wd2.fits')
hs=SkyCoord(np.asarray(H['MatchRA'],float)*u.deg,np.asarray(H['MatchDec'],float)*u.deg)
As=Table.read('ascenso.fits')
asc=SkyCoord(As['RAJ2000'],As['DEJ2000'],unit=(u.hourangle,u.deg))
def col(t,c):
    return np.ma.filled(np.ma.asarray(t[c]).astype(float),np.nan)
hm={'F125W':col(H,'W3_F125W'),'F160W':col(H,'W3_F160W')}
am={'H':col(As,'Hmag'),'Ks':col(As,'Ksmag')}
# JWST reference magnitude for isolation: brightest of dolphot SW bands
refmag=np.nanmin(np.vstack([A.ref[b] for b in ('115W','150W','162M','200W')]),axis=0)
def isolated(sky_ref,mag_ref,rad,mag_self,sel_idx):
    """True if no other source of sky_ref (mags mag_ref) within rad brighter than mag_self+3."""
    ok=np.zeros(len(sel_idx),bool)
    idx1,idx2,d,_=sky_ref.search_around_sky(sky_ref[sel_idx],rad*u.arcsec)
    bad=np.zeros(len(sel_idx),bool)
    other=sel_idx[idx1]!=idx2
    br=mag_ref[idx2]<mag_self[idx1]+3
    np.logical_or.at(bad,idx1[other&br],True)
    return ~bad
EDG=[13,14,15,16,17]
def binstat(res,mag,edges):
    out=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        k=(mag>=lo)&(mag<hi)&np.isfinite(res)
        n=k.sum()
        out.append((np.median(res[k]) if n>=3 else np.nan, 1.2533*an.mad(res[k])/np.sqrt(n) if n>=3 else np.nan, n))
    return out
def fitlin(c,d):
    k=np.isfinite(c)&np.isfinite(d)
    for _ in range(8):
        p=np.polyfit(c[k],d[k],1)
        r=d-np.polyval(p,c); s=an.mad(r[k])
        k=np.isfinite(c)&np.isfinite(d)&(np.abs(r)<3*s)
    return p,s,k.sum()
PAIRS=[('115W','HST','F125W','F125W','F160W'),
       ('150W','HST','F160W','F125W','F160W'),
       ('162M','HST','F160W','F125W','F160W'),
       ('200W','ASC','Ks','H','Ks')]
R={}
L=['# HST/ground anchor for satstars (main2kfpr)','']
for b,src,mname,c1,c2 in PAIRS:
    if src=='HST':
        ref,rm,rad,mt=hs,hm,0.10,hm
    else:
        ref,rm,rad,mt=asc,am,0.40,am
    m_ext=mt[mname]
    # match: nearest ref source to each dolphot row with a finite dolphot mag in band b
    cand=np.where(np.isfinite(A.ref[b]))[0]
    i,d,_=jw[cand].match_to_catalog_sky(ref)
    ok=(d<rad*u.arcsec)&np.isfinite(m_ext[i])
    # mutual: each ref source used once (keep closest)
    order=np.argsort(d.arcsec); seen=set(); keep=np.zeros(len(cand),bool)
    for j in order:
        if not ok[j]: continue
        if i[j] in seen: continue
        seen.add(i[j]); keep[j]=True
    sel=cand[keep]; ri=i[keep]
    nmatch=len(sel)
    # isolation in JWST (dolphot) and in external catalogue
    iso_rad=1.0 if src=='HST' else 1.5
    iso_j=isolated(jw,refmag,iso_rad,refmag,sel)
    # external: neighbours within iso_rad among external sources
    ext_m=m_ext
    ia,ib,dd,_=ref.search_around_sky(ref[ri],iso_rad*u.arcsec)
    bad=np.zeros(len(ri),bool)
    oth=ri[ia]!=ib
    brt=np.isfinite(ext_m[ib])&(ext_m[ib]<ext_m[ri][ia]+3)
    np.logical_or.at(bad,ia[oth&brt],True)
    iso=iso_j&~bad
    sel=sel[iso]; ri=ri[iso]
    miso=len(sel)
    colr=(rm[c1]-rm[c2])[ri]
    mext=m_ext[ri]
    s_flag=(A.sat[b]|A.rep[b])[sel]
    satmag=A.ref[b][sel][s_flag]
    L.append(f'## F{b} vs {mname} ({src}); colour {c1}-{c2}')
    L.append(f'dolphot rows with mag: {len(cand)}; matched within {rad}": {nmatch}; isolated: {miso}; with colour: {np.isfinite(colr).sum()}; satstars (flagged, isolated, matched): {np.isfinite(colr)[s_flag].sum()} of {s_flag.sum()}')
    if len(satmag)<5:
        L.append('too few'); continue
    lo_fit=17.0
    L.append(f'satstar dolphot mag 5/50/95 pct: {np.percentile(satmag,[5,50,95]).round(2)}; fit range for unsat: {lo_fit:.2f}-19.0 (dolphot mag)')
    res={}
    for name,mag in (('ours',A.our[b][sel]-A.zp[b]),('dolphot',A.ref[b][sel])):
        dm=mag-mext
        refm=A.ref[b][sel]
        uns=~s_flag&np.isfinite(dm)&np.isfinite(colr)&(refm>=lo_fit)&(refm<19.0)&(mag>0)
        p,s,n=fitlin(colr[uns],dm[uns])
        r=dm-np.polyval(p,colr)
        res[name]=(r,mag,p,s,n,uns)
        L.append(f'- {name}: fit unsat N={n}: m_JWST-m_ext = {p[1]:+.3f} {p[0]:+.3f}*({c1}-{c2}); rms (MAD) {s:.3f}; colour range unsat {np.nanpercentile(colr[uns],[2,98]).round(2)}, sat {np.nanpercentile(colr[s_flag&np.isfinite(colr)],[2,98]).round(2)}')
    refm=A.ref[b][sel]
    # unsat trend
    ue=np.arange(np.floor(lo_fit-1) ,19.01,1.0)
    ue=np.arange(np.floor(np.percentile(refm[~s_flag&np.isfinite(refm)],1)),19.01,1.0)
    L.append('')
    L.append('Unsaturated residual (data - colour relation; flat expected), by dolphot mag bin: median +/- err [N]')
    L.append('| anchor | '+' | '.join(f'{lo:.0f}-{hi:.0f}' for lo,hi in zip(ue[:-1],ue[1:]))+' |')
    L.append('|---|'+'---|'*(len(ue)-1))
    for name in ('ours','dolphot'):
        r=res[name][0]; bs=binstat(np.where(~s_flag,r,np.nan),refm,ue)
        L.append(f'| {name} | '+' | '.join(f'{m:+.3f}+/-{e:.3f} [{n}]' if n>=3 else f'- [{n}]' for m,e,n in bs)+' |')
    L.append('')
    L.append('Saturated residual by dolphot mag bin: median +/- err [N]')
    L.append('| anchor | '+' | '.join(f'{lo}-{hi}' for lo,hi in zip(EDG[:-1],EDG[1:]))+' | all |')
    L.append('|---|'+'---|'*(len(EDG)))
    tab={}
    for name in ('ours','dolphot'):
        r=res[name][0]; rs=np.where(s_flag,r,np.nan)
        bs=binstat(rs,refm,EDG); tab[name]=bs
        allb=binstat(rs,refm,[0,99])[0]
        L.append(f'| {name} | '+' | '.join(f'{m:+.3f}+/-{e:.3f} [{n}]' if n>=3 else f'- [{n}]' for m,e,n in bs)+f' | {allb[0]:+.3f}+/-{allb[1]:.3f} [{allb[2]}] |')
    ro=res['ours'][0]; rd=res['dolphot'][0]
    dif=np.where(s_flag,ro-rd,np.nan); bs=binstat(dif,refm,EDG)
    L.append('| ours-dolphot | '+' | '.join(f'{m:+.3f}+/-{e:.3f} [{n}]' if n>=3 else f'- [{n}]' for m,e,n in bs)+' | |')
    L.append('')
    R[b]=dict(res=res,refm=refm,s=s_flag,lo_fit=lo_fit,mname=mname,tab=tab,ue=ue)
open('hst_anchor.md','w').write('\n'.join(L)+'\n')
print('\n'.join(L))
fig,ax=plt.subplots(2,2,figsize=(11,8),sharex=True)
for axx,b in zip(ax.flat,[p[0] for p in PAIRS]):
    if b not in R: continue
    D=R[b]; refm=D['refm']; s=D['s']
    for name,c,off in (('ours','C3',-0.04),('dolphot','C0',0.04)):
        r=D['res'][name][0]
        axx.plot(refm[~s],r[~s],',',color=c,alpha=0.25)
        axx.plot(refm[s],r[s],'.',ms=2,color=c,alpha=0.25)
        for mask,mk,fill in ((~s,'o',False),(s,'s',True)):
            xs=[];ys=[];es=[]
            ed=D['ue'] if not fill else np.array(EDG,float)
            for lo,hi in zip(ed[:-1],ed[1:]):
                k=mask&(refm>=lo)&(refm<hi)&np.isfinite(r)
                if k.sum()>=3:
                    xs.append(np.median(refm[k])+off);ys.append(np.median(r[k]));es.append(1.2533*an.mad(r[k])/np.sqrt(k.sum()))
            axx.errorbar(xs,ys,es,fmt=mk+'-',color=c,mfc=c if fill else 'w',ms=7,capsize=2,label=f'{name} {"saturated" if fill else "unsaturated"}')
    axx.axhline(0,color='k',lw=0.5); axx.axvspan(0,D['lo_fit'],color='0.9',zorder=0)
    axx.set_ylim(-0.4,0.4); axx.set_xlim(12.5,19.2)
    axx.set_title(f'F{b} vs {D["mname"]}'); axx.set_ylabel('JWST - HST - colour relation (mag)')
    axx.legend(fontsize=7,loc='lower right')
for axx in ax[1]: axx.set_xlabel('dolphot mag')
plt.tight_layout(); plt.savefig('hst_anchor.png',dpi=130)
