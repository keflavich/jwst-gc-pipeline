"""Per-exposure FGS guiding vs per-frame ePSF shape (the obs 071/072/113 PSF outliers).

Reads the fine-guide centroid stream (``*_gs-fg_*_cal.fits``, FGS CENTROID PACKET) of
each observation, restricts it to each NIRCam exposure's DATE-BEG..DATE-END (from the
nrcblong cal primary header, cached in nircam_hdr.json), and plots guide-star motion
against the per-frame ePSF metrics of analyze_detector.py (result_<det>_stars.npz).

    EPSFMAP=<dir with result_*_stars.npz> DQA=<dir with gs-fg cal + nircam_hdr.json> python fgs_guiding.py
"""
import os, numpy as np, glob, json
from astropy.io import fits
from astropy.time import Time
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
EP=os.environ['EPSFMAP']; DQ=os.environ['DQA']
hdr=json.load(open(f'{DQ}/nircam_hdr.json'))
# per-frame PSF metrics: LW (both modules) and SW (8 detectors) means
met={}
for det in ['nrcalong','nrcblong','nrca1','nrca2','nrca3','nrca4','nrcb1','nrcb2','nrcb3','nrcb4']:
    z=np.load(f'{EP}/result_{det}_stars.npz',allow_pickle=True); ok=z['ok']
    key=np.array([f'{o}.{e}' for o,e in zip(z['obs'],z['expnum'])])
    band='lw' if 'long' in det else 'sw'
    for k in np.unique(key[ok]):
        m=ok&(key==k); d=met.setdefault(k,{})
        d.setdefault(band+'_wing',[]).append(np.median(z['dwing_grid'][m]))
        d.setdefault(band+'_chi2',[]).append(np.median(z['chi2'][m]))
        d.setdefault(band+'_de',[]).append(np.hypot(np.median(z['de1_grid'][m]),np.median(z['de2_grid'][m])))
        d.setdefault(band+'_psnr',[]).append(np.median(z['peak_snr'][m]))
tr={}
for fn in sorted(glob.glob(f'{DQ}/*gs-fg*_cal.fits')):
    with fits.open(fn) as f:
        c=f['FGS CENTROID PACKET'].data
        t=Time(np.char.replace(c['observatory_time'].astype(str),' ','T')).mjd
        x=c['guide_star_position_x']; y=c['guide_star_position_y']; g=(x!=0)&(y!=0)
        tr[fn]=(t[g],1000*x[g],1000*y[g])
rows={}
for k,h in hdr.items():
    t0=Time(h['DATE-BEG']).mjd; t1=Time(h['DATE-END']).mjd
    for fn,(t,x,y) in tr.items():
        m=(t>=t0)&(t<=t1)
        if m.sum()>100:
            xx=x[m]-np.median(x[m]); yy=y[m]-np.median(y[m])
            rows[k]=dict(rms=float(np.sqrt(np.var(xx)+np.var(yy))), span=float(np.percentile(np.hypot(xx,yy),99)), n=int(m.sum()))
for k in sorted(rows):
    d=met.get(k,{})
    print(k, f"motion rms={rows[k]['rms']:5.1f} mas p99|r|={rows[k]['span']:5.1f}", ' '.join(f"{q}={np.mean(d[q]):.3f}" for q in ['lw_wing','sw_wing','lw_chi2','sw_chi2'] if q in d))
json.dump(rows,open(f'{DQ}/motion_rows.json','w'),indent=1)

fig=plt.figure(figsize=(15,10))
gs=fig.add_gridspec(3,3,height_ratios=[1,1,1.25])
axes=[fig.add_subplot(gs[0,:]),fig.add_subplot(gs[1,:])]
for ax,obs in zip(axes,[['070','071','072','073'],['112','113']]):
    tmin=None
    for fn,(t,x,y) in tr.items():
        o=fn.split('jw10678')[1][:3]
        if o not in obs: continue
        tmin=min(tmin,t.min()) if tmin is not None else t.min()
    for fn,(t,x,y) in tr.items():
        o=fn.split('jw10678')[1][:3]
        if o not in obs: continue
        tm=(t-tmin)*1440
        ax.plot(tm,x-np.median(x),'C0',lw=.5); ax.plot(tm,y-np.median(y),'C3',lw=.5)
    for k,h in hdr.items():
        if k[:3] not in obs: continue
        a=(Time(h['DATE-BEG']).mjd-tmin)*1440; b=(Time(h['DATE-END']).mjd-tmin)*1440
        bad=k in ('071.2','071.4','071.6','072.2','072.5','072.6','113.1','113.3')
        ax.axvspan(a,b,color='r' if bad else '0.85',alpha=.25 if bad else .5,lw=0)
        ax.text((a+b)/2,56,k,ha='center',fontsize=7,color='r' if bad else 'k')
    ax.set_ylim(-60,62); ax.set_ylabel('guide-star position\n- median [mas]')
    ax.set_xlabel(f'minutes after {Time(tmin,format="mjd").isot[:16]} UT   (blue: FGS x, red: FGS y; shaded: NIRCam exposure, red = PSF outlier)')
axes[0].set_title('FGS fine-guide centroids (gs-fg_cal) through obs 070-073 and 112-113 of program 10678')
# scatter: motion rms vs PSF metrics
qs=[('sw_wing','F212N light at 3-10 px vs ePSF model',100),('lw_wing','F480M light at 3-10 px vs ePSF model',100),('lw_chi2','F480M median star chi2/pix',1)]
for j,(q,lab,sc) in enumerate(qs):
    ax=fig.add_subplot(gs[2,j])
    for k,r in rows.items():
        if k not in met or q not in met[k]: continue
        bad=k in ('071.2','071.4','071.6','072.2','072.5','072.6','113.1','113.3')
        ax.plot(r['rms'],sc*np.mean(met[k][q]),'o',color='r' if bad else 'k',ms=5)
        if r['rms']>4: ax.annotate(k,(r['rms'],sc*np.mean(met[k][q])),fontsize=7,xytext=(3,2),textcoords='offset points')
    ax.set_xlabel('guide-star motion during the exposure, rms [mas]'); ax.set_title(lab,fontsize=10)
    ax.set_ylabel('%' if sc==100 else '')
fig.tight_layout(); fig.savefig(f'{DQ}/fgs.png',dpi=80)
