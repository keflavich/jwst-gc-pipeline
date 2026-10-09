"""For b2/c stars: map each per-frame row within 1 px (0.063") of the star onto the base list built exactly as
combine_singleframe Loop 1 / Phase 1 do (merge_catalogs.py:~405-420 and ~590-600). usage: python basetrace.py BAND"""
import sys, glob, re, warnings
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
warnings.filterwarnings('ignore')
Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
sys.path.insert(0, Q)
import analyze as an
b = sys.argv[1].upper().lstrip('F')
T = f'{Q}/tree_main2kfpk'
an.ZPWIN.update(an.zp_windows())
A = an.Arm('main2kfpk')
rs = SkyCoord(np.asarray(A.m['RA'], float)*u.deg, np.asarray(A.m['DEC'], float)*u.deg)
pre = Table.read(f'/blue/adamginsburg/adamginsburg/tmp/claude-3663/pre_{b}.fits')
fin = Table.read(f'{T}/catalogs/f{b.lower()}_merged_indivexp_merged_resbgsub_m7_dao_basic.fits')
fns = sorted(glob.glob(f'{T}/F{b}/f{b.lower()}_*_visit*_vgroup*_exp*_resbgsub_m7_daophot_basic.fits'))
tabs = []
for fn in fns:
    t = Table.read(fn); mm = re.search(r'_(nrc[ab]long)_.*_exp(\d{5})_', fn); t.meta['fr'] = f'{mm.group(1)}_{mm.group(2)}'
    c = t['skycoord_centroid']; bad = ~np.isfinite(c.ra.deg) | ~np.isfinite(c.dec.deg)
    tabs.append(t[~bad])
min_off = 0.10*u.arcsec; max_off = 0.10*u.arcsec
base = None; birth = []
for ii, t in enumerate(tabs):
    c = t['skycoord_centroid']
    if base is None:
        base = c; birth += [ii]*len(c)
    else:
        _, sep, _ = c.match_to_catalog_sky(base, nthneighbor=1)
        keep = sep > min_off
        base = SkyCoord([base, c[keep]]); birth += [ii]*int(keep.sum())
birth = np.array(birth)
print('base', len(base), 'pre', len(pre), flush=True)
dev = base.separation(pre['skycoord_avg'])
print('pre avg pos vs base pos median/max shift arcsec', np.nanmedian(dev.arcsec), np.nanmax(dev.arcsec))
cls = Table.read(f'{Q}/lwrest/class_{b}.ecsv')
sel = cls[[x[:2] in ('b2', 'c:') for x in cls['cat']]]
out = []
for ii, t in enumerate(tabs):
    c = t['skycoord_centroid']
    mi, sep, _ = c.match_to_catalog_sky(base, nthneighbor=1)
    ri, _, _ = base.match_to_catalog_sky(c, nthneighbor=1)
    mutual = ri[mi] == np.arange(len(mi))
    keep = (sep < max_off) & mutual
    for s in sel:
        p = rs[int(s['dolphot_idx'])]
        d = c.separation(p).arcsec
        for k in np.where(d < 0.063)[0]:
            bi = int(mi[k])
            # who else claims base bi?
            rival = int(ri[bi])
            rs_sep = float(c[rival].separation(base[bi]).arcsec) if rival != k else 0.0
            out.append(dict(band=b, dolphot_idx=int(s['dolphot_idx']), cat=s['cat'][:2].rstrip(':'), frame=t.meta['fr'], row=int(k), sep_star=float(d[k]),
                            mag_inst=float(-2.5*np.log10(t['flux_fit'][k])), base_idx=bi, sep_base=float(sep[k].arcsec), mutual=bool(mutual[k]), kept=bool(keep[k]),
                            base_birth=int(birth[bi]), base_pos_to_star=float(base[bi].separation(p).arcsec), avg_pos_to_star=float(pre['skycoord_avg'][bi].separation(p).arcsec),
                            pre_nmatch=int(pre['nmatch'][bi]), pre_nmatch_good=int(pre['nmatch_good'][bi]),
                            rival_row_sep=rs_sep, rival_flux_ratio=float(t['flux_fit'][rival]/t['flux_fit'][k]) if rival != k else 1.0,
                            fin_rep=bool(fin['replaced_saturated'][0]) and False))
O = Table(out); O.write(f'basetrace_{b}.ecsv', overwrite=True); print(len(O))
