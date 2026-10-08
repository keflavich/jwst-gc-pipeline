"""Lost-star analysis: (a) orthogonal per-star presence flags; (b) nearest pk3 accepted satstar for stars lost in pk3
but recovered in pk2; (c) merge-stage fate of pk3 stars that have a per-frame satstar row at the star but no merged row."""
import json
import numpy as np
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
import astropy.units as u
O = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/pk_deblend_trace'
G = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/f277w_gap/dbl'
pf = Table.read(f'{O}/per_frame.ecsv')
ps = Table.read(f'{O}/per_star.ecsv')
ZP = json.load(open(f'{O}/zp.json'))['ZP']
N = len(ps)
star = SkyCoord(ps['ra'] * u.deg, ps['dec'] * u.deg)
print('== (a) per-star presence flags (any covering frame)')
flags = {}
for arm in ('pk2', 'pk3'):
    s = pf[pf['arm'] == arm]
    f = {k: np.zeros(N, bool) for k in ('sat008', 'sat05', 'rej008', 'rej05', 'dao008')}
    for r in s:
        i = r['idx']
        f['sat008'][i] |= r['sat_sep'] < 0.08
        f['sat05'][i] |= r['sat_sep'] < 0.5
        f['rej008'][i] |= r['rej_sep'] < 0.08
        f['rej05'][i] |= r['rej_sep'] < 0.5
        f['dao008'][i] |= r['dao_sep'] < 0.08
    f['mhit'] = np.asarray(ps[f'{arm}_mhit'], bool)
    flags[arm] = f
    print(arm, {k: int(v.sum()) for k, v in f.items()})
    # combos
    nothing = ~(f['sat05'] | f['rej05'] | f['dao008'])
    print('   none of sat<0.5, rej<0.5, dao<0.08:', nothing.sum())
    print('   dao008 & mhit', (f['dao008'] & f['mhit']).sum(), ' dao008 & ~mhit', (f['dao008'] & ~f['mhit']).sum())
    print('   sat008 & mhit', (f['sat008'] & f['mhit']).sum(), ' sat008 & ~mhit', (f['sat008'] & ~f['mhit']).sum())
    print('   rej008 only (no sat008, no dao008):', (f['rej008'] & ~f['sat008'] & ~f['dao008']).sum(), 'merged hit', (f['rej008'] & ~f['sat008'] & ~f['dao008'] & f['mhit']).sum())
p3, p2 = flags['pk3'], flags['pk2']
lost = p2['mhit'] & ~p3['mhit']
gained = ~p2['mhit'] & p3['mhit']
print('lost (pk2 hit, pk3 no):', lost.sum(), 'gained:', gained.sum(), 'both:', (p2['mhit'] & p3['mhit']).sum(), 'neither:', (~p2['mhit'] & ~p3['mhit']).sum())
print('lost stars, pk3 flags:', {k: int((v & lost).sum()) for k, v in p3.items()})
print('lost stars, pk2 flags:', {k: int((v & lost).sum()) for k, v in p2.items()})
print('lost stars: pk2 merged row replaced?', int(np.sum(ps['pk2_mrep'][lost])), ' pk2 merged dm pctl', np.nanpercentile(ps['pk2_mdm'][lost], [5, 16, 50, 84, 95]).round(2))
ps['lost'] = lost

# (b) nearest pk3 accepted satstar over all frames
FR = [(d, e) for d in 'ab' for e in (1, 2, 3, 4)]
def frame_cat(arm, d, e, kind):
    T = f'{G}/tree_dbl{2 if arm == "pk2" else 3}'
    return Table.read(f'{T}/F277W/pipeline/jw03523005001_10101_0000{e}_nrc{d}long_align_o005_crf_resbgsub_m7_satstar_{kind}.fits')
nn_sep = np.full(N, np.inf); nn_flux = np.full(N, np.nan); nn_n = np.zeros(N, int)
nn2_sep = np.full(N, np.inf)
for d, e in FR:
    c = frame_cat('pk3', d, e, 'catalog')
    sk = c['skycoord_fit']; fl = np.asarray(c['flux_fit'], float)
    j, dd, _ = star.match_to_catalog_sky(sk)
    dd = dd.arcsec
    upd = dd < nn_sep
    nn_sep[upd] = dd[upd]; nn_flux[upd] = fl[j][upd]
    # number of accepted satstars within 1.5 FWHM (F277W FWHM ~0.14 arcsec -> 0.21)
    idx1, idx2, d2, _ = sk.search_around_sky(star, 0.5 * u.arcsec)
    nn_n += np.bincount(idx1, minlength=N)
dol = np.asarray(ps['dol_flux'], float)
ps['pk3_nn_sat_sep'] = nn_sep; ps['pk3_nn_sat_fluxratio'] = nn_flux / dol
L = lost
print('\n== (b) lost stars: nearest pk3 accepted satstar (any frame), n =', L.sum())
print('sep pctl 5,25,50,75,95', np.percentile(nn_sep[L], [5, 25, 50, 75, 95]).round(3))
for lo, hi in ((0, 0.08), (0.08, 0.2), (0.2, 0.5), (0.5, 1.0), (1.0, 99)):
    m = L & (nn_sep >= lo) & (nn_sep < hi)
    print(f'  sep {lo}-{hi}: {m.sum()}', ('flux ratio pctl 5,50,95 ' + str(np.nanpercentile((nn_flux / dol)[m], [5, 50, 95]).round(2))) if m.any() else '')
print('flux ratio (all lost) pctl 5,16,50,84,95', np.nanpercentile((nn_flux / dol)[L], [5, 16, 50, 84, 95]).round(2))
# same-frame geometry: sat count per frame within 0.5 of star
print('frames-with-sat<0.5 mean per lost star', (nn_n[L] / 4).mean().round(2))
# pk2 accepted satstar nearest for the same stars
nn2 = np.full(N, np.inf); nn2f = np.full(N, np.nan)
for d, e in FR:
    c = frame_cat('pk2', d, e, 'catalog')
    j, dd, _ = star.match_to_catalog_sky(c['skycoord_fit']); dd = dd.arcsec
    upd = dd < nn2
    nn2[upd] = dd[upd]; nn2f[upd] = np.asarray(c['flux_fit'], float)[j][upd]
print('pk2 nearest accepted satstar for lost stars: sep pctl 5,50,95', np.percentile(nn2[L], [5, 50, 95]).round(3), ' <0.5:', int((nn2[L] < 0.5).sum()))
# is the nearest pk3 sat the same object in the pk2 catalog? (distinguish new deblend satstars)
ps['pk2_nn_sat_sep'] = nn2

# (c) merge-stage fate: pk3 stars with sat008 or rej008 or dao008 in per-frame products but no merged row <0.08
t3 = Table.read(f'{G}/tree_dbl3/catalogs/f277w_merged_indivexp_merged_resbgsub_m7_dao_basic.fits')
ok = np.isfinite(t3['skycoord'].ra.deg) & (np.asarray(t3['flux'], float) > 0)
print('\npk3 merged: rows', len(t3), 'finite/pos', ok.sum(), 'cols satstar_gate_rejected sum', int(np.sum(t3['satstar_gate_rejected'])), 'replaced', int(np.sum(t3['replaced_saturated'])), 'is_saturated', int(np.sum(t3['is_saturated'])))
t3 = t3[ok]
sk3 = t3['skycoord']
sel = p3['sat008'] & ~p3['mhit']
print('== (c) pk3 stars with a per-frame satstar row <0.08 but no merged row <0.08:', sel.sum())
rows = []
for i in np.where(sel)[0]:
    s = pf[(pf['arm'] == 'pk3') & (pf['idx'] == i)]
    sep = star[i].separation(sk3)
    k = np.argsort(sep)[:3]
    r = t3[k[0]]
    sats = [(fr, round(float(a), 3), round(float(b), 0)) for fr, a, b in zip(s['frame'], s['sat_sep'], s['sat_flux']) if a < 0.08]
    print(f"star {i} dolphot_idx {ps['dolphot_idx'][i]} ref {ps['ref_mag'][i]:.2f} dolflux {dol[i]:.0f}; sat frames {sats}; dao008 {p3['dao008'][i]}")
    for kk in k:
        rr = t3[kk]
        print(f"    merged row {kk}: sep {sep[kk].arcsec:.3f}\" flux {rr['flux']:.0f} rep {rr['replaced_saturated']} satstar_nframes {rr['satstar_nframes']} nmeas {rr['satstar_nmeas']} gate_rej {rr['satstar_gate_rejected']} sat_sat {rr['is_saturated']} flags {rr['flags']} nmatch {rr['nmatch']} satstar_match_sep {rr['satstar_match_sep']:.3f}")
    rows.append(i)
ps.write(f'{O}/per_star_ext.ecsv', overwrite=True)
