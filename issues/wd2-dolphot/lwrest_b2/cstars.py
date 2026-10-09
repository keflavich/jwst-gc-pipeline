"""c stars: per-frame fluxes at the star vs merged vs dolphot. -> c_detail.ecsv, prints mechanism counts."""
import numpy as np, warnings
from astropy.table import Table
warnings.filterwarnings('ignore')
Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
zp = {'277W': 24.128923927404763, '250M': 24.315577424137363, '300M': 23.970356429490028}
res = []
for b in ('277W', '250M', '300M'):
    S = Table.read(f'stars_{b}.ecsv'); R = Table.read(f'rows_{b}.ecsv'); B = Table.read(f'basetrace_{b}.ecsv')
    cl = Table.read(f'{Q}/lwrest/class_{b}.ecsv')
    pre = Table.read(f'/blue/adamginsburg/adamginsburg/tmp/claude-3663/pre_{b}.fits')
    for s in S[S['cat'] == 'c']:
        i = int(s['dolphot_idx']); ref = float(s['ref_mag']); r = R[R['dolphot_idx'] == i]; bt = B[B['dolphot_idx'] == i]
        fr = bt[bt['sep_star'] < 0.063]
        fm = fr['mag_inst'] + zp[b]
        fin = r[(r['kind'] == 'final') & (r['sep'] < 0.08)]
        fin = fin[np.argmin(fin['sep'])] if len(fin) else None
        prr = r[(r['kind'] == 'pre')]; prr = prr[np.argsort(prr['sep'])]
        d = dict(band=b, dolphot_idx=i, ref=ref, n_fr=len(fr), n_kept=int(fr['kept'].sum()), fr_mags=','.join(f'{x:.2f}' for x in sorted(fm)), fr_med=float(np.median(fm)) if len(fm) else np.nan,
                 fr_min=float(fm.min()) if len(fm) else np.nan, fr_max=float(fm.max()) if len(fm) else np.nan,
                 fin_mag=float(fin['mag']) if fin is not None else np.nan, fin_rep=bool(fin['rep']) if fin is not None else False, fin_nmatch=int(fin['nmatch']) if fin is not None else -1,
                 satsep=float(fin['y']) if fin is not None else np.nan, n_distinct_base=len(set(fr['base_idx'])) if len(fr) else 0)
        # pre row mapping to the final row: via nearest pre within 0.08 or the base idx
        if len(fr):
            bi = int(np.bincount(fr['base_idx']).argmax()); d['base_idx'] = bi
            d['pre_mag'] = float(-2.5*np.log10(pre['flux_fit_avg'][bi]) + zp[b]) if np.isfinite(pre['flux_fit_avg'][bi]) else np.nan
            d['pre_nmatch'] = int(pre['nmatch'][bi]); d['pre_nmatch_good'] = int(pre['nmatch_good'][bi])
            d['pre_pos_to_star'] = float(fr['avg_pos_to_star'][fr['base_idx'] == bi][0])
        else:
            d.update(base_idx=-1, pre_mag=np.nan, pre_nmatch=0, pre_nmatch_good=0, pre_pos_to_star=np.nan)
        d['dm_fin'] = d['fin_mag'] - ref; d['dm_frmed'] = d['fr_med'] - ref; d['dm_pre'] = d['pre_mag'] - ref
        # mechanism
        if d['fin_rep']: m = 'C1_satstar_replaced_value'
        elif d['n_fr'] == 0: m = 'C4_no_frame_row_at_star'
        elif abs(d['dm_frmed']) >= 0.3 and abs(d['fr_max'] - d['fr_min']) < 0.3: m = 'C2_all_frames_off'
        elif abs(d['dm_frmed']) >= 0.3: m = 'C2b_frames_off_and_scattered'
        elif abs(d['pre_mag'] - d['fr_med']) >= 0.3: m = 'C3_merge_combined_other_rows'
        else: m = 'C5_other'
        d['mech'] = m
        res.append(d)
C = Table(rows=res); C.write('c_detail.ecsv', overwrite=True)
for b in ('277W', '250M', '300M'):
    q = C[C['band'] == b]; print(b, {m: int((q['mech'] == m).sum()) for m in sorted(set(q['mech']))})
C['dm_fin','dm_frmed','dm_pre'].info
for r in C:
    print(r['band'], r['dolphot_idx'], f"ref={r['ref']:.2f} fin={r['fin_mag']:.2f} pre={r['pre_mag']:.2f} frames=[{r['fr_mags']}] rep={r['fin_rep']} n={r['fin_nmatch']}/{r['pre_nmatch_good']} nbase={r['n_distinct_base']} {r['mech']}")
