"""Classify b2 and c stars by merge mechanism. -> mech_BAND.ecsv, printed counts."""
import numpy as np, warnings
from astropy.table import Table, vstack
warnings.filterwarnings('ignore')
res = []
for b in ('277W', '250M', '300M'):
    S = Table.read(f'stars_{b}.ecsv'); R = Table.read(f'rows_{b}.ecsv'); B = Table.read(f'basetrace_{b}.ecsv')
    for s in S:
        i = s['dolphot_idx']; r = R[R['dolphot_idx'] == i]; bt = B[B['dolphot_idx'] == i]
        fr1 = bt  # per-frame rows within 1 px (0.063")
        info = dict(band=b, dolphot_idx=i, cat=s['cat'], ref_mag=s['ref_mag'], n_fr_rows_1px=len(fr1), n_kept=int(fr1['kept'].sum()) if len(fr1) else 0)
        pre = r[r['kind'] == 'pre']; fin = r[r['kind'] == 'final']
        # final within 0.08, best
        f08 = fin[fin['sep'] < 0.08]
        info['final_within_0.08'] = len(f08)
        info['fin_mag'] = float(f08['mag'][np.argmin(f08['sep'])]) if len(f08) else np.nan
        info['fin_nmatch'] = int(f08['nmatch'][np.argmin(f08['sep'])]) if len(f08) else -1
        if len(fr1):
            kept = fr1[fr1['kept']]
            info['n_notkept'] = int((~fr1['kept']).sum())
            bi = np.bincount(kept['base_idx']).argmax() if len(kept) else fr1['base_idx'][0]
            info['base_idx'] = int(bi)
            info['pre_nmatch'] = int(fr1['pre_nmatch'][0]); info['avg_pos_to_star'] = float(fr1['avg_pos_to_star'][fr1['base_idx'] == bi][0])
            info['n_distinct_base'] = len(set(fr1['base_idx']))
            pr = pre[pre['where'] == f'pre{bi}']
            info['pre_replaced'] = bool(pr['rep'][0]) if len(pr) else False
            info['satsep'] = float(pr['y'][0]) if len(pr) else np.nan
            info['pre_mag'] = float(pr['mag'][0]) if len(pr) else np.nan
        # mechanism
        if s['cat'] == 'b2':
            if len(fr1) == 0: m = 'noframerow'
            elif info['pre_replaced']: m = 'R1_replace_sat_first_pass' if info['satsep'] < 0.1 else 'R2_replace_sat_second_pass'
            elif info['n_notkept'] > 0 and info['n_kept'] == 0: m = 'M0_frame_rows_not_kept'
            elif not np.isfinite(info['avg_pos_to_star']): m = 'S_sigmaclip_masks_all_frames_NaN_position'
            elif info['avg_pos_to_star'] >= 0.08: m = 'P_merged_position_0.08-0.10_from_dolphot'
            else: m = 'other'
        else: m = ''
        info['mech'] = m
        res.append(info)
Rt = Table(rows=res)
Rt.write('mech_all.ecsv', overwrite=True)
for b in ('277W', '250M', '300M'):
    q = Rt[(Rt['band'] == b) & (Rt['cat'] == 'b2')]
    print(b, 'b2', {m: int((q['mech'] == m).sum()) for m in sorted(set(q['mech']))})
