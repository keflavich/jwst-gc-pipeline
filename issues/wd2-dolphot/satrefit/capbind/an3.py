"""Task 3: group-0 PSF photometry versus the H+h0+bgfree amplitude and the cap, LW."""
import pickle
import numpy as np
from cb_lib import Band, mad

BINS = [(12.3, 13.0), (13.0, 13.5), (13.5, 14.0), (14.0, 15.0), (15.0, 17.0)]
res = {}
L = []
for band in ('250M', '300M'):
    B = Band(band)
    rowref = np.full(B.nrow, np.nan)
    rowref[B.j] = B.ref[B.i]
    capH = B.cap_H * B.rcor
    capH = np.where(np.isfinite(capH), capH, np.inf)
    res[band] = {}
    for tag in ('g50', 'g30', 'g70'):
        ag = getattr(B, 'a_' + tag)
        a0 = B.a_H_h0_bgfree
        ok = B.good & np.isfinite(rowref) & np.isfinite(ag) & (ag > 0) & np.isfinite(a0) & (a0 > 0)
        cal = ok & (rowref >= 15) & (rowref < 17)
        c = np.median((ag / a0)[cal])
        res[band][tag] = c
        if tag == 'g50':
            L += [f'### F{band} (pixel cut g0 < 0.5 ceiling; calibration rows 15-17 mag: {int(cal.sum())}, median a_g0 / a_(H+h0+bgfree) = {c:.4f}, MAD {mad((ag / a0)[cal]):.4f})', '',
                  '| dolphot bin | rows | stars | median n px g0 fit | median err a_g0/a | median a_g0/a_H+h0+bgfree (raw) | calibrated | median a_g0/cap_H (calibrated; capped rows only) | a_H+h0+bgfree/cap_H (capped rows only) | frac rows cap_H < a_H | dm a_g0/c | dm H+h0+bgfree uncapped | dm H+h0+bgfree+cap | dm final |',
                  '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
            ag_c = ag / c
            dm_g = B.dm_of(ag_c)
            dm_u = B.dm_of(a0)
            cap0 = B.cap_H_h0 * B.rcor
            dm_cap = B.dm_of(np.minimum(a0, np.where(np.isfinite(cap0), cap0, np.inf)))
            have = B.have0 & np.isfinite(dm_g) & np.isfinite(dm_u) & np.isfinite(dm_cap)
            for lo, hi in BINS:
                rs = ok & (rowref >= lo) & (rowref < hi)
                st = have & (B.ref >= lo) & (B.ref < hi)
                bind = rs & np.isfinite(capH) & (capH < a0)
                bnd = rs & np.isfinite(capH) & (capH < np.inf)
                ratio = (ag / a0)[rs]
                L.append(f"| {lo}-{hi} | {int(rs.sum())} | {int(st.sum())} | {np.median(B.n_g50[rs]):.0f} | {np.median(np.array([r.get('ae_g50', np.nan) for r in B.rows])[rs] / ag[rs]):.4f} | "
                         f"{np.median(ratio):.4f} | {np.median(ratio) / c:.4f} | {np.median((ag_c / capH)[bind]) if bind.sum() else np.nan:.4f} | {np.median((a0 / capH)[bind]) if bind.sum() else np.nan:.4f} | {bind.sum() / max(bnd.sum(), 1):.2f} | "
                         f"{np.median(dm_g[st]):+.3f} | {np.median(dm_u[st]):+.3f} | {np.median(dm_cap[st]):+.3f} | {np.median(B.dm_final[st]):+.3f} |")
            L.append('')
            # finer mag trend of the ratio (calibrated)
            L += [f'F{band}: median calibrated a_g0 / a_(H+h0+bgfree) and a_(H+h0+bgfree)/cap_H by 0.25 mag bins', '', '| bin | rows | a_g0/a_H0bg (cal) | a_H0bg / cap_H (rows with finite cap) |', '|---|---|---|---|']
            for lo in np.arange(12.25, 15.0, 0.25):
                rs = ok & (rowref >= lo) & (rowref < lo + 0.25)
                if rs.sum() >= 8:
                    fc = rs & np.isfinite(capH) & (capH < np.inf)
                    L.append(f'| {lo:.2f}-{lo + 0.25:.2f} | {int(rs.sum())} | {np.median((ag / a0)[rs]) / c:.4f} | {np.median((a0 / capH)[fc]) if fc.sum() else np.nan:.4f} |')
            L.append('')
            res[band]['rows'] = dict(ok=ok, rowref=rowref, ag_c=ag_c, a0=a0, capH=capH)
        else:
            L.append(f'F{band} sensitivity {tag}: calibration {c:.4f}; calibrated ratio ' + ', '.join(
                f"{lo}-{hi}: {np.median((ag / a0)[ok & (rowref >= lo) & (rowref < hi)]) / c:.4f}" for lo, hi in BINS[:4]))
            L.append('')
open('an3_tables.md', 'w').write('\n'.join(L) + '\n')
pickle.dump({k: {t: v for t, v in d.items() if t != 'rows'} for k, d in res.items()}, open('an3_cal.pkl', 'wb'))
print('\n'.join(L))
