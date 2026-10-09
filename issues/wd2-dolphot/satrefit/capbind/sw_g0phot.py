"""Group-0 photometry check of the SW bright end: does a low-g0 wing fit side with the capped or the uncapped amplitude?"""
import sys
import numpy as np
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit/capbind')
from cb_lib import Band, mad
from an4 import bind_info, REFV

BINS = {'150W': [(14, 15), (15, 16), (16, 17), (17, 18), (18, 19)], '200W': [(13, 14), (14, 15), (15, 16), (16, 17), (17, 18)],
        '250M': [(12.3, 13), (13, 13.5), (13.5, 14), (14, 15), (15, 16), (16, 17)], '300M': [(12.3, 13), (13, 13.5), (13.5, 14), (14, 15), (15, 16), (16, 17)]}
CAL = {'150W': (17, 19), '200W': (16, 18), '250M': (15, 17), '300M': (15, 17)}
for band in ('150W', '200W', '250M', '300M'):
    B = Band(band)
    rowref = np.full(B.nrow, np.nan)
    rowref[B.j] = B.ref[B.i]
    cap0, _ = bind_info(B, 'cutH0')
    a0 = B.a_H_h0_bgfree
    c = cap0 * B.rcor
    c = np.where(np.isfinite(c), c, np.inf)
    acap = np.minimum(a0, c)
    print(f'\n### F{band} (reference {REFV[band]:+.3f})')
    for tag in ('g10', 'g20', 'f10'):
        ag = getattr(B, 'a_' + tag)
        nfin = int(np.isfinite(ag).sum())
        ok = B.good & np.isfinite(rowref) & np.isfinite(ag) & (ag > 0) & np.isfinite(a0) & (a0 > 0)
        lo_c, hi_c = CAL[band]
        cal = ok & (rowref >= lo_c) & (rowref < hi_c)
        if cal.sum() < 10:
            print(f'{tag}: finite rows {nfin}, calibration rows {int(cal.sum())}; skipped')
            continue
        k = np.median((ag / a0)[cal])
        dm_g = B.dm_of(ag / k)
        dm_c = B.dm_of(acap)
        dm_u = B.dm_of(a0)
        have = B.have0 & np.isfinite(dm_g) & np.isfinite(dm_c) & np.isfinite(dm_u)
        print(f'{tag}: finite rows {nfin}; calibration {lo_c}-{hi_c} mag rows {int(cal.sum())}, median a_g0/a_H0bg {k:.4f} (MAD {mad((ag / a0)[cal]):.4f})')
        print('| bin | rows | stars | a_g0/a_H0bg (cal) | a_g0/min(a,cap) (cal) | dm g0 | dm cap | dm uncap |')
        print('|---|---|---|---|---|---|---|---|')
        for lo, hi in BINS[band]:
            rs = ok & (rowref >= lo) & (rowref < hi)
            st = have & (B.ref >= lo) & (B.ref < hi)
            if rs.sum() < 5 or st.sum() < 5:
                print(f'| {lo}-{hi} | {int(rs.sum())} | {int(st.sum())} | - | - | - | - | - |')
                continue
            print(f'| {lo}-{hi} | {int(rs.sum())} | {int(st.sum())} | {np.median((ag / a0)[rs]) / k:.4f} | {np.median((ag / acap)[rs]) / k:.4f} | '
                  f'{np.median(dm_g[st]):+.3f} | {np.median(dm_c[st]):+.3f} | {np.median(dm_u[st]):+.3f} |')
