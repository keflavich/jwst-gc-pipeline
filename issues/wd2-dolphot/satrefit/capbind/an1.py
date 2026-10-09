"""Task 1: binding pixel of the H cap in LW satstar rows, by dolphot magnitude bin; why the capcore2 'b' exclusion did not raise the cap."""
import pickle, sys
import numpy as np
from cb_lib import Band, mad
from capfun import cap_arrays

BINS = [(12.3, 13.0), (13.0, 13.5), (13.5, 14.0), (14.0, 15.0), (15.0, 16.0), (16.0, 17.0)]
out = {}
L = []
for band in ('250M', '300M'):
    B = Band(band)
    # row -> star ref (a row may map to one star)
    rowref = np.full(B.nrow, np.nan)
    rowstar = np.full(B.nrow, -1)
    rowref[B.j] = B.ref[B.i]
    rowstar[B.j] = B.i
    recs = []
    for k, r in enumerate(B.rows):
        if r['label'] <= 0 or not np.isfinite(rowref[k]):
            continue
        g = r['reg']
        c, b = cap_arrays(g['cutH'], g['psf'], g['ur'], r['pkidx'], r['ppk'], return_bind=True)
        aH = B.a_H[k]
        rec = dict(k=k, star=rowstar[k], ref=rowref[k], aH=aH, a0=B.a_H_h0_bgfree[k], cap=c * B.rcor[k] if np.isfinite(c) else np.nan, nreg=r['nreg'])
        rec['bind'] = bool(np.isfinite(c) and c * B.rcor[k] < aH)
        if b >= 0:
            dY, dX = g['dy'][b], g['dx'][b]
            rec.update(off=float(np.hypot(dY, dX)), repl=bool(g['repl'][b]), g0=float(g['g0'][b]), ff=float(g['ff'][b]), g0c=float(g['g0'][b] / B.ceiling[k]),
                       g0sat=bool(g['g0sat'][b]), g0flag=bool(g['g0flag'][b]), srcf=float(B.src_frac(r, k)[b]), srcdn=float(g['src'][b]),
                       dmod=float(g['cutH'][b] / (aH * g['psf'][b])), dmod0=float(g['cutH'][b] / (rec['a0'] * g['psf'][b])), psfrel=float(g['psf'][b] / r['ppk']),
                       rim=bool(g['rim'][b]), ur=bool(g['ur'][b]), pkbind=bool(b == r['pkidx']))
        recs.append(rec)
    out[band] = recs
    # table
    L += [f'### F{band}: binding pixel of the H cap (rows where cap_H x rcor < a_H)', '',
          '| dolphot mag | rows | binds | frac replaced (ff) | frac g0 SAT flag | frac g0 SAT/DNU flag | median offset px | median g0 DN | median first-frame DN | median g0/ceiling | median src DN / FW | median data/model(a_H) | median data/model(a_H+h0+bgfree) | frac binding = model-peak pixel |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for lo, hi in BINS:
        rr = [x for x in recs if lo <= x['ref'] < hi]
        bb = [x for x in rr if x['bind'] and 'off' in x]
        if not rr:
            continue
        if bb:
            med = lambda key: np.median([x[key] for x in bb])
            frc = lambda key: np.mean([x[key] for x in bb])
            L.append(f"| {lo}-{hi} | {len(rr)} | {len(bb)} ({100*len(bb)/len(rr):.0f}%) | {frc('repl'):.2f} | {frc('g0sat'):.2f} | {frc('g0flag'):.2f} | {med('off'):.2f} | {med('g0'):.0f} | {med('ff'):.0f} | {med('g0c'):.2f} | {med('srcf'):.2f} | {med('dmod'):.3f} | {med('dmod0'):.3f} | {frc('pkbind'):.2f} |")
        else:
            L.append(f'| {lo}-{hi} | {len(rr)} | 0 | | | | | | | | | | | |')
    L.append('')
    # direct vs replaced split for 12.3-13
    L += [f'F{band} binding pixels, 12.3-13 mag, split by pixel type', '', '| type | n | median src DN / FW | median data/model(a_H) | median offset | median g0/ceiling |', '|---|---|---|---|---|---|']
    for nm, sel in (('replaced (first frame x k)', lambda x: x['repl']), ('direct g0', lambda x: not x['repl'])):
        bb = [x for x in recs if 12.3 <= x['ref'] < 13 and x['bind'] and 'off' in x and sel(x)]
        if bb:
            L.append(f"| {nm} | {len(bb)} | {np.median([x['srcf'] for x in bb]):.2f} | {np.median([x['dmod'] for x in bb]):.3f} | {np.median([x['off'] for x in bb]):.2f} | {np.median([x['g0c'] for x in bb]):.2f} |")
    L.append('')
    # --- why 'b' did not raise the cap
    L += [f'F{band}: capcore2 design-b style exclusion (replaced pixels with first frame > t x max first frame treated as unmeasured, inside the flux bound only), rows binding in 12.3-13 mag', '',
          '| t | rows | frac with binding pixel excluded | median cap_new/cap_old over rows with exclusion | median cap_new/cap_old all | frac cap_new < a_H | frac of rows whose new binding pixel is replaced | median src DN / FW of new binding pixel | median data/model(a_H) of new binding pixel |', '|---|---|---|---|---|---|---|---|---|']
    for t in (0.6, 0.8):
        n = ne = 0
        rat_e, rat_all, nb, newrepl, newsrc, newdm = [], [], 0, [], [], []
        for x in recs:
            if not (12.3 <= x['ref'] < 13) or not x['bind'] or 'off' not in x:
                continue
            k = x['k']
            r = B.rows[k]
            g = r['reg']
            ex = g['repl'] & (g['ff'] > t * B.FW_ff[k])
            c1, b1 = cap_arrays(g['cutH'], g['psf'], g['ur'], r['pkidx'], r['ppk'], excl=ex, excl_gate=False, return_bind=True)
            c0, b0 = cap_arrays(g['cutH'], g['psf'], g['ur'], r['pkidx'], r['ppk'], return_bind=True)
            n += 1
            ratio = c1 / c0 if np.isfinite(c1) else np.nan
            rat_all.append(ratio)
            if ex[b0]:
                ne += 1
                rat_e.append(ratio)
            if np.isfinite(c1) and c1 * B.rcor[k] < x['aH']:
                nb += 1
            if b1 >= 0:
                newrepl.append(bool(g['repl'][b1]))
                newsrc.append(float(B.src_frac(r, k)[b1]))
                newdm.append(float(g['cutH'][b1] / (x['aH'] * g['psf'][b1])))
        L.append(f"| {t} | {n} | {ne/n:.2f} | {np.nanmedian(rat_e) if rat_e else np.nan:.3f} | {np.nanmedian(rat_all):.3f} | {nb/n:.2f} | {np.mean(newrepl):.2f} | {np.median(newsrc):.2f} | {np.median(newdm):.3f} |")
    L.append('')
open('an1_tables.md', 'w').write('\n'.join(L) + '\n')
pickle.dump(out, open('an1_recs.pkl', 'wb'))
print('\n'.join(L))
