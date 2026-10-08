"""Task C part 2: group-0 and ZEROFRAME DN at the satstar cores per SW detector (uncal files, read-only).
Usage: nice -19 python -u g0.py -> g0_stats.json, g0_tables.md"""
import json
import numpy as np
from astropy.io import fits

OUT = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/nrcb3'
CRDS = '/orange/adamginsburg/jwst/crds/references/jwst/nircam'
BANDS = ['F150W', 'F162M', 'F182M', 'F200W']
SW = ['nrca1', 'nrca2', 'nrca3', 'nrca4', 'nrcb1', 'nrcb2', 'nrcb3', 'nrcb4']
refd = json.load(open(f'{OUT}/refdata.json'))


def ring_med(img, x, y, r0=15, r1=25):
    ny, nx = img.shape
    xa, xb, ya, yb = max(x - r1, 0), min(x + r1 + 1, nx), max(y - r1, 0), min(y + r1 + 1, ny)
    yy, xx = np.mgrid[ya:yb, xa:xb]
    rr = np.hypot(xx - x, yy - y)
    v = img[ya:yb, xa:xb][(rr >= r0) & (rr < r1)].astype(float)
    return float(np.median(v)) if v.size else np.nan


def main():
    rows = []
    for band in BANDS:
        z = np.load(f'{OUT}/rows_{band}.npz', allow_pickle=True)
        det, ex, vg, X, Y, ref, lab = z['S_det'], z['S_exp'], z['S_vg'], z['S_x'], z['S_y'], z['S_ref'], z['S_lab']
        for d in SW:
            sat = None
            for e in sorted(set(ex[det == d].tolist())):
                s = np.where((det == d) & (ex == e))[0]
                fn = f'/orange/adamginsburg/jwst/wd2/{band}/pipeline/jw03523005001_{vg[s[0]]}_{e:05d}_{d}_uncal.fits'
                if sat is None:
                    with fits.open(f"{CRDS}/{refd[f'{band}_{d}']['refs'][0].replace('crds://', '')}") as h:
                        sat = np.asarray(h['SCI'].data, float)
                with fits.open(fn, memmap=False) as h:
                    sci = h['SCI'].data
                    zf = np.asarray(h['ZEROFRAME'].data)[0].astype(float) if 'ZEROFRAME' in h else None
                    g = np.asarray(sci[0, :, :, :]).astype(float)   # (ngroup, ny, nx)
                ng = g.shape[0]
                for q in s:
                    xi, yi = int(round(X[q])), int(round(Y[q]))
                    if xi < 30 or yi < 30 or xi > 2017 or yi > 2017:
                        continue
                    win = (slice(yi - 2, yi + 3), slice(xi - 2, xi + 3))
                    py, px = np.unravel_index(np.argmax(g[0][win]), (5, 5))
                    py, px = yi - 2 + py, xi - 2 + px
                    bias = ring_med(g[0], xi, yi)
                    g0max = g[0][py, px] - bias
                    zmax = (zf[py, px] - ring_med(zf, xi, yi)) if zf is not None else np.nan
                    # groups before the peak pixel exceeds the saturation threshold (bias-subtracted by the ring median of that group's first frame)
                    ramp = g[:, py, px] - bias
                    over = np.where((ramp >= sat[py, px]) | (g[:, py, px] >= 64000))[0]
                    gsat = over[0] if over.size else ng
                    rows.append((band, d, e, float(ref[q]), g0max, zmax, float(sat[py, px]), gsat, ng, float(g[-1, py, px]), float(g[1, py, px] - g[0, py, px])))
                print(band, d, e, len(s), flush=True)
    R = np.array(rows, dtype=object)
    L = ['### Group-0 / ZEROFRAME DN at satstar cores (5x5 peak minus 15-25 px ring median), per detector',
         '', '| band | det | N rows | g0 peak DN median (16/84%) | g0 peak / sat ref | ZEROFRAME peak DN median | first saturated group, median (0 = saturated in group 0) | frac saturated in g0 |', '|---|---|---|---|---|---|---|---|']
    out = {}
    for band in BANDS:
        for d in SW:
            m = (R[:, 0] == band) & (R[:, 1] == d)
            if m.sum() == 0:
                continue
            g0 = R[m, 4].astype(float)
            zf = R[m, 5].astype(float)
            sat = R[m, 6].astype(float)
            gs = R[m, 7].astype(float)
            out[f'{band}_{d}'] = dict(n=int(m.sum()), g0=float(np.median(g0)), g0_16=float(np.percentile(g0, 16)), g0_84=float(np.percentile(g0, 84)),
                                      g0_over_sat=float(np.median(g0 / sat)), zf=float(np.nanmedian(zf)), gsat=float(np.median(gs)), frac_g0sat=float(np.mean(gs == 0)))
            q = out[f'{band}_{d}']
            L.append(f"| {band} | {d} | {q['n']} | {q['g0']:.0f} ({q['g0_16']:.0f}/{q['g0_84']:.0f}) | {q['g0_over_sat']:.3f} | {q['zf']:.0f} | {q['gsat']:.1f} | {q['frac_g0sat']:.2f} |")
    json.dump(out, open(f'{OUT}/g0_stats.json', 'w'), indent=1)
    open(f'{OUT}/g0_tables.md', 'w').write('\n'.join(L) + '\n')
    np.save(f'{OUT}/g0_rows.npy', R, allow_pickle=True)


if __name__ == '__main__':
    main()
