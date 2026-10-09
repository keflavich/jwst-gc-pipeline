"""Score out7f against dolphot: H+h0 (reproduction of round 7), Hf+h0f (flat applied to every group-0 rewrite) and the
rate-scale variants, all with bgfree and the hard recovered-core cap (LW also with the 0.96 floor).
usage: python score7f.py [BAND ...]"""
import sys
import pickle
import numpy as np
from astropy.table import Table
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit/capbind')
from cb_lib import Band, mad
from capfun import cap_arrays
from an4 import bind_info, REFV

D = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit/rstar/out7f'
BINS = {'150W': [(14, 15), (15, 16), (16, 17), (17, 18), (18, 19)], '200W': [(13, 14), (14, 15), (15, 16), (16, 17)],
        '250M': [(12.3, 13), (13, 13.5), (13.5, 14), (14, 15), (15, 16), (16, 17)],
        '300M': [(12.3, 13), (13, 13.5), (13.5, 14), (14, 15), (15, 16), (16, 17)]}
VARS = ['H+h0', 'Hf+h0f', 'Hfs0.97+h0f', 'Hfs1.03+h0f']


def load(B):
    T, G = [], []
    for fnm in B.map['files']:
        fn = f"{D}/{fnm.replace('_satrefit.fits', '_satrefit7f.fits')}"
        t = Table.read(fn)
        g = pickle.load(open(fn.replace('.fits', '_reg.pkl'), 'rb'))
        assert len(t) == len(g)
        T.append(t)
        G += g
    assert sum(len(t) for t in T) == B.nrow
    col = lambda c: np.concatenate([np.ma.filled(np.ma.asarray(t[c], float), np.nan) if c in t.colnames else np.full(len(t), np.nan) for t in T])
    lab = np.concatenate([np.asarray(t['label']) for t in T])
    assert np.array_equal(lab, B.label)
    return col, G


def caps(B, G, nm):
    out = np.full(B.nrow, np.nan)
    for k, g in enumerate(G):
        if g is None or B.label[k] <= 0:
            continue
        out[k] = cap_arrays(g['cut_' + nm], g['psf'], g['ur'], g['pkidx'], g['ppk'])
    return out


def main(bands):
    for band in bands:
        B = Band(band)
        col, G = load(B)
        lw = band in ('250M', '300M')
        D_ = {}
        a7 = B.a_H_h0_bgfree
        c7, _ = bind_info(B, 'cutH0')
        c7 = np.where(np.isfinite(c7 * B.rcor), c7 * B.rcor, np.inf)
        D_['round 7 H+h0'] = B.dm_of(np.minimum(a7, c7)) - REFV[band]
        for nm in VARS:
            a = col('a_' + nm + '+bgfree')
            c = caps(B, G, nm) * B.rcor
            c = np.where(np.isfinite(c), c, np.inf)
            acap = np.minimum(a, c)
            D_[nm] = B.dm_of(acap) - REFV[band]
            if lw:
                D_[nm + ' floor0.96'] = B.dm_of(np.maximum(acap, 0.96 * a)) - REFV[band]
        have = B.have0.copy()
        for d in D_.values():
            have &= np.isfinite(d)
        print(f'\n### F{band}: median dm - reference / MAD by dolphot bin ({int(have.sum())} stars; hard cap, bgfree)')
        print('| bin | N | ' + ' | '.join(D_) + ' |')
        print('|---|---|' + '---|' * len(D_))
        for lo, hi in BINS[band]:
            st = have & (B.ref >= lo) & (B.ref < hi)
            if st.sum() < 5:
                continue
            print(f'| {lo}-{hi} | {int(st.sum())} | ' + ' | '.join(f'{np.median(d[st]):+.3f} / {mad(d[st]):.3f}' for d in D_.values()) + ' |')
        st = have & (B.ref >= BINS[band][0][0]) & (B.ref < BINS[band][-1][1])
        print(f'| all | {int(st.sum())} | ' + ' | '.join(f'{np.median(d[st]):+.3f} / {mad(d[st]):.3f}' for d in D_.values()) + ' |')
        fr = col('flat_fitw')
        wr = col('wrim')
        print(f'rows: median flat (PSF^2-weighted over fit px) {np.nanmedian(fr):.4f}, p16-84 {np.nanpercentile(fr, 16):.4f}-{np.nanpercentile(fr, 84):.4f}; '
              f'median PSF^2 weight share in rim px {np.nanmedian(wr):.3f}')


if __name__ == '__main__':
    main(sys.argv[1:] or ['150W', '200W', '250M', '300M'])
