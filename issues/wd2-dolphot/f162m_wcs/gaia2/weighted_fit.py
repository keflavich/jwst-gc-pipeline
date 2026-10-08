"""Error-weighted J: per-star sigma^2 = s_int^2 + (pm_err*dt)^2 + gaia_pos_err^2 (mas, per component); 3-sigma clip on normalised residuals;
cluster bootstrap over Gaia sources.  Usage: python weighted_fit.py TAG FIELD(for E file) band ..."""
import sys, pickle
import numpy as np
TAG, FIELD = sys.argv[1], sys.argv[2]
BANDS_ = sys.argv[3:]
DTY = float(__import__('os').environ.get('DTY', '8.55'))
SINT = 1.0
sys.argv = [sys.argv[0]]
import datascale_gaia2 as D
P = pickle.load(open(f'gaia_fit_{TAG}.pkl', 'rb'))
data, G = P['data'], P['G']
sg2 = {}
for gid in range(len(G)):
    pass
epmra, epmde = np.asarray(G['pmra_error'], float), np.asarray(G['pmdec_error'], float)
era, ede = np.asarray(G['ra_error'], float), np.asarray(G['dec_error'], float)
sx = np.sqrt(SINT**2 + (epmra * DTY)**2 + era**2)
sy = np.sqrt(SINT**2 + (epmde * DTY)**2 + ede**2)
sig = np.sqrt(0.5 * (sx**2 + sy**2))   # one weight per star
res = {}
rng = np.random.default_rng(1135)
NB = 300
wgs = [np.bincount(rng.integers(0, len(G), len(G)), minlength=len(G)).astype(float) for _ in range(NB)]
for band in BANDS_:
    for det in [f'nrc{m}{i}' for m in 'ab' for i in range(1, 5)]:
        d = data.get((band, det))
        if d is None:
            continue
        w0 = 1 / sig[d['gid']] ** 2
        mask = np.ones(len(w0), bool)
        mask0, _ = D.clipped((d['fr'], d['gid'], d['x'], d['y'], d['rx'], d['ry']))
        mask = mask0.copy()
        for _ in range(4):
            J, A, cx, cy = D.fit(d['fr'], d['gid'], d['x'], d['y'], d['rx'], d['ry'], w0, mask)
            z = np.hypot((d['rx'] - A @ cx) / sig[d['gid']], (d['ry'] - A @ cy) / sig[d['gid']]) / np.sqrt(2)
            mask = z < 3
        if len(np.unique(d['gid'][mask])) < 8:
            continue
        bm = []
        for wg in wgs:
            w = w0 * wg[d['gid']]
            if (w * mask).sum() < 1e-9 or (wg[d['gid']] * mask).sum() < 8:
                continue
            bm.append(D.fit(d['fr'], d['gid'], d['x'], d['y'], d['rx'], d['ry'], w, mask)[0])
        bm = np.array(bm)
        res[(band, det)] = dict(J=J, boot=bm, nstar=len(np.unique(d['gid'][mask])), neff=float(w0[mask].sum()**2 / (w0[mask]**2).sum()))
        mb = np.array([D.inv(j)[2] for j in bm])
        print(f'{band} {det}  Nstar={res[(band,det)]["nstar"]:3d} Neff(pairs)={res[(band,det)]["neff"]:5.1f} m_w={D.inv(J)[2]:5.1f} (boot 16-84 {np.percentile(mb,16):.1f}-{np.percentile(mb,84):.1f})  tr-,curl+=({D.inv(J)[0]:+.4f},{D.inv(J)[1]:+.4f})', flush=True)
pickle.dump(res, open(f'weighted_{TAG}.pkl', 'wb'))
