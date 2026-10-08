"""Per-detector in-detector linear term J of (band m6 per-frame catalog - Gaia DR3),
r = c0_frame + J (p - p_centre), anchored on Gaia DR3 propagated to the JWST epoch.
Pair machinery (measure_offset + local_residual_map) as in ../datascale_field.py.
Exposures are combined with per-frame c0 and a shared J; uncertainties from a
cluster bootstrap over Gaia sources (shared across bands, so J_F212N - J_F150W
uncertainties include the common-star correlation).
Usage: FIELD_ROOT=... python datascale_gaia.py f212n f150w f162m
Env: MATCH (arcsec, 0.15), MAXEXP (4), NBOOT (500), GAIAFILE, GMAX, TAG"""
import glob, os, re, sys, pickle
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from jwst_gc_pipeline.photometry.astrometry_offsets import local_residual_map, measure_offset

R = os.environ.get('FIELD_ROOT', '/orange/adamginsburg/jwst/wd2')
STAGE = 'resbgsub_m6'
MATCH = float(os.environ.get('MATCH', '0.20'))
MAXEXP = os.environ.get('MAXEXP', '4')
NBOOT = int(os.environ.get('NBOOT', '500'))
GMAX = float(os.environ.get('GMAX', '21'))
TAG = os.environ.get('TAG', 'wd2')
GAIAFILE = os.environ.get('GAIAFILE', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gaia_dr3_wd2.fits'))
PROG = os.environ.get('PROGPAT', 'visit001_vgroup*')
BANDS = sys.argv[1:] or ['f212n', 'f150w', 'f162m']
USESAT = os.environ.get('USESAT', '1') == '1'
PIX = 31.0
RAD = 206264.806
MINPAIRS = 12   # measure_offset (module union)
MINLOC = int(os.environ.get('MINLOC', '6'))   # per-detector pairs
SATQ = float(os.environ.get('SATQ', '0.6'))
RCUT = float(os.environ.get('RCUT', '45'))

g3 = Table.read(GAIAFILE)
gk = (np.asarray(g3['ruwe'], float) < 1.4) & np.isin(np.asarray(g3['astrometric_params_solved']), (31, 95))
gk &= np.asarray(g3['phot_g_mean_mag'], float) < GMAX
gk &= np.isfinite(np.asarray(g3['pmra'], float)) & (np.asarray(g3['ra_error'], float) < 5) & (np.asarray(g3['dec_error'], float) < 5)
G = g3[gk]
print(f'Gaia DR3: {len(g3)} rows, {len(G)} after ruwe<1.4, 5/6-param, G<{GMAX}, pos err<5 mas', flush=True)
_cache = {}


def gaia_at(mjd):
    """Gaia positions linearly propagated to mjd (pm includes cos(dec); parallax neglected)."""
    key = round(float(mjd), 1)
    if key not in _cache:
        dt = (Time(mjd, format='mjd').jyear - np.asarray(G['ref_epoch'], float))
        dec = np.asarray(G['dec'], float)
        ra = np.asarray(G['ra'], float) + np.asarray(G['pmra'], float) * dt / 3.6e6 / np.cos(np.deg2rad(dec))
        de = dec + np.asarray(G['pmdec'], float) * dt / 3.6e6
        _cache[key] = SkyCoord(ra * u.deg, de * u.deg)
    return _cache[key]


def epoch(t):
    fn = t.meta['FILENAME']
    if not os.path.exists(fn):
        fn = os.path.join(R, 'pipeline', os.path.basename(fn))
    h = fits.getheader(fn, 1)
    return h['MJD-BEG'] if 'MJD-BEG' in h else Time(h['DATE-BEG']).mjd


def load_frame(path):
    t = Table.read(path)
    mjd = epoch(t)
    anc = gaia_at(mjd)
    q = np.asarray(t['qfit'], float)
    with np.errstate(divide='ignore', invalid='ignore'):
        snr = np.asarray(t['flux_fit'], float) / np.asarray(t['flux_err'], float)
    k = np.isfinite(q) & (q <= 0.1) & np.isfinite(snr) & (snr >= 20)
    sc = SkyCoord(t['skycoord_centroid'][k])
    x = np.asarray(t['x_fit'], float)[k]
    y = np.asarray(t['y_fit'], float)[k]
    nuns = int(k.sum())
    if USESAT:
        # Gaia stars are bright; at F150W/F212N they are saturated and absent from the
        # unsaturated list, so add the satstar_catalog (wing-fit) positions.
        sp = t.meta['FILENAME'] if os.path.exists(t.meta['FILENAME']) else os.path.join(R, 'pipeline', os.path.basename(t.meta['FILENAME']))
        sp = sp.replace('.fits', f'_{STAGE}_satstar_catalog.fits')
        if os.path.exists(sp):
            st = Table.read(sp)
            ssc = SkyCoord(st['skycoord_fit'])
            sk = np.isfinite(ssc.ra.deg) & np.isfinite(np.asarray(st['x_0'], float)) & (np.asarray(st['qfit'], float) <= SATQ)
            sc = SkyCoord(np.r_[sc.ra.deg, ssc.ra.deg[sk]] * u.deg, np.r_[sc.dec.deg, ssc.dec.deg[sk]] * u.deg)
            x = np.r_[x, np.asarray(st['x_0'], float)[sk]]   # x_fit is in cutout coords; x_0 is the detector pixel
            y = np.r_[y, np.asarray(st['y_0'], float)[sk]]
        else:
            print(f'  no satstar catalog {sp}')
    ok = np.isfinite(sc.ra.deg) & np.isfinite(x)
    return dict(sc=sc[ok], x=x[ok], y=y[ok], anc=anc, nuns=nuns, nm=os.path.basename(path))


def module_tie(frames):
    """Bulk Gaia tie from the union of one exposure's detectors in a module (sky positions only)."""
    sc = SkyCoord(np.concatenate([f['sc'].ra.deg for f in frames]) * u.deg,
                  np.concatenate([f['sc'].dec.deg for f in frames]) * u.deg)
    g = measure_offset(sc, frames[0]['anc'], sweep=True, min_pairs=MINPAIRS, context=frames[0]['nm'] + '+module')
    return g


def frame_pairs(f, g):
    sc, x, y, anc, nm, nuns = f['sc'], f['x'], f['y'], f['anc'], f['nm'], f['nuns']
    if g is None or not g.get('ok') or g.get('swept') or g['off'] > MATCH * 1000 / 3:
        print(f'  {nm[:44]} skip: no verified Gaia tie ({None if g is None else (g.get("ok"), g.get("swept"), round(g["off"],1), round(g.get("contrast",0),1))})'); return None
    lrm = local_residual_map(sc, anc, g, cell_arcsec=1e9, match_radius=MATCH * u.arcsec,
                             min_stars=MINLOC, tol_mas=np.inf, return_pairs=True, context=nm)
    pr = lrm.get('pairs')
    if not pr or len(pr['ia']) < MINLOC:
        print(f'  {nm[:44]} skip: few pairs'); return None
    ia, ib = np.asarray(pr['ia']), np.asarray(pr['ib'])
    cosd = np.cos(np.deg2rad(anc.dec.deg[ib]))
    rx = (sc.ra.deg[ia] - anc.ra.deg[ib]) * cosd * 3.6e6
    ry = (sc.dec.deg[ia] - anc.dec.deg[ib]) * 3.6e6
    print(f'  {nm[:44]} tie=({g["dra"]:+.1f},{g["ddec"]:+.1f}) mas off={g["off"]:.1f} pairs={len(ia)} (unsat stars {nuns}) '
          f'median(frame-gaia)=({np.median(rx):+.1f},{np.median(ry):+.1f})', flush=True)
    return dict(gid=ib, x=x[ia] - 1023.5, y=y[ia] - 1023.5, rx=rx, ry=ry)


def fit(fr, gid, px, py, rx, ry, w, mask):
    """Weighted LS with per-frame c0 and shared J.  Returns (cx[2: J row for RA], ...)"""
    nf = fr.max() + 1
    A = np.zeros((len(px), nf + 2))
    A[np.arange(len(px)), fr] = 1
    A[:, nf], A[:, nf + 1] = px, py
    sw = np.sqrt(w * mask)
    cx = np.linalg.lstsq(A * sw[:, None], rx * sw, rcond=None)[0]
    cy = np.linalg.lstsq(A * sw[:, None], ry * sw, rcond=None)[0]
    return np.array([[cx[nf], cx[nf + 1]], [cy[nf], cy[nf + 1]]]), A, cx, cy


def clipped(d):
    fr, gid, px, py, rx, ry = d
    # robust start: drop random false matches (>RCUT mas from the per-frame median offset)
    mask = np.zeros(len(px), bool)
    for f in np.unique(fr):
        s_ = fr == f
        mask[s_] = np.hypot(rx[s_] - np.median(rx[s_]), ry[s_] - np.median(ry[s_])) < RCUT
    for _ in range(5):
        J, A, cx, cy = fit(fr, gid, px, py, rx, ry, np.ones(len(px)), mask)
        ex, ey = rx - A @ cx, ry - A @ cy
        s = 1.4826 * np.median(np.abs(np.r_[ex[mask], ey[mask]]))
        mask = (np.abs(ex) < 3 * s) & (np.abs(ey) < 3 * s)
    return mask, s


def inv(J):
    t, c = J[0, 0] - J[1, 1], J[1, 0] + J[0, 1]
    return t, c, np.hypot(t, c) / (2 * PIX) * RAD, np.degrees(np.arctan2(c, t))


if __name__ == '__main__':
    data = {}   # (band, det) -> arrays
    for band in BANDS:
        files = sorted(glob.glob(f'{R}/{band.upper()}/{band}_nrc[ab][1-4]_{PROG}_exp0000[1-{MAXEXP}]_{STAGE}_daophot_basic.fits'))
        print(f'### {band}: {len(files)} frame catalogs', flush=True)
        per = {}
        byexp = {}
        for p in files:
            mm = re.search(r'_(nrc[ab][1-4])_visit\d+_vgroup\w+_exp(\d+)_', os.path.basename(p))
            byexp.setdefault((mm.group(2), mm.group(1)[:4]), []).append((mm.group(1), load_frame(p)))
        for (e, mod), lst in sorted(byexp.items()):
            g = module_tie([f for _d, f in lst])
            for det, f in lst:
                r = frame_pairs(f, g)
                if r is not None:
                    per.setdefault(det, []).append(r)
        for det, lst in per.items():
            data[(band, det)] = dict(
                fr=np.concatenate([np.full(len(r['x']), i) for i, r in enumerate(lst)]),
                gid=np.concatenate([r['gid'] for r in lst]), x=np.concatenate([r['x'] for r in lst]),
                y=np.concatenate([r['y'] for r in lst]), rx=np.concatenate([r['rx'] for r in lst]),
                ry=np.concatenate([r['ry'] for r in lst]), nfr=len(lst))
    # nominal fits
    res = {}
    for key, d in data.items():
        arr = (d['fr'], d['gid'], d['x'], d['y'], d['rx'], d['ry'])
        mask, s = clipped(arr)
        d['mask'] = mask
        J, *_ = fit(*arr, np.ones(len(mask)), mask)
        res[key] = dict(J=J, sigma=s, nfr=d['nfr'], npair=int(mask.sum()), nstar=len(np.unique(d['gid'][mask])))
    # cluster bootstrap over Gaia sources, shared across bands/detectors
    rng = np.random.default_rng(1135)
    ng = len(G)
    boot = {k: [] for k in data}
    for b in range(NBOOT):
        wg = np.bincount(rng.integers(0, ng, ng), minlength=ng).astype(float)
        for key, d in data.items():
            w = wg[d['gid']]
            if (w * d['mask']).sum() < 8:
                boot[key].append(np.full((2, 2), np.nan)); continue
            boot[key].append(fit(d['fr'], d['gid'], d['x'], d['y'], d['rx'], d['ry'], w, d['mask'])[0])
    boot = {k: np.array(v) for k, v in boot.items()}
    pickle.dump(dict(res=res, boot=boot), open(f'gaia_fit_{TAG}.pkl', 'wb'))
    print(f'\n## {TAG}: J (mas/pix), tr-/curl+ and m (arcsec) per detector; errors = cluster-bootstrap std ({NBOOT} draws)')
    DETS = [f'nrc{m}{i}' for m in 'ab' for i in range(1, 5)]
    for band in BANDS:
        print(f'\n### {band} - gaia')
        print('det     Nstars Npairs nfr sigma | J00 J01 J10 J11 (+-) | tr- curl+ | m[arcsec] +- | angle')
        for det in DETS:
            if (band, det) not in res:
                print(det, 'no data'); continue
            r, B = res[(band, det)], boot[(band, det)]
            J = r['J']
            ok = np.isfinite(B[:, 0, 0])
            eJ = B[ok].reshape(-1, 4).std(0)
            t, c, m, ang = inv(J)
            mb = np.array([inv(j)[2] for j in B[ok]])
            print(f'{det}  {r["nstar"]:4d} {r["npair"]:5d} {r["nfr"]} {r["sigma"]:5.1f} | '
                  f'J=[{J[0,0]:+.4f} {J[0,1]:+.4f}; {J[1,0]:+.4f} {J[1,1]:+.4f}] +-({eJ[0]:.4f},{eJ[1]:.4f},{eJ[2]:.4f},{eJ[3]:.4f}) | '
                  f'{t:+.4f} {c:+.4f} | m={m:5.1f} +-{mb.std():4.1f} (boot 16-84: {np.percentile(mb,16):.1f}-{np.percentile(mb,84):.1f}) | ang={ang:+.0f}')
    if 'f212n' in BANDS and 'f150w' in BANDS:
        print('\n### J_F212N - J_F150W (both Gaia-anchored)')
        for det in DETS:
            if ('f212n', det) in res and ('f150w', det) in res:
                D = res[('f212n', det)]['J'] - res[('f150w', det)]['J']
                BD = boot[('f212n', det)] - boot[('f150w', det)]
                ok = np.isfinite(BD[:, 0, 0]) & np.isfinite(boot[('f150w', det)][:, 0, 0])
                t, c, m, ang = inv(D)
                mb = np.array([inv(j)[2] for j in BD[ok]])
                print(f'{det}  m_diff={m:5.1f} +-{mb.std():4.1f} arcsec (boot 16-84: {np.percentile(mb,16):.1f}-{np.percentile(mb,84):.1f}) '
                      f'tr-={t:+.4f} curl+={c:+.4f} ang={ang:+.0f}')
