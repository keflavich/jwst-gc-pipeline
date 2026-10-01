"""Per-exposure physical-optics modes fitted to the held-out dither of the LOO model.

For a held-out dither k the leave-one-dither-out prediction of patternfit.py (static
pattern Q from the other five dithers + per-exposure nuisance, <tag>_e<k>.npz) leaves the
residual  r_k = d_k - pred_k.  If the dither-to-dither change of the halo is a small
physical change of the pupil or wavefront, r_k is (to first order) a combination of

    dP_j = a_k * D[ dI/dtheta_j ](q_k)          (theta_j: a few pupil/wavefront parameters)

evaluated through the static physical-optics PSF (physfit.py) at the dither's own distorted
pixel positions q_k.  The coefficients are fitted on the held-out dither only, together with
a refit of its smooth nuisance (flux, pedestal, gradient, star-centred halo B-splines), by
iteratively reweighted least squares.  The mode families:

    zern<n>   Zernikes n=2..<n> (no piston/tip/tilt)          global low-order WFE
    seg       piston/tip/tilt of the 18 segments (54)           telescope WFE
    shear     shift of the whole static OPD relative to the pupil amplitude (2)
    shearhp   shift of the high-pass (>6 cycles/pupil) OPD only (2)   out-of-pupil screen
    hpscale   scale of the high-pass OPD (1)
              (shear/shearhp/hpscale need a non-zero static OPD, i.e. a phase-retrieved
              static; about the geometry-only static of the README, OPD = 0, they are
              identically zero and physloo.py raises)
    ampz<n>   pupil transmission Zernikes n=1..<n> (vignetting, apodisation)
    stop      edge of a circular stop at the pupil rim: radius and shear (3)

Control: the same families built on a random OPD with the same power spectrum
(CONTROL=1), i.e. physically-shaped maps that know nothing about this star's field.
CONTROL=1 also needs a non-zero static OPD (it raises otherwise); the README's control is
CONTROL_ROT (families built on a rotated pupil), which works about OPD = 0.

    python physloo.py <static.npz> <looTag> <k> <families, e.g. zern6,seg,shear>
"""
import numpy as np, sys, os, time
import scipy.sparse as sp
import patternfit as pf
from patternfit import QE, secondary_centres, to_F
from exposure import Exposure
from physpsf import PhysPSF, f480m_band, DEFAULT_GEOM
from physfit import random_opd

N = pf.N
BINS = [(30, 50), (50, 80), (80, 120), (120, 200), (200, 300)]


def load_static(fn, J1, nlam=12):
    z = np.load(fn, allow_pickle=True)
    geom = z['geom'].item(); det = z['det'].item()
    lams, wl = f480m_band(nlam)
    G = int(round(np.sqrt(z['opd'].size)))
    ps = PhysPSF(N, J1, lams, wl, geom=geom, G=G)
    ps.det.update(det)
    return ps, z['opd'], [tuple(s) for s in z['src']], z['f']


def highpass(ps, opd, fcut=6.0):
    """OPD above fcut cycles per 6.6 m pupil"""
    o = opd.reshape(ps.G, ps.G)
    k = np.fft.fftfreq(ps.G, d=2*ps.half/(ps.G-1))*6.6
    kk = np.hypot(k[None, :], k[:, None])
    return np.fft.ifft2(np.fft.fft2(o)*(kk > fcut)).real.ravel()


def grid_grad(ps, o):
    h = 2*ps.half/(ps.G-1)
    gy, gx = np.gradient(o.reshape(ps.G, ps.G), h)
    return gx.ravel(), gy.ravel()


def mode_images(ps, opd, fams):
    """list of (name, dI) for the requested families at the static OPD"""
    inert = [f for f in fams if f in ('shear', 'shearhp', 'hpscale')]
    if inert and not np.any(highpass(ps, opd) if inert != ['shear'] else opd):
        # these families act on (the high-pass part of) the static OPD; about a geometry-only
        # static (OPD = 0, e.g. geo2.npz from `physfit.py geometry`) they are identically zero
        raise ValueError(f'mode families {inert} are identically zero for this static OPD '
                         '(OPD = 0 or no high-pass content); they need a phase-retrieved static')
    I, E = ps.intensity(opd, return_fields=True)
    out = []
    X, Y = ps.grid_xy()
    for fam in fams:
        if fam.startswith('zern'):
            Z = ps.zernike_basis(int(fam[4:]))
            out += [(f'{fam}_{j}', ps.jvp(opd, E, dopd=z*50e-9)) for j, z in enumerate(Z)]
        elif fam == 'seg':
            Sg = ps.segment_basis()
            out += [(f'seg_{j}', ps.jvp(opd, E, dopd=z*(50e-9 if j % 3 == 0 else 50e-9/0.66))) for j, z in enumerate(Sg)]
        elif fam in ('shear', 'shearhp'):
            o = opd if fam == 'shear' else highpass(ps, opd)
            gx, gy = grid_grad(ps, o)
            out += [(f'{fam}_x', ps.jvp(opd, E, dopd=-gx*0.01)), (f'{fam}_y', ps.jvp(opd, E, dopd=-gy*0.01))]
        elif fam == 'hpscale':
            out += [('hpscale', ps.jvp(opd, E, dopd=0.1*highpass(ps, opd)))]
        elif fam.startswith('ampz'):
            n = int(fam[4:])
            rho = np.hypot(X, Y)/3.3; th = np.arctan2(Y, X)
            maps = [rho*np.cos(th), rho*np.sin(th)]
            if n >= 2:
                maps += [rho**2, rho**2*np.cos(2*th), rho**2*np.sin(2*th)]
            if n >= 3:
                maps += [rho**3*np.cos(th), rho**3*np.sin(th), rho**3*np.cos(3*th), rho**3*np.sin(3*th)]
            out += [(f'{fam}_{j}', ps.jvp(opd, E, damp=0.05*m)) for j, m in enumerate(maps)]
        elif fam == 'stop':
            # an extra circular stop just inside the pupil corners: radius, dx, dy derivatives
            base = dict(ps.geom)
            def I_of(**kw):
                g = dict(base, stop_r=3.25); g.update(kw)
                ps.set_geom(g); return ps.intensity(opd)
            I0s = I_of()
            for nm, st in (('stop_r', 0.02), ('stop_dx', 0.02), ('stop_dy', 0.02)):
                out.append((nm, (I_of(**{nm: base.get(nm, 0.0)+st if nm != 'stop_r' else 3.25+st})-I0s)))
            ps.set_geom(base)
        elif fam == 'geomd':
            # per-exposure change of the pupil geometry: magnification (= effective
            # wavelength), rotation, strut width, segment size, and a red/blue spectral tilt
            base = dict(ps.geom)
            for nm, st in (('scale', 0.002), ('rot', 0.05), ('strut_w', 0.01), ('flat', 0.003)):
                ps.set_geom(dict(base, **{nm: base[nm]+st})); Ip = ps.intensity(opd)
                ps.set_geom(dict(base, **{nm: base[nm]-st})); Im = ps.intensity(opd)
                out.append((f'geomd_{nm}', (Ip-Im)/2))
            ps.set_geom(base)
            wl0 = ps.wl.copy(); t = np.linspace(-1, 1, len(wl0))
            ps.wl = wl0*(1+0.05*t); Ip = ps.intensity(opd); ps.wl = wl0*(1-0.05*t); Im = ps.intensity(opd); ps.wl = wl0
            out.append(('geomd_tilt', (Ip-Im)/2))
        else:
            raise ValueError(fam)
    return out


def exposure_columns(e, ps, src, flux, imgs):
    """a_k * [K_k * D(shift_s(dI))](q_k) for every mode image"""
    cols = []
    mult = e.mult()
    for nm, dI in imgs:
        P = sum(f*ps.detect(dI, shift=s) for s, f in zip(src, flux))
        cols.append(e.a*pf.nufft2(e.qx, e.qy, to_F(P)*mult))
    return np.stack(cols, 1) if cols else np.zeros((len(e.d), 0))


def rstd(x):
    return 1.4826*np.median(np.abs(x-np.median(x)))


SPIKES = (0, 30, 90, 150, 180, 210, 270, 330)


def spike_distance(qx, qy):
    th = np.degrees(np.arctan2(qy, qx)) % 360
    return np.min([np.abs((th-a+180) % 360-180) for a in SPIKES], 0)


def fit_and_score(e, r, var, extra, clean, R, fitmask, niter=4, dth=None):
    """IRLS: r ~ [extra | 1, X, Y, halo basis] c; returns robust sigma(chi) per radial bin"""
    base = sp.hstack([sp.csc_matrix(np.stack([np.ones_like(r), e.X, e.Y], 1)), e.B], format='csc')
    A = sp.hstack([sp.csc_matrix(extra), base], format='csc') if extra.shape[1] else base
    A = A[fitmask]; y = r[fitmask]; v = var[fitmask]
    rw = np.ones(len(y))
    for it in range(niter):
        sw = np.sqrt(rw/v)
        Aw = A.multiply(sw[:, None]).tocsc()
        cn = np.sqrt(np.asarray(Aw.multiply(Aw).sum(0))).ravel(); cn[cn == 0] = 1
        Aw = Aw@sp.diags(1/cn)
        Gm = (Aw.T@Aw).toarray(); Gm[np.diag_indices_from(Gm)] += 1e-8
        c = np.linalg.solve(Gm, Aw.T@(sw*y))/cn
        chi = (y-A@c)/np.sqrt(v)
        rw = 1/(1+(chi/4)**2)
    full = sp.hstack([sp.csc_matrix(extra), base], format='csc') if extra.shape[1] else base
    chi = (r-full@c)/np.sqrt(var)
    tab = [rstd(chi[clean & (R >= lo) & (R < hi)]) for lo, hi in BINS]
    if dth is not None:
        tab += [rstd(chi[clean & (R >= lo) & (R < hi) & (dth < 1.5)]) for lo, hi in BINS[2:]]
    return tab, c[:extra.shape[1]], chi


if __name__ == '__main__':
    stat, tag, k, fams = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4].split(',')
    files = [f'tgt_e{i}.npz' for i in range(1, 7)]
    J1 = np.load(files[0])['J']
    ps, opd, src, flux = load_static(stat, J1)
    if os.environ.get('GEO'):
        opd = opd*0
    if int(os.environ.get('CONTROL', 0)):
        rms_nm = np.std(highpass(ps, opd))*1e9
        if not rms_nm > 0:
            raise ValueError('CONTROL=1 draws a random screen with the rms of the high-pass static '
                             'OPD, which is 0 here (geometry-only static); use CONTROL_ROT instead')
        rng_opd = random_opd(ps.G, ps.half, rms_nm=rms_nm, seed=int(os.environ.get('SEED', 1)))*1e-9
        opd = rng_opd
    ex = Exposure(files[k-1], J1)
    e = QE(ex, [(ex.xt0, ex.yt0, 720., 4)]+secondary_centres(ex, 4))
    z = np.load(f'{tag}_e{k}.npz')
    e.a = float(z['a']); e.cx = z['cx']; e.cy = z['cy']; e.beta = z['beta']
    pred = z['pred'].ravel()[e.pix]; varQ = z['varQ'].ravel()[e.pix]; cov = z['cov'].ravel()[e.pix]
    r = e.d-pred; var = 1/e.w+varQ
    dq = ex.dq.ravel()[e.pix]
    R = np.hypot(e.qx-src[0][0], e.qy-src[0][1])
    dth = spike_distance(e.qx-src[0][0], e.qy-src[0][1])
    ok = np.isfinite(r) & np.isfinite(var) & (cov >= 3)
    clean = ok & ((dq & 6) == 0)
    fitmask = ok & (R >= 20) & (R < 400)
    # restrict everything to the fit region (speed)
    sel = np.flatnonzero(fitmask | (clean & (R < 300)))
    e.pix = e.pix[sel]; e.qx0 = e.qx0[sel]; e.qy0 = e.qy0[sel]; e.d = e.d[sel]; e.w = e.w[sel]
    e.X = e.X[sel]; e.Y = e.Y[sel]; e.phi = e.phi[sel]; e.B = e.B[sel]
    r, var, clean, R, fitmask, dth = r[sel], var[sel], clean[sel], R[sel], fitmask[sel], dth[sel]
    keepB = np.asarray(abs(e.B).sum(0)).ravel() > 0
    e.B = e.B[:, np.flatnonzero(keepB)]
    if os.environ.get('INJECT'):
        # injection-recovery: add the exact (non-linear) change of a 'true' PSF whose static
        # OPD differs from the fiducial one (INJ_OPD nm rms random screen) by a random
        # per-exposure perturbation of family INJECT=<fam>:<rms nm>
        fam, amp = os.environ['INJECT'].split(':'); amp = float(amp)*1e-9
        rng = np.random.default_rng(int(os.environ.get('INJ_SEED', 5)))
        opd_t = opd+random_opd(ps.G, ps.half, rms_nm=float(os.environ.get('INJ_OPD', 0)), seed=11)*1e-9
        if fam == 'seg':
            Sg = ps.segment_basis()[::3]
            d_opd = (rng.normal(size=len(Sg))@Sg)*amp
        else:
            Z = ps.zernike_basis(int(fam[4:]))
            d_opd = (rng.normal(size=len(Z))/np.sqrt(len(Z))@Z)*amp
        dI = ps.intensity(opd_t+d_opd)-ps.intensity(opd_t)
        inj = exposure_columns(e, ps, src, flux, [('inj', dI)])[:, 0]
        m = clean & (dth < 1.5) & (R > 80) & (R < 300)
        print(f'   injected {fam} {amp*1e9:.0f} nm on a {os.environ.get("INJ_OPD", 0)} nm-rms static screen: '
              f'rms(inj)/rms(resid) on spikes = {np.std(inj[m])/rstd(r[m]):.2f}', flush=True)
        r = r+inj
    tab0, _, _ = fit_and_score(e, r, var, np.zeros((len(r), 0)), clean, R, fitmask, dth=dth)
    print('   bins', BINS, '+ on-spike', BINS[2:])
    print(f'e{k} baseline (nuisance refit only):', ' '.join(f'{v:.2f}' for v in tab0), flush=True)
    t0 = time.time()
    if os.environ.get('CONTROL_ROT'):
        # control: the same families built on a pupil rotated by CONTROL_ROT deg (physically
        # shaped maps whose spikes and fine structure do not line up with the star's)
        ps.set_geom(dict(rot=ps.geom['rot']+float(os.environ['CONTROL_ROT'])))
    imgs = mode_images(ps, opd, fams)
    cols = exposure_columns(e, ps, src, flux, imgs)
    print(f'   {len(imgs)} modes built in {time.time()-t0:.0f}s', flush=True)
    if os.environ.get('MTFQ'):
        # empirical control family: per-exposure detector MTF polynomial (2nd-4th order, even
        # and odd) applied to the whole static pattern Q (star + scene)
        Qf = to_F(np.load(os.environ['MTFQ']))*e.mult()
        KX, KY = pf.KX, pf.KY
        polys = [KX**2, KY**2, KX*KY, KX**4, KY**4, KX**2*KY**2, KX**3*KY, KX*KY**3,
                 1j*KX**3, 1j*KY**3, 1j*KX**2*KY, 1j*KX*KY**2]
        mcols = np.stack([e.a*pf.nufft2(e.qx, e.qy, Qf*pp) for pp in polys], 1)
        tab, c, _ = fit_and_score(e, r, var, mcols, clean, R, fitmask, dth=dth)
        print(f'e{k} +mtfQ     ({mcols.shape[1]:3d}):', ' '.join(f'{v:.2f}' for v in tab), flush=True)
    for fam in fams:
        m = [i for i, (nm, _) in enumerate(imgs) if nm.startswith(fam+'_') or nm == fam or (fam == 'stop' and nm.startswith('stop'))]
        if not m:
            continue
        tab, c, _ = fit_and_score(e, r, var, cols[:, m], clean, R, fitmask, dth=dth)
        print(f'e{k} +{fam:8s} ({len(m):3d}):', ' '.join(f'{v:.2f}' for v in tab), flush=True)
    tab, c, chi = fit_and_score(e, r, var, cols, clean, R, fitmask, dth=dth)
    print(f'e{k} +ALL      ({cols.shape[1]:3d}):', ' '.join(f'{v:.2f}' for v in tab), flush=True)
    np.savez_compressed(f'physloo_{os.environ.get("OUT", "run")}_e{k}.npz', coef=c, names=[n for n, _ in imgs],
                        chi=chi, pix=e.pix, R=R, clean=clean, r=r, var=var, cols=cols.astype(np.float32))
