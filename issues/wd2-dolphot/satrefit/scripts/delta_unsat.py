"""Derive Delta_unsat(r) for bands radprof did not measure (F200W, F300M), and validate the aggregation on F150W/F250M.

Reuses radprof_measure.process_frame (imported, unmodified) and the stack() logic of radprof_analyze.py (copied):
per radial bin, over isolated ('iso') unsaturated daophot stars (flags 0/1, dolphot-matched) in the 2 mag below the
satstar faint edge, neighbours removed (dsumb), fit  sum(data-bg-nbrs) = (1+S) sum(model)  with weights 1/npix and
4-sigma clipping.  Output: out/delta_unsat_<band>.txt  (r_arcsec  Delta  N).
usage: python delta_unsat.py validate BAND        # aggregate existing radprof/data/prof_main2_F<band>.fits
       python delta_unsat.py measure BAND DET1,DET2
"""
import sys
import numpy as np
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
import astropy.units as u

Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
sys.path.insert(0, Q + '/radprof')
sys.path.insert(0, Q)
import radprof_measure as RM  # noqa: E402
import analyze as an  # noqa: E402

PIX = {'F150W': 0.031, 'F200W': 0.031, 'F250M': 0.063, 'F300M': 0.063}
SW_EDGES = list(np.arange(0, 4.01, 0.5)) + [5, 6, 8, 10, 12, 15, 18, 22, 26, 30, 36, 43, 50]
LW_EDGES = list(np.arange(0, 4.01, 0.5)) + [5, 6, 8, 10, 12, 15, 18, 22, 25]
NEWCFG = {
    'F200W': dict(vg='12101', dets=['nrcb1', 'nrcb3'], pix=0.031, key='200W', iso_as=0.6, rmax=50, fwhm=2.2, satfov=512, edges=SW_EDGES),
    'F300M': dict(vg='12101', dets=['nrcblong'], pix=0.063, key='300M', iso_as=1.0, rmax=25, fwhm=1.6, satfov=1024, edges=LW_EDGES),
}


def starparts(t):
    ntot = np.asarray(t['ntot'], float)
    nval = np.asarray(t['nval'], float)
    mp = np.asarray(t['msum'], float)
    ok = (nval >= 0.9 * np.maximum(ntot, 1)) & (mp > 0)
    return np.where(ok, np.asarray(t['dsumb']) - mp, 0.0), np.where(ok, mp, 0.0), ok


def stack_bin(t, m, b, minn=8):
    num, den, ok = starparts(t)
    nv = np.asarray(t['nval'], float)
    good = m & ok[:, b]
    n = int(good.sum())
    if n < minn:
        return np.nan, n
    x = den[good, b]
    npx = nv[good, b]
    y = num[good, b] + den[good, b]
    w = 1.0 / npx
    keep = np.ones(n, bool)
    for _ in range(3):
        k1 = (w[keep] * x[keep] * y[keep]).sum() / (w[keep] * x[keep] ** 2).sum()
        r = (y - k1 * x) * np.sqrt(w)
        sd = 1.4826 * np.median(np.abs(r - np.median(r)))
        keep = np.abs(r - np.median(r)) < 4 * sd
    return k1 - 1, n


def aggregate(t, edges, band):
    sel = np.asarray((t['kind'] == 'dao') & (t['iso'] == 1))
    c = 0.5 * (edges[1:] + edges[:-1]) * PIX[band]
    rows = []
    for b in range(len(c)):
        d, n = stack_bin(t, sel, b)
        rows.append((c[b], d, n))
    return rows


def write(rows, band, tag=''):
    fn = f'{Q}/satrefit/out/delta_unsat_{band}{tag}.txt'
    with open(fn, 'w') as f:
        f.write('# r_arcsec Delta_unsat N  (iso unsaturated daophot stars, neighbours removed, per-bin weighted fit)\n')
        for r, d, n in rows:
            f.write(f'{r:.4f} {d:+.4f} {n}\n')
    print(open(fn).read())


if __name__ == '__main__':
    mode, band = sys.argv[1], 'F' + sys.argv[2]
    if mode == 'validate':
        t = Table.read(f'{Q}/radprof/data/prof_main2_{band}.fits')
        edges = np.array([float(e) for e in t.meta['EDGES'].split(',')])
        write(aggregate(t, edges, band), band, '_validate')
    else:
        dets = sys.argv[3].split(',')
        cfg = dict(NEWCFG[band])
        cfg['dets'] = dets
        an.ZPWIN.update(an.zp_windows())
        A = an.Arm('main2')
        bnd = band[1:]
        sel = A.matched & A.rep[bnd] & np.isfinite(A.ref[bnd])
        edge = float(np.nanpercentile(A.ref[bnd][sel], 95))
        print(band, 'satstar faint edge (95th pct of dolphot mag of replaced matched stars)', edge, flush=True)
        cfg['dao_mag'] = (edge, edge + 2.0)
        m = Table.read('/orange/adamginsburg/jwst/wd2/dolphot_benchmark/matched_Q_main2kf.fits')
        mag = an.fl(m['ref_' + cfg['key']])
        ok = np.isfinite(mag)
        ref = dict(sky=SkyCoord(np.asarray(m['RA'], float)[ok] * u.deg, np.asarray(m['DEC'], float)[ok] * u.deg), mag=mag[ok])
        out = []
        for det in dets:
            for exp in (1, 2, 3, 4):
                t = RM.process_frame('main2', band, det, exp, ref, cfg)
                print(band, det, exp, len(t), flush=True)
                out.append(t)
        T = vstack(out, metadata_conflicts='silent')
        edges = np.array(cfg['edges'], float)
        T.meta['EDGES'] = ','.join(f'{e:g}' for e in edges)
        T.write(f'{Q}/satrefit/out/prof_main2_{band}.fits', overwrite=True)
        write(aggregate(T, edges, band), band)
