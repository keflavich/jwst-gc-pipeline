"""Round 2.  usage: python run_frames2.py BAND DET1,DET2 [EXP...]   (writes out2/<band>_<stem>_satrefit.fits)
New variants (v7b combinations), peak ratios pk_<variant> for the cap scaling, and the wing self-calibration emulation."""
import re, sys, os
import numpy as np
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit')
import satrefit_core as C

band = sys.argv[1]
dets = sys.argv[2].split(',')
exps = [int(x) for x in sys.argv[3:]] or [1, 2, 3, 4]
lw = band in ('250M', '277W', '300M', '323N', '335M', '405N', '410M', '444W', '466N')
vgroup = {'150W': '10101', '200W': '12101', '250M': '04101', '300M': '12101'}[band]
RM = 0.5 if lw else 0.35
rmid = 12 if lw else 20
V = {}
if band in ('150W', '250M'):
    V['bgfree+v7b'] = {'bg': 'free', 'psfcorr': 'b'}
    V['uni+v7b'] = {'uniform': 1, 'psfcorr': 'b'}
    V['uni+bgfree+v7b'] = {'uniform': 1, 'bg': 'free', 'psfcorr': 'b'}
    V[f'r{rmid}+bgfree+v7b'] = {'rmax': rmid, 'bg': 'free', 'psfcorr': 'b'}
    if lw:
        V['r6+bgfree+v7b'] = {'rmax': 6, 'bg': 'free', 'psfcorr': 'b'}
    V['g4+bgfree+v7b'] = {'guard': 4, 'bg': 'free', 'psfcorr': 'b'}
    V['bgfree'] = {'bg': 'free'}
    V['uni+bgfree'] = {'uniform': 1, 'bg': 'free'}
    V['v7b'] = {'psfcorr': 'b'}
else:
    V['uni'] = {'uniform': 1}
    V['bgfree'] = {'bg': 'free'}
    V['v7b'] = {'psfcorr': 'b'}
    V['bgfree+v7b'] = {'bg': 'free', 'psfcorr': 'b'}
    V['uni+bgfree+v7b'] = {'uniform': 1, 'bg': 'free', 'psfcorr': 'b'}


def parse_delta_radprof(b):
    txt = open(C.Q + '/radprof/radprof.md').read().split('\n')
    key = f'### F{b} main2 per-bin Delta(r) (neighbours removed), iso, satstars precap'
    i = [k for k, l in enumerate(txt) if l.startswith(key)]
    r, d = [], []
    for l in txt[i[0] + 4:]:
        if not l.startswith('|'):
            break
        cells = [c.strip() for c in l.strip('|').split('|')]
        m = re.match(r'([+-]?\d+\.\d+)', cells[-1])
        if m:
            r.append(float(cells[0]))
            d.append(float(m.group(1)))
    return np.array(r), np.array(d)


def parse_delta_file(b):
    a = np.loadtxt(f'{C.Q}/satrefit/out/delta_unsat_F{b}.txt')
    a = a[np.isfinite(a[:, 1])]
    return a[:, 0], a[:, 1]


dtab = parse_delta_radprof(band) if band in ('150W', '250M') else parse_delta_file(band)
print('Delta_unsat', band, dtab, flush=True)
tree = C.Q + '/tree_main2'
radii = [1, 2, 3, 4, 5, 6, 8]
for det in dets:
    grid = None
    for e in exps:
        stem = f'jw03523005001_{vgroup}_{e:05d}_{det}_align_o005_crf'
        fn = f'{tree}/F{band}/pipeline/{stem}.fits'
        outfn = f'{C.Q}/satrefit/out2/{band}_{stem}_satrefit.fits'
        if os.path.exists(outfn):
            print('exists', outfn, flush=True)
            continue
        if grid is None:
            hdr = C.fits.getheader(fn)
            grid, gf = C.load_grid(tree + '/psfs', hdr, lw)
            C.log('grid', gf)
        t = C.run_frame(fn, band, V, grid, dtab=dtab, rmax_arcsec=RM, wing_radii=radii)
        t.write(outfn, overwrite=True)
        C.log('wrote', outfn)
