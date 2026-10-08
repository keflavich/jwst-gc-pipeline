"""usage: python run_frames.py BAND DET1,DET2 [EXP...]   e.g. 150W nrcb1,nrcb3 1 2 3 4"""
import re, sys, os
import numpy as np
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit')
import satrefit_core as C

band = sys.argv[1]
dets = sys.argv[2].split(',')
exps = [int(x) for x in sys.argv[3:]] or [1, 2, 3, 4]
lw = band in ('250M', '277W', '300M', '323N', '335M', '405N', '410M', '444W', '466N')
vgroup = {'150W': '10101', '200W': '12101', '250M': '04101', '300M': '12101'}[band]
rad = {False: [10, 20, 40], True: [6, 12, 25]}[lw]
far = (30, 40) if lw else (58, 77)
RM = 0.5 if lw else 0.35

V = {}
for g in (2, 4, 6, 10):
    V[f'g{g}'] = {'guard': g}
for s in (2, 4, 8):
    V[f'w{s}'] = {'wsig': s}
V['bgfar'] = {'bg': 'far', 'far': far}
V['bgfree'] = {'bg': 'free'}
V['uni'] = {'uniform': 1}
for r in rad:
    V[f'r{r}'] = {'rmax': r}
V['nbr_all'] = {'nbr': 'all'}
V['nbr_none'] = {'nbr': 'none'}
V['fill'] = {'fill': 1}
V['v7a'] = {'psfcorr': 'a'}
V['v7b'] = {'psfcorr': 'b'}
# combinations
for g in (4, 6):
    V[f'g{g}+bgfree'] = {'guard': g, 'bg': 'free'}
    V[f'g{g}+bgfar'] = {'guard': g, 'bg': 'far', 'far': far}
    V[f'g{g}+uni'] = {'guard': g, 'uniform': 1}
    V[f'g{g}+uni+bgfree'] = {'guard': g, 'uniform': 1, 'bg': 'free'}
V['uni+bgfree'] = {'uniform': 1, 'bg': 'free'}
V['uni+bgfar'] = {'uniform': 1, 'bg': 'far', 'far': far}
V[f'r{rad[1]}+bgfree'] = {'rmax': rad[1], 'bg': 'free'}
V[f'r{rad[1]}+uni+bgfree'] = {'rmax': rad[1], 'uniform': 1, 'bg': 'free'}
V[f'g4+r{rad[1]}+bgfree'] = {'guard': 4, 'rmax': rad[1], 'bg': 'free'}
V['bgfree+v7a'] = {'bg': 'free', 'psfcorr': 'a'}
V['g4+bgfree+v7a'] = {'guard': 4, 'bg': 'free', 'psfcorr': 'a'}


def parse_delta(b):
    txt = open(C.Q + '/radprof/radprof.md').read().split('\n')
    key = f'### F{b} main2 per-bin Delta(r) (neighbours removed), iso, satstars precap'
    i = [k for k, l in enumerate(txt) if l.startswith(key)]
    if not i:
        return None
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


dtab = parse_delta(band)
if dtab is None:
    for k in [k for k in V if 'v7' in k]:
        V.pop(k)
else:
    print('Delta_unsat table', band, dtab, flush=True)
tree = C.Q + '/tree_main2'
for det in dets:
    grid = None
    for e in exps:
        stem = f'jw03523005001_{vgroup}_{e:05d}_{det}_align_o005_crf'
        fn = f'{tree}/F{band}/pipeline/{stem}.fits'
        outfn = f'{C.Q}/satrefit/out/{band}_{stem}_satrefit.fits'
        if os.path.exists(outfn):
            print('exists', outfn, flush=True)
            continue
        if grid is None:
            hdr = C.fits.getheader(fn)
            grid, gf = C.load_grid(tree + '/psfs', hdr, lw)
            C.log('grid', gf)
        t = C.run_frame(fn, band, V, grid, dtab=dtab, rmax_arcsec=RM)
        t.write(outfn, overwrite=True)
        C.log('wrote', outfn)
