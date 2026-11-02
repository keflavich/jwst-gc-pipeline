"""Fate of the i2d residual seeds admitted only by the loose roundness window
(``seed_round_loose``) in the reference-field runs of one variant.

For each star field (the loose window is off on the extended-emission field)
and each run (clean + injection seeds), the new i2d detections (``seed_origin
== 'i2d'``) of every phase's i2d seed catalog of the scored band are pooled.
A position detected at several phases counts once (friends-of-friends at one
pixel), and it is loose-only when every detection of it was loose-only.  Only
positions inside the evaluated inner box count.

- survives: the final m7 vetted catalog has a source within one pixel;
- at injected (injection runs only): an injected star lies within one pixel.
  The chance count is the number of surviving seeds times the injected-star
  density in the inner box times the one-pixel match area.

usage: python loose_survival.py <variant> <out.json>
"""
import glob
import json
import sys

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import fisher_exact

from jwst_gc_pipeline.photometry import reference_fields as RF

PIX_AS = 0.031      # NIRCam SW pixel; every scored band here is SW
FIELDS = ('superdense', 'dense_bright', 'dark')


def _within(a, b, radius_as):
    """Boolean per row of ``a``: some row of ``b`` within ``radius_as``."""
    out = np.zeros(len(a), bool)
    if len(a) and len(b):
        ia, _, _, _ = b.search_around_sky(a, radius_as * u.arcsec)
        out[np.unique(ia)] = True
    return out


def _inner(sc, spec):
    """Inside the evaluated inner box, and the box area in arcsec^2."""
    c = SkyCoord(spec['ra'] * u.deg, spec['dec'] * u.deg)
    dx, dy = c.spherical_offsets_to(sc)
    half = spec['size_arcsec'] / 2 - spec['inner_margin_arcsec']
    ins = (np.abs(dx.to_value(u.arcsec)) < half) & (np.abs(dy.to_value(u.arcsec)) < half)
    return ins, (2 * half) ** 2


def _new_seed_positions(rdir, filt):
    """Distinct new i2d seed positions of one run and their loose-only flag."""
    ra, dec, loose = [], [], []
    for f in sorted(glob.glob(f'{rdir}/catalogs/*{filt}*i2dseed.fits')):
        t = Table.read(f)
        new = np.asarray(t['seed_origin']).astype(str) == 'i2d'
        sc = SkyCoord(t['skycoord'])[new]
        ra.append(sc.ra.deg)
        dec.append(sc.dec.deg)
        lc = (np.asarray(t['seed_round_loose'], bool) if 'seed_round_loose' in t.colnames
              else np.zeros(len(t), bool))
        loose.append(lc[new])
    if not ra:
        return None, None
    sc = SkyCoord(np.concatenate(ra) * u.deg, np.concatenate(dec) * u.deg)
    loose = np.concatenate(loose)
    i1, i2, _, _ = sc.search_around_sky(sc, PIX_AS * u.arcsec)
    graph = coo_matrix((np.ones(len(i1)), (i1, i2)), shape=(len(sc), len(sc)))
    ncomp, lab = connected_components(graph, directed=False)
    first = np.array([np.flatnonzero(lab == k)[0] for k in range(ncomp)], int)
    uloose = np.array([loose[lab == k].all() for k in range(ncomp)], bool)
    return sc[first], uloose


def field_fate(spec, variant):
    filt = spec['filters'][0].lower()
    acc = {k: dict(n=0, survive=0, n_injruns=0, at_inj=0, surv_injruns=0,
                   surv_at_inj=0, chance=0.0, runs=0)
           for k in ('loose', 'tight')}
    for seed in [0] + list(spec['seeds']):
        rdir = RF.run_dir(spec, variant, seed)
        vet = glob.glob(f'{rdir}/catalogs/{filt}_*_m7_dao_basic_vetted.fits')
        if len(vet) != 1:
            print('skip', rdir, vet)
            continue
        final = SkyCoord(Table.read(vet[0])['skycoord'])
        usc, uloose = _new_seed_positions(rdir, filt)
        if usc is None:
            continue
        ins, area = _inner(usc, spec)
        surv = _within(usc, final, PIX_AS)
        inj = None
        if seed:
            t = Table.read(RF.injection_table_path(spec['name'], seed))
            inj = SkyCoord(np.asarray(t['ra'], float) * u.deg,
                           np.asarray(t['dec'], float) * u.deg)
            inj = inj[_inner(inj, spec)[0]]
            at_inj = _within(usc, inj, PIX_AS)
        for key, sel in (('loose', uloose & ins), ('tight', ~uloose & ins)):
            a = acc[key]
            a['runs'] += 1
            a['n'] += int(sel.sum())
            a['survive'] += int((surv & sel).sum())
            if inj is not None:
                a['n_injruns'] += int(sel.sum())
                a['at_inj'] += int((at_inj & sel).sum())
                a['surv_injruns'] += int((surv & sel).sum())
                a['surv_at_inj'] += int((at_inj & surv & sel).sum())
                a['chance'] += float((surv & sel).sum()) * len(inj) * np.pi * PIX_AS ** 2 / area
    lo, ti = acc['loose'], acc['tight']
    acc['p_survive'] = fisher_exact([[lo['survive'], lo['n'] - lo['survive']],
                                     [ti['survive'], ti['n'] - ti['survive']]])[1]
    acc['p_surv_at_inj'] = fisher_exact(
        [[lo['surv_at_inj'], lo['surv_injruns'] - lo['surv_at_inj']],
         [ti['surv_at_inj'], ti['surv_injruns'] - ti['surv_at_inj']]])[1]
    return acc


def main(variant, out):
    _, fields = RF.load_config()
    res = {name: field_fate(fields[name], variant) for name in FIELDS}
    print(f'variant {variant}')
    print('| field | seeds | distinct new i2d seeds | reach m7 vetted | at an injected star (injection runs, any fate) '
          '| survivors at an injected star (chance) |')
    print('|---|---|---|---|---|---|')
    for name, acc in res.items():
        for key in ('loose', 'tight'):
            a = acc[key]
            print(f"| `{name}` | {key} | {a['n']} | {a['survive']} ({a['survive'] / max(a['n'], 1):.2f}) "
                  f"| {a['at_inj']}/{a['n_injruns']} ({a['at_inj'] / max(a['n_injruns'], 1):.3f}) "
                  f"| {a['surv_at_inj']}/{a['surv_injruns']} ({a['chance']:.1f}) |")
        print(f"| | Fisher p (loose vs tight) | | {acc['p_survive']:.2g} | | {acc['p_surv_at_inj']:.2g} |")
    pool = {key: {k: sum(res[n][key][k] for n in FIELDS)
                  for k in ('n', 'survive', 'n_injruns', 'at_inj', 'surv_injruns', 'surv_at_inj')}
            for key in ('loose', 'tight')}
    lo, ti = pool['loose'], pool['tight']
    for col, num, den in (('reach m7 vetted', 'survive', 'n'),
                          ('at an injected star, any fate', 'at_inj', 'n_injruns'),
                          ('survivors at an injected star', 'surv_at_inj', 'surv_injruns')):
        p = fisher_exact([[lo[num], lo[den] - lo[num]], [ti[num], ti[den] - ti[num]]])[1]
        print(f"pooled {col}: loose {lo[num]}/{lo[den]} ({lo[num] / max(lo[den], 1):.3f}), "
              f"tight {ti[num]}/{ti[den]} ({ti[num] / max(ti[den], 1):.3f}), Fisher p {p:.2g}")
    res['pooled'] = pool
    with open(out, 'w') as fh:
        json.dump(res, fh, indent=1)


if __name__ == '__main__':
    main(*sys.argv[1:])
