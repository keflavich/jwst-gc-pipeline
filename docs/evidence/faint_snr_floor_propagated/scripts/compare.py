"""Compare per-branch m6 vetting replays (vet_variant.py) with the #1015 base.

For each branch: sources the branch keeps that the base drops ('added') and
the reverse ('lost'), split by distance to the nearest saturated star, with
  * realness: match fraction against a reference catalog within 60 mas,
    chance from shifted positions (evid1015/realness.SHIFTS_AS), and the
    flux-matched expectation from the base-kept sources in the same flux
    bins.  rel = (m - ch) / (m_exp - ch_exp): 1 = as real as base-kept
    sources of the same flux, 0 = chance.
      Brick F182M: Brick 1182/o004 F200W m7 vetted (independent visit).
      Sgr B2 F187N: Sgr B2 F182M m6 vetted (same visit, other filter; PSF
      artifacts of bright stars appear in both, so this overstates realness).
  * peak: the brightest pixel of the 7x7 data_i2d box lies within 1 px.
  * star-like reason the base vetting would assign (first that applies).

usage: python compare.py [field ...]
"""
import glob
import json
import os
import sys
import warnings

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy import wcs

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'evid1015'))
from realness import match_fraction, in_footprint  # noqa: E402

warnings.simplefilter('ignore', wcs.FITSFixedWarning)
R = '/orange/adamginsburg/jwst'
HERE = os.path.dirname(os.path.abspath(__file__))
FIELDS = {
    'brick': dict(band='f182m',
                  ref=f'{R}/brick/catalogs/f200w_merged_o004_indivexp_merged_resbgsub_m7_dao_basic_vetted.fits',
                  ref_label='F200W 1182/o004 m7 vetted (independent visit)'),
    'sgrb2': dict(band='f187n',
                  ref=f'{R}/sgrb2/catalogs/f182m_merged_indivexp_merged_resbgsub_m6_dao_basic_vetted.fits',
                  ref_label='F182M m6 vetted (same visit)'),
}
VARS = {'snr': '#1016', 'qsnr': '#1017', 'prom': '#1018', 'lsky': '#1019'}
SAT_BINS = ((0, 1), (1, 2), (2, np.inf))     # arcsec to the nearest saturated star
FLUX_EDGES = np.arange(0, 7.01, 0.25)        # log10 flux


def sky(t):
    sc = t['skycoord']
    return sc if isinstance(sc, SkyCoord) else SkyCoord(sc)


def local_peak(path, sc):
    with fits.open(path) as h:
        d = h['SCI'].data.astype(float)
        w = wcs.WCS(h['SCI'].header)
    x, y = w.world_to_pixel(sc)
    out = np.zeros(len(sc), bool)
    for j, (xi, yi) in enumerate(zip(x, y)):
        if not (np.isfinite(xi) and np.isfinite(yi)):
            continue
        ix, iy = int(round(xi)), int(round(yi))
        if 3 <= ix < d.shape[1] - 3 and 3 <= iy < d.shape[0] - 3:
            box = d[iy - 3:iy + 4, ix - 3:ix + 4]
            if np.isfinite(box[2:5, 2:5]).any():
                out[j] = np.nanmax(box[2:5, 2:5]) >= np.nanmax(box)
    return out


def realness(sc, flux, ref, base_sc, base_flux):
    """match, chance, flux-matched expected match and chance, rel."""
    if len(sc) == 0:
        return dict(n=0)
    m, ch = match_fraction(sc, ref)
    lf, lb = np.log10(np.clip(flux, 1e-30, None)), np.log10(np.clip(base_flux, 1e-30, None))
    w_exp, mexp, chexp = 0, 0.0, 0.0
    for lo, hi in zip(FLUX_EDGES[:-1], FLUX_EDGES[1:]):
        k = (lf >= lo) & (lf < hi)
        if not k.any():
            continue
        kb = np.flatnonzero((lb >= lo) & (lb < hi))
        if kb.size == 0:
            continue
        if kb.size > 3000:
            kb = np.random.default_rng(0).choice(kb, 3000, replace=False)
        mb, cb = match_fraction(base_sc[kb], ref)
        w_exp += k.sum()
        mexp += k.sum() * mb
        chexp += k.sum() * cb
    mexp, chexp = (mexp / w_exp, chexp / w_exp) if w_exp else (np.nan, np.nan)
    rel = (m - ch) / (mexp - chexp) if np.isfinite(mexp) and mexp > chexp else np.nan
    return dict(n=int(len(sc)), match=float(m), chance=float(ch), match_exp=float(mexp),
                chance_exp=float(chexp), rel=float(rel))


def reason(t):
    qf = np.asarray(t['qfit'], float)
    flg = np.asarray(t['flags'], float)
    pk = np.asarray(t['peak_sb'], float) if 'peak_sb' in t.colnames else np.full(len(t), np.nan)
    lb = np.asarray(t['local_bkg'], float)
    r = np.full(len(t), 'other', dtype='U10')
    r[(lb > 0) & np.isfinite(pk) & (pk > 20 * lb)] = 'peakSB'
    r[flg == 1] = 'flags==1'
    r[qf <= 0.2] = 'qfit<=0.2'
    return r


def main(fields):
    for field in fields:
        cfg = FIELDS[field]
        band = cfg['band']
        basep = f'{HERE}/out/{field}_{band}_seed.fits'
        if not os.path.exists(basep):
            print(f'{field}: no base replay yet')
            continue
        base = Table.read(basep)
        prov_base = json.loads(base.meta['PROVJSON'])
        ref = sky(Table.read(cfg['ref']))
        allsc = sky(base)
        kb = np.asarray(base['kept'], bool)
        sat = np.asarray(base['is_saturated'], bool) if 'is_saturated' in base.colnames else np.zeros(len(base), bool)
        satsc = allsc[sat]
        idx, d2d, _ = allsc.match_to_catalog_sky(satsc, nthneighbor=1)
        dsat = d2d.arcsec
        # a saturated star's own distance is to the next saturated star
        if sat.any():
            _, d2, _ = allsc[sat].match_to_catalog_sky(satsc, nthneighbor=2)
            dsat[sat] = d2.arcsec
        infp = in_footprint(allsc, ref)
        flux = np.asarray(base['flux'], float)
        bsel = kb & infp
        res = dict(field=field, band=band, ref=cfg['ref_label'], base=prov_base,
                   n_merged=len(base), n_base_kept=int(kb.sum()), frac_in_ref_footprint=float(infp.mean()),
                   base_kept_within_1as_of_satstar=float(np.mean(dsat[kb] < 1)),
                   variants={})
        lines = [f'== {field} {band.upper()}: base (#1015 replay, {prov_base["commit"][:8]}'
                 f'{" DIRTY" if prov_base["dirty"] else ""}) keeps {kb.sum()}/{len(base)}; '
                 f'realness ref: {cfg["ref_label"]}; {infp.mean():.2f} of sources in its footprint; '
                 f'base-kept within 1" of a satstar: {np.mean(dsat[kb] < 1):.2f}']
        for v, pr in VARS.items():
            p = f'{HERE}/out/{field}_{band}_{v}.fits'
            if not os.path.exists(p):
                lines.append(f'  {pr} ({v}): not run yet')
                continue
            t = Table.read(p)
            assert np.array_equal(np.asarray(t['rowid']), np.asarray(base['rowid']))
            prov = json.loads(t.meta['PROVJSON'])
            kv = np.asarray(t['kept'], bool)
            add, lost = kv & ~kb, kb & ~kv
            pk = np.zeros(len(t), bool)
            if add.any() or lost.any():
                pk[add | lost] = local_peak(prov['data_i2d'], allsc[add | lost])
            rsn = reason(t)
            vres = dict(pr=pr, commit=prov['commit'], dirty=prov['dirty'], slurm_job_id=prov.get('slurm_job_id'),
                        n_kept=int(kv.sum()), n_added=int(add.sum()), n_lost=int(lost.sum()),
                        added_reason={r: int((rsn[add] == r).sum()) for r in ('qfit<=0.2', 'flags==1', 'peakSB', 'other')},
                        added_peak=float(pk[add].mean()) if add.any() else np.nan,
                        lost_peak=float(pk[lost].mean()) if lost.any() else np.nan,
                        sat_bins={})
            lines.append(f'  {pr} ({v}, {prov["commit"][:8]}{" DIRTY" if prov["dirty"] else ""}): '
                         f'kept {kv.sum()} = base {kb.sum():+d} {add.sum():+d} added {-lost.sum():+d} lost; '
                         f'added on a data peak {vres["added_peak"]:.2f}, lost {vres["lost_peak"]:.2f}; '
                         f'added reason {vres["added_reason"]}')
            for lo, hi in SAT_BINS:
                ks = (dsat >= lo) & (dsat < hi)
                ra = realness(allsc[add & ks & infp], flux[add & ks & infp], ref,
                              allsc[bsel & ks], flux[bsel & ks])
                rl = realness(allsc[lost & ks & infp], flux[lost & ks & infp], ref,
                              allsc[bsel & ks], flux[bsel & ks])
                vres['sat_bins'][f'{lo}-{hi:g}'] = dict(n_added=int((add & ks).sum()), n_lost=int((lost & ks).sum()),
                                                        added_peak=float(pk[add & ks].mean()) if (add & ks).any() else np.nan,
                                                        added_real=ra, lost_real=rl)

                def fmt(r):
                    if r.get('n', 0) == 0:
                        return 'n=0'
                    return (f'n={r["n"]} match {r["match"]:.2f} (chance {r["chance"]:.2f}; '
                            f'flux-matched base {r["match_exp"]:.2f}) rel {r["rel"]:.2f}')
                lines.append(f'      satstar {lo}-{hi:g}": added {(add & ks).sum():6d} '
                             f'[peak {vres["sat_bins"][f"{lo}-{hi:g}"]["added_peak"]:.2f}; in-ref-footprint {fmt(ra)}]   '
                             f'lost {(lost & ks).sum():5d} [{fmt(rl)}]')
            res['variants'][v] = vres
            np.save(f'{HERE}/out/{field}_{band}_{v}_added_rowid.npy', np.flatnonzero(add))
        np.save(f'{HERE}/out/{field}_{band}_dsat.npy', dsat)
        txt = '\n'.join(lines)
        print(txt, flush=True)
        with open(f'{HERE}/compare_{field}.txt', 'w') as fh:
            fh.write(txt + '\n')
        with open(f'{HERE}/compare_{field}.json', 'w') as fh:
            json.dump(res, fh, indent=1, default=float)


if __name__ == '__main__':
    main(sys.argv[1:] or list(FIELDS))
