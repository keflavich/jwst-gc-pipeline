"""Does an independent visit confirm the own-band sources a far-companion exemption would add?

Brick 2221/o001 F182M m6 vetted catalog against the m6 vetted catalog of 1182/o004 F200W (another visit of the same field).  The companion rule of
``_build_m7_band_seed`` is applied inside the F182M catalog itself: a row is *dominated* when another row within 2.5 FWHM has more than 10x its flux
(the builder's pool also holds cross-band positions; here it is the own-band catalog alone).  Groups, all inside the o004 footprint:
  control            rows that are not dominated (the sources the cut keeps);
  cut_all            dominated rows;
  cut_blocked        dominated rows with a dominating row closer than KEEP FWHM or saturated (the exemption never returns these);
  cut_far_unsat_anysnr   dominated rows whose every dominating row is at least KEEP FWHM away and unsaturated, at any S/N;
  exempt_snr<s>      those rows with flux / flux_err >= s (the rows the exemption returns at --manual-m7-seed-own-band-companion-keep-snr s);
  far_unsat_below_snr50  far, unsaturated rows with S/N < 50 (left cut at 50).
purity = (matched - chance) / (1 - chance): matched = fraction of the group with an o004 source within 60 mas, chance = the same fraction for the group's
positions shifted by a few arcsec.  control_purity = the purity of control rows inside the group's 10th-90th percentile flux range.  pa_m2 and pa_m6 are
|<exp(i m PA)>| of the position angle of the row around its brightest dominating row (m = 2, 6; PSF-feature fits sit at fixed angles; isotropic
positions give about pa_iso = sqrt(pi / 4n)).

usage: python confirm_keep.py OUT.json [KEEP_FWHM=1.75] [--cat /orange/adamginsburg/jwst/brick/catalogs]
"""
import json
import os
import sys

sys.dont_write_bytecode = True
args = sys.argv[1:]
CAT = '/orange/adamginsburg/jwst/brick/catalogs'
if '--cat' in args:
    i = args.index('--cat')
    CAT = args[i + 1]
    del args[i:i + 2]
OUT = args[0]
KEEP = float(args[1]) if len(args) > 1 else 1.75
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'faint_m7_seed_union', 'scripts'))

import astropy.units as u  # noqa: E402
import numpy as np  # noqa: E402
from astropy.coordinates import SkyCoord, search_around_sky  # noqa: E402
from astropy.table import Table  # noqa: E402
from realness import in_footprint, match_fraction  # noqa: E402

FWHM = 1.99 * 0.0309           # F182M, arcsec (as m7_companion_ratio/scripts/companion_conf.py)
t = Table.read(f'{CAT}/f182m_merged_o001_indivexp_merged_resbgsub_m6_dao_basic_vetted.fits')
ref = Table.read(f'{CAT}/f200w_merged_o004_indivexp_merged_resbgsub_m6_dao_basic_vetted.fits')['skycoord']
ref = ref if isinstance(ref, SkyCoord) else SkyCoord(ref)
flux = np.asarray(t['flux'], float)
ok = np.isfinite(flux) & (flux > 0)
t = t[ok]
flux = flux[ok]
sc = t['skycoord'] if isinstance(t['skycoord'], SkyCoord) else SkyCoord(t['skycoord'])
ferr = np.asarray(t['flux_err'], float)
snr = np.where(np.isfinite(ferr) & (ferr > 0), flux / np.where(ferr > 0, ferr, 1), 0.0)


def flag(c):
    a = np.asarray(np.ma.filled(np.ma.asarray(c), False))
    return (np.char.lower(a.astype(str)) == 'true') if a.dtype.kind in 'US' else a.astype(bool)


sat = np.zeros(len(t), bool)
for c in ('is_saturated', 'replaced_saturated'):
    if c in t.colnames:
        sat |= flag(t[c])
print('rows', len(t), 'saturated', int(sat.sum()), flush=True)
ia, ib, sep, _ = search_around_sky(sc, sc, 2.5 * FWHM * u.arcsec)
sep = sep.to_value(u.arcsec)
dom_pair = (ia != ib) & (flux[ia] < 0.1 * flux[ib])
ia, ib, sep = ia[dom_pair], ib[dom_pair], sep[dom_pair]
dom = np.zeros(len(t), bool)
dom[ia] = True
block = np.zeros(len(t), bool)
bp = (sep < KEEP * FWHM) | sat[ib]
block[ia[bp]] = True
fp = in_footprint(sc, ref)
print('dominated', int(dom.sum()), 'in footprint', int((dom & fp).sum()), 'blocked', int((dom & block).sum()), flush=True)
control = ~dom & fp
# position angle of a dominated row around its brightest dominating row (radians)
order = np.lexsort((-flux[ib], ia))
ia_s, ib_s = ia[order], ib[order]
first = np.r_[True, ia_s[1:] != ia_s[:-1]]
nbr = np.full(len(t), -1)
nbr[ia_s[first]] = ib_s[first]
pa = np.full(len(t), np.nan)
dd = np.flatnonzero(nbr >= 0)
pa[dd] = sc[nbr[dd]].position_angle(sc[dd]).to_value(u.rad)


def harm(g, m):
    p = pa[g]
    p = p[np.isfinite(p)]
    return float(np.abs(np.mean(np.exp(1j * m * p)))) if len(p) else float('nan')


def purity(g):
    f, c = match_fraction(sc[g], ref)
    return float(f), float(c), float((f - c) / (1 - c)) if c < 1 else float('nan')


rows = []


def add(name, g):
    n = int(g.sum())
    if n < 5:
        rows.append(dict(group=name, n=n))
        print(f'{name:22s} n={n}', flush=True)
        return
    f, c, p = purity(g)
    lo, hi = np.percentile(flux[g], [10, 90])
    cs = control & (flux >= lo) & (flux <= hi)
    fc, cc, pc = purity(cs) if cs.sum() >= 20 else (np.nan, np.nan, np.nan)
    rows.append(dict(group=name, n=n, matched=f, chance=c, purity=p, n_control=int(cs.sum()), control_purity=pc,
                     flux_p10=float(lo), flux_p90=float(hi), snr_median=float(np.median(snr[g])),
                     pa_m2=harm(g, 2), pa_m6=harm(g, 6), pa_iso=float(np.sqrt(np.pi / (4 * n)))))
    print(f"{name:22s} n={n:6d} matched {f:.3f} chance {c:.3f} purity {p:.3f} | control same flux n={int(cs.sum())} purity {pc:.3f} "
          f"| snr med {np.median(snr[g]):.0f} | PA m2 {harm(g, 2):.3f} m6 {harm(g, 6):.3f} (isotropic {np.sqrt(np.pi / (4 * n)):.3f})", flush=True)


add('control', control)
add('cut_all', dom & fp)
add('cut_blocked', dom & block & fp)
add('cut_far_unsat_anysnr', dom & ~block & fp)
for s in (20, 30, 40, 50, 60, 80, 100):
    add(f'exempt_snr{s}', dom & ~block & (snr >= s) & fp)
add('far_unsat_below_snr50', dom & ~block & (snr < 50) & fp)
json.dump(dict(keep_fwhm=KEEP, fwhm_arcsec=FWHM, rows=rows), open(OUT, 'w'), indent=1)
