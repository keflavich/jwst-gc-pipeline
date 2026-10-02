"""Per-bin purity of the #1021 prominence-floor exemption region (W51 m6).

Exempt region: qfit <= 0.2, prominence 2-3 (below the ext-emission floor 3,
above the exempt floor 2).  Label: a continuum counterpart within 60 mas
(F187N vs F210M, F480M vs F410M; Pa-alpha / Br-alpha+CO knots have none).
Purity = (m - ch) / (cmax - ch): m = match fraction, ch = chance from +-1.5"
shifts of the same positions, cmax = match fraction of unambiguous stars
(S/N_prop > 30, prominence > 10).  68% interval: Jeffreys interval on the
match fraction (chance and cmax held fixed).
Third column: the exempt region restricted to sources on a local data_i2d
peak (brightest pixel of the 7x7 box within 1 px), the rule the branch uses.
"""
import json
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.stats import beta
import glob
import os
import warnings
from astropy.io import fits
from astropy import wcs

warnings.simplefilter('ignore', wcs.FITSFixedWarning)


def local_peak(t, fld, band):
    """Brightest pixel of the 7x7 data_i2d box within 1 px of each position."""
    # the data i2d the prominence was measured on (newer build_pl.py tables
    # record it; the two earliest W51 tables use the o001 merged mosaic)
    path = t.meta.get('DATAI2D') or sorted(glob.glob(
        f'{R}/{fld}/{band.upper()}/pipeline/jw*-o00?_t001_nircam_clear-{band}-merged_data_i2d.fits'))[0]
    with fits.open(path) as h:
        d = h['SCI'].data.astype(float)
        w = wcs.WCS(h['SCI'].header)
    x, y = w.world_to_pixel(SkyCoord(t['skycoord']))
    out = np.zeros(len(t), bool)
    for j, (xi, yi) in enumerate(zip(x, y)):
        if not (np.isfinite(xi) and np.isfinite(yi)):
            continue
        ix, iy = int(round(xi)), int(round(yi))
        if 3 <= ix < d.shape[1] - 3 and 3 <= iy < d.shape[0] - 3:
            box = d[iy - 3:iy + 4, ix - 3:ix + 4]
            if np.isfinite(box[2:5, 2:5]).any():
                out[j] = np.nanmax(box[2:5, 2:5]) >= np.nanmax(box)
    return out

R = '/orange/adamginsburg/jwst'
BINS = ((20, 25), (25, 30), (30, 40), (40, 60), (60, np.inf))
out = {}
SETS = (('w51', 'f187n', 'f210m', 'pl_w51.fits'), ('w51', 'f480m', 'f410m', 'pl_w51_f480m.fits'),
        ('sgrb2', 'f187n', 'f182m', 'pl_sgrb2.fits'), ('sgrb2', 'f480m', 'f410m', 'pl_sgrb2_f480m.fits'),
        ('sickle', 'f187n', 'f210m', 'pl_sickle_f187n.fits'), ('ngc6334', 'f187n', 'f182m', 'pl_ngc6334_f187n.fits'),
        ('wd2', 'f187n', 'f182m', 'pl_wd2_f187n.fits'), ('wd2', 'f405n', 'f410m', 'pl_wd2_f405n.fits'))
for fld, band, cont, tab in SETS:
    if not os.path.exists(tab):
        print('missing', tab)
        continue
    t = Table.read(tab)
    cstem = t.meta.get('CSTEM', f'{cont}_merged')
    c = Table.read(f'{R}/{fld}/catalogs/{cstem}_indivexp_merged_resbgsub_m6_dao_basic.fits')
    cs = np.asarray(c['flux'] / c['flux_err'], float)
    cc = SkyCoord(c['skycoord'][np.isfinite(cs) & (cs >= 3)])
    sc = SkyCoord(t['skycoord'])
    # inside the continuum catalog's footprint: a continuum source within 1"
    infp = sc.match_to_catalog_sky(SkyCoord(c['skycoord']))[1].arcsec < 1.0
    ch = np.mean([sc.spherical_offsets_by(dx * u.arcsec, dy * u.arcsec).match_to_catalog_sky(cc)[1].to(u.mas).value < 60
                  for dx, dy in [(1.5, 0), (-1.5, 0), (0, 1.5), (0, -1.5)]], axis=0)
    snrp = np.asarray(t['flux'] / t['flux_err'], float) * np.sqrt(np.clip(np.asarray(t['nmatch'], float), 1, None))
    qf = np.asarray(t['qfit'], float)
    pr = np.asarray(t['prominence'], float)
    m = np.asarray(t['cont_match'], bool)
    reg = (qf <= 0.2) & (pr >= 2) & (pr < 3)
    pk = np.zeros(len(t), bool)
    pk[reg] = local_peak(t[reg], fld, band)
    sel_c = (snrp > 30) & (pr > 10) & infp
    cmax = m[sel_c].mean()

    def pur(s):
        idx = np.flatnonzero(s)
        if idx.size == 0:
            return dict(n=0)
        k, n = int(m[idx].sum()), int(idx.size)
        chs = float(ch[idx].mean())
        p = (k / n - chs) / (cmax - chs)
        lo, hi = beta.ppf([0.16, 0.84], k + 0.5, n - k + 0.5)
        return dict(n=n, k=k, match=k / n, chance=chs, purity=float(p),
                    p16=float((lo - chs) / (cmax - chs)), p84=float((hi - chs) / (cmax - chs)))

    rows = {}
    for lo, hi in BINS:
        sb = (snrp >= lo) & (snrp < hi) & (qf <= 0.2) & infp
        rows[f'{lo}-{hi:g}'] = dict(exempt=pur(sb & (pr >= 2) & (pr < 3)),
                                    exempt_peak=pur(sb & (pr >= 2) & (pr < 3) & pk),
                                    floor_kept=pur(sb & (pr >= 3) & (pr < 5)))
    out[f'{fld}_{band}'] = dict(cont=cont, cmax=float(cmax), n_cmax=int(sel_c.sum()),
                                frac_in_footprint=float(infp.mean()), bins=rows)
    print(f'== {fld} {band} vs {cont}: cmax {cmax:.2f} (n={sel_c.sum()}); {infp.mean():.2f} of sources in the continuum footprint')
    print(f'  {"S/N_prop":>9}  {"exempt (prom 2-3)":>28}  {"exempt + local peak":>28}  {"floor-kept (prom 3-5)":>28}')
    for k, v in rows.items():
        def fmt(r):
            if r['n'] == 0:
                return f'{"--":>28}'
            return f'{r["purity"]:5.2f} [{r["p16"]:4.2f},{r["p84"]:4.2f}] {r["k"]:3d}/{r["n"]:<4d}'.rjust(28)
        print(f'  {k:>9}  {fmt(v["exempt"])}  {fmt(v["exempt_peak"])}  {fmt(v["floor_kept"])}')
with open('evid1021/exempt_bins.json', 'w') as fh:
    json.dump(out, fh, indent=1)
