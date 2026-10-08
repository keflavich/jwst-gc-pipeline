"""For each star and nrcblong frame: is the star inside a DQ SATURATED component (3x3), how big is the component, and does it
hold an accepted satstar centroid in pk2 / pk3?  Tests the proposed explanation that _handoff_restore_pixels skips components
holding an accepted centre."""
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from scipy import ndimage as ndi
import astropy.units as u
O = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/pk_deblend_trace'
G = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/f277w_gap/dbl'
ps = Table.read(f'{O}/per_star_ext.ecsv')
N = len(ps)
star = SkyCoord(ps['ra'] * u.deg, ps['dec'] * u.deg)
lost = np.asarray(ps['lost'], bool)
kept = np.asarray(ps['pk2_mhit'], bool) & np.asarray(ps['pk3_mhit'], bool)
res = []
for e in (1, 2, 3, 4):
    stem = f'jw03523005001_10101_0000{e}_nrcblong_align_o005_crf'
    with fits.open(f'{G}/tree_dbl3/F277W/pipeline/{stem}.fits') as h:
        w = WCS(h['SCI'].header)
        dq = h['DQ'].data
    sat = (dq & 2) != 0
    lab, n = ndi.label(sat)
    area = np.bincount(lab.ravel(), minlength=n + 1)
    x, y = w.world_to_pixel(star)
    ok = (x > 1) & (x < 2046) & (y > 1) & (y < 2046)
    xi = np.round(x).astype(int); yi = np.round(y).astype(int)
    slab = np.zeros(N, int)
    for i in np.where(ok)[0]:
        slab[i] = lab[yi[i] - 1:yi[i] + 2, xi[i] - 1:xi[i] + 2].max()
    out = {'slab': slab, 'area': area[slab], 'ok': ok}
    for arm, tree in (('pk2', 'tree_dbl2'), ('pk3', 'tree_dbl3')):
        c = Table.read(f'{G}/{tree}/F277W/pipeline/{stem}_resbgsub_m7_satstar_catalog.fits')
        cx = np.round(np.asarray(c['xcentroid'], float)).astype(int); cy = np.round(np.asarray(c['ycentroid'], float)).astype(int)
        good = (cx >= 0) & (cx < 2048) & (cy >= 0) & (cy < 2048)
        cl = np.zeros(len(c), int)
        cl[good] = lab[cy[good], cx[good]]
        nacc = np.bincount(cl[cl > 0], minlength=n + 1)
        out[arm] = nacc[slab]
    res.append(out)
print('per-frame: star inside SATURATED component (3x3), component area >= 50, number of accepted satstars in that component')
for name, m in (('lost (pk2 hit, pk3 none)', lost), ('kept (both hit)', kept)):
    print('--', name, m.sum())
    for e, o in zip((1, 2, 3, 4), res):
        s = m & o['ok']
        ins = s & (o['slab'] > 0)
        big = ins & (o['area'] >= 50)
        print(f' exp{e}: covered {s.sum()}, in sat comp {ins.sum()}, comp>=50px {big.sum()}, '
              f'median area {np.median(o["area"][big]) if big.any() else 0:.0f}; '
              f'comp holds >=1 accepted: pk2 {np.sum(big & (o["pk2"] > 0))}, pk3 {np.sum(big & (o["pk3"] > 0))}; '
              f'median n accepted in comp: pk2 {np.median(o["pk2"][big]) if big.any() else 0:.0f} pk3 {np.median(o["pk3"][big]) if big.any() else 0:.0f}')
