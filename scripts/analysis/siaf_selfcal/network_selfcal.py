"""SIAF accuracy via network self-calibration.

Model: measured sky position of a star in exposure i, detector d =
   true + att_i + S_(d,band)   (+ intra-detector distortion residual -> pair scatter)
For every overlapping catalog pair, the matched-star median offset gives one
observation  off = (att_j - att_i) + (S_dj - S_di).
Least squares over all pairs separates per-exposure attitude (gauge: mean att = 0)
from per-detector SIAF shifts (gauge: mean S = 0 per band-group).
Invariant to any rigid per-exposure WCS manipulation (alignment etc.).

Usage: python network_selfcal.py BAND1[,BAND2,...] [--stage m7] [--out tag]
Bands may span proposals (f212n=2221, f200w=1182): detector unknowns are
(band, detector) so cross-band pairs measure filteroffset+module consistency.
"""
import sys, glob, os, re, itertools, warnings
warnings.filterwarnings('ignore')
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

B = '/blue/adamginsburg/adamginsburg/jwst/brick'
# bands: comma list of band[:stage], default stage resbgsub_m7
spec = [(b.split(':')[0], b.split(':')[1] if ':' in b else 'resbgsub_m7') for b in sys.argv[1].split(',')]
bands = [b for b, _ in spec]
args = sys.argv[2:]
tag = args[args.index('--out')+1] if '--out' in args else '_'.join(bands)

cats = {}   # (band, det, visit, exp) -> dict(ra, dec, n)
for band, stage in spec:
    for f in sorted(glob.glob(f'{B}/{band.upper()}/{band}_nrc*_visit*_vgroup*_exp*_{stage}_daophot_basic.fits')):
        m = re.search(rf'{band}_(nrc[ab](?:[0-9]|long))_visit(\d+)_vgroup\d+_exp(\d+)_', os.path.basename(f))
        if not m: continue
        det, visit, exp = m.group(1), m.group(2), m.group(3)
        t = Table.read(f)
        if 'skycoord_centroid' not in t.colnames or len(t) < 50: continue
        q = np.asarray(t['qfit'], float)
        fl = np.asarray(t['flux_fit'], float); fe = np.asarray(t['flux_err'], float)
        with np.errstate(divide='ignore', invalid='ignore'):
            snr = fl/fe
        good = np.isfinite(q) & (q < 0.15) & np.isfinite(snr) & (snr > 20)
        if good.sum() < 50: continue
        sc = SkyCoord(t['skycoord_centroid'][good])
        cats[(band, det, visit, exp)] = dict(ra=np.asarray(sc.ra.deg), dec=np.asarray(sc.dec.deg),
                                             n=int(good.sum()))
print(f'{len(cats)} catalogs loaded', flush=True)

keys = sorted(cats)
# bounding boxes for overlap pruning
bb = {}
for k in keys:
    c = cats[k]
    bb[k] = (c['ra'].min(), c['ra'].max(), c['dec'].min(), c['dec'].max())

def overlap(k1, k2, pad=2/3600):
    a, b = bb[k1], bb[k2]
    return not (a[1] < b[0]-pad or b[1] < a[0]-pad or a[3] < b[2]-pad or b[3] < a[2]-pad)

def pair_offset(c1, c2):
    """median matched-star offset (c2 - c1) in mas, via histogram rigid + tight mutual match."""
    s1 = SkyCoord(c1['ra']*u.deg, c1['dec']*u.deg); s2 = SkyCoord(c2['ra']*u.deg, c2['dec']*u.deg)
    i2, i1, sep, _ = s1.search_around_sky(s2, 1.5*u.arcsec)
    if len(i1) < 40: return None
    cosd = np.cos(np.radians(np.median(c1['dec'])))
    dra = (c2['ra'][i2]-c1['ra'][i1])*cosd*3.6e6
    dde = (c2['dec'][i2]-c1['dec'][i1])*3.6e6
    e = np.arange(-1500, 1502, 8.)
    H, xe, ye = np.histogram2d(dra, dde, bins=[e, e])
    i, j = np.unravel_index(H.argmax(), H.shape)
    px, py = (xe[i]+xe[i+1])/2, (ye[j]+ye[j+1])/2
    core = (np.abs(dra-px) < 40) & (np.abs(dde-py) < 40)
    if core.sum() < 40: return None
    # iterate median in shrinking window
    for w in (25, 12):
        mx, my = np.median(dra[core]), np.median(dde[core])
        core = (np.abs(dra-mx) < w) & (np.abs(dde-my) < w)
        if core.sum() < 30: return None
    dr, dd = dra[core], dde[core]
    n = core.sum()
    return (np.median(dr), np.median(dd),
            1.4826*np.median(np.abs(dr-np.median(dr))), 1.4826*np.median(np.abs(dd-np.median(dd))), int(n))

obs = []  # (k1, k2, dra, dde, sra, sde, n)
npairs_checked = 0
for a, b2 in itertools.combinations(range(len(keys)), 2):
    k1, k2 = keys[a], keys[b2]
    if (k1[0], k1[2], k1[3]) == (k2[0], k2[2], k2[3]):  # same band+visit+exposure, different detector: no overlap
        continue
    if not overlap(k1, k2): continue
    npairs_checked += 1
    r = pair_offset(cats[k1], cats[k2])
    if r is None: continue
    obs.append((k1, k2) + r)
print(f'{npairs_checked} overlapping pairs checked, {len(obs)} usable', flush=True)

# unknowns: attitude per (band? no -- per EXPOSURE = (visit,exp) is shared across detectors
# BUT across bands, exposures are distinct; attitude key = (band, visit, exp).
attkeys = sorted({(k[0], k[2], k[3]) for k in keys})
detkeys = sorted({(k[0], k[1]) for k in keys})
ai = {k: i for i, k in enumerate(attkeys)}
di = {k: i for i, k in enumerate(detkeys)}
na, nd = len(attkeys), len(detkeys)
rows = []
vals_ra, vals_de, wts = [], [], []
for (k1, k2, dra, dde, sra, sde, n) in obs:
    row = np.zeros(na + nd)
    row[ai[(k2[0], k2[2], k2[3])]] += 1; row[ai[(k1[0], k1[2], k1[3])]] -= 1
    row[na + di[(k2[0], k2[1])]] += 1;   row[na + di[(k1[0], k1[1])]] -= 1
    rows.append(row)
    vals_ra.append(dra); vals_de.append(dde)
    wts.append(np.sqrt(n) / max(np.hypot(sra, sde), 1.0))
# gauge: mean attitude = 0, mean detector shift = 0 (per band group for dets)
gauge_rows = []
g = np.zeros(na+nd); g[:na] = 1; gauge_rows.append(g)
for band in bands:
    g = np.zeros(na+nd)
    for k, idx in di.items():
        if k[0] == band: g[na+idx] = 1
    gauge_rows.append(g)
A = np.vstack(rows + gauge_rows)
W = np.array(wts + [100.0]*len(gauge_rows))
sol = {}
for lab, vals in (('ra', vals_ra), ('dec', vals_de)):
    y = np.array(vals + [0.0]*len(gauge_rows))
    x, *_ = np.linalg.lstsq(A * W[:, None], y * W, rcond=None)
    sol[lab] = x
    pred = A[:len(obs)] @ x
    res = np.array(vals) - pred
    print(f'[{lab}] pair-residual rms={res.std():.2f} mas, MAD={1.4826*np.median(np.abs(res-np.median(res))):.2f} mas', flush=True)

print('\n=== per-detector SIAF shifts (mas; gauge: per-band mean = 0) ===')
for k, idx in sorted(di.items()):
    print(f'  {k[0]:6s} {k[1]:6s}: dRA={sol["ra"][na+idx]:+7.2f}  dDec={sol["dec"][na+idx]:+7.2f}')
att_ra = sol['ra'][:na]; att_de = sol['dec'][:na]
print(f'\nattitude spread: RA rms {att_ra.std():.1f} mas, Dec rms {att_de.std():.1f} mas over {na} exposures')

# save residual detail per pair for flex analysis
out = Table(rows=[(str(k1), str(k2), dra, dde, sra, sde, n) for (k1,k2,dra,dde,sra,sde,n) in obs],
            names=['k1','k2','dra','dde','sra','sde','n'])
out.write(f'{os.path.dirname(os.path.abspath(__file__))}/pairs_{tag}.ecsv', overwrite=True)
np.savez(f'{os.path.dirname(os.path.abspath(__file__))}/solution_{tag}.npz',
         attkeys=np.array([str(k) for k in attkeys]), detkeys=np.array([str(k) for k in detkeys]),
         att_ra=att_ra, att_de=att_de,
         det_ra=sol['ra'][na:], det_de=sol['dec'][na:])
print('NETSOLVE_DONE', flush=True)
