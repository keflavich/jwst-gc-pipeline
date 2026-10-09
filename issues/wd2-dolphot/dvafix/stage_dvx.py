"""Stage trees for the DVA-consistent m7 seed A/B on the main2gt configuration (#1128).

The earlier attempt (forced_refit/dva_ab) ran m7 with --filternames=F150W, and m7 exists
only for a multi-filter run, so it stopped at the phase check.  This version stages
F150W plus F323N (8 LW frames, cheap) so the run is multi-filter; only F150W is scored.

Trees follow stage_trees.py (main2 / main2gt): symlinks to the production inputs, never
anything m7/m8 writes, and regular copies of the m6 smoothed backgrounds from tree_integbg.

Arms:
  dvx0  control: production m6 vetted catalogs for all 16 bands (= main2gt seed inputs)
  dvx1  the 15 non-F150W m6 vetted catalogs replaced by copies whose 'skycoord' carries the
        DVA shift the F150W crf frames received: per source, the mean (DVASHRA, DVASHDE)
        of the F150W frames containing it (mean of all 32 outside the F150W footprint).
        Copies are stamped WCSGDVA=True.  F150W itself is untouched.

usage: python stage_dvx.py dvx0 dvx1
"""
import json
import os
import re
import shutil
import sys

import glob
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

R = '/orange/adamginsburg/jwst/wd2'
Q = f'{R}/dolphot_benchmark/Q_integ'
H = f'{Q}/dvafix'
ARMS = sys.argv[1:]
BANDS = ['F150W', 'F323N']
ALL16 = ['F115W', 'F150W', 'F162M', 'F164N', 'F182M', 'F187N', 'F200W', 'F212N',
         'F250M', 'F277W', 'F300M', 'F323N', 'F335M', 'F405N', 'F410M', 'F466N']
DENY = re.compile(r'(_m7(?![0-9])|m7_|_m8|m8_|wingcal|consolidated_satstar|crossband_seed|_perframe_markers)')
COPY = re.compile(r'(_mergedcat_grid_.*\.asdf$|^gaia.*refcat.*\.fits$)')
SKIP_SUFFIX = ('_asn.json',)
VET = '{f}_merged_indivexp_merged_resbgsub_m6_dao_basic_vetted.fits'
BG = 'jw03523-o005_t001_nircam_clear-{f}-merged_resbgsub_m6_daophot_basic_mergedcat_residual_smoothed_bg_i2d.fits'
manifest = {}


def place(src, dstdir):
    name = os.path.basename(src)
    dst = os.path.join(dstdir, name)
    if os.path.lexists(dst):
        os.remove(dst)
    if COPY.search(name):
        shutil.copy2(src, dst)
        return
    os.symlink(src, dst)
    real = os.path.realpath(src)
    st = os.stat(real)
    manifest[real] = [st.st_size, st.st_mtime_ns]


def stage_dir(srcdir, dstdir):
    os.makedirs(dstdir, exist_ok=True)
    n = 0
    for name in sorted(os.listdir(srcdir)):
        src = os.path.join(srcdir, name)
        if not os.path.isfile(src) or DENY.search(name) or name.endswith(SKIP_SUFFIX):
            continue
        place(src, dstdir)
        n += 1
    return n


def f150w_frames():
    out = []
    for p in sorted(glob.glob(f'{R}/F150W/pipeline/jw03523005*_align_o005_crf.fits')):
        h = fits.getheader(p, 'SCI')
        assert h.get('DVACORR') is True, p
        assert np.isclose(h['CRVAL1'] - h['OLCRVAL1'], h['DVASHRA'], atol=1e-9), p
        assert np.isclose(h['CRVAL2'] - h['OLCRVAL2'], h['DVASHDE'], atol=1e-9), p
        out.append((os.path.basename(p), WCS(h), (h['NAXIS2'], h['NAXIS1']),
                    float(h['DVASHRA']), float(h['DVASHDE'])))
    assert len(out) == 32, len(out)
    return out


def dva_shift(sc, frames):
    """Per-source mean (dRA, dDec) coordinate shift [deg] of the containing frames."""
    sra = np.zeros(len(sc))
    sde = np.zeros(len(sc))
    n = np.zeros(len(sc), int)
    for _, w, (ny, nx), dra, dde in frames:
        x, y = w.world_to_pixel(sc)
        ins = (x >= -0.5) & (x < nx - 0.5) & (y >= -0.5) & (y < ny - 0.5)
        sra[ins] += dra
        sde[ins] += dde
        n[ins] += 1
    mra = np.mean([f[3] for f in frames])
    mde = np.mean([f[4] for f in frames])
    ok = n > 0
    dra = np.where(ok, sra / np.maximum(n, 1), mra)
    dde = np.where(ok, sde / np.maximum(n, 1), mde)
    return dra, dde, n


def shifted_catalog(src, dst, frames):
    t = Table.read(src)
    sc = t['skycoord'] if isinstance(t['skycoord'], SkyCoord) else SkyCoord(t['skycoord'])
    dra, dde, n = dva_shift(sc, frames)
    t['skycoord'] = SkyCoord(sc.ra.deg + dra, sc.dec.deg + dde, unit='deg', frame=sc.frame.name)
    assert t.meta.get('WCSGDVA') in (None, False), (src, t.meta.get('WCSGDVA'))
    t.meta['WCSGDVA'] = True
    t.meta['DVASEED'] = 'shifted into the F150W DVA frame by dvafix/stage_dvx.py (#1128)'
    t.write(dst, overwrite=False)
    cosd = np.cos(np.deg2rad(sc.dec.deg))
    return (len(t), int((n == 0).sum()),
            float(np.median(dra * cosd) * 3.6e6), float(np.median(dde) * 3.6e6))


frames = f150w_frames()
for arm in ARMS:
    assert arm in ('dvx0', 'dvx1'), arm
    T = f'{H}/tree_{arm}'
    for b in BANDS:
        print(arm, b, stage_dir(f'{R}/{b}/pipeline', f'{T}/{b}/pipeline'), flush=True)
    for d in ('catalogs', 'psfs', 'regions_'):
        print(arm, d, stage_dir(f'{R}/{d}', f'{T}/{d}'), flush=True)
    os.makedirs(f'{T}/offsets', exist_ok=True)
    for f in os.listdir(f'{R}/offsets'):
        shutil.copy2(f'{R}/offsets/{f}', f'{T}/offsets/{f}')
    for b in BANDS:
        name = BG.format(f=b.lower())
        src = f'{Q}/tree_integbg/{b}/pipeline/{name}'
        assert os.path.isfile(src) and not os.path.islink(src), src
        dst = f'{T}/{b}/pipeline/{name}'
        if os.path.islink(dst):
            os.remove(dst)
        assert not os.path.exists(dst), dst
        shutil.copy2(src, dst)
        print(arm, 'copied m6 smoothed bg', name, flush=True)
    if arm != 'dvx1':
        continue
    for b in ALL16:
        if b == 'F150W':
            continue
        name = VET.format(f=b.lower())
        src = f'{R}/catalogs/{name}'
        dst = f'{T}/catalogs/{name}'
        assert os.path.islink(dst) and os.path.realpath(dst) == os.path.realpath(src), dst
        os.remove(dst)
        st = shifted_catalog(src, dst, frames)
        print(arm, 'shifted', name, 'rows %d outside-F150W %d  dRA* med %.2f mas  dDec med %.2f mas' % st,
              flush=True)
json.dump(manifest, open(f'{H}/manifest_{"_".join(ARMS)}.json', 'w'))
print(len(manifest), 'symlink targets in manifest')
