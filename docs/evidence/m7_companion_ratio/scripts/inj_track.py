"""Injected stars across the phases of the reference-field runs.

For every injected star inside the evaluated inner box (seeds 1-10, both
bands): the nearest vetted source of each phase (m2..m7, within 1 px), the
brightest m6 vetted source within ``comp`` FWHM that is brighter than the
star's m6 match, and the PSF-matched S/N left at the injected position in the
m6 and m7 residuals.

usage: python inj_track.py <variant> <out.fits>
"""
import glob
import os
import re
import sys

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jwst_gc_pipeline.photometry import reference_fields as RF  # noqa: E402
from phase_loss import PHASES, FWHM_PIX, Mosaic, phase_files, read_cat, stamp_metrics, _in_box  # noqa: E402

variant, outpath = sys.argv[1], sys.argv[2]
COMP = 2.5
rows = []
_, fields = RF.load_config()
for name, spec in fields.items():
    half = spec['size_arcsec'] / 2 - spec['inner_margin_arcsec']
    for seed in spec['seeds']:
        d = RF.run_dir(spec, variant, seed)
        inj = Table.read(RF.injection_table_path(name, seed))
        isc = SkyCoord(np.asarray(inj['ra']) * u.deg, np.asarray(inj['dec']) * u.deg)
        keep = _in_box(isc, (spec['ra'], spec['dec'], half))
        for filt in spec['filters']:
            f = filt.lower()
            hits = glob.glob(f'{d}/catalogs/{f}_merged*_indivexp_merged_resbgsub_m7_dao_basic_vetted.fits')
            if len(hits) != 1:
                print('MISSING', d, filt, flush=True)
                continue
            obstok = re.match(rf'{f}_merged(.*)_indivexp', os.path.basename(hits[0])).group(1)
            files = phase_files(f'{d}/catalogs', f'{d}/{filt}/pipeline', filt, 'merged', obstok)
            data = Mosaic(files['data'])
            ix, iy = data.xy(isc[keep])
            t = Table(dict(field=[name] * len(ix), variant=[variant] * len(ix), seed=[seed] * len(ix),
                           filt=[filt] * len(ix), id=np.asarray(inj['id'])[keep], x=ix, y=iy))
            if f'snr_true_{filt}' in inj.colnames:
                t['snr_true'] = np.asarray(inj[f'snr_true_{filt}'])[keep]
            else:
                t['snr_true'] = np.nan
            fw = FWHM_PIX[filt]
            for ph in PHASES:
                v = read_cat(files[ph]['vetted'], data)
                dd, ii = cKDTree(np.c_[v['_x'], v['_y']]).query(np.c_[ix, iy])
                t[f'found_{ph}'] = dd <= 1.0
                t[f'flux_{ph}'] = np.where(dd <= 1.0, np.asarray(v['flux'])[ii], np.nan)
                if ph == 'resbgsub_m6':
                    # brightest brighter m6 vetted neighbour within COMP FWHM
                    tree = cKDTree(np.c_[v['_x'], v['_y']])
                    fl = np.asarray(v['flux'], float)
                    nb_sep = np.full(len(ix), np.nan)
                    nb_ratio = np.full(len(ix), np.nan)
                    for k in range(len(ix)):
                        own = fl[ii[k]] if dd[k] <= 1.0 else 0.0
                        js = [j for j in tree.query_ball_point([ix[k], iy[k]], COMP * fw)
                              if (j != ii[k] or dd[k] > 1.0) and fl[j] > own]
                        if js:
                            nb_sep[k] = np.min(np.hypot(v['_x'][js] - ix[k], v['_y'][js] - iy[k])) / fw
                            # the cut's test: own flux vs the brightest such neighbour
                            nb_ratio[k] = own / np.max(fl[js]) if own > 0 else np.nan
                    t['brighter_nb_fwhm'] = nb_sep
                    t['brighter_nb_ratio'] = nb_ratio
            sd = cKDTree(np.c_[read_cat(files['m7_seed'], data)['_x'], read_cat(files['m7_seed'], data)['_y']])
            t['m7_seed_dist'] = sd.query(np.c_[ix, iy])[0]
            for ph, pre in (('resbgsub_m6', 'r6'), ('resbgsub_m7', 'r7')):
                m = Mosaic(files[ph]['resid'])
                met = stamp_metrics(m, ix, iy, fw)
                m.close()
                t[f'{pre}_snr'] = met['snr']
                t[f'{pre}_flux'] = met['flux']
            met = stamp_metrics(data, ix, iy, fw)
            t['data_flux'] = met['flux']
            data.close()
            rows.append(t)
            print(name, seed, filt, len(t), flush=True)
T = vstack(rows)
T.write(outpath, overwrite=True)
