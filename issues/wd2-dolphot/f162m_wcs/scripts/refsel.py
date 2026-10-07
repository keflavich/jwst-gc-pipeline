"""Distortion / filteroffset references used by each wd2 SW band (from the
_cal headers) and what the cached CRDS contexts select for the same header."""
import glob
import os

os.environ.setdefault('CRDS_PATH', '/orange/adamginsburg/jwst/crds')
os.environ.setdefault('CRDS_SERVER_URL', 'https://jwst-crds.stsci.edu')
import crds
from astropy.io import fits

R = '/orange/adamginsburg/jwst/wd2'
BANDS = ['F115W', 'F150W', 'F162M', 'F164N', 'F182M', 'F187N', 'F200W', 'F212N']
CTXS = ['jwst_1568.pmap', 'jwst_1595.pmap', 'jwst_1596.pmap']
KEYS = ['META.INSTRUMENT.NAME', 'META.INSTRUMENT.DETECTOR', 'META.INSTRUMENT.FILTER',
        'META.INSTRUMENT.PUPIL', 'META.EXPOSURE.TYPE', 'META.OBSERVATION.DATE',
        'META.OBSERVATION.TIME', 'META.SUBARRAY.NAME', 'META.INSTRUMENT.CHANNEL',
        'META.INSTRUMENT.MODULE']
for b in BANDS:
    for det in ('nrca1', 'nrcb1'):
        fs = sorted(glob.glob(f'{R}/{b}/pipeline/jw03523005001_*_00001_{det}_cal.fits'))
        if not fs:
            continue
        h = fits.getheader(fs[0])
        used = (h.get('CRDS_CTX'), os.path.basename(str(h.get('R_DISTOR'))),
                os.path.basename(str(h.get('R_FILOFF'))))
        hdr = {'META.INSTRUMENT.NAME': 'NIRCAM', 'META.INSTRUMENT.DETECTOR': h['DETECTOR'],
               'META.INSTRUMENT.FILTER': h['FILTER'], 'META.INSTRUMENT.PUPIL': h['PUPIL'],
               'META.EXPOSURE.TYPE': h['EXP_TYPE'], 'META.OBSERVATION.DATE': h['DATE-OBS'],
               'META.OBSERVATION.TIME': h['TIME-OBS'], 'META.SUBARRAY.NAME': h['SUBARRAY'],
               'META.INSTRUMENT.CHANNEL': h['CHANNEL'], 'META.INSTRUMENT.MODULE': h['MODULE']}
        sel = []
        for c in CTXS:
            try:
                r = crds.getrecommendations(hdr, reftypes=['distortion', 'filteroffset'],
                                            context=c, observatory='jwst')
            except crds.CrdsError as e:
                sel.append(f'{c}: {e}')
                continue
            sel.append(f"{c.split('.')[0]}: {r['distortion']} {r['filteroffset']}")
        print(f"{b} {det} {h['FILTER']}/{h['PUPIL']} used ctx={used[0]} {used[1]} {used[2]}")
        for s in sel:
            print('    ', s)
