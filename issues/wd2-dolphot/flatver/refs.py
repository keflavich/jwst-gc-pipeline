"""List CRDS-selected flat/photom/area references per context for representative (detector, filter)."""
import os, sys, json
os.environ['CRDS_PATH'] = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/flatver/crds_cache'
os.environ['CRDS_SERVER_URL'] = 'https://jwst-crds.stsci.edu'
from crds.client import api
CTX = sys.argv[1:] or ['jwst_1298.pmap', 'jwst_1322.pmap', 'jwst_1568.pmap', 'jwst_1595.pmap']
BANDS = {'F150W': ('SW', 'CLEAR', 'F150W'), 'F200W': ('SW', 'CLEAR', 'F200W'), 'F212N': ('SW', 'CLEAR', 'F212N'),
         'F250M': ('LW', 'CLEAR', 'F250M'), 'F300M': ('LW', 'CLEAR', 'F300M'), 'F410M': ('LW', 'CLEAR', 'F410M')}
DETS = {'SW': ['NRCA1', 'NRCA2', 'NRCA3', 'NRCA4', 'NRCB1', 'NRCB2', 'NRCB3', 'NRCB4'], 'LW': ['NRCALONG', 'NRCBLONG']}
out = {}
for ctx in CTX:
    for band, (ch, pup, fil) in BANDS.items():
        for det in DETS[ch]:
            hd = {'META.INSTRUMENT.NAME': 'NIRCAM', 'META.INSTRUMENT.DETECTOR': det, 'META.INSTRUMENT.FILTER': fil,
                  'META.INSTRUMENT.PUPIL': pup, 'META.EXPOSURE.TYPE': 'NRC_IMAGE', 'META.EXPOSURE.READPATT': 'SHALLOW4',
                  'META.SUBARRAY.NAME': 'FULL', 'META.INSTRUMENT.CHANNEL': 'SHORT' if ch == 'SW' else 'LONG',
                  'META.INSTRUMENT.MODULE': det[3], 'META.OBSERVATION.DATE': '2024-07-14', 'META.OBSERVATION.TIME': '04:48:30',
                  'META.EXPOSURE.START_TIME': 60505.2}
            r = api.get_best_references(ctx, hd, reftypes=['flat', 'photom', 'area'])
            out[f'{ctx}|{band}|{det}'] = {k: str(v) for k, v in r.items()}
            print(ctx, band, det, {k: str(v) for k, v in r.items()}, flush=True)
json.dump(out, open('refs.json', 'w'), indent=1)
