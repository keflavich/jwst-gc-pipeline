"""Replay one branch's m6 vetting on a production m6 merged catalog.

The vetting call in run_manual_pipeline (from the ``_ext_prom_opt`` line to
the ``_filter_extended_emission(...)`` call) is cut out of the worktree's own
cataloging.py and executed with pipeline-default options, so every branch is
replayed with its own call-site arguments and AUTO resolution.  Writes the
input catalog with a ``kept`` column and the diagnostics the vetting computed
(prominence, prominence_robust, local_emission_snr, ...).

usage: [REPLAY_OPTS=<json>] python vet_variant.py <pipeline_worktree> <field> <band> <out.fits>
"""
import contextlib
import io
import json
import os
import subprocess
import sys
import time
import types

WT, FIELD, BAND, OUTF = sys.argv[1:5]
sys.path.insert(0, WT)
import numpy as np                                       # noqa: E402
from astropy.io import fits                              # noqa: E402
from astropy.table import Table                          # noqa: E402
from astropy import wcs                                  # noqa: E402
from jwst_gc_pipeline.photometry import cataloging as C  # noqa: E402

assert os.path.realpath(C.__file__).startswith(os.path.realpath(WT)), C.__file__
R = '/orange/adamginsburg/jwst'
FIELDS = {
    'brick': dict(cat='{fl}_merged_o001_indivexp_merged_resbgsub_m6_dao_basic.fits',
                  i2d='jw02221-o001_t001_nircam_clear-{fl}-merged_data_i2d.fits'),
    'sgrb2': dict(cat='{fl}_merged_indivexp_merged_resbgsub_m6_dao_basic.fits',
                  i2d='jw05365-o001_t001_nircam_clear-{fl}-merged_data_i2d.fits'),
    'w51': dict(cat='{fl}_merged_indivexp_merged_resbgsub_m6_dao_basic.fits',
                i2d='jw06151-o001_t001_nircam_clear-{fl}-merged_data_i2d.fits'),
}
fl = BAND.lower()
cfg = FIELDS[FIELD]
catpath = f'{R}/{FIELD}/catalogs/' + cfg['cat'].format(fl=fl)
dpath = f'{R}/{FIELD}/{BAND.upper()}/pipeline/' + cfg['i2d'].format(fl=fl)


def git(*a):
    return subprocess.run(['git', '-C', WT, *a], capture_output=True, text=True).stdout.strip()


prov = dict(wt=WT, commit=git('rev-parse', 'HEAD'), branch=git('rev-parse', '--abbrev-ref', 'HEAD'),
            dirty=bool(git('status', '--porcelain', '--untracked-files=no')),
            catalog=catpath, data_i2d=dpath, started=time.strftime('%Y-%m-%dT%H:%M:%S'),
            slurm_job_id=os.environ.get('SLURM_JOB_ID'))

# --- cut the vetting call out of the branch's own run_manual_pipeline
src = open(C.__file__).read().splitlines()
i0 = next(i for i, s in enumerate(src) if "_ext_prom_opt = float(mopt(opts_phase, 'manual_ext_prom_min'))" in s)
i1 = next(i for i in range(i0, len(src)) if "label=f'{phase}:{filt}')" in src[i])
block = src[i0:i1 + 1]
ind = len(block[0]) - len(block[0].lstrip())
code = '\n'.join(s[ind:] if s.strip() else '' for s in block)
assert 'vetted = _filter_extended_emission(' in code, code

merged = Table.read(catpath)
merged['rowid'] = np.arange(len(merged))
if os.environ.get('REPLAY_MAXROWS'):            # smoke test only
    merged = merged[:int(os.environ['REPLAY_MAXROWS'])]
with fits.open(dpath) as dh:
    d_i2d = dh['SCI'].data.astype(float)
    ww_i2d = wcs.WCS(dh['SCI'].header)
    # the i2d ERR plane (read by branches whose call-site passes err_i2d_image)
    e_i2d = dh['ERR'].data.astype(float) if 'ERR' in dh else None

opts = types.SimpleNamespace(target=FIELD)
# option overrides for threshold sweeps, e.g. REPLAY_OPTS='{"manual_ext_star_prom_min": 7}'
overrides = json.loads(os.environ.get('REPLAY_OPTS', '{}'))
for k, v in overrides.items():
    assert k in C.MANUAL_DEFAULTS, k
    setattr(opts, k, v)
prov['overrides'] = overrides
ns = dict(vars(C))
ns.update(merged=merged, d_i2d=d_i2d, ww_i2d=ww_i2d, e_i2d=e_i2d, opts_phase=opts, _miri_field=False,
          phase='m6', filt=BAND.upper(), module='merged')
log = io.StringIO()
t0 = time.time()
with contextlib.redirect_stdout(log):
    exec(compile(code, f'{C.__file__}:vetting-call', 'exec'), ns)
vetted = ns['vetted']
prov['seconds'] = round(time.time() - t0)
prov['n_merged'], prov['n_vetted'] = len(merged), len(vetted)
prov['callsite'] = code

kept = np.zeros(len(merged), bool)
kept[np.asarray(vetted['rowid'])] = True
merged['kept'] = kept
cols = ['rowid', 'skycoord', 'flux', 'flux_err', 'flux_err_prop', 'qfit', 'flags', 'nmatch',
        'local_bkg', 'group_size', 'is_saturated', 'replaced_saturated', 'prominence',
        'prominence_robust', 'peak_sb', 'local_emission_snr', 'sky_clean', 'core_concentration', 'kept']
out = merged[[c for c in cols if c in merged.colnames]]
for c in out.colnames:
    if c != 'skycoord' and out[c].dtype == np.float64:
        out[c] = out[c].astype(np.float32)
out.meta = {'PROVJSON': json.dumps({k: v for k, v in prov.items() if k != 'callsite'})}
out.write(OUTF, overwrite=True)
with open(OUTF.replace('.fits', '.json'), 'w') as fh:
    json.dump(dict(prov, log=log.getvalue()), fh, indent=1)
print(json.dumps({k: v for k, v in prov.items() if k != 'callsite'}))
print(log.getvalue())
