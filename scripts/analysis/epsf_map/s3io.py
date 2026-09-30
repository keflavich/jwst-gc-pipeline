"""Streaming access to public JWST products and CRDS references on the MAST AWS mirror.

MAST (mast.stsci.edu) and CRDS may be unreachable from sandboxed machines; the public
bucket ``s3://stpubdata`` serves the same files over plain HTTPS:

    https://stpubdata.s3.amazonaws.com/jwst/public/jw10678/jw10678061001/<file>
    https://stpubdata.s3.amazonaws.com/jwst/public/references/<crds file>
    https://stpubdata.s3.amazonaws.com/jwst/public/R2026091802/<WSS OPD file>

Files are fetched with curl (it honours the proxy / CA environment) into a scratch
directory and deleted by the caller once measured -- a full cal frame is 117 MB.
"""
import json
import os
import subprocess
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from jwst_gc_pipeline.mast_names import jw_prefix

BASE = 'https://stpubdata.s3.amazonaws.com'
NS = {'s': 'http://s3.amazonaws.com/doc/2006-03-01/'}


def list_keys(prefix):
    """All (key, size) under a prefix (paginates the S3 v2 listing)."""
    out, tok = [], None
    while True:
        url = f'{BASE}/?list-type=2&max-keys=1000&prefix={urllib.parse.quote(prefix)}'
        if tok:
            url += '&continuation-token=' + urllib.parse.quote(tok)
        root = ET.fromstring(urllib.request.urlopen(url).read())
        for c in root.findall('s:Contents', NS):
            out.append((c.find('s:Key', NS).text, int(c.find('s:Size', NS).text)))
        t = root.find('s:NextContinuationToken', NS)
        if t is None:
            return out
        tok = t.text


def fetch(key, outdir, retries=4):
    """Download one key into outdir (curl, retry with backoff); returns the local path."""
    fn = os.path.join(outdir, os.path.basename(key))
    if os.path.exists(fn):
        return fn
    tmp = f'{fn}.{os.getpid()}.part'
    for i in range(retries + 1):
        r = subprocess.run(['curl', '-sS', '--fail', '-o', tmp, f'{BASE}/{key}'],
                           capture_output=True, text=True)
        if r.returncode == 0:
            os.replace(tmp, fn)
            return fn
        if i == retries:
            raise OSError(f'curl failed for {key}: {r.stderr.strip()}')
        time.sleep(2 ** (i + 1))
    return fn


def crds_reference(crds_uri, cachedir):
    """Local path of a CRDS reference named like 'crds://jwst_nircam_saturation_0115.fits'."""
    name = crds_uri.replace('crds://', '')
    os.makedirs(cachedir, exist_ok=True)
    return fetch(f'jwst/public/references/{name}', cachedir)


def program_cal_keys(program, detectors, cachefile):
    """Cached list of (key, size) of all _cal frames of `program` on `detectors`."""
    if os.path.exists(cachefile):
        with open(cachefile) as f:
            return [tuple(x) for x in json.load(f)]
    allk = list_keys(f'jwst/public/{jw_prefix(program)}/')
    keys = [(k, s) for k, s in allk
            if k.endswith('_cal.fits') and any(f'_{d}_cal' in k for d in detectors)]
    with open(cachefile, 'w') as f:
        json.dump(keys, f)
    return keys
