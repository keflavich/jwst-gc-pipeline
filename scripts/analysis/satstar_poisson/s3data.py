"""List / fetch public JWST products from the MAST open-data mirror on AWS (s3://stpubdata).

MAST itself (mast.stsci.edu) may be unreachable from sandboxed environments; the
public bucket serves the same files over plain HTTPS:
    https://stpubdata.s3.amazonaws.com/jwst/public/jw<PPPPP>/jw<PPPPP><OOO>001/<file>
"""
import os, subprocess, sys, urllib.parse, urllib.request
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
            url += '&continuation-token='+urllib.parse.quote(tok)
        root = ET.fromstring(urllib.request.urlopen(url).read())
        for c in root.findall('s:Contents', NS):
            out.append((c.find('s:Key', NS).text, int(c.find('s:Size', NS).text)))
        t = root.find('s:NextContinuationToken', NS)
        if t is None:
            return out
        tok = t.text


def fetch(key, outdir='.'):
    fn = os.path.join(outdir, os.path.basename(key))
    if not os.path.exists(fn):
        subprocess.run(['curl', '-sS', '-o', fn, f'{BASE}/{key}'], check=True)
    return fn


def fetch_obs(program, obs, detector, suffix, dithers=range(1, 7), outdir='.', visit='001', seq='02101'):
    jw = jw_prefix(program)
    return [fetch(f'jwst/public/{jw}/{jw}{obs}{visit}/{jw}{obs}{visit}_{seq}_{i:05d}_{detector}_{suffix}.fits', outdir)
            for i in dithers]


if __name__ == '__main__':
    # python s3data.py [--outdir DIR] 10678 061 nrcblong cal uncal rateints
    args = sys.argv[1:]
    outdir = '.'
    if args[:1] == ['--outdir']:
        outdir = args[1]; args = args[2:]
        os.makedirs(outdir, exist_ok=True)
    prog, obs, det = args[:3]
    for suf in args[3:]:
        print(fetch_obs(prog, obs, det, suf, outdir=outdir))
