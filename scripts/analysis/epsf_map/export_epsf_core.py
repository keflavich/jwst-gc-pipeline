"""Export the spatial ePSF core of an epsf_map run as a pipeline core file.

    python export_epsf_core.py <result_<det>.npz> <outdir> [--filter F480M]

Writes ``<outdir>/epsf_core_<stpsf detector>_<filter>.fits`` (format in
``jwst_gc_pipeline.photometry.epsf_hybrid``): the FULL-sample core ``PG`` (all
frames, not one held-out half) on its G x G cell-centre lattice, R = 12 px,
O = 4, normalised to unit flux inside r_norm = 10 px.  Point
``PSF_EPSF_CORE_DIR`` at ``<outdir>`` to fit with the hybrid
(ePSF core + STPSF wing).  ``--half a|b`` exports one of the two held-out
halves instead (for testing on a frame that is in the other half).

Provenance (written to the header): program, number of frames and observations
that contributed, and the detector name mapping (the epsf_map runs use the
file-name detector, e.g. ``nrcblong``; stpsf and the pipeline's PSF cache use
``NRCB5``).
"""
import argparse
import os

import numpy as np

from jwst_gc_pipeline.photometry.epsf_hybrid import EPSFCore, epsf_core_filename, write_epsf_core

# the epsf_map runs (README): LW = F480M, SW = F212N
DEFAULT_FILTER = {'long': 'F480M', 'short': 'F212N'}


def stpsf_detector(det):
    det = det.lower()
    if det.endswith('long'):
        return det[:4].upper() + '5'          # nrcalong -> NRCA5
    return det.upper()                        # nrcb1 -> NRCB1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('result')
    ap.add_argument('outdir')
    ap.add_argument('--filter', default=None)
    ap.add_argument('--half', choices=['a', 'b'], default=None,
                    help='export a held-out half instead of the full sample: '
                         'a = PGa (built from EVEN-index frames), b = PGb (ODD); for validation only')
    a = ap.parse_args()
    z = np.load(a.result, allow_pickle=True)
    det = os.path.basename(a.result)[len('result_'):-len('.npz')]
    filt = a.filter or DEFAULT_FILTER['long' if det.endswith('long') else 'short']
    G, O, R = int(z['Gbest']), int(z['O']), int(z['R'])
    if not z['bilinear_best']:
        raise ValueError(f'{a.result}: best spatial model is not the bilinear lattice')
    nodes = (np.arange(G) + 0.5) * 2048 / G
    frames = [os.path.basename(str(f)) for f in z['frames']]
    meta = dict(PROGRAM=','.join(sorted({f[2:7] for f in frames})),
                NFRAMES=len(frames), NOBS=len({f[2:10] for f in frames}),
                NSTARS=int(np.sum(z['nG'])), RNORM=float(z['r_norm']),
                SRCDET=det, SRCFILE=os.path.basename(a.result))
    key = {'a': 'PGa', 'b': 'PGb'}.get(a.half, 'PG')
    meta['SAMPLE'] = (key, 'PG = all frames; PGa/PGb = even/odd-index halves')
    core = EPSFCore(z[key], O, R, nodes, nodes, meta=meta)
    sdet = stpsf_detector(det)
    os.makedirs(a.outdir, exist_ok=True)
    out = os.path.join(a.outdir, epsf_core_filename(sdet, filt))
    write_epsf_core(out, core, sdet, filt)
    print(f'{out}: {sdet} {filt}, {G}x{G} nodes, {meta["NSTARS"]} stars, '
          f'{meta["NFRAMES"]} frames / {meta["NOBS"]} observations of {meta["PROGRAM"]}')


if __name__ == '__main__':
    main()
