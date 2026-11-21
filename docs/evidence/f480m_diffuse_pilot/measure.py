"""Read-only Treasury F480M lineage audit and calibrated-exposure injection pilot."""
import argparse
from collections import Counter
import json
from pathlib import Path
import os
import warnings

import numpy as np
from astropy.io import fits
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import map_coordinates, median_filter

PILOT = Path(os.environ.get('F480M_PILOT_ROOT',
                           '/blue/adamginsburg/adamginsburg/tmp/destreakpilot'))
JC = Path('/orange/adamginsburg/jwst/gc-treasury')
from jwst_gc_pipeline.frame_wcs import frame_wcs
from jwst_gc_pipeline.reduction.destreak import destreak_data, nozero_percentile

OBSERVATIONS = ('065', '072', '079')
WINDOWS = (256, 2048)
CENTER = (1250, 1024)
HALF = 384
WIDTH = 2048
SCALE = np.sqrt(8 * np.log(2))
AMPLITUDE = 0.6
TEMPLATES = {
    **{f'filament_{w}px_{direction}': (direction, w)
       for w in (8, 32, 128) for direction in ('parallel', 'perpendicular')},
    'knot_30px': ('knot', 30),
    'wave_512px_parallel': ('wave_y', 512),
    'wave_512px_perpendicular': ('wave_x', 512),
}

def audit(out):
    recipe = JC / 'mosaics/recipe/make_residual_mosaics.py'
    source = ''.join(line for line in recipe.read_text().splitlines(keepends=True)
                     if not line.startswith(('import numpy', 'from astropy', 'from reproject')))
    ns = {'__name__': 'audit', 'print': lambda *args: None}
    exec(compile(source, str(recipe), 'exec'), ns)
    tiles = {}
    for filename in ns['select']('F480M'):
        image = Path(filename)
        obs = ns['OBSID_RE'].search(image.name)[1]
        ledger = image.with_name(image.name.replace('_i2d', '_perframe_inputs').replace('.fits', '.json'))
        record = {'image': str(image), 'ledger': str(ledger)}
        if ledger.exists():
            inputs = json.loads(ledger.read_text())
            frames = [row['frame'] for row in inputs['frames']]
            record['frames'] = frames
            lineage = Counter('destreak' if '_destreak_' in f else
                              'align' if '_align_' in f else 'other' for f in frames)
            record['counts'] = dict(lineage)
            record['lineage'] = next(iter(lineage)) if len(lineage) == 1 else 'mixed_or_empty'
            record['ledger_names_image'] = inputs['i2d'] == image.name
        else:
            record['lineage'] = 'missing_ledger'
        tiles[obs] = record

    summaries = {name: json.loads((PILOT / f'analysis/survey_i2d_{name}.json').read_text())
                 for name in ('prod', 'woff', 'wjoint')}
    pairs = sorted(set.intersection(*[{pair for pair, row in table.items()
                                        if row.get('n_blocks', 0) >= 5
                                        and row.get('block_median') is not None}
                                       for table in summaries.values()]))
    scores = {}
    for name, table in summaries.items():
        offsets = np.abs([table[pair]['block_median'] for pair in pairs])
        scores[name] = dict(n=len(pairs), median_abs=float(np.median(offsets)),
                            p90_abs=float(np.percentile(offsets, 90)),
                            above_one=int(np.sum(offsets > 1)),
                            within_block_mad=float(np.mean(
                                [table[pair]['within_block_mad_mean'] for pair in pairs])))
    result = {'selection_recipe': str(recipe), 'tile_lineage': tiles, 'archived_overlap': scores}
    (out / 'lineage.json').write_text(json.dumps(result, indent=2) + '\n')
    print('Selected F480M lineage:', Counter(t['lineage'] for t in tiles.values()), flush=True)
    print('Ledger-name mismatch:', [o for o, t in tiles.items() if t.get('ledger_names_image') is False], flush=True)
    print('Archived overlaps:', scores, flush=True)

def axis_grid(low, high):
    return np.append(np.arange(low, high, 32), high)

def warp_on_grid(transform, x_nodes, y_nodes, x_out, y_out):
    xx, yy = np.meshgrid(x_nodes, y_nodes)
    coords = transform(xx, yy)
    if not np.isfinite(coords).all():
        raise ValueError('Non-finite coordinates at WCS interpolation nodes')
    dense = [RectBivariateSpline(y_nodes, x_nodes, a, kx=1, ky=1)(y_out, x_out).astype(np.float32)
             for a in coords]
    x_mid = (x_nodes[:-1] + x_nodes[1:]) / 2
    y_mid = (y_nodes[:-1] + y_nodes[1:]) / 2
    xm, ym = np.meshgrid(x_mid, y_mid)
    exact = transform(xm, ym)
    interpolated = [(a[:-1, :-1] + a[:-1, 1:] + a[1:, :-1] + a[1:, 1:]) / 4
                    for a in coords]
    error = float(np.max(np.hypot(*(np.subtract(exact, interpolated)))))
    if error > .1:
        raise ValueError(f'WCS interpolation error {error:.3f} pix exceeds 0.1 pix')
    return dense, error

def profiles(image, noutputs):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "All-NaN slice encountered", RuntimeWarning)
        return [nozero_percentile(chunk, 10, axis=1)
                for chunk in np.split(image, noutputs, axis=1)]

def corrections(row_profiles, window):
    return [p - median_filter(p, min(window, len(p))) for p in row_profiles]

def template(kind, width, x, y):
    if kind in ('wave_x', 'wave_y'):
        coord = x if kind == 'wave_x' else y
        return AMPLITUDE * np.sin(2 * np.pi * coord / width)
    if kind == 'knot':
        return AMPLITUDE * np.exp(-0.5 * (x**2 + y**2) * (SCALE / width)**2)
    along, across = (x, y) if kind == 'parallel' else (y, x)
    return AMPLITUDE * np.exp(-0.5 * ((across * SCALE / width)**2 + (along * SCALE / 960)**2))

def measure(obs):
    root = PILOT / 'woff/gc-treasury/F480M/pipeline'
    frames = sorted(root.glob(f'jw10678{obs}001_*_nrcblong_cal.fits'))
    if len(frames) != 6:
        raise ValueError(f'o{obs}: expected six NRCBLONG inputs, found {len(frames)}')
    wcs_files = [f.with_name(f.name.replace('_cal.fits', f'_align_o{obs}_crf.fits')) for f in frames]
    transforms = [frame_wcs(str(f)) for f in wcs_files]
    ref = transforms[0]
    xc, yc = CENTER
    ra, dec = ref.all_pix2world(np.array([xc, xc + 32, xc]),
                                np.array([yc, yc, yc + 32]), 0)
    sky_origin = (ra[0], dec[0])
    cos_dec = np.cos(np.deg2rad(dec[0]))
    detector_to_sky = np.array([(ra[1:] - ra[0]) * cos_dec, dec[1:] - dec[0]]) / 32
    sky_to_detector = np.linalg.inv(detector_to_sky)
    rx = np.arange(xc - HALF, xc + HALF)
    ry = np.arange(yc - HALF, yc + HALF)
    nodes = axis_grid(0, WIDTH - 1)
    ndet = np.arange(WIDTH)
    coadds = {name: np.zeros((len(WINDOWS) + 1, len(ry), len(rx)), dtype=np.float64)
              for name in TEMPLATES}
    coverage = np.zeros((len(ry), len(rx)), dtype=np.int16)
    errors = []
    for f, crf, w in zip(frames, wcs_files, transforms):
        def to_ref(x, y):
            ra, dec = w.all_pix2world(x.ravel(), y.ravel(), 0)
            xy = sky_to_detector @ np.array([(ra - sky_origin[0]) * cos_dec,
                                              dec - sky_origin[1]])
            return [value.reshape(x.shape) for value in xy]

        def to_detector(x, y):
            ra, dec = ref.all_pix2world(x.ravel(), y.ravel(), 0)
            if w.gwcs is None:
                xd, yd = w.all_world2pix(ra, dec, 0)
            else:
                xd, yd = w.gwcs.invert(ra, dec, with_bounding_box=False)
            return xd.reshape(x.shape), yd.reshape(y.shape)

        (dx, dy), sky_error = warp_on_grid(to_ref, nodes, nodes, ndet, ndet)
        (x_det, y_det), det_error = warp_on_grid(to_detector, axis_grid(rx[0], rx[-1]),
                                                axis_grid(ry[0], ry[-1]), rx, ry)
        errors.extend((sky_error, det_error))
        with fits.open(f, memmap=False) as h, fits.open(crf, memmap=False) as c:
            img = h['SCI'].data.astype(np.float32, copy=True)
            noutputs = int(h[0].header['NOUTPUTS'])
            valid = (np.isfinite(img) & (img != 0) & ((c['DQ'].data & 3) == 0))
        valid[:4] = False
        valid[-4:] = False
        valid[:, :4] = False
        valid[:, -4:] = False
        sample_coordinates = (y_det, x_det)
        use = map_coordinates(valid.astype(np.float32), sample_coordinates, order=1) >= .999
        coverage += use
        baseline = profiles(img, noutputs)
        base_corr = {window: corrections(baseline, window) for window in WINDOWS}

        for name, (kind, width) in TEMPLATES.items():
            add = template(kind, width, dx, dy).astype(np.float32)
            injected = np.where(np.isfinite(img) & (img != 0), img + add, img)
            p_in = profiles(injected, noutputs)
            # The linear projection and validity mask are held fixed across arms.
            # A simulated zero in the measured SCI stays a zero in the injected arm.
            for index, window in enumerate((None, *WINDOWS)):
                recovered = add.copy()
                if window is not None:
                    for dest, new, old in zip(np.split(recovered, noutputs, axis=1),
                                              corrections(p_in, window), base_corr[window]):
                        dest -= (new - old)[:, None]
                    if f == frames[0] and name == next(iter(TEMPLATES)):
                        expected = (destreak_data(injected.copy(), median_filter_size=window, noutputs=noutputs)
                                    - destreak_data(img.copy(), median_filter_size=window, noutputs=noutputs))
                        np.testing.assert_allclose(recovered[valid], expected[valid], atol=5e-4, rtol=0)
                resampled = map_coordinates(recovered, sample_coordinates, order=1)
                coadds[name][index, use] += resampled[use]
        print(f'o{obs} {f.name}: injection complete', flush=True)

    yy, xx = np.meshgrid(ry - yc, rx - xc, indexing='ij')
    common = coverage >= 3
    scores = {}
    for name, stack in coadds.items():
        kind, width = TEMPLATES[name]
        truth, *recovered = (s[common] / coverage[common] for s in stack)
        centered = truth - truth.mean()
        if kind == 'knot':
            aperture = (xx**2 + yy**2) ** .5 <= 1.5 * width
        elif kind == 'parallel':
            aperture = (np.abs(yy) <= 1.5 * width) & (np.abs(xx) <= 288)
        elif kind == 'perpendicular':
            aperture = (np.abs(xx) <= 1.5 * width) & (np.abs(yy) <= 288)
        else:
            aperture = None
        scores[name] = {}
        for window, result in zip(WINDOWS, recovered):
            metrics = dict(contrast_gain=float(centered @ (result - result.mean()) / (centered @ centered)),
                           error_rms_mjysr=float(np.sqrt(np.mean((result - truth)**2))))
            if aperture is not None:
                ap = aperture[common]
                metrics['aperture_flux_ratio'] = float(result[ap].sum() / truth[ap].sum())
            scores[name][str(window)] = metrics
    return dict(obs=obs, cal_paths=list(map(str, frames)), crf_wcs_paths=list(map(str, wcs_files)),
                max_wcs_interpolation_error_px=max(errors), common_coverage_fraction=float(common.mean()),
                peak_mjysr=AMPLITUDE, metrics=scores)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--obs', nargs='+', default=OBSERVATIONS)
    parser.add_argument('--skip-audit', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.skip_audit:
        audit(args.out)
    for observation in args.obs:
        result = measure(observation)
        (args.out / f'o{observation}.json').write_text(json.dumps(result, indent=2) + '\n')
        print(f'o{observation} complete', flush=True)

if __name__ == '__main__':
    main()
