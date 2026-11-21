"""Grade tile offsets on excluded overlap edges of the archived F480M mosaics."""
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

INPUT = Path(__file__).with_name('overlap_input.json')

def check_edges(data, edges):
    pairs = [p.split('-') for p in edges]
    nodes = sorted({tile for p in pairs for tile in p})
    index = {tile: i for i, tile in enumerate(nodes)}
    a, b = np.array([[index[t] for t in p] for p in pairs]).T
    n = len(nodes)
    design = np.zeros((len(edges), n))
    design[np.arange(len(edges)), a] = 1.
    design[np.arange(len(edges)), b] = -1.
    offsets = np.array([data[p]['block_median'] for p in edges])
    # Equal weight per edge, and a robust residual on the native brightness
    # scale. A block count is not an independent-sample count.
    results = []
    for held in range(len(edges)):
        train = np.arange(len(edges)) != held
        graph = csr_matrix((np.ones(train.sum()), (a[train], b[train])), shape=(n, n))
        _, label = connected_components(graph, directed=False)
        if label[a[held]] != label[b[held]]:
            continue
        group = np.flatnonzero(label == label[a[held]])
        keep = train & (label[a] == label[a[held]])
        X = design[keep][:, group[1:]]
        y = offsets[keep]
        initial = np.linalg.lstsq(X, y, rcond=None)[0]
        x = least_squares(lambda x: X @ x - y, initial, jac=lambda x: X,
                          loss='soft_l1', f_scale=.25).x
        pred = float(design[held, group[1:]] @ x)
        results.append(dict(edge=edges[held], offset=float(offsets[held]), prediction=pred,
                            residual=float(offsets[held] - pred)))
    return dict(n_tiles=n, n_overlaps=len(edges), n_held_out=len(results), predictions=results)

def metrics(values):
    absolute = np.abs(values)
    return dict(median_abs=float(np.median(absolute)),
                p90_abs=float(np.percentile(absolute, 90)),
                above_one=int((absolute > 1).sum()))

def main(out):
    data = json.loads(INPUT.read_text())
    edges = sorted(set.intersection(*({p for p, rec in rows.items()
                                       if rec.get('n_blocks', 0) >= 5
                                       and rec.get('block_median') is not None}
                                      for rows in data.values())))
    result = {}
    for v, rows in data.items():
        grade = check_edges(rows, edges)
        grade['uncorrected'] = metrics([p['offset'] for p in grade['predictions']])
        grade['corrected'] = metrics([p['residual'] for p in grade['predictions']])
        result[v] = grade
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    for v, grade in result.items():
        print(v, {k: x for k, x in grade.items() if k != 'predictions'})

if __name__ == '__main__':
    main(Path(sys.argv[1]))
