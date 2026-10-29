"""Condensed catalog from PCAT chain samples (Portillo et al. 2017, Sect. 5).

Algorithm (after Lion ``retr_catlseed`` / ``retr_catlcond``):

1. Seed graph.  Every source in every used sample is a node.  Two nodes from
   different samples within ``radius`` (0.75 px) are joined when each is the
   other's closest source in the other's sample (Lion keeps, for each node and
   each other sample, only the edge to the closest source; on its undirected
   graph that leaves the mutual-closest pairs).
2. Seeds.  Repeatedly take the node of highest current degree, record it as a
   seed with that degree, and delete it and its neighbours (degrees update).
   Seeds with degree < ``min_degree`` (2) are dropped.
3. Condensation.  For each seed, in seed order, and for each sample, take the
   brightest not-yet-assigned source within ``radius`` of the seed.  The
   prevalence is the number of samples with an assigned source divided by the
   number of samples used.

Differences from Lion (which are fixes): the seed graph uses samples spread
evenly over the whole post-burn-in chain (Lion: the first 10 samples);
uncertainties are standard deviations (Lion's ``stdv`` is the 84-16 percentile
range, i.e. 2 sigma); percentiles are reported separately.

Memory: the seed graph has up to ``n_src * m^2 / 2`` candidate pairs for ``m``
seed samples (every source in every sample pairs with its counterparts in all
other samples), about 1.3 GB of index arrays per 1e8 pairs.  ``seed_max_samples``
(default 200) thins the samples that build the seed graph; step 3 (prevalence,
moments) always uses every sample in ``sample_idx`` / ``max_samples``.  With
``m`` seed samples a source needs prevalence of order ``min_degree / m`` to
seed (1 % for m = 200).
"""
import heapq

import numpy as np
from scipy.spatial import cKDTree

__all__ = ['condense', 'merge_bright', 'n_above_samples']


def _stack(xs, ys, fs):
    """Flatten per-sample arrays (NaN padded) into points with sample ids."""
    ns = xs.shape[0]
    good = np.isfinite(xs)
    sid = np.repeat(np.arange(ns), xs.shape[1]).reshape(xs.shape)[good]
    X = xs[good]
    Y = ys[good]
    F = np.stack([fs[:, b, :][good] for b in range(fs.shape[1])], axis=0)
    return X, Y, F, sid


def _mutual_closest_edges(X, Y, sid, radius):
    tree = cKDTree(np.column_stack([X, Y]))
    pairs = tree.query_pairs(radius, output_type='ndarray')
    if pairs.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    i, j = pairs[:, 0], pairs[:, 1]
    ok = sid[i] != sid[j]
    i, j = i[ok], j[ok]
    d2 = (X[i] - X[j]) ** 2 + (Y[i] - Y[j]) ** 2
    # directed both ways
    a = np.concatenate([i, j])
    b = np.concatenate([j, i])
    d = np.concatenate([d2, d2])
    # closest b per (a, sample of b)
    order = np.lexsort((b, d, sid[b], a))
    a, b, d = a[order], b[order], d[order]
    key_new = np.ones(a.size, bool)
    key_new[1:] = (a[1:] != a[:-1]) | (sid[b][1:] != sid[b][:-1])
    a, b = a[key_new], b[key_new]
    # mutual: keep (a, b) if (b, a) is also present
    n = X.size
    code = a.astype(np.int64) * n + b
    rcode = b.astype(np.int64) * n + a
    mutual = np.isin(code, rcode)
    a, b = a[mutual], b[mutual]
    keep = a < b
    return np.column_stack([a[keep], b[keep]])


def _seeds(n, edges):
    """Greedy max-degree seed selection with degree updates (lazy heap)."""
    if edges.size == 0:
        return np.arange(n), np.zeros(n, int)
    a = np.concatenate([edges[:, 0], edges[:, 1]])
    b = np.concatenate([edges[:, 1], edges[:, 0]])
    order = np.argsort(a, kind='stable')
    a, b = a[order], b[order]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(indptr, a + 1, 1)
    indptr = np.cumsum(indptr)
    deg = np.diff(indptr).astype(np.int64)
    alive = np.ones(n, bool)
    heap = [(-int(deg[k]), k) for k in range(n)]
    heapq.heapify(heap)
    seeds, sdeg = [], []
    while heap:
        negd, k = heapq.heappop(heap)
        if not alive[k]:
            continue
        if -negd != deg[k]:
            heapq.heappush(heap, (-int(deg[k]), k))
            continue
        nbr = b[indptr[k]:indptr[k + 1]]
        nbr = nbr[alive[nbr]]
        seeds.append(k)
        sdeg.append(int(deg[k]))
        kill = np.concatenate([[k], nbr])
        alive[kill] = False
        # decrement degrees of the living neighbours of every removed node
        for q in kill:
            nn = b[indptr[q]:indptr[q + 1]]
            nn = nn[alive[nn]]
            if nn.size:
                np.subtract.at(deg, nn, 1)
    return np.array(seeds, dtype=np.int64), np.array(sdeg, dtype=np.int64)


def condense(x, y, f, radius=0.75, min_degree=2, max_samples=None, sample_idx=None,
             seed_max_samples=200, return_samples=False):
    """Condensed catalog from chain samples.

    Parameters
    ----------
    x, y : (nsamp, nmax) positions (NaN padded).
    f : (nsamp, nbands, nmax) fluxes (NaN padded).
    sample_idx : indices of samples to use (default all).
    max_samples : thin evenly to at most this many samples.
    seed_max_samples : at most this many (evenly spaced) of the used samples
        build the seed graph (``None``: all); see the module docstring.

    return_samples : also return the per-sample assignment arrays
        ``samp_x, samp_y`` (n, nsamp_used) and ``samp_f`` (n, nbands,
        nsamp_used), NaN where a sample holds no source for that entry, and
        ``samp_idx`` (the chain indices of the used samples); needed by
        :func:`merge_bright`.

    Returns a dict of arrays, one entry per condensed source: ``x, y`` (mean),
    ``x_err, y_err`` (std), ``x_med, y_med``, ``flux`` (nbands, n; median),
    ``flux_err`` (std), ``flux_mean``, ``flux_p16``, ``flux_p84``, ``prob``
    (prevalence), ``nmatch``, ``degree``; plus ``nsamp_used``.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    f = np.asarray(f, float)
    if f.ndim == 2:
        f = f[:, None, :]
    idx = np.arange(x.shape[0]) if sample_idx is None else np.asarray(sample_idx)
    if max_samples is not None and idx.size > max_samples:
        idx = idx[np.round(np.linspace(0, idx.size - 1, max_samples)).astype(int)]
    xs, ys, fs = x[idx], y[idx], f[idx]
    ns = idx.size
    nb = fs.shape[1]
    X, Y, F, sid = _stack(xs, ys, fs)
    nseed = int(ns if seed_max_samples is None else min(ns, seed_max_samples))
    empty = dict(x=np.zeros(0), y=np.zeros(0), x_err=np.zeros(0), y_err=np.zeros(0),
                 x_med=np.zeros(0), y_med=np.zeros(0), flux=np.zeros((nb, 0)),
                 flux_err=np.zeros((nb, 0)), flux_mean=np.zeros((nb, 0)),
                 flux_p16=np.zeros((nb, 0)), flux_p84=np.zeros((nb, 0)), prob=np.zeros(0),
                 nmatch=np.zeros(0, int), degree=np.zeros(0, int), nsamp_used=ns,
                 nsamp_seed=nseed)
    if return_samples:
        empty.update(samp_x=np.zeros((0, ns)), samp_y=np.zeros((0, ns)),
                     samp_f=np.zeros((0, nb, ns)), samp_idx=idx)
    if X.size == 0:
        return empty
    if seed_max_samples is not None and ns > seed_max_samples:
        sseed = np.unique(np.round(np.linspace(0, ns - 1, seed_max_samples)).astype(int))
        node = np.flatnonzero(np.isin(sid, sseed))
    else:
        node = np.arange(X.size)
    edges = _mutual_closest_edges(X[node], Y[node], sid[node], radius)
    seeds, sdeg = _seeds(node.size, edges)
    seeds = node[seeds]
    keep = sdeg >= min_degree
    seeds, sdeg = seeds[keep], sdeg[keep]
    if seeds.size == 0:
        return empty
    tree = cKDTree(np.column_stack([X, Y]))
    cand = tree.query_ball_point(np.column_stack([X[seeds], Y[seeds]]), radius)
    assigned = np.zeros(X.size, bool)
    ksrc = len(seeds)
    mx = np.full((ksrc, ns), np.nan)
    my = np.full((ksrc, ns), np.nan)
    mf = np.full((ksrc, nb, ns), np.nan)
    for k in range(ksrc):
        c = np.asarray(cand[k], dtype=np.int64)
        if c.size == 0:
            continue
        c = c[~assigned[c]]
        if c.size == 0:
            continue
        # brightest (pivot flux) per sample
        o = np.lexsort((-F[0, c], sid[c]))
        c = c[o]
        first = np.ones(c.size, bool)
        first[1:] = sid[c][1:] != sid[c][:-1]
        c = c[first]
        assigned[c] = True
        mx[k, sid[c]] = X[c]
        my[k, sid[c]] = Y[c]
        mf[k, :, sid[c]] = F[:, c].T
    nm = np.sum(np.isfinite(mx), axis=1)
    ok = nm > 0
    mx, my, mf, sdeg = mx[ok], my[ok], mf[ok], sdeg[ok]
    out = _summarise(mx, my, mf, ns)
    out.update(degree=sdeg, nsamp_used=ns, nsamp_seed=nseed)
    if return_samples:
        out.update(samp_x=mx, samp_y=my, samp_f=mf, samp_idx=idx)
    return out


def _std(a, axis=-1):
    n = np.sum(np.isfinite(a), axis=axis)
    s = np.nanstd(a, axis=axis, ddof=1) if a.shape[axis] > 1 else np.full(n.shape, np.nan)
    return np.where(n > 1, s, np.nan)


def _summarise(mx, my, mf, ns):
    """Per-source posterior summaries from per-sample assignment arrays
    ``mx, my`` (k, ns) and ``mf`` (k, nb, ns), NaN where absent."""
    nm = np.sum(np.isfinite(mx), axis=1)
    import warnings
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return dict(
            x=np.nanmean(mx, axis=1), y=np.nanmean(my, axis=1),
            x_err=_std(mx), y_err=_std(my),
            x_med=np.nanmedian(mx, axis=1), y_med=np.nanmedian(my, axis=1),
            flux=np.nanmedian(mf, axis=2).T, flux_err=_std(mf).T,
            flux_mean=np.nanmean(mf, axis=2).T,
            flux_p16=np.nanpercentile(mf, 16, axis=2).T,
            flux_p84=np.nanpercentile(mf, 84, axis=2).T,
            prob=nm / ns, nmatch=nm)


def merge_bright(cond, f_bright, r_merge=1.5, var_frac=0.8, band=0):
    """Merge condensed components that trade flux with a bright source.

    Around a bright star with a PSF-model mismatch the sampler often holds the
    star as two or more components (e.g. 20000 + 13000 counts 0.1 px apart, or
    20000 + 1300 counts 0.7 px apart on frame P) whose fluxes anti-correlate
    sample by sample while their sum stays constant.  Condensation keeps them
    as separate sources.  This step merges them.

    For every row ``i`` with median flux (band ``band``) ``>= f_bright``, in
    descending flux order, each not-yet-merged row ``j`` with lower flux within
    ``r_merge`` px of the current group position is merged when the per-sample
    summed flux is more stable than the group alone:
    ``Var(F_i + F_j) <= var_frac * Var(F_i)``, a sample without the component
    counting as zero flux.  The test is repeated against the updated group
    until no candidate passes.  Components that are uncorrelated with the
    bright star (real neighbours, or PSF-ring residuals) give a variance ratio
    ``>= 1`` and are kept.

    ``cond`` must come from :func:`condense` with ``return_samples=True``.
    Returns a new dict with the same keys (merged rows removed, summaries of
    each group recomputed from its per-sample summed flux and flux-weighted
    position, ``prob`` = fraction of samples holding any member) plus
    ``nmerge`` (components absorbed per row), ``merged_from`` (list of
    original row indices per output row) and ``merge_log``.
    """
    for k in ('samp_x', 'samp_y', 'samp_f'):
        if k not in cond:
            raise ValueError('merge_bright needs condense(..., return_samples=True)')
    mx = np.array(cond['samp_x'], float)
    my = np.array(cond['samp_y'], float)
    mf = np.array(cond['samp_f'], float)
    k, nb, ns = mf.shape
    fmed = np.asarray(cond['flux'], float)[band]
    gx = np.array(cond['x'], float)
    gy = np.array(cond['y'], float)
    alive = np.ones(k, bool)
    members = [[i] for i in range(k)]
    log = []
    tree = cKDTree(np.column_stack([gx, gy])) if k else None
    order = np.argsort(-np.nan_to_num(fmed, nan=-np.inf))
    for i in order:
        if not (alive[i] and np.isfinite(fmed[i]) and fmed[i] >= f_bright):
            continue
        Fi = np.nan_to_num(mf[i, band], nan=0.0)
        while True:
            cand = np.asarray(tree.query_ball_point([gx[i], gy[i]], r_merge + 1e-9), dtype=np.int64)
            cand = cand[(cand != i) & alive[cand]]
            cand = cand[~(np.nan_to_num(fmed[cand], nan=0.0) > fmed[i])]
            if cand.size == 0:
                break
            vi = np.var(Fi)
            best, best_v = -1, np.inf
            for j in cand:
                v = np.var(Fi + np.nan_to_num(mf[j, band], nan=0.0))
                if v < best_v:
                    best, best_v = int(j), float(v)
            if not best_v <= var_frac * vi:
                break
            j = best
            # per-sample flux-weighted position, summed flux
            wi = np.nan_to_num(mf[i, band], nan=0.0)
            wj = np.nan_to_num(mf[j, band], nan=0.0)
            wsum = wi + wj
            with np.errstate(invalid='ignore', divide='ignore'):
                nxp = (np.nan_to_num(mx[i]) * wi + np.nan_to_num(mx[j]) * wj) / wsum
                nyp = (np.nan_to_num(my[i]) * wi + np.nan_to_num(my[j]) * wj) / wsum
            pres = np.isfinite(mx[i]) | np.isfinite(mx[j])
            mx[i] = np.where(pres, nxp, np.nan)
            my[i] = np.where(pres, nyp, np.nan)
            mf[i] = np.where(pres[None, :], np.nan_to_num(mf[i]) + np.nan_to_num(mf[j]), np.nan)
            alive[j] = False
            members[i] += members[j]
            log.append(dict(into=int(i), row=int(j), d=float(np.hypot(gx[j] - gx[i], gy[j] - gy[i])),
                            flux=float(fmed[j]), var_ratio=float(best_v / vi) if vi > 0 else None))
            Fi = np.nan_to_num(mf[i, band], nan=0.0)
            with np.errstate(invalid='ignore'):
                gx[i] = np.nanmean(mx[i])
                gy[i] = np.nanmean(my[i])
    keep = np.flatnonzero(alive)
    out = _summarise(mx[keep], my[keep], mf[keep], ns)
    out.update(degree=np.asarray(cond['degree'])[keep], nsamp_used=cond['nsamp_used'],
               nsamp_seed=cond.get('nsamp_seed', cond['nsamp_used']),
               samp_x=mx[keep], samp_y=my[keep], samp_f=mf[keep],
               nmerge=np.array([len(members[i]) - 1 for i in keep], int),
               merged_from=[members[i] for i in keep], merge_log=log,
               merge_params=dict(f_bright=float(f_bright), r_merge=float(r_merge),
                                 var_frac=float(var_frac), band=int(band)))
    if 'samp_idx' in cond:
        out['samp_idx'] = cond['samp_idx']
    for key in cond:
        if key not in out and np.ndim(cond[key]) == 1 and len(cond[key]) == k:
            out[key] = np.asarray(cond[key])[keep]
    return out


def n_above_samples(cond, flims, band=0, region=None):
    """Per-sample number of condensed sources above each flux limit.

    Counts, for every used sample, the rows of ``cond`` (``return_samples=True``
    output of :func:`condense` or :func:`merge_bright`) whose sample flux is
    ``>= flim``; ``region = (x0, x1, y0, y1)`` restricts to rows whose
    posterior mean position lies inside.  Returns ``{flim: (nsamp,) int}``.
    """
    f = np.asarray(cond['samp_f'], float)[:, band, :]
    sel = np.ones(f.shape[0], bool)
    if region is not None:
        x, y = np.asarray(cond['x']), np.asarray(cond['y'])
        sel = (x >= region[0]) & (x < region[1]) & (y >= region[2]) & (y < region[3])
    with np.errstate(invalid='ignore'):
        return {fl: np.sum(np.nan_to_num(f[sel], nan=-np.inf) >= fl, axis=0) for fl in flims}
