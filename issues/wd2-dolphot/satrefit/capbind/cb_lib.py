"""Shared loader/scorer for capbind: joins out7 row tables, stage-1 region records and the star mapping (score7 conventions)."""
import os, pickle, re
import numpy as np
from astropy.table import Table
from capfun import cap_arrays

Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
CB = Q + '/satrefit/capbind'
SWB = ('150W', '200W')


class Band:
    def __init__(self, band, s1=True):
        self.band = band
        m = pickle.load(open(f'{CB}/map_{band}.pkl', 'rb'))
        self.map = m
        self.ref, self.dm_final, self.i, self.j = m['ref'], m['dm_final'], m['i'], m['j']
        self.n = len(self.ref)
        cols = {}
        S1 = []
        T = []
        self.frame_of = []
        for k, fnm in enumerate(m['files']):
            t6 = Table.read(f"{Q}/satrefit/out7/{fnm.replace('_satrefit.fits', '_satrefit7.fits')}")
            T.append(t6)
            mm = re.match(r'(\w+?)_jw03523005001_(\d+)_(\d+)_(\w+?)_align', fnm)
            b_, vg, e_, det = mm.groups()
            d = pickle.load(open(f'{CB}/s1_{band}_{det}_{int(e_)}.pkl', 'rb')) if s1 else dict(meta=dict(FW_ff=1, FW_g0=1, ceiling=1), rows=[dict(label=l) for l in t6['label']])
            assert len(d['rows']) == len(t6)
            S1.append(d)
            self.frame_of += [k] * len(t6)
        self.frame_of = np.array(self.frame_of)
        self.metas = [d['meta'] for d in S1]
        self.rows = [r for d in S1 for r in d['rows']]
        rowmeta = [d['meta'] for d in S1 for _ in d['rows']]
        self.FW_ff = np.array([m_['FW_ff'] for m_ in rowmeta])
        self.FW_g0 = np.array([m_['FW_g0'] for m_ in rowmeta])
        self.ceiling = np.array([m_['ceiling'] for m_ in rowmeta])
        self.nrow = len(self.rows)
        col = lambda c: np.concatenate([np.asarray(t[c], float) for t in T])
        self.col = col
        self.label = np.concatenate([np.asarray(t['label']) for t in T])
        assert len(self.label) == sum(m['lens'])
        self.good = self.label > 0
        self.acat, self.araw, self.abase, self.cbase = col('a_cat'), col('a_raw'), col('a_base'), col('cap_base')
        self.capped = self.araw < 0.999 * self.acat
        self.rcor = np.where(self.capped & np.isfinite(self.cbase) & (self.cbase > 0), self.araw / self.cbase, 1.0)
        for c in ('a_H', 'a_H+bgfree', 'a_H+h0', 'a_H+h0+bgfree', 'cap_H', 'cap_H+h0'):
            setattr(self, c.replace('+', '_'), col(c))
        self.capH_pipe = np.array([r.get('cap_H_pipe', np.nan) for r in self.rows], float)
        for tag in ('g10','g20','g30','g50','g70','f10','f20','f30','f50'):
            setattr(self, 'a_' + tag, np.array([r.get('a_' + tag, np.nan) for r in self.rows], float))
            setattr(self, 'n_' + tag, np.array([r.get('n_' + tag, 0) for r in self.rows], float))
        n = self.n
        self.star_det = None
        # score7 baseline
        s_cap = self.med_per_star(-2.5 * np.log10(self.acat / self.araw))
        self.s_cap = s_cap
        self.dm_unc = self.dm_final + s_cap
        self.have0 = np.isfinite(s_cap)
        self.unsat_dm = m['unsat_dm']

    def med_per_star(self, x):
        n = self.n
        out = np.full(n, np.nan)
        i, j = self.i, self.j
        msk = self.good[j] & np.isfinite(x[j])
        ii, xx = i[msk], x[j][msk]
        order = np.argsort(ii, kind='stable')
        ii, xx = ii[order], xx[order]
        u_, st = np.unique(ii, return_index=True)
        for k, s, e in zip(u_, st, list(st[1:]) + [len(ii)]):
            out[k] = np.median(xx[s:e])
        return out

    def dm_of(self, a_eff):
        with np.errstate(invalid='ignore', divide='ignore'):
            return self.dm_unc + self.med_per_star(-2.5 * np.log10(a_eff / self.abase))

    def dm_capped(self, a, cap_raw):
        """cap_raw: per-row cap (NaN = no cap) before the rcor correction, as score7.cap_of."""
        c = cap_raw * self.rcor
        c = np.where(np.isfinite(c), c, np.inf)
        return self.dm_of(np.minimum(a, c))

    # ---- per-row cap helpers ----
    def src_frac(self, r, k):
        g = r['reg']
        return g['src'] / np.where(g['repl'], self.FW_ff[k], self.FW_g0[k])

    def cap_rows(self, fn):
        """fn(row, rowindex) -> cap (NaN = skipped)"""
        out = np.full(self.nrow, np.nan)
        for k, r in enumerate(self.rows):
            if r['label'] > 0:
                out[k] = fn(r, k)
        return out

    def base_cap(self, cutname='cutH'):
        return self.cap_rows(lambda r, k: cap_arrays(r['reg'][cutname], r['reg']['psf'], r['reg']['ur'], r['pkidx'], r['ppk']))

    def rule_i(self, f, gate, cutname='cutH'):
        def fn(r, k):
            g = r['reg']
            ex = self.src_frac(r, k) > f
            return cap_arrays(g[cutname], g['psf'], g['ur'], r['pkidx'], r['ppk'], excl=ex, excl_gate=gate)
        return self.cap_rows(fn)

    def rule_ii(self, f, cutname='cutH'):
        def fn(r, k):
            g = r['reg']
            c, b = cap_arrays(g[cutname], g['psf'], g['ur'], r['pkidx'], r['ppk'], return_bind=True)
            if b >= 0 and self.src_frac(r, k)[b] > f:
                return np.nan
            return c
        return self.cap_rows(fn)


def mad(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))
