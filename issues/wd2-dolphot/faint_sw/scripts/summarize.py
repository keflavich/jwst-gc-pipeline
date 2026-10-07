"""Summaries from faint_sw.py output.  usage: python summarize.py A B  -> tables_{A}_{B}.md"""
import sys
import numpy as np
from astropy.table import Table
from common import *
A_, B_ = sys.argv[1:3]
S = Table.read(f'{FS}/stars_{A_}_{B_}.ecsv'); F = Table.read(f'{FS}/frames_{A_}_{B_}.ecsv')
L = [f'# Faint SW movers {B_} vs {A_} (dolphot mag > 20, moved = |B-A| > 0.1 mag)']
def med(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return np.median(x) if len(x) else np.nan
L += ['', '## Q1 sign and closeness to dolphot', f'| band | moved | median dolphot | {B_} brighter (B-A<0) | {B_} fainter | median B-A | closer to dolphot: B / A, B brighter | B / A, B fainter | median abs(dm) A -> B (all moved) |', '|---|---|---|---|---|---|---|---|---|']
for b in BANDS4:
    m = (S['band'] == b) & S['moved']
    br = m & (S['dBA'] < 0); fa = m & (S['dBA'] > 0)
    cl = np.abs(S['dmB']) < np.abs(S['dmA'])
    L.append(f'| F{b} | {m.sum()} | {med(S["ref"][m]):.1f} | {br.sum()} | {fa.sum()} | {med(S["dBA"][m]):+.2f} | {(br & cl).sum()} / {(br & ~cl).sum()} | {(fa & cl).sum()} / {(fa & ~cl).sum()} | {med(np.abs(S["dmA"][m])):.3f} -> {med(np.abs(S["dmB"][m])):.3f} |')
L += ['', '## Q1b signed dm to dolphot (median), moved stars', '| band | median dmA | median dmB | median dmA, B brighter | median dmB, B brighter | median dmA, B fainter | median dmB, B fainter |', '|---|---|---|---|---|---|---|']
for b in BANDS4:
    m = (S['band'] == b) & S['moved']; br = m & (S['dBA'] < 0); fa = m & (S['dBA'] > 0)
    L.append(f'| F{b} | {med(S["dmA"][m]):+.2f} | {med(S["dmB"][m]):+.2f} | {med(S["dmA"][br]):+.2f} | {med(S["dmB"][br]):+.2f} | {med(S["dmA"][fa]):+.2f} | {med(S["dmB"][fa]):+.2f} |')
L += ['', '## Q2 distance to nearest satstar (same-band per-frame m7 satstar catalogs, A and B pooled; arcsec)',
      '| band | group | N | median dsat | frac < 2" | frac < 5" | frac < 10" | median dsat A only | median dsat B only |', '|---|---|---|---|---|---|---|---|---|']
for b in BANDS4:
    for nm, m in (('moved', S['moved']), ('control', ~S['moved'])):
        s = (S['band'] == b) & m
        d = S['dsat'][s]
        L.append(f'| F{b} | {nm} | {s.sum()} | {np.median(d):.1f} | {(d < 2).mean():.3f} | {(d < 5).mean():.3f} | {(d < 10).mean():.3f} | {np.median(S["dsatA"][s]):.1f} | {np.median(S["dsatB"][s]):.1f} |')
L += ['', '## Q3 detections on collapsed frames (B/A per-frame satstar count > 2); A m7 daophot rows within 0.1"',
      '| band | group | stars | A detections | on collapsed frames | frac | stars with any collapsed detection | stars with all detections collapsed |', '|---|---|---|---|---|---|---|---|']
cf = {}
for b in BANDS4:
    f = F[F['band'] == b]
    for nm, flag in (('moved', True), ('control', False)):
        ids = S['i'][(S['band'] == b) & (S['moved'] == flag)]
        g = f[np.isin(f['i'], ids)]
        per = {}
        for i, c in zip(g['i'], g['collapsed']):
            per.setdefault(i, []).append(bool(c))
        L.append(f'| F{b} | {nm} | {len(ids)} | {len(g)} | {int(g["collapsed"].sum())} | {g["collapsed"].mean():.3f} | {sum(any(v) for v in per.values())} | {sum(all(v) for v in per.values())} |')
nfr = {}
L += ['', 'frames per band: (det exp nsatA nsatB collapsed)']
for b in BANDS4:
    f = F[F['band'] == b]
    seen = {}
    for r in f:
        seen[(r['det'], r['exp'])] = (r['nsatA'], r['nsatB'], bool(r['collapsed']))
    ncol = sum(v[2] for v in seen.values())
    L.append(f'- F{b}: {len(seen)} frames with detections, {ncol} collapsed: ' + '; '.join(f'{k[0]}e{k[1]} {v[0]}/{v[1]}' for k, v in sorted(seen.items()) if v[2]))
L += ['', '## Q4 B model - A model at the star position (7x7 PSF-weighted, Gaussian PSF, counts in image units) vs star flux',
      'Dpsf = sum(D*P)/sum(P^2) with P a normalized Gaussian of the nominal band FWHM; predicted fB = fA - Dpsf (B residual = data - Bmodel).',
      '| band | group | star-frames | median Dpsf/fA | frac Dpsf/fA < -0.1 | frac > +0.1 | frac abs < 0.02 | median abs(Dpsf/fA) | median (fB-fA)/fA observed (B row found) | median predicted (-Dpsf/fA) same rows |', '|---|---|---|---|---|---|---|---|---|---|']
mvset = {(r['band'], int(r['i'])): bool(r['moved']) for r in S}
F['moved'] = [mvset[(b, int(i))] for b, i in zip(F['band'], F['i'])]
F['rel'] = F['Dpsf'] / F['fA']
F['obs'] = (F['fB'] - F['fA']) / F['fA']
for b in BANDS4:
    for nm, flag in (('moved', True), ('control', False)):
        g = F[(F['band'] == b) & (F['moved'] == flag) & (F['fA'] > 0)]
        r = g['rel']; both = np.isfinite(g['fB'])
        L.append(f'| F{b} | {nm} | {len(g)} | {np.median(r):+.3f} | {(r < -0.1).mean():.3f} | {(r > 0.1).mean():.3f} | {(np.abs(r) < 0.02).mean():.3f} | {np.median(np.abs(r)):.3f} | {med(g["obs"][both]):+.3f} | {med(-r[both]):+.3f} |')
# per-star prediction vs observed m8 shift
L += ['', '## Q4b per-star predicted vs observed B-A (mag); predicted = -2.5 log10(sum(fA - Dpsf) / sum(fA)) over A-detected frames',
      '| band | moved stars with frames | Spearman rho(pred, observed) | sign agreement | median observed | median predicted | median abs(pred) / abs(obs) | fraction |pred| > 0.05 |', '|---|---|---|---|---|---|---|---|']
from scipy.stats import spearmanr
pred_all = {}
for b in BANDS4:
    sm = S[(S['band'] == b) & S['moved']]
    pr, ob = [], []
    for r in sm:
        g = F[(F['band'] == b) & (F['i'] == r['i']) & (F['fA'] > 0)]
        if not len(g):
            continue
        x = (g['fA'] - g['Dpsf']).sum() / g['fA'].sum()
        p = -2.5 * np.log10(x) if x > 0 else np.nan
        pr.append(p); ob.append(r['dBA']); pred_all[(b, int(r['i']))] = p
    pr, ob = np.array(pr), np.array(ob); k = np.isfinite(pr)
    rho = spearmanr(pr[k], ob[k])[0]
    L.append(f'| F{b} | {k.sum()} | {rho:.2f} | {(np.sign(pr[k]) == np.sign(ob[k])).mean():.2f} | {np.median(ob[k]):+.2f} | {np.median(pr[k]):+.3f} | {np.median(np.abs(pr[k])) / np.median(np.abs(ob[k])):.2f} | {(np.abs(pr[k]) > 0.05).mean():.2f} |')
L += ['', '## Q5 other properties (medians; moved vs control)', '| band | group | N | m8 nmatch A | nmatch B | qfit A | qfit B | spike_artifact A / B | near_saturated A / B | forced_filled A / B | neighbours within 1" (m8, A) | satstar_nframes A / B | rep.sat A / B |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
def fr(x):
    x = np.asarray(x, float); return np.nanmean(x) if np.isfinite(x).any() else np.nan
for b in BANDS4:
    lo = 'f' + b.lower()
    for nm, flag in (('moved', True), ('control', False)):
        s = S[(S['band'] == b) & (S['moved'] == flag)]
        def c(n):
            return s[n] if n in s.colnames else np.full(len(s), np.nan)
        L.append(f'| F{b} | {nm} | {len(s)} | {med(c("nmatch_A")):.0f} | {med(c("nmatch_B")):.0f} | {med(c("qfit_A")):.2f} | {med(c("qfit_B")):.2f} | {fr(c("spike_A")):.3f} / {fr(c("spike_B")):.3f} | '
                 f'{fr(c(f"near_saturated_{lo}_{lo}_A")):.3f} / {fr(c(f"near_saturated_{lo}_{lo}_B")):.3f} | {fr(c("forced_filled_A")):.3f} / {fr(c("forced_filled_B")):.3f} | {med(c("nneigh1")):.0f} | '
                 f'{med(c("satstar_nframes_A")):.0f} / {med(c("satstar_nframes_B")):.0f} | {fr(c("rep_A")):.3f} / {fr(c("rep_B")):.3f} |')
L += ['', '## Q5b change in contributing frames (nmatch B - nmatch A), moved vs control', '| band | group | median | frac B < A | frac B > A |', '|---|---|---|---|---|']
for b in BANDS4:
    for nm, flag in (('moved', True), ('control', False)):
        s = S[(S['band'] == b) & (S['moved'] == flag)]
        dn = np.asarray(s['nmatch_B'], float) - np.asarray(s['nmatch_A'], float)
        L.append(f'| F{b} | {nm} | {np.nanmedian(dn):+.0f} | {np.nanmean(dn < 0):.3f} | {np.nanmean(dn > 0):.3f} |')
open(f'{FS}/tables_{A_}_{B_}.md', 'w').write('\n'.join(L))
print('\n'.join(L))
