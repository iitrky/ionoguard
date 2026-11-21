"""Aggregate the extended evaluation into results/summary.json (10 nights), results/mu_summary.json,
results/ood_summary.json and results/stats.json (95% confidence intervals and paired Wilcoxon tests)."""
import json, glob, os, numpy as np
from scipy import stats
from _common import RES

KEYS = ('gp', 'bler', 'dbler', 'defer', 'ho', 'wviol', 'dgp', 'dfrac')
MUK = ('gp', 'gp_min', 'jain', 'bler', 'dbler', 'defer', 'blocked', 'ho', 'dgp', 'dfrac')


def ci(x):
    x = np.asarray(x, float); n = len(x)
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n) if n > 1 else float('nan')
    return [float(x.mean()), float(x.std(ddof=1)) if n > 1 else 0.0, float(h)]


def wilcox(a, b):
    """Paired two-sided Wilcoxon signed-rank test (exact for small n) and number of nights a > b."""
    a, b = np.asarray(a), np.asarray(b)
    if np.allclose(a, b):
        return {'p': 1.0, 'wins': 0, 'n': len(a)}
    return {'p': float(stats.wilcoxon(a, b).pvalue), 'wins': int((a > b).sum()), 'n': len(a)}


def load(pattern, key):
    import re
    fs = sorted(glob.glob(pattern), key=lambda f: int(re.findall(r'\d+', os.path.basename(f).split(key, 1)[1])[0]))
    return [json.load(open(f)) for f in fs]


S, ST = {}, {}
for sc in ('S1', 'S2', 'S3'):
    D = load(os.path.join(RES, 'nights', f'{sc}_night*.json'), 'night')
    if not D:
        continue
    S[sc] = {'n': len(D)}
    for p in D[0]:
        S[sc][p] = {k: ci([d[p][k] for d in D]) for k in KEYS}
        S[sc][p]['nights_gp'] = [d[p]['gp'] for d in D]
        S[sc][p]['nights_bler'] = [d[p]['bler'] for d in D]
    ST[sc] = {q: wilcox([d['IONOGUARD']['gp'] for d in D], [d[q]['gp'] for d in D]) for q in D[0] if q != 'IONOGUARD'}
    ST[sc]['bler_below_eps'] = int(sum(d['IONOGUARD']['bler'] <= 0.05 for d in D))
    ST[sc]['dbler_below_eps'] = int(sum(d['IONOGUARD']['dbler'] <= 0.05 for d in D))
    B = load(os.path.join(RES, 'mu', f'su_{sc}_*.json'), f'su_{sc}_')
    if B:
        S[sc]['bound'] = ci([b['bound'] for b in B])
        S[sc]['bound_n'] = len(B)
        S[sc]['mindef_max'] = float(max(b['mindef_all'] for b in B))
        k = min(len(B), len(D))
        S[sc]['ratio'] = {p: ci([D[i][p]['gp'] / B[i]['bound'] for i in range(k)]) for p in D[0]}
json.dump(S, open(os.path.join(RES, 'summary.json'), 'w'), indent=1)

M = {}
for K in (1, 2, 4):
    D = load(os.path.join(RES, 'mu', f'mu_*_K{K}.json'), 'mu_')
    if not D:
        continue
    pols = [p for p in D[0] if p not in ('bound', 'bound_nocap')]
    M[K] = {'n': len(D), 'bound': ci([d['bound']['gp'] for d in D]), 'bound_nocap': ci([d['bound_nocap']['gp'] for d in D])}
    for p in pols:
        M[K][p] = {k: ci([d[p][k] for d in D]) for k in MUK}
        M[K][p]['gp_users'] = np.mean([d[p]['gp_users'] for d in D], 0).tolist()
        M[K][p]['nights_gp'] = [d[p]['gp'] for d in D]
        M[K][p]['ratio'] = ci([d[p]['gp'] / d['bound']['gp'] for d in D])
    if len(D) > 1:
        M[K]['tests'] = {f'{a}>{b}:{k}': wilcox([d[a][k] for d in D], [d[b][k] for d in D]) | {'metric': k}
                         for a, b, k in (('IONOGUARD-MU', 'IG-Indep', 'gp'), ('IONOGUARD-MU', 'A3-OLLA', 'gp'),
                                         ('IONOGUARD-PF', 'IG-Indep', 'gp'), ('IONOGUARD-PF', 'A3-OLLA', 'gp'),
                                         ('IONOGUARD-PF', 'IONOGUARD-MU', 'gp_min'), ('IONOGUARD-PF', 'IONOGUARD-MU', 'jain'),
                                         ('IONOGUARD-PF', 'IG-Indep', 'gp_min'))}
json.dump(M, open(os.path.join(RES, 'mu_summary.json'), 'w'), indent=1)

O = {}
for name in ('S3toS1', 'S1toS2'):
    D = load(os.path.join(RES, 'ood', f'{name}_night*.json'), 'night')
    if D:
        O[name] = {'n': len(D)}
        for p in D[0]:
            O[name][p] = {k: ci([d[p][k] for d in D]) for k in KEYS}
json.dump(O, open(os.path.join(RES, 'ood_summary.json'), 'w'), indent=1)
json.dump(ST, open(os.path.join(RES, 'stats.json'), 'w'), indent=1)
print('extended summary written:', {k: v['n'] for k, v in S.items()}, {k: v['n'] for k, v in M.items()}, {k: v['n'] for k, v in O.items()})
