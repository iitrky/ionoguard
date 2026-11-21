"""Summary of the online risk-control runs (Theorem 1) -> results/ol_summary.json.
For every night: goodput and block failure rates of IONOGUARD and IONOGUARD-OL (paired), the per-group
failure rates and the certificate eps' + B*delta_h(T+1)/(gamma*N_h) of Theorem 1."""
import json, glob, os, re, numpy as np
from scipy import stats
from _common import RES
import run
from sim import SUB
EPSP = 0.05 - run.OL_MARGIN
O = {}
for name in ('S1', 'S2', 'S3', 'S3toS1', 'S1toS2'):
    fs = sorted(glob.glob(os.path.join(RES, 'ol', f'{name}_night*.json')), key=lambda f: int(re.findall(r'night(\d+)', f)[0]))
    if not fs:
        continue
    N = []
    for f in fs:
        d = json.load(open(f)); a, b = d['IONOGUARD'], d['IONOGUARD-OL']
        att, fail, dend = np.array(b['grp_att']), np.array(b['grp_fail']), np.array(b['d_end'])
        cert = [float(EPSP + SUB * dend[h] / (run.OL_GAMMA * att[h])) if att[h] else None for h in range(2)]
        N.append(dict(gp=a['gp'], gp_ol=b['gp'], bler=a['bler'], bler_ol=b['bler'], dbler=a['dbler'], dbler_ol=b['dbler'],
                      grp_bfr=[float(fail[h] / att[h]) if att[h] else None for h in range(2)], cert=cert,
                      d_max=b['d_max'], d_end=b['d_end'], defer=a['defer'], defer_ol=b['defer']))
    k = lambda key: np.array([n[key] for n in N])
    O[name] = dict(n=len(N), nights=N,
                   gp=float(k('gp').mean()), gp_ol=float(k('gp_ol').mean()), loss=float(100 * (1 - k('gp_ol').mean() / k('gp').mean())),
                   max_bler=float(k('bler').max()), max_bler_ol=float(k('bler_ol').max()),
                   max_dbler=float(k('dbler').max()), max_dbler_ol=float(k('dbler_ol').max()),
                   over_dbler=int((k('dbler') > 0.05).sum()), over_dbler_ol=int((k('dbler_ol') > 0.05).sum()),
                   over_bler=int((k('bler') > 0.05).sum()), over_bler_ol=int((k('bler_ol') > 0.05).sum()),
                   max_grp_ol=float(max(max(x for x in n['grp_bfr'] if x is not None) for n in N)),
                   max_cert=float(max(max(x for x in n['cert'] if x is not None) for n in N)),
                   active=int(sum(max(n['d_max']) > 0.05 for n in N)),
                   p_gp=float(stats.wilcoxon(k('gp'), k('gp_ol')).pvalue) if len(N) > 1 and not np.allclose(k('gp'), k('gp_ol')) else 1.0)
json.dump(O, open(os.path.join(RES, 'ol_summary.json'), 'w'), indent=1)
for n, v in O.items():
    print(n, {k: (round(x, 4) if isinstance(x, float) else x) for k, x in v.items() if k != 'nights'})

# ---------------- multi-user (K = 2): online layer on top of IONOGUARD-MU and IONOGUARD-PF
fs = sorted(glob.glob(os.path.join(RES, 'mu_ol', 'mu_*.json')))
if fs:
    D = [json.load(open(f)) for f in fs]
    MU = {'n': len(D), 'seeds': [int(re.findall(r'mu_(\d+)', f)[0]) for f in fs]}
    for base in ('IONOGUARD-MU', 'IONOGUARD-PF'):
        ol = base + '-OL'
        a = np.array([d[base]['gp'] for d in D]); b = np.array([d[ol]['gp'] for d in D])
        cert = [max(EPSP + SUB * d[ol]['d_end'][h] / (run.OL_GAMMA * d[ol]['grp_att'][h]) for h in range(2) if d[ol]['grp_att'][h])
                for d in D]
        MU[base] = dict(gp=float(a.mean()), gp_ol=float(b.mean()), loss=float(100 * (1 - b.mean() / a.mean())),
                        p_gp=float(stats.wilcoxon(a, b).pvalue) if not np.allclose(a, b) else 1.0,
                        max_bler=max(d[base]['bler'] for d in D), max_bler_ol=max(d[ol]['bler'] for d in D),
                        max_dbler=max(d[base]['dbler'] for d in D), max_dbler_ol=max(d[ol]['dbler'] for d in D),
                        over_dbler=int(sum(d[base]['dbler'] > 0.05 for d in D)), over_dbler_ol=int(sum(d[ol]['dbler'] > 0.05 for d in D)),
                        gp_min=float(np.mean([d[base]['gp_min'] for d in D])), gp_min_ol=float(np.mean([d[ol]['gp_min'] for d in D])),
                        jain=float(np.mean([d[base]['jain'] for d in D])), jain_ol=float(np.mean([d[ol]['jain'] for d in D])),
                        max_cert=float(max(cert)))
    O['MU'] = MU
    json.dump(O, open(os.path.join(RES, 'ol_summary.json'), 'w'), indent=1)
    print('MU', json.dumps({k: v for k, v in MU.items()}, indent=0)[:1500])
