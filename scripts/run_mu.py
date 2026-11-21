"""Multi-user experiments and theoretical bounds.
Usage: python run_mu.py <worker> <n_workers>"""
import sys, os, json, pickle, time, numpy as np
from _common import RES
import sim, run, mu
sim.FADE = sim.build_fade_table()
OUT = os.path.join(RES, 'mu'); os.makedirs(OUT, exist_ok=True)
models = pickle.load(open(os.path.join(RES, 'models_S1.pkl'), 'rb'))
JOBS = [('mu', 5101 + s, 2) for s in range(3)] + [('mu', 5101, 1), ('mu', 5101, 4)]
JOBS += [('su_bound', sc, s) for sc in ('S1', 'S2', 'S3') for s in range(3)]
SC = {'S1': ('strong', 0.8e9, 1101), 'S2': ('strong', 1.99e9, 2101), 'S3': ('moderate', 0.8e9, 3101)}
w, nw = int(sys.argv[1]), int(sys.argv[2])
for i, job in enumerate(JOBS):
    if i % nw != w:
        continue
    t0 = time.time()
    if job[0] == 'mu':
        _, seed, cap = job
        out = os.path.join(OUT, f'mu_{seed}_K{cap}.json')
        if os.path.exists(out):
            continue
        n = mu.MultiNight(seed)
        res = mu.run_multi(n, models, cap=cap, seed=seed)
        b = mu.lagrangian_bound(n.geo, n.T, 0.05, cap=cap)
        res['bound'] = {'gp': b['bound'], 'mu': b['mu']}
        res['bound_nocap'] = {'gp': mu.lagrangian_bound(n.geo, n.T, 0.05, cap=None)['bound']}
        json.dump(res, open(out, 'w'))
    else:
        _, sc, s = job
        out = os.path.join(OUT, f'su_{sc}_{s}.json')
        if os.path.exists(out):
            continue
        inten, f, base = SC[sc]
        n = run.Night(base + s, inten, f)
        dmask = np.array([g['s4'].max() > 0.4 if len(g['ids']) else False for g in n.geo])
        b = mu.lagrangian_bound([n.geo], n.T, 0.05)
        fd = dmask.mean()
        bq = mu.lagrangian_bound([n.geo], n.T, 0.05, mask=dmask) if dmask.any() else {'bound': float('nan')}
        bn = mu.lagrangian_bound([n.geo], n.T, 0.05, mask=~dmask) if (~dmask).any() else {'bound': float('nan')}
        md_all = mu.min_deferral([n.geo], n.T, 0.05)[0]
        md_dist = mu.min_deferral([n.geo], n.T, 0.05, mask=dmask)[0] if dmask.any() else 0.0
        json.dump({'bound': b['bound'], 'bound_regime': float((fd * bq['bound'] if fd > 0 else 0.0) + ((1 - fd) * bn['bound'] if fd < 1 else 0.0)),
                   'bound_dist': bq['bound'], 'dfrac': float(fd), 'mindef_all': md_all, 'mindef_dist': md_dist,
                   'curve': b['curve']}, open(out, 'w'))
    print(job, 'done', round(time.time() - t0), flush=True)
print('WORKER', w, 'FINISHED', flush=True)
