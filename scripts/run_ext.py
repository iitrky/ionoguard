"""Extended evaluation (revision 3): 10 held-out test nights per scenario, multi-user runs with the
proportional-fair variant, capacity sweep, single-UE bounds for every test night and
out-of-distribution (calibration-shift) tests.

Usage: OMP_NUM_THREADS=1 python run_ext.py <worker_id>
Several workers can run in parallel; jobs are claimed through lock files and skipped when the
output already exists, so the script can be restarted at any time."""
import sys, os, json, pickle, time, numpy as np
from _common import RES
import sim, run, train, mu
sim.FADE = sim.build_fade_table()

NN = 10                                   # test nights per scenario
SCEN = {'S1': ('strong', 0.8e9, 1000), 'S2': ('strong', 1.99e9, 2000), 'S3': ('moderate', 0.8e9, 3000)}
OOD = {'S3toS1': ('S3', 'S1'), 'S1toS2': ('S1', 'S2')}     # (models from, deployed in)
LOCK = os.path.join(RES, 'locks'); os.makedirs(LOCK, exist_ok=True)
for d in ('nights', 'mu', 'ood'):
    os.makedirs(os.path.join(RES, d), exist_ok=True)

JOBS = [('train', 'S2'), ('train', 'S3')]
JOBS += [('su', 'S1', s) for s in range(NN)]
JOBS += [('mu', 5101 + s, 2) for s in range(NN)]
JOBS += [('su', sc, s) for sc in ('S2', 'S3') for s in range(NN)]
JOBS += [('mu', 5101, 1), ('mu', 5101, 4)]
JOBS += [('ood', k, s) for k in OOD for s in range(3)]
JOBS += [('bound', sc, s) for sc in SCEN for s in range(NN)]


def out_of(job):
    k = job[0]
    if k == 'train':
        return os.path.join(RES, f'models_{job[1]}.pkl')
    if k == 'su':
        return os.path.join(RES, 'nights', f'{job[1]}_night{job[2]}.json')
    if k == 'mu':
        return os.path.join(RES, 'mu', f'mu_{job[1]}_K{job[2]}.json')
    if k == 'ood':
        return os.path.join(RES, 'ood', f'{job[1]}_night{job[2]}.json')
    return os.path.join(RES, 'mu', f'su_{job[1]}_{job[2]}.json')


def models(sc):
    p = os.path.join(RES, f'models_{sc}.pkl')
    while not os.path.exists(p):
        time.sleep(20)
    time.sleep(2)
    return pickle.load(open(p, 'rb'))


def tofloat(res):
    return {p: {k: (float(v) if np.isscalar(v) else v) for k, v in d.items()} for p, d in res.items()}


def do(job):
    k = job[0]
    if k == 'train':
        inten, f, b = SCEN[job[1]]
        return train.build_models([b + 1 + i for i in range(3)], [b + 11 + i for i in range(3)], inten, f)
    if k == 'su':
        inten, f, b = SCEN[job[1]]
        return tofloat(run.run_night(run.Night(b + 101 + job[2], inten, f), models(job[1])))
    if k == 'mu':
        _, seed, cap = job
        n = mu.MultiNight(seed)
        res = mu.run_multi(n, models('S1'), cap=cap, seed=seed)
        res['bound'] = {'gp': mu.lagrangian_bound(n.geo, n.T, 0.05, cap=cap)['bound']}
        res['bound_nocap'] = {'gp': mu.lagrangian_bound(n.geo, n.T, 0.05, cap=None)['bound']}
        return res
    if k == 'ood':
        src, dst = OOD[job[1]]
        inten, f, b = SCEN[dst]
        return tofloat(run.run_night(run.Night(b + 101 + job[2], inten, f), models(src)))
    _, sc, s = job
    inten, f, b = SCEN[sc]
    n = run.Night(b + 101 + s, inten, f)
    bd = mu.lagrangian_bound([n.geo], n.T, 0.05)
    dmask = np.array([g['s4'].max() > 0.4 if len(g['ids']) else False for g in n.geo])
    return {'bound': bd['bound'], 'mu': bd['mu'], 'curve': bd['curve'], 'dfrac': float(dmask.mean()),
            'mindef_all': mu.min_deferral([n.geo], n.T, 0.05)[0]}


if __name__ == '__main__':
    w = sys.argv[1]; maxj = int(sys.argv[2]) if len(sys.argv) > 2 else 10 ** 6
    kinds = sys.argv[3].split(',') if len(sys.argv) > 3 else None
    for job in JOBS:
        if maxj <= 0:
            break
        if kinds and job[0] not in kinds:
            continue
        out = out_of(job)
        if os.path.exists(out):
            continue
        lk = os.path.join(LOCK, '_'.join(map(str, job)))
        try:
            os.close(os.open(lk, os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            continue
        t0 = time.time(); print(w, 'start', job, time.strftime('%H:%M:%S'), flush=True)
        res = do(job)
        if job[0] == 'train':
            pickle.dump(res, open(out + '.tmp', 'wb'))
        else:
            json.dump(res, open(out + '.tmp', 'w'))
        os.replace(out + '.tmp', out)
        print(w, 'done', job, round(time.time() - t0), 's', flush=True)
        maxj -= 1
    print('WORKER', w, 'FINISHED', flush=True)
