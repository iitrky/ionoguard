"""Online risk control (IONOGUARD-OL, Theorem 1): paired runs with IONOGUARD on every test night and on the
calibration-shift nights. Usage: OMP_NUM_THREADS=1 python run_ol.py <worker_id> [max_jobs]"""
import sys, os, json, pickle, time, numpy as np
from _common import RES
import sim, run
sim.FADE = sim.build_fade_table()
SCEN = {'S1': ('strong', 0.8e9, 1000), 'S2': ('strong', 1.99e9, 2000), 'S3': ('moderate', 0.8e9, 3000)}
OOD = {'S3toS1': ('S3', 'S1'), 'S1toS2': ('S1', 'S2')}
OUT = os.path.join(RES, 'ol'); os.makedirs(OUT, exist_ok=True)
LOCK = os.path.join(RES, 'locks'); os.makedirs(LOCK, exist_ok=True)
JOBS = [(sc, s) for s in range(10) for sc in SCEN] + [(k, s) for k in OOD for s in range(3)]
w = sys.argv[1]; maxj = int(sys.argv[2]) if len(sys.argv) > 2 else 10 ** 6
for job in JOBS:
    if maxj <= 0:
        break
    out = os.path.join(OUT, f'{job[0]}_night{job[1]}.json')
    lk = os.path.join(LOCK, 'ol_' + '_'.join(map(str, job)))
    if os.path.exists(out):
        continue
    try:
        os.close(os.open(lk, os.O_CREAT | os.O_EXCL))
    except FileExistsError:
        continue
    t0 = time.time()
    src, dst = (OOD[job[0]] if job[0] in OOD else (job[0], job[0]))
    inten, f, b = SCEN[dst]
    mdl = pickle.load(open(os.path.join(RES, f'models_{src}.pkl'), 'rb'))
    res = run.run_night(run.Night(b + 101 + job[1], inten, f), mdl, policies=['IONOGUARD', 'IONOGUARD-OL'])
    res = {p: {k: (float(v) if np.isscalar(v) else v) for k, v in d.items()} for p, d in res.items()}
    json.dump(res, open(out + '.tmp', 'w')); os.replace(out + '.tmp', out)
    print(w, 'done', job, round(time.time() - t0), 's', flush=True); maxj -= 1
print('WORKER', w, 'FINISHED', flush=True)
