"""Online risk control in the multi-user scenario (K = 2): IONOGUARD-MU and -PF with and without the online layer,
paired on the ten multi-user test nights. Usage: OMP_NUM_THREADS=1 python run_mu_ol.py <worker_id> [max_jobs]"""
import sys, os, json, pickle, time
from _common import RES
import sim, mu
sim.FADE = sim.build_fade_table()
OUT = os.path.join(RES, 'mu_ol'); os.makedirs(OUT, exist_ok=True)
LOCK = os.path.join(RES, 'locks'); os.makedirs(LOCK, exist_ok=True)
POL = ['IONOGUARD-MU', 'IONOGUARD-PF', 'IONOGUARD-MU-OL', 'IONOGUARD-PF-OL']
w = sys.argv[1]; maxj = int(sys.argv[2]) if len(sys.argv) > 2 else 10 ** 6
m = pickle.load(open(os.path.join(RES, 'models_S1.pkl'), 'rb'))
for seed in range(5101, 5111):
    if maxj <= 0:
        break
    out = os.path.join(OUT, f'mu_{seed}.json'); lk = os.path.join(LOCK, f'muol_{seed}')
    if os.path.exists(out):
        continue
    try:
        os.close(os.open(lk, os.O_CREAT | os.O_EXCL))
    except FileExistsError:
        continue
    t0 = time.time()
    res = mu.run_multi(mu.MultiNight(seed), m, cap=2, policies=POL, seed=seed)
    json.dump(res, open(out + '.tmp', 'w')); os.replace(out + '.tmp', out)
    print(w, 'done', seed, round(time.time() - t0), 's', flush=True); maxj -= 1
print('WORKER', w, 'FINISHED', flush=True)
