"""Quick end-to-end check (about 2-3 minutes on one core): single-UE controller with and without the online
layer (Theorem 1), multi-user joint assignment, and the Lagrangian bound (Proposition 3), on shortened nights.
Uses results/models_S1.pkl if present, otherwise trains a small model first.
Usage: OMP_NUM_THREADS=1 python smoke_test.py"""
import os, time, pickle
from _common import RES
import sim, run, train, mu
sim.FADE = sim.build_fade_table()
t0 = time.time()
p = os.path.join(RES, 'models_S1.pkl')
if os.path.exists(p):
    models = pickle.load(open(p, 'rb')); print('loaded', p)
else:
    print('models_S1.pkl not found: training a small model on shortened nights')
    models = train.build_models([1001], [1011], 'strong', 0.8e9, T=5400)
T = 3600   # 19:00-20:00, the onset of bubble activity
r = run.run_night(run.Night(1101, 'strong', 0.8e9, T=T), models, policies=['A3-OLLA', 'IONOGUARD', 'IONOGUARD-OL'])
print('\nSingle UE, S1, first hour of test night 1101')
for k, d in r.items():
    print(f"  {k:13s} goodput {d['gp']:.2f} Mbit/s  BFR {100 * d['bler']:.2f}%  deferral {100 * d['defer']:.1f}%")
n = mu.MultiNight(5101, T=1200)
r = mu.run_multi(n, models, cap=2, policies=['IG-Indep', 'IONOGUARD-MU', 'IONOGUARD-PF'], seed=5101)
print('\nMulti-user (8 UEs, K = 2), first 20 min of test night 5101')
for k, d in r.items():
    print(f"  {k:13s} goodput/UE {d['gp']:.2f} Mbit/s  worst UE {d['gp_min']:.2f}  Jain {d['jain']:.3f}  BFR {100 * d['bler']:.2f}%")
b = mu.lagrangian_bound(n.geo, n.T, 0.05, cap=2)['bound']
print(f'  Lagrangian goodput bound (Proposition 3): {b:.2f} Mbit/s per UE')
assert all(d['bler'] <= 0.10 for d in r.values()), 'unexpectedly high failure rate'
print(f'\nSmoke test passed in {time.time() - t0:.0f} s')
