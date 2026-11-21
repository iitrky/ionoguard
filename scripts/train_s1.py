import pickle, os, time
from _common import RES
import sim, train
sim.FADE = sim.build_fade_table()
t0 = time.time()
m = train.build_models([1001, 1002, 1003], [1011, 1012, 1013], 'strong', 0.8e9)
pickle.dump(m, open(os.path.join(RES, 'models_S1.pkl'), 'wb'))
print('S1 models done', round(time.time() - t0), flush=True)
