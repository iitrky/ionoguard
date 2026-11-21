"""Multi-user extension of IONOGUARD.

Several handsets in South India share the same LEO constellation. Each satellite
beam can serve at most K users per 1-s epoch (per-satellite capacity). Satellite
selection therefore becomes a joint, capacity-constrained assignment problem.

Contents
  MultiNight            shared constellation + bubble field, per-user link geometry
  run_multi             IONOGUARD-MU (joint b-matching) and multi-user baselines
  lagrangian_bound      Proposition 2: goodput upper bound for any eps-reliable policy
  min_deferral          Proposition 3: deferral any eps-reliable policy must incur
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
import sim
import train
from sim import MCS_SE, MCS_REQ, SUB, HO_LOSS, BW
from run import HSTEPS, GOLD

# eight handsets in Tamil Nadu, Puducherry, Andhra Pradesh and Karnataka (lat, lon)
SITES = {
    'Chennai': (13.08, 80.27), 'Kanchipuram': (12.83, 79.70), 'Vellore': (12.92, 79.13),
    'Puducherry': (11.94, 79.81), 'Tiruchirappalli': (10.79, 78.70), 'Tirupati': (13.63, 79.42),
    'Nellore': (14.44, 79.99), 'Bengaluru': (12.97, 77.59)}
REF = SITES['Chennai']
BIG = 1e6


class MultiNight:
    def __init__(self, seed, intensity='strong', f=0.8e9, s0=9.0, T=21600, M=120, latency=60, sites=None):
        self.rng = np.random.default_rng(seed)
        self.T, self.f = T, f
        self.sites = sites or SITES
        self.names = list(self.sites)
        self.Uu = len(self.names)
        const = sim.Constellation(rng=self.rng)
        self.field = sim.BubbleField(self.rng, *REF, intensity=intensity, T=T)
        self.geo = []
        for name in self.names:
            g_u = sim.link_geometry(*self.sites[name], const, np.arange(T, dtype=float))
            lens = np.array([len(g['ids']) for g in g_u])
            ilat = np.concatenate([g['ilat'] for g in g_u]); ilon = np.concatenate([g['ilon'] for g in g_u])
            x, y = self.field.xy(ilat, ilon)
            tt = np.repeat(np.arange(T), lens)
            s4v = np.empty(len(x))
            for a in range(0, len(x), 20000):
                s4v[a:a + 20000] = self.field.s4(x[a:a + 20000], y[a:a + 20000], tt[a:a + 20000])
            cosx = np.concatenate([g['cosx'] for g in g_u])
            s4s = sim.slant_s4(s4v, cosx, f)
            snr = sim.mean_snr(np.concatenate([g['rng'] for g in g_u]), np.concatenate([g['scan'] for g in g_u]), s0)
            off = np.concatenate([[0], np.cumsum(lens)])
            for k, g in enumerate(g_u):
                sl = slice(off[k], off[k + 1])
                g['x'], g['y'], g['s4v'], g['s4'], g['snr'] = x[sl], y[sl], s4v[sl], s4s[sl], snr[sl]
                g['pos'] = {int(i): j for j, i in enumerate(g['ids'])}
            self.geo.append(g_u)
        self.Ufade = self.rng.random((self.Uu, T, SUB))
        self.mon = sim.Monitors(self.field, np.random.default_rng(seed + 777), M=M, latency=latency)

    def fades(self, u, t, sid, s4):
        s4g, xg, cdf = sim.FADE
        uu = (self.Ufade[u, t] + GOLD * sid) % 1.0
        i = int(round(min(max(s4, 0), 1) * 100))
        return np.interp(uu, cdf[i], xg)


def _rows(night, u, t):
    rows = []
    for h in HSTEPS:
        tq = t + h
        if tq >= night.T:
            continue
        g = night.geo[u][tq]
        for s in night.geo[u][t]['ids']:
            j = g['pos'].get(int(s))
            if j is not None:
                rows.append((u, int(s), int(h), g['x'][j], g['y'][j], g['cosx'][j], g['snr'][j], g['s4'][j], g['s4v'][j]))
    return rows


def _score(rows_u, s4col, eps):
    """J_un = sum over look-ahead of guaranteed efficiency (Eq. 10)."""
    J = {}
    for r, s4 in zip(rows_u, s4col):
        J[r[1]] = J.get(r[1], 0.0) + sim.exp_goodput(r[6], s4, eps)
    return J


def joint_assign(J, cur, cap, kappa=0.05, kappa0=0.1):
    """Exact capacity-constrained assignment (Proposition 4).
    J: list over users of dict sat->score. cur: current sat per user.
    Hysteresis is folded in as a bonus for keeping the current satellite, so that
    an unconstrained single user reproduces the rule J_n* > (1+k)J_c + k0."""
    U = len(J)
    sats = sorted({s for d in J for s in d})
    if not sats:
        return [-1] * U
    col = {s: i for i, s in enumerate(sats)}
    W = np.full((U, len(sats) * cap), -BIG)
    for u, d in enumerate(J):
        for s, v in d.items():
            val = v * (1 + kappa) + kappa0 if s == cur[u] else v
            for k in range(cap):
                W[u, col[s] * cap + k] = val
    r, c = linear_sum_assignment(-W)
    out = [-1] * U
    for ui, ci in zip(r, c):
        if W[ui, ci] > -BIG / 2:
            out[ui] = sats[ci // cap]
    return out


def indep_assign(J, cur, cap, order, kappa=0.05, kappa0=0.1):
    """Each user applies its own selection rule independently (no coordination).
    Handover admission control as in cellular networks: a user keeps its serving
    satellite (slot reserved) and moves only if its preferred target has a free
    slot; users without a serving satellite take the best satellite with a free slot."""
    load = {}; out = [-1] * len(J)
    key = lambda u, s: J[u][s] * (1 + kappa) + kappa0 if s == cur[u] else J[u][s]
    for u in range(len(J)):
        if cur[u] in J[u]:
            out[u] = cur[u]; load[cur[u]] = load.get(cur[u], 0) + 1
    for u in order:
        if not J[u]:
            continue
        for s in sorted(J[u], key=lambda s: -key(u, s)):
            if s == out[u]:
                break                                   # preferred over current is full: stay
            if load.get(s, 0) < cap:
                if out[u] >= 0:
                    load[out[u]] -= 1
                out[u] = s; load[s] = load.get(s, 0) + 1
                break
    return out


POLICIES_MU = ['A3-OLLA', 'Risk-ACM', 'IG-Indep', 'IONOGUARD-MU', 'IONOGUARD-PF', 'Genie-MU']
PF_TAU = 300.0      # averaging window (s) of the proportional-fair weights
OL_POL = {'IONOGUARD-MU-OL': 'mu', 'IONOGUARD-PF-OL': 'pf'}   # online risk control (Theorem 1) on top of joint assignment
PF_FLOOR = 0.1      # Mbit/s, floor on the average rate in the weight


def run_multi(night, models, cap=2, policies=POLICIES_MU, eps=0.05, seed=0):
    T, U = night.T, night.Uu
    S4R2 = (2 * sim.K_RICE + 1) / (sim.K_RICE + 1) ** 2
    st = {p: dict(cur=[-1] * U, bits=np.zeros(U), att=np.zeros(U), fail=np.zeros(U), ho=np.zeros(U), defer=np.zeros(U),
                  blk=np.zeros(U), dbits=0.0, datt=0, dfail=0, ddef=0, dopp=0, olla=np.zeros(U), a3=np.zeros(U, int),
                  s4e=np.full(U, 0.3), last=[None] * U, rbar=np.ones(U),
                  old=np.zeros(2), oldmax=np.zeros(2), ga=np.zeros(2), gf=np.zeros(2)) for p in policies}
    from run import OL_GAMMA, OL_MARGIN, OL_PDIST
    mp = sim.DriftMap(use_drift=True)
    mrng = np.random.default_rng(seed + 4242)
    nb, lb = -1, None
    lam = models['lam'][('drift', 'p', eps)]
    import time as _tm; _t0 = _tm.time()
    for t in range(T):
        if t % 3600 == 0 and t:
            print(f'  epoch {t} ({_tm.time() - _t0:.0f} s)', flush=True)
        if t % night.mon.cad == 0:
            night.mon.collect(t)
        av = night.mon.available(t)
        if av and (len(av) != nb or av[-1][0] != lb):
            mp.update(av); nb, lb = len(av), av[-1][0]
        rows = [_rows(night, u, t) for u in range(U)]
        flat = [r for ru in rows for r in ru]
        R = np.array(flat) if flat else np.zeros((0, 9))
        if len(R) and av:
            F = mp.features(R[:, 3], R[:, 4], t + R[:, 2], t, t / 3600.0)
            pd_ = models['drift']['p'].predict(F)
            base = train.apply_lambda(pd_, R[:, 5], night.f, lam)
            up = sim.slant_s4(np.clip(base, 0, 1), R[:, 5], night.f)
            spt = sim.slant_s4(np.clip(pd_, 0, 1), R[:, 5], night.f)
        else:
            up = np.ones(len(R)); base = np.full(len(R), 2.0); spt = np.ones(len(R))
        # split per user
        ofs = np.cumsum([0] + [len(r) for r in rows])
        up_u = [up[ofs[u]:ofs[u + 1]] for u in range(U)]
        tr_u = [np.array([r[7] for r in rows[u]]) for u in range(U)]
        dist_u = [night.geo[u][t]['s4'].max() > 0.4 for u in range(U)]
        J_ig = [_score(rows[u], up_u[u], eps) for u in range(U)]
        J_gn = [_score(rows[u], tr_u[u], eps) for u in range(U)]
        # observable epoch group per UE: a bubble predicted on any visible link (point forecast, tau = 0)
        hu = np.array([int(np.any((spt[ofs[u]:ofs[u + 1]] > OL_PDIST) & (R[ofs[u]:ofs[u + 1], 2] == 0))) if ofs[u + 1] > ofs[u] else 1
                       for u in range(U)], int)
        rowh = np.repeat(hu, np.diff(ofs))
        order = mrng.permutation(U)
        # A3 measurements
        meas = []
        for u in range(U):
            g = night.geo[u][t]
            m = []
            for j in range(len(g['ids'])):
                mm = 1 / max(g['s4'][j], 0.05) ** 2
                m.append(g['snr'][j] + 10 * np.log10(np.mean(mrng.gamma(mm, 1 / mm, SUB))))
            meas.append(np.array(m))
        for p in policies:
            S = st[p]; prev = list(S['cur']); UPU = up_u
            if p in OL_POL:
                upo = sim.slant_s4(np.clip(base + S['old'][rowh], 0, 1), R[:, 5], night.f) if len(R) else up
                UPU = [upo[ofs[u]:ofs[u + 1]] for u in range(U)]
                Jo = [_score(rows[u], UPU[u], eps) for u in range(U)]
                if OL_POL[p] == 'pf':
                    wpf = 1.0 / np.maximum(S['rbar'], PF_FLOOR)
                    Jo = [{s: v * wpf[u] for s, v in Jo[u].items()} for u in range(U)]
                choice = joint_assign(Jo, prev, cap)
            elif p == 'IONOGUARD-MU':
                choice = joint_assign(J_ig, prev, cap)
            elif p == 'IONOGUARD-PF':
                wpf = 1.0 / np.maximum(S['rbar'], PF_FLOOR)       # gradient of sum_u log(Rbar_u)
                choice = joint_assign([{s: v * wpf[u] for s, v in J_ig[u].items()} for u in range(U)], prev, cap)
            elif p == 'Genie-MU':
                choice = joint_assign(J_gn, prev, cap)
            elif p == 'IG-Indep':
                choice = indep_assign(J_ig, prev, cap, order)
            elif p == 'Risk-ACM':
                Jel = []
                for u in range(U):
                    g = night.geo[u][t]
                    d = {int(s): float(e) for s, e in zip(g['ids'], g['el'])}
                    if prev[u] in d and max(d.values()) <= d[prev[u]] + 10:
                        d[prev[u]] += 100.0          # keep unless another is 10 deg higher
                    Jel.append(d)
                choice = indep_assign(Jel, prev, cap, order, kappa=0.0, kappa0=0.0)
            elif p == 'A3-OLLA':
                Jm = []
                for u in range(U):
                    g = night.geo[u][t]
                    d = {int(s): float(m) for s, m in zip(g['ids'], meas[u])}
                    if prev[u] in d:
                        S['a3'][u] = S['a3'][u] + 1 if max(d.values()) > d[prev[u]] + 3 else 0
                        if S['a3'][u] < 2:
                            d[prev[u]] += 100.0       # time-to-trigger not met: stay
                    Jm.append({s: v + 200.0 for s, v in d.items()})
                choice = indep_assign(Jm, prev, cap, order, kappa=0.0, kappa0=0.0)
            ebits = np.zeros(U); efail = np.zeros(U); eatt = np.zeros(U)
            for u in range(U):
                cur = choice[u]; g = night.geo[u][t]
                if cur < 0 or cur not in g['pos']:
                    S['blk'][u] += SUB; S['cur'][u] = -1
                    if dist_u[u]:
                        S['dopp'] += SUB; S['ddef'] += SUB
                    continue
                ho = prev[u] != -1 and cur != prev[u]
                if ho:
                    S['ho'][u] += 1
                    if p == 'A3-OLLA':
                        S['a3'][u] = 0
                S['cur'][u] = cur
                j = g['pos'][cur]; snr, s4t = g['snr'][j], g['s4'][j]
                fd = night.fades(u, t, cur, s4t)
                start = HO_LOSS if ho else 0
                n = SUB - start; fails = 0; bits = 0.0; dfr = 0
                if p == 'A3-OLLA':
                    ok_n = 0
                    for kk in range(start, SUB):
                        eff = snr - S['olla'][u]
                        idx = np.where(MCS_REQ <= eff)[0]; idx = idx.max() if len(idx) else 0
                        if snr + fd[kk] >= MCS_REQ[idx]:
                            bits += MCS_SE[idx] * BW / SUB; ok_n += 1
                            S['olla'][u] = max(S['olla'][u] - 0.5 * eps / (1 - eps), -5)
                        else:
                            S['olla'][u] = min(S['olla'][u] + 0.5, 30)
                    fails = n - ok_n
                else:
                    if p == 'Risk-ACM':
                        if ho:
                            S['s4e'][u] = 0.3
                        k, _ = sim.best_mcs(snr, min(S['s4e'][u], 1.0), eps)
                    elif p == 'Genie-MU':
                        k, _ = sim.best_mcs(snr, s4t, eps)
                    else:
                        i0 = [i for i, r in enumerate(rows[u]) if r[1] == cur and r[2] == 0]
                        k, _ = sim.best_mcs(snr, UPU[u][i0[0]] if i0 else 1.0, eps)
                    if k < 0:
                        S['defer'][u] += n; dfr = n; n = 0
                    else:
                        okv = (snr + fd[start:]) >= MCS_REQ[k]
                        fails = int(n - okv.sum()); bits = okv.sum() * MCS_SE[k] * BW / SUB
                    if p == 'Risk-ACM':
                        lin = 10 ** (fd / 10); s4t2 = lin.var() / lin.mean() ** 2
                        s4s = np.sqrt(max((1 + s4t2) / (1 + S4R2) - 1, 0))
                        ema = 0.6 * S['s4e'][u] / 1.15 + 0.4 * s4s if S['last'][u] is not None else s4s
                        S['s4e'][u] = 1.15 * max(ema, s4s); S['last'][u] = s4s
                S['bits'][u] += bits; S['att'][u] += n; S['fail'][u] += fails; ebits[u] = bits; efail[u] = fails; eatt[u] = n
                if dist_u[u]:
                    S['dbits'] += bits; S['datt'] += n; S['dfail'] += fails; S['ddef'] += dfr; S['dopp'] += SUB
            S['rbar'] = (1 - 1 / PF_TAU) * S['rbar'] + (1 / PF_TAU) * ebits / 1e6
            if p in OL_POL:
                for h in (0, 1):
                    m_ = hu == h
                    if eatt[m_].sum() > 0:
                        S['old'][h] = max(S['old'][h] + OL_GAMMA * (efail[m_].sum() - (eps - OL_MARGIN) * eatt[m_].sum()) / SUB, 0.0)
                        S['ga'][h] += eatt[m_].sum(); S['gf'][h] += efail[m_].sum(); S['oldmax'][h] = max(S['oldmax'][h], S['old'][h])
    res = {}
    for p in policies:
        S = st[p]
        gp_u = S['bits'] / T / 1e6
        res[p] = dict(gp=float(gp_u.mean()), gp_min=float(gp_u.min()),
                      jain=float(gp_u.sum() ** 2 / (U * (gp_u ** 2).sum() + 1e-12)),
                      bler=float(S['fail'].sum() / max(S['att'].sum(), 1)),
                      fpo=float(S['fail'].sum() / (T * SUB * U)),
                      defer=float(S['defer'].sum() / (T * SUB * U)), blocked=float(S['blk'].sum() / (T * SUB * U)),
                      ho=float(S['ho'].sum() / U / (T / 60)),
                      dgp=float(S['dbits'] / max(S['dopp'] / SUB, 1) / 1e6),
                      dbler=float(S['dfail'] / max(S['datt'], 1)), ddefer=float(S['ddef'] / max(S['dopp'], 1)),
                      dfrac=float(S['dopp'] / (T * SUB * U)), gp_users=[float(x) for x in gp_u])
        if p in OL_POL:
            res[p].update(d_end=S['old'].tolist(), d_max=S['oldmax'].tolist(), grp_att=S['ga'].tolist(), grp_fail=S['gf'].tolist())
    return res


# ----------------------------------------------------------------------------- theory
def _epoch_tables(geos, T):
    """Per epoch: list of (user, sat, r_j array, c_j array) with r = eta(1-q) [b/s/Hz], c = q."""
    out = []
    for t in range(T):
        items = []
        for u, gu in enumerate(geos):
            g = gu[t]
            if not len(g['ids']):
                continue
            P = sim.best_mcs_vec(g['snr'], g['s4'])            # (n_vis, J) failure probabilities
            items.append((u, g['ids'], MCS_SE[None, :] * (1 - P), P))
        out.append(items)
    return out


def _epoch_value(items, mu, cap, U):
    """max over capacity-feasible assignment and MCS/defer of sum_u (r - mu c)."""
    best = []; tot = 0.0; need = False; load = {}
    vals = []
    for u, ids, r, c in items:
        v = np.maximum((r - mu * c).max(1), 0.0)               # defer gives 0
        vals.append((u, ids, v))
        k = int(np.argmax(v)); tot += v[k]
        s = int(ids[k]); load[s] = load.get(s, 0) + 1
        if v[k] > 0 and load[s] > cap:
            need = True
    if not need or cap is None:
        return tot
    sats = sorted({int(s) for _, ids, _ in vals for s in ids}); col = {s: i for i, s in enumerate(sats)}
    W = np.zeros((len(vals), len(sats) * cap))
    for i, (u, ids, v) in enumerate(vals):
        for s, vv in zip(ids, v):
            W[i, col[int(s)] * cap:(col[int(s)] + 1) * cap] = vv
    rr, cc = linear_sum_assignment(-W)
    return float(W[rr, cc].sum())


def lagrangian_bound(geos, T, eps, cap=None, mask=None, mus=None):
    """Proposition 2. Upper bound on average goodput (Mbit/s per user) of ANY policy whose
    long-run failure fraction per opportunity is <= eps (over the epochs in `mask`).
    Weak duality: G* <= min_mu>=0 [ (1/|E|U) sum_t V_t(mu) + mu*eps ] * BW."""
    U = len(geos)
    tabs = _epoch_tables(geos, T)
    idx = np.arange(T) if mask is None else np.where(mask)[0]
    if mus is None:
        mus = np.concatenate([[0], np.geomspace(0.05, 400, 60)])
    D = []
    for mu in mus:
        V = sum(_epoch_value(tabs[t], mu, cap if cap else 10 ** 6, U) for t in idx)
        D.append(V / (len(idx) * U) + mu * eps)
    D = np.array(D); k = int(np.argmin(D))
    return dict(bound=float(D[k] * BW / 1e6), mu=float(mus[k]), curve=[(float(m), float(d * BW / 1e6)) for m, d in zip(mus, D)])


def min_deferral(geos, T, eps, mask=None):
    """Proposition 3. Minimum fraction of opportunities that ANY policy must defer to keep the
    failure fraction <= eps over the epochs in `mask` (fractional knapsack on q_min)."""
    qmin = []
    for u, gu in enumerate(geos):
        for t in (range(T) if mask is None else np.where(mask)[0]):
            g = gu[t]
            if not len(g['ids']):
                qmin.append(1.0); continue
            q = sim.outage_prob(g['s4'], g['snr'] - MCS_REQ[0])
            qmin.append(float(q.min()))
    q = np.sort(np.array(qmin)); n = len(q)
    budget = eps * n; cs = np.cumsum(q)
    full = int(np.searchsorted(cs, budget, side='right'))      # epochs transmitted fully
    if full >= n:
        return 0.0, q
    rem = budget - (cs[full - 1] if full > 0 else 0.0)
    frac = rem / q[full] if q[full] > 0 else 1.0
    return float((n - full - frac) / n), q
