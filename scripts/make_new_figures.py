"""Figures for the performance limits (Fig. 12) and the multi-user results (Fig. 13).
Inputs: results/summary.json, results/mu_summary.json, results/mu/*.json (run summarize_ext.py first)."""
import os, json, glob, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Liberation Sans', 'DejaVu Sans'], 'font.size': 8.5})
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results'); FIG = os.path.join(ROOT, 'figures')
S = json.load(open(os.path.join(RES, 'summary.json')))
M = json.load(open(os.path.join(RES, 'mu_summary.json')))
SC = ('S1', 'S2', 'S3')
SU0 = {sc: json.load(open(os.path.join(RES, 'mu', f'su_{sc}_0.json'))) for sc in SC}

# ---------------- Fig 12: single-user goodput vs Lagrangian bound (mean and 95% CI over test nights)
fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.8))
pols = [('A3-OLLA', 'A3-OLLA'), ('Risk-ACM', 'Risk-ACM'), ('Oracle', 'Genie-PL'), ('IONOGUARD', 'IONOGUARD')]
x = np.arange(3); w = 0.16
ek = dict(ecolor='black', capsize=1.5, elinewidth=0.6)
for i, (k, lab) in enumerate(pols):
    axs[0].bar(x + (i - 2) * w, [S[sc][k]['gp'][0] for sc in SC], w, yerr=[S[sc][k]['gp'][2] for sc in SC], error_kw=ek, label=lab,
               hatch=['//', '..', 'xx', ''][i], edgecolor='black', lw=0.4, color=['#bbbbbb', '#999999', '#666666', '#c0392b'][i])
axs[0].bar(x + 2 * w, [S[sc]['bound'][0] for sc in SC], w, yerr=[S[sc]['bound'][2] for sc in SC], error_kw=ek,
           label='Bound (16)', color='white', edgecolor='black', lw=0.8)
axs[0].set_xticks(x); axs[0].set_xticklabels(SC); axs[0].set_ylabel('Goodput (Mbit/s)')
axs[0].set_ylim(0, max(S[sc]['bound'][0] for sc in SC) * 1.3); axs[0].legend(fontsize=6.6, ncol=2, loc='upper left', frameon=False)
axs[0].text(0.98, 0.02, '(a)', transform=axs[0].transAxes, ha='right', weight='bold')
lo = np.inf
for sc, mk, ls in (('S1', 'o', '-'), ('S2', 's', '--'), ('S3', '^', ':')):
    c = np.array(SU0[sc]['curve']); lo = min(lo, c[:, 1].min())
    axs[1].plot(c[1:, 0], c[1:, 1], ls, marker=mk, ms=2.5, lw=1, label=f'{sc}, night 1', color='black')
    k = int(np.argmin(c[:, 1])); axs[1].plot(max(c[k, 0], 0.05), c[k, 1], 'o', mfc='none', mec='#c0392b', ms=7, mew=1.2)
axs[1].set_ylim(lo * 0.85, lo * 2.0); axs[1].set_xlim(0.04, 60)
axs[1].set_xscale('log'); axs[1].set_xlabel('Multiplier μ'); axs[1].set_ylabel('Dual function (Mbit/s)')
axs[1].legend(fontsize=7, frameon=False); axs[1].grid(alpha=0.3, which='both')
axs[1].text(0.98, 0.02, '(b)', transform=axs[1].transAxes, ha='right', weight='bold')
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'Fig10.png'), dpi=600); plt.close()

# ---------------- Fig 13: multi-user results
P = [('A3-OLLA', 's', '#bbbbbb', '//'), ('Risk-ACM', 'D', '#999999', '..'), ('IG-Indep', 'v', '#5d6d7e', 'xx'),
     ('IONOGUARD-MU', '*', '#c0392b', ''), ('IONOGUARD-PF', 'o', '#e59866', '++'), ('Genie-MU', 'P', '#333333', '\\\\')]
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0))
names = ['Chennai', 'Kanchipuram', 'Vellore', 'Puducherry', 'Tiruchirappalli', 'Tirupati', 'Nellore', 'Bengaluru']
xs = np.arange(len(names)); w = 0.135
for i, (p, mk, col, h) in enumerate(P):
    axs[0].bar(xs + (i - 2.5) * w, M['2'][p]['gp_users'], w, label=p, color=col, hatch=h, edgecolor='black', lw=0.35)
axs[0].set_xticks(xs); axs[0].set_xticklabels(names, rotation=35, ha='right', fontsize=7)
axs[0].set_ylabel('Goodput per UE (Mbit/s)'); axs[0].legend(fontsize=6.3, ncol=2, frameon=False, loc='upper left')
axs[0].set_ylim(0, axs[0].get_ylim()[1] * 1.35)
axs[0].text(0.98, 0.97, '(a)', transform=axs[0].transAxes, ha='right', va='top', weight='bold')
SW = {K: json.load(open(os.path.join(RES, 'mu', f'mu_5101_K{K}.json'))) for K in (1, 2, 4)}
Ks = [1, 2, 4]
for p, mk, col, h in P:
    axs[1].plot(Ks, [SW[K][p]['gp'] for K in Ks], marker=mk, color=col if col != '#bbbbbb' else '#888888', label=p, lw=1.2)
axs[1].plot(Ks, [SW[K]['bound']['gp'] for K in Ks], 'k--', lw=1, label='Bound (16)')
axs[1].set_xticks(Ks); axs[1].set_xlabel('Capacity K (UEs per satellite)'); axs[1].set_ylabel('Mean goodput per UE (Mbit/s)')
axs[1].legend(fontsize=6.3, frameon=False); axs[1].grid(alpha=0.3)
axs[1].text(0.02, 0.97, '(b)', transform=axs[1].transAxes, va='top', weight='bold')
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'Fig11.png'), dpi=600); plt.close()
print('ok')
