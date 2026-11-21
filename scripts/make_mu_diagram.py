"""Block diagram of the capacity-constrained joint assignment (multi-user IONOGUARD)."""
import os
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Liberation Sans', 'DejaVu Sans'], 'font.size': 8})
F = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures')
fig, ax = plt.subplots(figsize=(7.2, 3.6)); ax.axis('off'); ax.set_xlim(0, 12.4); ax.set_ylim(0, 6.2)
def box(x, y, w, h, t, fc, fs=7.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.03,rounding_size=0.1', fc=fc, ec='#333', lw=0.9))
    ax.text(x + w / 2, y + h / 2, t, ha='center', va='center', fontsize=fs)
def arr(x1, y1, x2, y2, c='#333', ls='-'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='->', lw=1.0, color=c, linestyle=ls))
users = ['Chennai', 'Kanchipuram', 'Vellore', 'Puducherry', 'Tiruchirappalli', 'Tirupati', 'Nellore', 'Bengaluru']
sats = ['A', 'B', 'C', 'D', 'E']
ys_u = [5.6 - 0.66 * i for i in range(8)]; ys_s = [5.2 - 1.0 * i for i in range(5)]
xu, xs = 3.3, 6.1
for y, n in zip(ys_u, users):
    ax.add_patch(Circle((xu, y), 0.13, fc='#1f618d')); ax.text(xu - 0.22, y, n, ha='right', va='center', fontsize=7)
for y, n in zip(ys_s, sats):
    ax.add_patch(FancyBboxPatch((xs - 0.05, y - 0.22), 0.9, 0.44, boxstyle='round,pad=0.02', fc='#fde3e3' if n == 'A' else '#e3f4e3', ec='#333', lw=0.8))
    ax.text(xs + 0.4, y, f'Sat {n}\nK = 2', ha='center', va='center', fontsize=6.6)
edges = {0: 1, 1: 1, 2: 2, 3: 3, 4: 3, 5: 2, 6: 4, 7: 4}
for u, y in enumerate(ys_u):
    for s, ysat in enumerate(ys_s):
        if abs(u * 0.6 - s) < 1.6:
            ax.plot([xu + 0.13, xs - 0.05], [y, ysat], c='0.85', lw=0.6, zorder=0)
    ax.plot([xu + 0.13, xs - 0.05], [y, ys_s[edges[u]]], c='#1e8449', lw=1.6)
ax.text(xs + 0.4, ys_s[0] + 0.45, 'bubble', fontsize=6.6, color='#c0392b', ha='center')
ax.text(4.7, 0.05, 'grey: feasible links (weight J_un); green: selected b-matching', ha='center', fontsize=7)
box(7.6, 4.1, 2.2, 1.3, 'Per-user scores J_un\nfrom Algorithm 1\n(calibrated forecast,\n15 s look-ahead)', '#fff2d9', fs=7)
box(7.6, 2.35, 2.2, 1.3, 'Hysteresis bonus for\nthe serving satellite:\nW_un = (1+κ)J_un + κ0', '#fff2d9', fs=7)
box(7.6, 0.6, 2.2, 1.3, 'Replicate each satellite\nK times; Hungarian\nmethod (Algorithm 3)', '#e8f0fa', fs=7)
box(10.2, 2.35, 2.1, 1.3, 'Serving satellite per\nuser, then MCS or\ndeferral per link\n(Mondrian CRC)', '#fde3e3', fs=7)
arr(8.7, 4.1, 8.7, 3.65); arr(8.7, 2.35, 8.7, 1.9); arr(9.8, 1.25, 10.6, 2.35)
plt.tight_layout(); plt.savefig(os.path.join(F, 'Fig5.png'), dpi=600); plt.close()
print('ok')
