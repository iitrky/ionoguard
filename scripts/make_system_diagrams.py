import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Liberation Sans', 'DejaVu Sans'], 'font.size': 8})
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = os.path.join(ROOT, 'figures') + '/'
RES = os.path.join(ROOT, 'results') + '/'
def box(ax, x, y, w, h, t, fc, fs=7.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.03,rounding_size=0.1', fc=fc, ec='#333', lw=0.9))
    ax.text(x + w/2, y + h/2, t, ha='center', va='center', fontsize=fs)
def arr(ax, x1, y1, x2, y2, ls='-', c='#333'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='->', lw=1.0, color=c, linestyle=ls))

# ---------------- Fig 3: downlink and channel model chain
fig, ax = plt.subplots(figsize=(7.2, 2.9)); ax.axis('off'); ax.set_xlim(0, 12.4); ax.set_ylim(0, 5)
y = 2.0; h = 1.25
box(ax, 0.05, y, 1.95, h, 'LEO satellite\nphased array\nMCS j, 5 MHz', '#e8f0fa')
box(ax, 2.45, y, 2.0, h, 'Path loss and\nscan loss\nγ̄ = γ0 − 20 log(r/h)\n+ 12 log cos ψ', '#fff2d9', fs=7)
box(ax, 4.9, y, 2.25, h, 'Ionospheric\nscintillation\nGamma, m = 1/S4²', '#fde3e3')
box(ax, 7.6, y, 2.0, h, 'Handset\nmultipath\nRician, K = 10 dB', '#fff2d9')
box(ax, 10.05, y, 2.25, h, 'Receiver\nγ = γ̄ + F\nblock ok if γ ≥ Γj', '#e3f4e3')
for x1, x2 in [(2.0, 2.45), (4.45, 4.9), (7.15, 7.6), (9.6, 10.05)]:
    arr(ax, x1, y + h/2, x2, y + h/2)
box(ax, 3.9, 3.95, 1.9, 0.85, 'Plasma bubble\nfield σ(x, y, t)', '#f2f2f2')
box(ax, 6.25, 3.95, 2.25, 0.85, 'Pierce point p_n(t),\nzenith angle χ_n', '#f2f2f2')
arr(ax, 4.85, 3.95, 5.6, y + h); arr(ax, 7.35, 3.95, 6.6, y + h)
ax.text(6.0, 0.95, 'S4 = σ(p_n, t) · (sec χ_n)^1/2 · (f / f0)^−3/2, saturating at 1', ha='center', fontsize=7.4, style='italic')
arr(ax, 6.0, 1.15, 6.0, y)
box(ax, 10.05, 0.25, 2.25, 0.9, 'Block success or\nfailure, goodput', '#f2f2f2')
arr(ax, 11.17, y, 11.17, 1.15)
box(ax, 0.05, 0.25, 1.95, 0.9, 'IONOGUARD:\nsatellite n, MCS j', '#fde3e3')
arr(ax, 1.0, 1.15, 1.0, y, ls='--', c='#7d3c98')
plt.tight_layout(); plt.savefig(F + 'Fig3.png', dpi=600); plt.close()

# ---------------- Fig 6: timing structure
fig, ax = plt.subplots(figsize=(7.2, 3.0)); ax.axis('off'); ax.set_xlim(-130, 30); ax.set_ylim(0, 5.2)
ax.annotate('', xy=(28, 3.6), xytext=(-128, 3.6), arrowprops=dict(arrowstyle='->', lw=1.2))
ax.text(28, 3.25, 'time (s)', ha='right', fontsize=7.5)
for t in range(-120, 1, 30):
    ax.plot([t, t], [3.45, 3.75], c='#d68910', lw=2)
ax.text(-120, 4.0, 'monitor reports\nevery 30 s', fontsize=7.2, color='#9c640c', ha='center')
ax.annotate('', xy=(0, 4.55), xytext=(-60, 4.55), arrowprops=dict(arrowstyle='<->', lw=1.0, color='#9c640c'))
ax.text(-30, 4.7, 'latency L = 60 s', ha='center', fontsize=7.4, color='#9c640c')
ax.plot([-60, -60], [3.6, 4.5], ls=':', c='#9c640c')
ax.axvspan(-300 + 0.0, -300, color='none')
ax.plot([0, 0], [3.45, 4.9], c='crimson', lw=1.4); ax.text(0, 4.95, 'decision epoch t', ha='center', fontsize=7.6, color='crimson')
for tau in (0, 5, 10, 15):
    ax.plot(tau, 3.6, 'o', c='#1f618d', ms=5)
ax.annotate('', xy=(15, 3.05), xytext=(0, 3.05), arrowprops=dict(arrowstyle='<->', lw=1.0, color='#1f618d'))
ax.text(2, 4.0, 'look-ahead\nτ = 0, 5, 10, 15 s', ha='left', fontsize=7, color='#1f618d')
ax.text(-94, 2.55, 'report window: last 300 s (partly shown),\nused for the drift-compensated map', ha='center', fontsize=7.2)
ax.annotate('', xy=(-60, 2.25), xytext=(-129, 2.25), arrowprops=dict(arrowstyle='<-', lw=0.9))
# zoom of one epoch
x0, w = -55, 50
for k in range(20):
    fc = '#bbbbbb' if k < 2 else ('#a9dfbf' if k not in (7, 13) else '#f5b7b1')
    ax.add_patch(plt.Rectangle((x0 + k * w / 20, 0.55), w / 20, 0.55, fc=fc, ec='white', lw=0.6))
ax.annotate('', xy=(x0 + w + 1, 1.0), xytext=(-2, 3.4), arrowprops=dict(arrowstyle='->', lw=0.8, color='crimson', ls=':'))
ax.text(-12, 1.6, 'epoch t\n(zoom)', fontsize=7, color='crimson')
ax.text(x0 + w / 2, 0.15, 'one 1-s epoch = 20 transport blocks of 50 ms (grey: 100-ms handover interruption; red: failed blocks)', ha='center', fontsize=7)
plt.tight_layout(); plt.savefig(F + 'Fig6.png', dpi=600); plt.close()
print('ok')
