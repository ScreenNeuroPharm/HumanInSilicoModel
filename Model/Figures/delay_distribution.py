'''
delay_distr.py - Script for visualizing synaptic delay distributions
    across Random (RND), Small-World (SW), and Scale-Free (SF) networks.
    The outputs are used to generate panels for Figure 2.
'''

# ================================================================
# Imports
# ================================================================
import numpy as np
import matplotlib.pyplot as plt
from brian2 import *

# ================================================================
# Compute Delay Data
# ================================================================
Ne = 100
radius = 160 * um
Vmax = 25 * mm / second  # Axonal conduction velocity

# Load connectivity matrices for RND, SW, SF networks
sources_files = ['source_RND_RND.npy', 'source_SW.npy', 'source_SF2_SF2.npy']
targets_files = ['target_RND_RND.npy', 'target_SW.npy', 'target_SF2_SF2.npy']

delays = []
for src_file, trg_file in zip(sources_files, targets_files):
    sources = np.load(src_file)
    targets = np.load(trg_file)

    # Random spatial layout (same method as in your main simulation)
    x = np.sqrt(np.random.rand(Ne)) * radius * np.cos(2 * np.pi * np.random.rand(Ne))
    y = np.sqrt(np.random.rand(Ne)) * radius * np.sin(2 * np.pi * np.random.rand(Ne))

    # Compute Euclidean distance and delay for each connection
    distances = np.sqrt((x[sources] - x[targets])**2 + (y[sources] - y[targets])**2)
    delay_values = (distances / Vmax).to(ms)  # Convert to milliseconds
    delays.append(delay_values / ms)  # store as float array in ms

# ================================================================
# Figure Configuration
# ================================================================
titles = ['RND', 'SW', 'SF']
fig, axes = plt.subplots(1, 3, figsize=(10/2.54, 5/2.54))
titsize = 10
labsize = 8
ticsize = 6

colors = ['#d90429',
          '#fb8500',
          '#ffb703']

# ================================================================
# Plot and Save Delay Distributions
# ================================================================
for delay, ax, title, color in zip(delays, axes.flat, titles, colors):
    bins = range(1,14)
    delay_hist, bins = np.histogram(delay, bins=bins)
    bins = (bins[:-1] + bins[1:]) / 2
    delay_smooth = np.convolve(delay_hist, np.ones(3)/3, mode='valid')
    ax.plot(bins[1:-1], 100*delay_smooth/np.sum(delay_smooth), color=color, zorder=0)
    ax.fill_between(bins[1:-1], 100*delay_smooth/np.sum(delay_smooth), color=color, alpha=0.5, zorder=0)
    # ax.plot(bins, 100*delay_hist/np.sum(delay_hist))
    ax.vlines(np.mean(delay), 0, np.interp(np.mean(delay), bins[1:-1], 100*delay_smooth/np.sum(delay_smooth)), linewidth=0.8, color='k', zorder=1)
    ax.scatter(np.mean(delay), np.interp(np.mean(delay), bins[1:-1], 100*delay_smooth/np.sum(delay_smooth)), marker='o', s=6, color='k', zorder=1)
    print(f'{np.mean(delay):.2f} ms')
    # ax.set_title(title, fontsize=titsize)
    ax.set_xlim(2, 12)
    ax.set_ylim(0, 20)
    ax.set_xticks(ticks=[3, 6, 9, 12], labels=[3, 6, 9, 12], fontsize=ticsize)
    ax.set_yticks(ticks=[5, 10, 15, 20], labels=[5, 10, 15, 20], fontsize=ticsize)
    ax.spines[['right', 'top']].set_visible(False)
    ax.tick_params(width=1)
    ax.spines[['left', 'bottom']].set_linewidth(1)
    if title == 'RND':
        ax.set_ylabel('p(delay) (%)', fontsize=labsize)
    ax.set_xlabel('Delay (ms)', fontsize=labsize)

fig.tight_layout()
# plt.show()
fig.savefig('delay_distr_new.tif', format="tiff", pil_kwargs={"compression": "tiff_lzw"}, dpi=600)
