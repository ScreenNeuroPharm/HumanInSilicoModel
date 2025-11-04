'''
degree_distribution.py - Script for generating and visualizing degree distributions
    of Random, Small-World, and Scale-Free network topologies.
    The outputs are used to generate panels for Figure 2.
'''

# ================================================================
# Imports
# ================================================================
from matplotlib import pyplot as plt
import numpy as np
import networkx as nx
import scipy
from scipy.optimize import curve_fit
from scipy.stats import skewnorm


# ================================================================
# Distribution Functions
# ================================================================
def powerlaw(t, b):
    return t**(-b)


def linear(t, a, b):
    return a*t + b


def gaussian(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))


def skewed_gaussian(x, c, x0, sigma, a):
    skewed_distr = c*skewnorm.pdf((x-x0)/sigma, a=a)/sigma
    return skewed_distr

# ================================================================
# Visualization Settings
# ================================================================
colors = ['#d90429',
          # '#00b4d8',  # in vitro
          '#fb8500',
          '#ffb703']

Ne = 100

# ================================================================
# Scale-Free Distribution (Power-law)
# ================================================================
x2 = np.linspace(6, 99, 94)
y2 = powerlaw(x2, 2)
prob2 = y2 / sum(y2)

np.random.seed(1223)
sf_distr = (np.random.choice(x2, Ne, p=prob2)).astype(int)

# ================================================================
# Random Distribution (Gaussian-like)
# ================================================================
np.random.seed(130)
rnd_distr = (19 + 3.5*np.random.randn(Ne)).astype(int)

# Ensure the two distributions have the same total degree sum
assert sum(sf_distr) == sum(rnd_distr)

# ================================================================
# Generate Directed Network: Random → Scale-Free (mixed topology)
# ================================================================
G = nx.directed_havel_hakimi_graph(rnd_distr.tolist(), sf_distr.tolist())
C = nx.adjacency_matrix(G, nodelist=None, weight=1)
C = scipy.sparse.dia_matrix.toarray(C)
source, target = C.nonzero()

in_deg = [d for _, d in G.in_degree()]
out_deg = [d for _, d in G.out_degree()]

# ================================================================
# Fit In-degree Distribution (Random)
# ================================================================
in_deg_hist, bins = np.histogram(in_deg, bins=7)
bins_in = (bins[:-1] + bins[1:]) / 2
popt_in, _ = curve_fit(gaussian, bins_in, in_deg_hist, p0=[29, 19, 3.5])

# ================================================================
# Fit Out-degree Distribution (Scale-Free)
# ================================================================
out_deg_log = np.log10(out_deg)
out_deg_hist, bins = np.histogram(out_deg_log, bins=7)
bins_out = (bins[:-1] + bins[1:]) / 2
popt_out, _ = curve_fit(linear, bins_out, np.log10(out_deg_hist))

# ================================================================
# Generate Small-World Network
# ================================================================
G = nx.connected_watts_strogatz_graph(100, 10, p=0.5, seed=1893)
C = nx.adjacency_matrix(G, nodelist=None, weight=1)
C = scipy.sparse.dia_matrix.toarray(C)
source, target = C.nonzero()
in_deg = [d for _, d in G.degree()]
in_deg_hist_SW, bins = np.histogram(in_deg)
bins_in_SW = (bins[:-1] + bins[1:]) / 2
popt_in_skew, _ = curve_fit(skewed_gaussian, bins_in_SW, in_deg_hist_SW, p0=[100,16,5,3])
x_range = np.linspace(0, 30, 100)

# ================================================================
# Plotting Degree Distributions
# ================================================================
fig, ax = plt.subplots(1, 3, figsize=(10/2.54, 5/2.54))
titsize = 10
labsize = 8
ticsize = 6

# ------------------------------------------------
# (A) Random Network
# ------------------------------------------------
ax[0].plot(bins_in, np.convolve(in_deg_hist, np.ones(3)/3, mode='same'), '.k', markersize=3)
ax[0].plot(range(10,30), gaussian(range(10,30), *popt_in), color=colors[0], linewidth=1)
# ax[0].set_title('Random', fontsize=titsize)
ax[0].set_xlim((8, 30))
ax[0].set_ylim((0, 30))
ax[0].set_ylabel('# connections', fontsize=labsize)
ax[0].set_xlabel('Degree', fontsize=labsize)
ax[0].set_xticks(ticks=[10, 20, 30], labels=[10, 20, 30], fontsize=ticsize)
ax[0].set_yticks(ticks=[10, 20, 30], labels=[10, 20, 30], fontsize=ticsize)
ax[0].spines[['right', 'top']].set_visible(False)
ax[0].tick_params(width=1)
ax[0].spines[['left', 'bottom']].set_linewidth(1)
ax[0].set_title('Random', fontsize=titsize)

# ------------------------------------------------
# (B) Small-World Network
# ------------------------------------------------
ax[1].plot(bins_in_SW+10, np.convolve(in_deg_hist_SW, np.ones(3)/3, mode='same'), '.k', markersize=3)
ax[1].plot(x_range+10, skewed_gaussian(x_range, *popt_in_skew), color=colors[1], linewidth=1)
# ax[1].plot(x_range, skewed_gaussian(x_range, *[120,17,6,2]), color=colors[1], linewidth=3.5)
# ax[1].set_title('Small World', fontsize=titsize)
ax[1].set_xlim((8, 30))
ax[1].set_ylim((0, 30))
ax[1].set_xlabel('Degree', fontsize=labsize)
ax[1].set_xticks(ticks=[10, 20, 30], labels=[10, 20, 30], fontsize=ticsize)
ax[1].set_yticks(ticks=[10, 20, 30], labels=[10, 20, 30], fontsize=ticsize)
ax[1].spines[['right', 'top']].set_visible(False)
ax[1].tick_params(width=1)
ax[1].spines[['left', 'bottom']].set_linewidth(1)
ax[1].set_title('Small-world', fontsize=titsize)


# ------------------------------------------------
# (C) Scale-Free Network
# ------------------------------------------------
ax[2].loglog(10**bins_out, out_deg_hist, '.k', markersize=3)
ax[2].loglog(10**bins_out, 10**linear(bins_out, *popt_out), color=colors[2], linewidth=1)
# ax[2].set_title('Scale Free', fontsize=titsize)
ax[2].set_xlim([8, 60])
ax[2].set_ylim([0.9, 60])
ax[2].set_xlabel('Degree', fontsize=labsize)
ax[2].set_xticks(ticks=[10, 60], labels=[10, 60], fontsize=ticsize)
ax[2].set_yticks(ticks=[10, 60], labels=[10, 60], fontsize=ticsize)
ax[2].spines[['right', 'top']].set_visible(False)
ax[2].tick_params(width=1)
ax[2].spines[['left', 'bottom']].set_linewidth(1)
# ax[2].minorticks_off()
ax[2].tick_params(which='minor', labelleft=False, labelbottom=False)
ax[2].set_title('Scale-free', fontsize=titsize)

# ================================================================
# Save Figure
# ================================================================
fig.tight_layout()
plt.savefig('distribution_degree_new.tif', format="tiff", pil_kwargs={"compression": "tiff_lzw"}, dpi=600)
# plt.show()
