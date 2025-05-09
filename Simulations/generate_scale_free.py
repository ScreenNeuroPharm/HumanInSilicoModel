'''
generate_scale_free.py - Script for generating directed network graph
    featuring heterogeneous incoming and outgoing connectivity

Written by Valerio Barabino - valerio.barabino@edu.unige.it
NNT Group, DIBRIS, University of Genoa (UNIGE), Italy
last edit: 08-05-2025

This software is provided as-is, without any warranty.
You are free to modify or share this file, provided that the above
copyright notice is kept intact.
'''

import numpy as np
import networkx as nx
import scipy
import os


def powerlaw(t, b):

    return t**(-b)


def linear(t, a, b):

    return a*t + b


def gaussian(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))


Ne = 100
x_min = 1
x_max = 99

alpha = 2

x = np.linspace(x_min, x_max, x_max-x_min+1)
y = powerlaw(x, alpha)
prob2 = y / sum(y)

''' Find seed Scale-Free '''

conn_mean = []
seeds = range(2000)
for seed in seeds:
    np.random.seed(seed)
    sf_distr = (np.random.choice(x, Ne, p=prob2)).astype(int)
    conn_mean.append(sum(sf_distr))
err = np.where(np.asarray(conn_mean) == 1900)

np.random.seed(108)
distr_SF = (np.random.choice(x, Ne, p=prob2)).astype(int)
# distr_SF_shuffled = distr_SF.copy()
# np.random.shuffle(distr_SF_shuffled)

''' Find seed RND '''

conn_mean = []
seeds = range(2000)
for seed in seeds:
    np.random.seed(seed)
    rnd_distr = (19 + 3.5*np.random.randn(Ne)).astype(int)
    conn_mean.append(sum(rnd_distr))
err = np.where(np.asarray(conn_mean) == 1900)
np.random.seed(130)
distr_RND = (19 + 3.5*np.random.randn(Ne)).astype(int)
# distr_RND_shuffled = distr_RND.copy()
# np.random.shuffle(distr_RND_shuffled)

G = nx.directed_havel_hakimi_graph(distr_SF.tolist(), distr_RND.tolist())
C = nx.adjacency_matrix(G, nodelist=None, weight=1)
C = scipy.sparse.dia_matrix.toarray(C)
source, target = C.nonzero()

# fig, ax = plt.subplots(1,2)
# in_deg = [d for _, d in G.in_degree()]
# out_deg = [d for _, d in G.out_degree()]
# in_deg_log = np.log10(in_deg)
# out_deg_log = np.log10(out_deg)
# in_deg_hist, _ = np.histogram(in_deg_log, bins=7)
# out_deg_hist, bins = np.histogram(out_deg_log, bins=7)
# bins = (bins[:-1] + bins[1:]) / 2
# popt_in, _ = curve_fit(linear, bins, np.log10(in_deg_hist))
# popt_out, _ = curve_fit(linear, bins, np.log10(out_deg_hist))
# ax[0].loglog(10**bins, in_deg_hist, '.b')
# ax[0].loglog(10**bins, 10**linear(bins, *popt_in), 'r', label=rf'$\alpha$ = {-popt_in[0]:.2f}')
# ax[0].set_ylabel('SF')
# ax[0].set_xlim([0.5, 100])
# ax[0].set_ylim([0.5, 100])
# ax[0].legend()
# ax[1].loglog(10**bins, out_deg_hist, '.b')
# ax[1].loglog(10**bins, 10**linear(bins, *popt_out), 'r', label=rf'$\alpha$ = {-popt_out[0]:.2f}')
# ax[1].set_ylabel('SF')
# ax[1].set_xlim([0.5, 100])
# ax[1].set_ylim([0.5, 100])
# ax[1].legend()
# plt.show()
