"""
Script for generating the Synchronous Time Histogram (STH) 
from detected spike events across multiple channels.

The pipeline performs:
1. Spike accumulation and smoothing.
2. Network burst detection and alignment.
3. Correlation-based averaging to obtain a representative STH profile.

Requires:
- spikesT.npy  (array of spike times, in seconds)
- spikesI.npy  (array of spike channel indices)

Output:
- sth.npy  (average synchronous burst profile)

Dependencies:
    numpy, matplotlib, scipy, sklearn, pickle, os
"""

from matplotlib import pyplot as plt
from scipy.signal import correlate, find_peaks, argrelmin
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import numpy as np
import pickle
import os

# ============================================================
# Define exponential functions (for later fitting, optional)
# ============================================================
def single_exp(x, a, b, d):
    return a * (np.exp(b * x)) + d

def double_exp(x, a0, b0, a1, b1, d):
    return a0 * (np.exp(b0 * x)) + a1 * (np.exp(b1 * x)) + d

def exp2(x, a, b, c, d):
    return a*np.exp(-b*x) + c*np.exp(-d*x)

def exp1(x, a, b):
    return a*np.exp(-b*x)

# ============================================================
# General parameters
# ============================================================
bin = 25  # ms
freqSam = 10_000  # Hz
acqTime = 180  # seconds
numCh = 100
bin_sam = int(bin * freqSam / 1000)  # samples
smooth_window = 1000  # samples
min_peak_height = 0.05  # 5 %
min_peak_distance = 8000  # samples
min_peak_th = 0.05  # 5 %
n_sam = int(freqSam * acqTime)

# ============================================================
# Load spike data
# ============================================================
spikes_time = (np.load('spikesT.npy')*freqSam).astype(int)
cell_index = np.load('spikesI.npy')

# ============================================================
# Create cumulative spike count and activity arrays
# ============================================================
cum_spikes, sp_count = np.unique(spikes_time, return_counts=True)
cum_peak = np.zeros(n_sam)
cum_peak[cum_spikes] = sp_count


# Boolean array indicating time intervals when each electrode is active
el_att = np.full((numCh, n_sam), False,  dtype=bool)

for s, el in zip(spikes_time, cell_index):
    el_att[el, np.clip(s-125, 0, n_sam):np.clip(s+125, 0, n_sam)] = True

# ============================================================
# Compute smoothed instantaneous firing rate (IFR)
# ===========================================================
cum_ifr = np.convolve(cum_peak, np.ones(smooth_window)/smooth_window*freqSam, mode='same')
cum_el_att = np.sum(el_att, axis=0)
net_burst_profile = cum_ifr*cum_el_att

# ============================================================
# Detect network bursts using peak finding
# ===========================================================
peaks, _ = find_peaks(net_burst_profile, height=min_peak_height * np.amax(net_burst_profile), distance=min_peak_distance)

# Identify local minima to define burst boundaries
minima = argrelmin(net_burst_profile)[0]
minima = minima[net_burst_profile[minima] < min_peak_th * np.amax(net_burst_profile[peaks])]

min_pre_peak = -np.ones(len(peaks), dtype=int)
min_post_peak = -np.ones(len(peaks), dtype=int)

# ============================================================
# Associate each peak with pre- and post-burst minima
# ============================================================
for idx, peak in enumerate(peaks):

    min_pre = minima[minima < peak]
    min_pre_th = min_pre[net_burst_profile[min_pre] < min_peak_th * net_burst_profile[peak]]
    min_post = minima[minima > peak]
    min_post_th = min_post[net_burst_profile[min_post] < min_peak_th * net_burst_profile[peak]]

    if min_pre_th.size == 0 or min_post_th.size == 0:
        continue

    min_pre_peak[idx] = min_pre_th[-1]
    min_post_peak[idx] = min_post_th[0]

# Keep only bursts with valid start and end points
peaks = peaks[min_post_peak != -1]
min_pre_peak = min_pre_peak[min_post_peak != -1]
min_post_peak = min_post_peak[min_post_peak != -1]

net_burst_start = min_pre_peak
net_burst_end = min_post_peak - min_pre_peak

# ============================================================
# Build burst-aligned profiles
# ============================================================
max_nbd = np.amax(net_burst_end)
t_pad = int(500 * freqSam / 1000)
bursts_profile = np.zeros((len(net_burst_start), max_nbd+2*t_pad))

for idx, nb_start in enumerate(net_burst_start):
    nb_profile = cum_ifr[np.clip(nb_start - t_pad, 0, None).astype(int):nb_start + max_nbd + t_pad]
    if len(nb_profile) < max_nbd + 2*t_pad:
        nb_profile = np.concatenate((nb_profile, np.full(max_nbd + 2*t_pad - len(nb_profile), nb_profile[-1])))
    bursts_profile[idx, :] = nb_profile

net_burst_profile_norm = bursts_profile/np.amax(bursts_profile)

# ============================================================
# Find reference burst via average correlation
# ============================================================
R_idx = []
for idx_r, nb_ref in enumerate(net_burst_profile_norm):

    R_after = []
    shifted_burst_profile = np.zeros_like(bursts_profile)
    shifted_burst_profile_norm = np.zeros_like(bursts_profile)

    for idx, (norm_profile, profile) in enumerate(zip(net_burst_profile_norm, bursts_profile)):

        shift = np.argmax(correlate(nb_ref, norm_profile))-len(nb_ref)+1

        if shift > 0:  # reference starts after
            shifted_burst_profile[idx, shift:] = profile[shift:]
            shifted_burst_profile_norm[idx, shift:] = norm_profile[shift:]
        else:
            shifted_burst_profile[idx, :shift] = profile[:shift]
            shifted_burst_profile_norm[idx, :shift] = norm_profile[:shift]

        R_after.append(np.corrcoef(nb_ref, shifted_burst_profile_norm[idx, :])[0, 1])
    # print(idx_r)
    R_idx.append(np.nanmean(R_after))

# Identify reference burst (highest mean correlation)
ref_idx = np.argmax(R_idx)
nb_ref = net_burst_profile_norm[ref_idx]


# ============================================================
# Align all bursts to reference burst
# ============================================================
shifted_burst_profile = np.zeros_like(bursts_profile)
shifted_burst_profile_norm = np.zeros_like(bursts_profile)

for idx, (norm_profile, profile) in enumerate(zip(net_burst_profile_norm, bursts_profile)):

    shift = np.argmax(correlate(nb_ref, norm_profile))-len(nb_ref)+1

    if shift > 0:  # reference starts after
        shifted_burst_profile[idx, shift:] = profile[shift:]
        shifted_burst_profile_norm[idx, shift:] = norm_profile[shift:]
    else:
        shifted_burst_profile[idx, :shift] = profile[:shift]
        shifted_burst_profile_norm[idx, :shift] = norm_profile[:shift]

# Compute mean synchronous time histogram (STH)
sth = np.mean(shifted_burst_profile, axis=0)

# ============================================================
# Plot STH and save results
# ============================================================
plt.plot(np.arange(len(sth))*1000/freqSam, sth)
plt.xlabel('ms')
plt.ylabel('spikes/s')
plt.title('STH')
plt.show()

np.save('sth.npy', sth)
