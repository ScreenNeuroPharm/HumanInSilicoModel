# ================================================================
# Burst Detection Analysis
# ================================================================
# This script detects bursting activity from spike train data.
# It implements three main functions:
#   1. BurstDetection() – detects bursts based on ISI and MBR thresholds
#   2. BurstEvent() – identifies burst onset and offset indices
#   3. LogISIHistogram() – estimates ISI thresholds from ISI histograms
# The script loads spike data, applies burst detection, and saves the results.
# ================================================================

from scipy.sparse import csr_matrix
import numpy as np
from scipy.signal import find_peaks
import os

# ================================================================
# Burst Detection Function
# ================================================================
def BurstDetection(spikes, n_spikes_min=5, isi_th=100., mbr_min=0.4, freqSam=10000, acqTime=60, log_isi_max_th_flag=False, min_mfb_flag=False):
    """
    Detects bursting activity in multi-channel spike data.

    Parameters:
        spikes (list of lists): Spike times for each channel (in samples)
        n_spikes_min (int): Minimum number of spikes per burst
        isi_th (float): Inter-spike interval threshold in ms
        mbr_min (float): Minimum mean burst rate (bursts per minute)
        freqSam (float): Sampling frequency in Hz
        acqTime (float): Total acquisition time in seconds
        log_isi_max_th_flag (bool): Whether to use ISI histogram thresholding
        min_mfb_flag (bool): Whether to filter out bursts with low spike rate

    Returns:
        chBurstStart, chBurstEnd, chMBR, chBD, chSPB, chRS (lists): Burst metrics
    """

    chBurstStart = [np.nan for _ in range(len(spikes))]
    chBurstEnd = [np.nan for _ in range(len(spikes))]
    chMBR = [np.nan for _ in range(len(spikes))]
    chBD = [np.nan for _ in range(len(spikes))]
    chSPB = [np.nan for _ in range(len(spikes))]
    chRS = [np.nan for _ in range(len(spikes))]

    # Compute ISI thresholds from ISI histograms if flag is enabled
    if log_isi_max_th_flag:
        chISITh = LogISIHistogram(spikes, freqSam=freqSam)

    # Process each channel independently
    for el, sp in enumerate(spikes):

        if len(sp) <= 1:
            continue

        sp = np.asarray(sp)

        # Initial burst detection with a fixed ISI threshold
        up_small, down_small = BurstEvent(spikes=sp,
                                          n_spikes_min=n_spikes_min,
                                          isi_th=isi_th,
                                          freqSam=freqSam)

        # Use the log(ISI) histogram threshold if available
        if np.isnan(chISITh[el]) or not log_isi_max_th_flag:
            burst_up = up_small
            burst_down = down_small

        else:
            # Merge short bursts if necessary
            if len(up_small) >= 2:
                cond_merge = (sp[up_small][1:] - sp[down_small][:-1]) > 500  # self.chISITh[el]
                up_small = up_small[np.concatenate(([True], cond_merge))]
                # down_small = down_small[np.concatenate((cond_merge, [True]))]  # Not needed
            
            # Perform a second burst detection using adaptive ISI threshold
            up_large, down_large = BurstEvent(spikes=sp,
                                              n_spikes_min=n_spikes_min,
                                              isi_th=chISITh[el])

            burst_up = []
            burst_down = []

            # Combine results from both detections
            for u_large, d_large in zip(up_large, down_large):

                burst_nested = np.logical_and(u_large <= up_small, d_large >= up_small)

                if sum(burst_nested) < 2:
                    burst_up.append(u_large)
                    burst_down.append(d_large)
                else:
                    burst_up.append(u_large)
                    burst_up += up_small[burst_nested][1:].tolist()
                    burst_down += (up_small[burst_nested][1:] - 1).tolist()
                    burst_down.append(d_large)

        burst_up = np.asarray(burst_up)
        burst_down = np.asarray(burst_down)
        assert len(burst_up) == len(burst_down)

        # Filter bursts by minimum mean firing rate if required
        if min_mfb_flag and burst_up.size > 0:
            cond_min_mfb = freqSam*(burst_down - burst_up + 1)/(sp[burst_down] - sp[burst_up]) >= 50  # sp/s
            burst_up = burst_up[cond_min_mfb]
            burst_down = burst_down[cond_min_mfb]

        # Skip channels with too few bursts
        if len(burst_up)/(acqTime/60) < mbr_min:
            continue

        # Store burst metrics
        chBurstStart[el] = sp[burst_up]-1  # samples
        chBurstEnd[el] = sp[burst_down]+1  # samples
        chMBR[el] = len(burst_up)/(acqTime/60)  # bursts/min
        chBD[el] = 1000*(sp[burst_down] - sp[burst_up])/freqSam  # ms
        chSPB[el] = burst_down - burst_up + 1  # number of spikes
        chRS[el] = 100*(1 - sum(burst_down - burst_up+1) / len(sp))  # %

    return chBurstStart, chBurstEnd, chMBR, chBD, chSPB, chRS

# ================================================================
# Burst Event Detection
# ================================================================
def BurstEvent(spikes, n_spikes_min=5, isi_th=100., freqSam=10000):
    """
    Identifies burst onset and offset indices based on ISI threshold.

    Parameters:
        spikes (array): Spike times in samples
        n_spikes_min (int): Minimum number of spikes in a burst
        isi_th (float): ISI threshold in ms
        freqSam (float): Sampling frequency in Hz

    Returns:
        burst_up, burst_down (arrays): Start and end indices of detected bursts
    """
    isi_th = int(isi_th/1000*freqSam)  # from ms to samples

    burst_train = np.concatenate(([0], np.diff(spikes) <= isi_th, [0]))
    burst_up = np.where(np.diff(burst_train) == 1)[0]
    burst_down = np.where(np.diff(burst_train) == -1)[0]

    cond = burst_down - burst_up + 1 > n_spikes_min
    burst_up = burst_up[cond]
    burst_down = burst_down[cond]

    assert len(burst_up) == len(burst_down)

    return burst_up, burst_down

# ================================================================
# Log(ISI) Histogram Thresholding
# ================================================================
def LogISIHistogram(spikes, freqSam=10000):
    """
    Determines ISI thresholds using log(ISI) histograms.

    Parameters:
        spikes (list of lists): Spike trains for all channels
        freqSam (float): Sampling frequency in Hz

    Returns:
        chISITh (array): Estimated ISI thresholds per channel
    """
    bins_per_decade = 10
    smooth_size = 3
    min_peak_dist = 2
    void_param_th = 0.7
    chISITh = np.zeros(len(spikes))

    for el, sp in enumerate(spikes):

        if len(sp) <= 1:
            continue

        # Compute ISI in ms
        all_isi = np.diff(sp)/freqSam*1000  # in ms
        max_isi = np.ceil(np.log10(np.amax(all_isi)))
        bins = np.logspace(0, max_isi, int(bins_per_decade*max_isi))
        isi_hist, _ = np.histogram(all_isi, bins=np.concatenate((bins, [np.inf])))
        isi_smooth = np.convolve(isi_hist/sum(isi_hist), np.ones(smooth_size) / smooth_size, mode='same')

        isi_peak, _ = find_peaks(isi_smooth, distance=min_peak_dist)

        # Skip channels without sufficient long ISI peaks
        if not np.any(bins[isi_peak] > 100):
            continue

        isi_peak = isi_peak[np.where(bins[isi_peak] > 100)[0][0]-1:]

        if len(isi_peak) <= 1:
            continue

        void_param = np.zeros(len(isi_peak) - 1)
        idx_minima = np.zeros(len(isi_peak) - 1, dtype=int)

        # Compute void parameter between peaks
        for idx, (peak_l, peak_r) in enumerate(zip(isi_peak[:-1], isi_peak[1:])):
            idx_minima[idx] = peak_l + np.argmin(isi_smooth[peak_l:peak_r])
            void_param[idx] = 1 - isi_smooth[idx_minima[idx]] / np.sqrt(isi_smooth[peak_l] * isi_smooth[peak_r])

        if not np.any(void_param >= void_param_th):
            continue

        isi_th = bins[idx_minima[np.argmax(void_param)]]

        if isi_th > 1000:
            continue

        chISITh[el] = isi_th

    return chISITh

# ================================================================
# Main Script Execution
# ================================================================
# Load spike time and index data from files
spike_time = np.load('spikesT.npy')
cell_index = np.load('spikesI.npy')

# Define simulation parameters
N = 100
fs = 10000
acq_time = 60  # s

# Build spike list per neuron
Spikes = [[] for _ in range(N)]

for spike, cell in enumerate(cell_index):
    Spikes[cell].append(int(spike_time[spike] * fs))

# Run burst detection
burst_up, burst_down, mbr, bd, spb, rs = BurstDetection(Spikes,
                                                                n_spikes_min=10,
                                                                isi_th=100.,
                                                                mbr_min=0.4,
                                                                freqSam=fs,
                                                                acqTime=acq_time,
                                                                log_isi_max_th_flag=True,
                                                                min_mfb_flag=True)

# Save all results as NumPy array
burst_detection = np.array([burst_up, burst_down, mbr, bd, spb, rs], dtype=np.ndarray)
np.save('burst_detection.npy', burst_detection)
