# ================================================================
# STH Exponential Fitting
# ================================================================
# This script analyzes the Spike Time Histogram (STH) to extract
# rise and decay kinetics using exponential fitting.
# Both single and double exponential models are implemented.
# The script fits the rising and decaying phases of the STH curve
# and computes the corresponding time constants (τ) and R² scores.
# ================================================================

from matplotlib import pyplot as plt
from scipy.signal import correlate, find_peaks, argrelmin
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import pickle
import os

# ================================================================
# Exponential Models
# ================================================================
def single_exp(x, a, b, d):
    return a * (np.exp(b * x)) + d

def double_exp(x, a0, b0, a1, b1, d):
    return a0 * (np.exp(b0 * x)) + a1 * (np.exp(b1 * x)) + d

def exp2(x, a, b, c, d):
    return a*np.exp(-b*x) + c*np.exp(-d*x)

def exp1(x, a, b):
    return a*np.exp(-b*x)


# ================================================================
# Parameters and Data Loading
# ================================================================
freqSam = 10_000
sth = np.load('sth.npy')

# ================================================================
# Rising Phase Fitting
# ================================================================
# Identify the start and end of the rising phase based on relative amplitude
idx_peak = np.argmax(sth)
tmp = np.where(sth > 0.1*sth[idx_peak])[0]
start_rise = tmp[tmp < idx_peak][0]
tmp = np.where(sth > 0.8 * sth[idx_peak])[0]
end_rise = tmp[tmp < idx_peak][0]

# Extract rising phase segment
sth_rise = sth[start_rise:end_rise]

# Fit the rising phase using a double exponential model
bounds = ((0,0,0,0,-1000),(1000, 0.01, 1000, 0.01, 1000))
param_rise, cov_rise = curve_fit(double_exp, np.arange(len(sth_rise)), sth_rise, bounds=bounds, maxfev=100000)

# Extract the steepest slope from fitted parameters
slope_rise = max(param_rise[1], param_rise[3])

# Compute fitted curve and R² score
fitted_rise = double_exp(np.arange(len(sth_rise)), *param_rise)
r2_rise = r2_score(fitted_rise, sth_rise)

# plt.plot(np.arange(start_rise, end_rise)*1000/freqSam, sth_rise, label='data')
# plt.plot(np.arange(start_rise, end_rise)*1000/freqSam, fitted_rise, label='fitted')
# plt.title('Rise Phase')
# plt.xlabel('ms')
# plt.ylabel('spikes/s')
# plt.legend()
# plt.show()

# print(f'Slope = {1/(slope_rise)/10:.2f} ms')
# print(f'r-squared: {r2_rise:.4f}')

# Convert slope to time constant (τ = 1/slope) and format values
df_rise_tau = round(1/(slope_rise)/10, 1)
df_rise_r2= round(r2_rise, 4)

# ================================================================
# Decaying Phase Fitting
# ================================================================
# Identify the start and end of the decay phase
tmp = np.where(sth < 0.8 * sth[idx_peak])[0]
start_decay = tmp[tmp > idx_peak][0]
tmp = np.where(sth < 0.1 * sth[idx_peak])[0]
end_decay = tmp[tmp > idx_peak][0]

# Extract decaying phase segment
sth_decay = sth[start_decay:end_decay]
# if netID == 'SF2_RND':
#     bounds = ((0,0,0,0),(10000, 0.01, 10000, 0.01))
#     param_decay, cov_decay = curve_fit(exp2, np.arange(len(sth_decay)), sth_decay, p0=(1000,0.0005,1000,0.0005), bounds=bounds)
# else:

# Fit decay using double exponential model
bounds = (0, np.inf)
param_decay, cov_decay = curve_fit(exp2, np.arange(len(sth_decay)), sth_decay)

# Extract slope (steepest decay rate)
slope_decay = max(param_decay[1], param_decay[3])

# Compute fitted curve and adjusted R²
fitted_decay = exp2(np.arange(len(sth_decay)), *param_decay)
r2_decay = r2_score(fitted_decay, sth_decay)
adj_r2_decay = 1-(1-r2_decay)*(len(sth_decay)-1)/(len(sth_decay)-len(param_decay)-1)

# plt.plot(np.arange(start_decay, end_decay)*1000/freqSam, sth_decay, label='data')
# plt.plot(np.arange(start_decay, end_decay)*1000/freqSam, fitted_decay, label='fitted')
# plt.title('Decay Phase')
# plt.xlabel('ms')
# plt.ylabel('spikes/s')
# plt.legend()
# plt.show()

# print(f'Slope = {1/(slope_decay)/10:.2f} ms')
# print(f'r-squared: {r2_decay:.4f}')

# Convert slope to time constant (τ = 1/slope)
df_decay_tau = round(1/(slope_decay)/10, 1)
df_decay_r2 = round(r2_decay, 4)

# ================================================================
# Optional Plotting Section (commented out)
# ================================================================

# plt.plot(np.arange(len(sth))*1000/freqSam, sth, linewidth=3)
#
# plt.plot(start_rise*1000/freqSam, sth[start_rise], 'ro', markersize=7)
# plt.plot(np.arange(start_rise, end_rise)*1000/freqSam, fitted_rise, '--r', linewidth=1.5)
# plt.plot(end_rise*1000/freqSam, sth[end_rise], 'ro', markersize=7)
#
# plt.plot(start_decay*1000/freqSam, sth[start_decay], 'ro', markersize=7)
# plt.plot(np.arange(start_decay, end_decay)*1000/freqSam, fitted_decay, '--r', linewidth=1.5)
# plt.plot(end_decay*1000/freqSam, sth[end_decay], 'ro', markersize=7)
#
# plt.xlabel('ms')
# plt.ylabel('spikes/s')
# plt.title('STH')
# plt.show()

# with pd.ExcelWriter('Fitting_STH.xlsx', engine='openpyxl') as writer:
#     df_rise_tau.to_excel(writer, sheet_name='Rise (slope)')
#     df_rise_r2.to_excel(writer, sheet_name='Rise (r2)')
#     df_decay_tau.to_excel(writer, sheet_name='Decay (slope)')
#     df_decay_r2.to_excel(writer, sheet_name='Decay (r2)')
