"""
FI_curve_noise.py
Generates a current–firing rate (F–I) curve for a Hodgkin-Huxley neuron model
with noisy current input (sigma). Uses Brian2.
"""

# ================================================================
# Imports and Settings
# ================================================================
from brian2 import *
import numpy as np
import matplotlib.pyplot as plt

prefs.codegen.target = "numpy"
seed(1893)
defaultclock.dt = 0.01*ms

# ================================================================
# Biophysical Parameters
# ================================================================
area = 220*umetre**2
Cm = (1*uF*cm**-2) * area
g_na = (50*mS*cm**-2) * area
g_kd = (5*mS*cm**-2) * area
gl = (0.3*mS*cm**-2) * area
El = -39.2*mV
EK = -80*mV
ENa = 70*mV
VT = -30.4*mV

# ================================================================
# Model Equations
# ================================================================
eqs = '''
dV/dt = (-gl*(V-El) - g_na*(m**3)*h*(V-ENa) - g_kd*(n**4)*(V-EK) + noise)/Cm : volt
dm/dt = alpha_m*(1-m)-beta_m*m : 1
dh/dt = alpha_h*(1-h)-beta_h*h : 1
dn/dt = alpha_n*(1-n)-beta_n*n : 1

alpha_m = 0.32*(mV**-1)*4*mV/exprel((13*mV-V+VT)/(4*mV))/ms : Hz
beta_m  = 0.28*(mV**-1)*5*mV/exprel((V-VT-40*mV)/(5*mV))/ms : Hz
alpha_h = 0.128*exp((17*mV-V+VT)/(18*mV))/ms : Hz
beta_h  = 4./(1+exp((40*mV-V+VT)/(5*mV)))/ms : Hz
alpha_n = 0.032*(mV**-1)*5*mV/exprel((15*mV-V+VT)/(5*mV))/ms : Hz
beta_n  = .5*exp((10*mV-V+VT)/(40*mV))/ms : Hz

noise = sigma*(2*gl*Cm)**.5*randn()/sqrt(dt) : amp (constant over dt)
sigma : volt
'''

# ================================================================
# Simulation Parameters
# ================================================================
n_cells = 100
duration = 10*second
sigma_values = np.linspace(0, 10, 21)*mV  # da 0 a 10 mV in 21 step

# ================================================================
# Simulation Loop
# ================================================================
mfr_list = []

for sigma_val in sigma_values:
    print(f"Running simulation for sigma = {sigma_val/mV:.2f} mV")

    cells = NeuronGroup(n_cells, model=eqs, threshold='V > 0*mV', refractory=2*ms, method='exponential_euler')
    cells.V = El
    cells.sigma = sigma_val

    spikes = SpikeMonitor(cells)

    run(duration, report='text')

    # Mean Firing Rate per cell
    mfr = np.mean(spikes.count / duration)
    mfr_list.append(mfr / Hz)  # convert to Hz

# ================================================================
# Save & Plot
# ================================================================
mfr_array = np.array(mfr_list)
np.save('mfr_sigma.npy', mfr_array)

plt.figure(figsize=(5,4))
plt.plot(sigma_values/mV, mfr_array, '-o', color='#D35400')
plt.xlabel('Noise amplitude (mV)')
plt.ylabel('Mean firing rate (Hz)')
plt.title('F-I Curve (Noise-driven activity)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()