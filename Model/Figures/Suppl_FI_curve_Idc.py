
'''
FI_curve.py - Script for generating a current-firing rate (F I) curve
    using a Hodgkin Huxley type neuron model with DC input in Brian2.
    The outputs are used to generate panels for Supplementary Figure S2.
'''

# ================================================================
# Imports and Global Settings
# ================================================================
from brian2 import *
import numpy as np
from matplotlib import pyplot as plt

prefs.codegen.target = "numpy"

# ================================================================
# Biophysical Parameters
# ================================================================
area = 220*um**2
Cm = (1*uF*cm**-2) * area
g_na = (50*mS*cm**-2) * area
g_kd = (5*mS*cm**-2) * area
gl = (0.3*mS*cm**-2) * area
El = -39.2 * mV                     # Nernst potential of leaky ions
EK = -80 * mV                       # Nernst potential of potassium
ENa = 70 * mV                       # Nernst potential of sodium
VT = -30.4*mV                       # alters firing threshold of neurons
sigma = 0*mV                        # standard deviation of the noisy voltage fluctuations
I_DC = np.linspace(8,10,100)*pA

# ================================================================
# Hodgkin–Huxley Model Equations
# ================================================================
eqs = '''
dV/dt = (-gl*(V-El) - g_na*(m**3)*h*(V-ENa) - g_kd*(n**4)*(V-EK) + I)/Cm : volt
dm/dt = alpha_m*(1-m)-beta_m*m : 1
dh/dt = alpha_h*(1-h)-beta_h*h : 1
dn/dt = alpha_n*(1-n)-beta_n*n : 1
alpha_m = 0.32*(mV**-1)*4*mV/exprel((13*mV-V+VT)/(4*mV))/ms : Hz
beta_m = 0.28*(mV**-1)*5*mV/exprel((V-VT-40*mV)/(5*mV))/ms : Hz
alpha_h = 0.128*exp((17*mV-V+VT)/(18*mV))/ms : Hz
beta_h = 4./(1+exp((40*mV-V+VT)/(5*mV)))/ms : Hz
alpha_n = 0.032*(mV**-1)*5*mV/exprel((15*mV-V+VT)/(5*mV))/ms : Hz
beta_n = .5*exp((10*mV-V+VT)/(40*mV))/ms : Hz
I : amp
'''

# ================================================================
# Neuron Group Initialization
# ================================================================
cellsExc = NeuronGroup(100, model=eqs, threshold='V>0*mV', refractory=2*ms, method='exponential_euler')
cellsExc.V = El
cellsExc.I = I_DC

# ================================================================
# Simulation
# ================================================================
spikes = SpikeMonitor(cellsExc)
mon = StateMonitor(cellsExc, 'V', record=True)

dur = 1*second
run(dur, report='text')

firing_rate = spikes.count/Hz
color = '#003366'


fig, ax = plt.subplots(1, 1, figsize=(5.3/2.54, 5.3/2.54))
ax.plot(I_DC[1:-1]/pA, np.convolve(firing_rate, np.ones(3)/3, mode='valid'), color='k', linewidth=1.5)
# ax.set_title('Current-driven Activity', fontsize=10)
# ax.set_title('Mean Firing Rate', fontsize=8)
ax.set_ylabel('MFR (spikes/s)', fontsize=8)
ax.set_xlabel('Current intensity (pA)', fontsize=8)
ax.spines[['right', 'top']].set_visible(False)
ax.spines[['left', 'bottom']].set_linewidth(1)
ax.tick_params(width=1)
ax.set_xlim(8,10)
ax.set_ylim(-1,50)
ax.set_xticks(ticks=[8,8.5,9,9.5,10], labels=[8,8.5,9,9.5,10], fontsize=6)
ax.set_yticks(ticks=[0,10,20,30,40,50], labels=[0,10,20,30,40,50], fontsize=6)

fig.tight_layout()
# plt.show()
fig.savefig('FI_curve1_new.tif', format="tiff", pil_kwargs={"compression": "tiff_lzw"}, dpi=600)
