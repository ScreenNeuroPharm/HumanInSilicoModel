import matplotlib.pyplot as plt
from brian2 import *

# ================================================================
# Global Settings
# ================================================================
prefs.codegen.target = "numpy"  # use NumPy backend for code generation

# ================================================================
# Biophysical Parameters
# ================================================================
area = 300*um**2
Cm = (1*uF*cm**-2) * area           # membrane capacitance
El = -39.2 * mV                     # Nernst potential of leak ions
EK = -80 * mV                       # Nernst potential of potassium
ENa = 70 * mV                       # Nernst potential of sodium
g_na = (50*mS*cm**-2) * area        # maximal sodium conductance
g_kd = (5*mS*cm**-2) * area         # maximal potassium conductance
gl = (0.3*mS*cm**-2) * area         # leak conductance
VT = -30.4*mV                       # threshold offset
sigma = 4.1*mV                      # standard deviation of noisy voltage fluctuations

# ================================================================
# Synaptic Parameters (AMPA)
# ================================================================
E_ampa = 0*mV                       # AMPA reversal potential
g_ampa = 0.2808*nS                  # synaptic conductance increment per presynaptic spike
tau_ampa = 2*ms                     # AMPA decay time constant

# ================================================================
# Short-Term Depression Parameters
# ================================================================
tau_d = 800*ms                      # recovery time constant for synaptic depression

# ================================================================
# Postsynaptic Neuron Model (Hodgkin–Huxley type + AMPA input)
# ================================================================
eqs = '''
dV/dt = (-gl*(V-El)-g_na*(m*m*m)*h*(V-ENa)-g_kd*(n*n*n*n)*(V-EK)-I_ampa)/Cm : volt
dm/dt = alpha_m*(1-m)-beta_m*m : 1
dh/dt = alpha_h*(1-h)-beta_h*h : 1
dn/dt = alpha_n*(1-n)-beta_n*n : 1

alpha_m = 0.32*(mV**-1)*4*mV/exprel((13*mV-V+VT)/(4*mV))/ms : Hz
beta_m = 0.28*(mV**-1)*5*mV/exprel((V-VT-40*mV)/(5*mV))/ms : Hz
alpha_h = 0.128*exp((17*mV-V+VT)/(18*mV))/ms : Hz
beta_h = 4./(1+exp((40*mV-V+VT)/(5*mV)))/ms : Hz
alpha_n = 0.032*(mV**-1)*5*mV/exprel((15*mV-V+VT)/(5*mV))/ms : Hz
beta_n = .5*exp((10*mV-V+VT)/(40*mV))/ms : Hz

I_ampa = g_ampa*(V-E_ampa)*s_ampa : amp
ds_ampa/dt = -s_ampa/tau_ampa : 1
'''

# ================================================================
# Presynaptic Neuron Model (simple spike generator)
# ================================================================
eqs_pre = '''
dV/dt = (1-V)/(1*ms) : 1
'''

# ================================================================
# Network Definition
# ================================================================
P1 = NeuronGroup(1, model=eqs_pre, threshold='V>0.99', refractory=10*ms, reset='V=0',
                 method='exponential_euler')  # presynaptic neuron (periodic spiking)
P1.V = 0

P2 = NeuronGroup(1, model=eqs, threshold='V>-20*mV', refractory=3*ms, method='exponential_euler')  # postsynaptic neuron
P2.V = El

# ================================================================
# Synapse Model (Short-Term Depression)
# ================================================================
eqs_synapsmodel = '''
dx_d/dt = (1-x_d)/tau_d : 1 (clock-driven)    # recovery of available vesicles
S : 1                                         # synaptic scaling factor
U : 1                                         # utilization factor
'''

eqs_onpre = '''
x_d *= (1-U)          # vesicle depletion upon presynaptic spike
s_ampa += S * x_d     # AMPA conductance increment scaled by available vesicles
'''

# ================================================================
# Synapse Connection and Initialization
# ================================================================
C = Synapses(P1, P2, model=eqs_synapsmodel, on_pre=eqs_onpre, method='euler')
C.connect()
C.x_d = 1  # all vesicles initially available

# ================================================================
# Monitors
# ================================================================
M1 = SpikeMonitor(P1)                       # presynaptic spikes
M2 = StateMonitor(P2, ['V'], record=True)   # postsynaptic membrane potential
Msyn = StateMonitor(C, 'x_d', record=True)  # synaptic depression variable

# ================================================================
# Simulation
# ================================================================
dur = 1000*ms
C.S = 1.65
C.U = 0.15
run(dur)

# ================================================================
# Plotting
# ================================================================
x1 = 0
x2 = 100

colors = ['#1f77b4', '#ff7f0e', '#bcbd22']
csyn = np.array((255, 127, 102))/255     # color for synaptic variable
cvolt = np.array((1, 140, 149))/255      # color for voltage trace

fig, ax = plt.subplots(2, 1, figsize=(6/2.54, 5/2.54), layout='constrained')

# ---------- Panel 1: Synaptic Depression ----------
ax[0].set_title('Synaptic depression', fontsize=10)
ax[0].plot(Msyn.t/ms, 100*Msyn[0].x_d, color=csyn, linewidth=1)
ax[0].set_ylabel('Available\nvesicles (%)', fontsize=8)
ax[0].set_xlim([x1, x2])
ax[0].set_ylim(0, 101)
ax[0].spines['left'].set_bounds(0, 100)
ax[0].spines[['right', 'top']].set_visible(False)
ax[0].spines[['left', 'bottom']].set_linewidth(1)
ax[0].tick_params(width=1)
ax[0].set_yticks(ticks=[25, 50, 75, 100], labels=[25, 50, 75, 100], fontsize=6)
ax[0].set_xticks([])

# ---------- Panel 2: Postsynaptic Voltage ----------
ax[1].plot(M2.t/ms, M2[0].V/mV + 39.2, color=cvolt, linewidth=1)
ax[1].set_ylabel('EPSPs (mV)', fontsize=8)
ax[1].set_xlim([x1, x2])
ax[1].spines[['right', 'top']].set_visible(False)
ax[1].spines[['left', 'bottom']].set_linewidth(1)
ax[1].tick_params(width=1)
ax[1].set_yticks(ticks=[2, 4, 6], labels=[2, 4, 6], fontsize=6)
ax[1].set_xlabel('Time (ms)', fontsize=8)
ax[1].set_xticks(ticks=[20, 40, 60, 80, 100], labels=[20, 40, 60, 80, 100], fontsize=6)

fig.align_ylabels()

# ================================================================
# Save Figure
# ================================================================
# plt.show()
fig.savefig('STD_new.tif', format="tiff", pil_kwargs={"compression": "tiff_lzw"}, dpi=600)
