
'''
EPSP_EPSC_simulations.py

This script simulates the postsynaptic response (EPSC and EPSP) of a Hodgkin-Huxley-type neuron
driven by a single presynaptic excitatory input, using the Brian2 simulator.
The outputs are used to generate panels for Figure 2.
'''
from brian2 import *

prefs.codegen.target = "numpy"
seed(3)

# ---------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------
# neuron parameters
area = 300*um**2
Cm = (1*uF*cm**-2) * area           # membrane capacitance
El = -39.2 * mV                     # Nernst potential of leaky ions
EK = -80 * mV                       # Nernst potential of potassium
ENa = 70 * mV                       # Nernst potential of sodium
g_na = (50*mS*cm**-2) * area        # maximal conductance of sodium channels
g_kd = (5*mS*cm**-2) * area         # maximal conductance of potassium
gl = (0.3*mS*cm**-2) * area         # maximal leak conductance
VT = -30.4*mV                       # alters firing threshold of neurons
sigma = 4.1*mV                      # standard deviation of the noisy voltage fluctuations

# synaptic parameters
E_ampa = 0*mV                       # reverse synaptic potential
g_ampa = 0.2808*nS                  # conductance increment when spike on pre
tau_ampa = 2*ms                     # synaptic time constant AMPA
S = 1.65                            # weight scaling parameter
E_nmda = 0 * mV                     # Nernst potentials of synaptic channels
g_nmda = 0.0981*nS                  # conductance increment when spike on pre
taud_nmda = 100 * ms                # decay time constant of nmda conductance
taur_nmda = 2 * ms                  # rise time constant of nmda conductance
alpha_nmda = 0.5 * kHz
E_gaba = -80*mV                     # inhibitory reversal potential
g_gaba = 0.05*nS                     # conductance increment when spike on pre
tau_gaba = 10*ms                    # synaptic time constant GABA

# ================================================================
# Definition of Cell Intrinsic and Synaptic Equations
# ================================================================
eqs = '''
dV/dt = (-gl*(V-El)-g_na*(m*m*m)*h*(V-ENa)-g_kd*(n*n*n*n)*(V-EK)-I_syn)/Cm : volt
dm/dt = alpha_m*(1-m)-beta_m*m : 1
dh/dt = alpha_h*(1-h)-beta_h*h : 1
dn/dt = alpha_n*(1-n)-beta_n*n : 1
alpha_m = 0.32*(mV**-1)*4*mV/exprel((13*mV-V+VT)/(4*mV))/ms : Hz
beta_m = 0.28*(mV**-1)*5*mV/exprel((V-VT-40*mV)/(5*mV))/ms : Hz
alpha_h = 0.128*exp((17*mV-V+VT)/(18*mV))/ms : Hz
beta_h = 4./(1+exp((40*mV-V+VT)/(5*mV)))/ms : Hz
alpha_n = 0.032*(mV**-1)*5*mV/exprel((15*mV-V+VT)/(5*mV))/ms : Hz
beta_n = .5*exp((10*mV-V+VT)/(40*mV))/ms : Hz
noise = sigma*(2*gl*Cm)**.5*randn()/sqrt(dt) : amp (constant over dt)
I_syn = I_ampa + I_nmda : amp
ds_gaba/dt = -s_gaba/tau_gaba :1 
I_ampa = g_ampa*(V-E_ampa)*s_ampa : amp
ds_ampa/dt = -s_ampa/tau_ampa : 1
I_nmda = g_nmda*(V-E_nmda)*s_nmda_tot/(1+exp(-0.062*V/mV)/3.57) : amp
s_nmda_tot : 1
x : meter
y : meter
g_AHP : siemens
'''

eqs_pre = '''
dV/dt = (1-V)/(100*ms) : 1
'''
eqs_synE_model = '''
s_nmda_tot_post = S * s_nmda : 1 (summed)
ds_nmda/dt = -s_nmda/(taud_nmda)+alpha_nmda*x_nmda*(1-s_nmda) : 1 (clock-driven)
dx_nmda/dt = -x_nmda/(taur_nmda) : 1 (clock-driven)
w : 1
'''

eqs_synE_onpre = '''
s_ampa += S
x_nmda += 1
'''

# ================================================================
# Network Construction
# ================================================================
P1 = NeuronGroup(1, model=eqs_pre, threshold='V>0.99', refractory=300*second, reset= 'V=0',
                method='exponential_euler')
P1.V = 0.95

P2 = NeuronGroup(1, model=eqs, threshold='V>-20*mV', refractory=3*ms,
                method='exponential_euler')
P2.V = El

syn_EE = Synapses(P1, P2, model=eqs_synE_model, on_pre=eqs_synE_onpre, method='euler')
syn_EE.connect()

# ---------------------------------------------------------------------
# Simulation and Output generation
# ---------------------------------------------------------------------
M1 = SpikeMonitor(P1)
M2 = StateMonitor(P2, ['V','I_syn','I_ampa','I_nmda'], record=True)

dur = 1000*ms
run(dur)

# fig1, ax1 = plt.subplots(2,1,figsize=(12,6))
# fig1.suptitle('Excitatory synaptic contribution', fontsize=24)
#
# ax1[0].plot(M2.t/ms, -M2[0].I_syn/pA, linewidth=3.5)
# ax1[0].fill_between(M2.t/ms, -M2[0].I_syn/pA, alpha=0.7, linewidth=3.5)
# ax1[0].set_ylabel('EPSC (pA)', fontsize=16)
# ax1[0].set_xlim([150,250])
# ax1[0].spines[['right', 'top']].set_visible(False)
# ax1[0].spines[['left', 'bottom']].set_linewidth(2)
# ax1[0].tick_params(width=2)
# ax1[0].set_yticks(ticks=[0,5,10,15,20], labels=[0,5,10,15,20], fontsize=12)
# ax1[0].set_xticks(ticks=[], labels=[])
#
# ax1[1].plot(M2.t/ms, M2[0].V/mV+39.2, color='tab:orange', linewidth=3.5)
# ax1[1].fill_between(M2.t/ms, 0, M2[0].V/mV+39.2, color='tab:orange', alpha=0.7, linewidth=3.5)
# ax1[1].set_xlabel('Time (ms)', fontsize=16)
# ax1[1].set_ylabel('EPSP (mV)', fontsize=16)
# ax1[1].set_xlim([150,250])
# ax1[1].spines[['right', 'top']].set_visible(False)
# ax1[1].spines[['left', 'bottom']].set_linewidth(2)
# ax1[1].tick_params(width=2)
# ax1[1].set_yticks(ticks=[0,2,4,6], labels=[0,2,4,6], fontsize=12)
# ax1[1].set_xticks(ticks=[175,200,225,250], labels=[25,50,75,100], fontsize=12)

# fig1.savefig('EPSC.tif', format="tiff", pil_kwargs={"compression": "tiff_lzw"}, dpi=600)

colors = ['#1f77b4', '#ff7f0e', '#bcbd22']

csyn = np.array((255, 127, 102))/255
cvolt = np.array((1,140,149))/255

fig2, ax2 = plt.subplots(2,1,figsize=(6/2.54, 5/2.54), layout='constrained')
ax2[0].set_title('Synaptic contribution', fontsize=10)

ax2[0].plot(M2.t/ms, -M2[0].I_syn/pA, color=csyn, label='Overall', linewidth=1)
ax2[0].plot(M2.t/ms, -M2[0].I_nmda/pA, color='k', linestyle='dashed', label='NMDA', linewidth=1)
ax2[0].plot(M2.t/ms, -M2[0].I_ampa/pA, color='k', linestyle='dotted', label='AMPA', linewidth=1)
ax2[0].fill_between(M2.t/ms, -M2[0].I_syn/pA, color=csyn, alpha=0.7, linewidth=1)
ax2[0].set_ylabel('EPSC (pA)', fontsize=8)
ax2[0].legend(loc='best', frameon=False, fontsize=6)
ax2[0].set_xlim([150, 200])
ax2[0].set_ylim([-1, 19])
ax2[0].spines[['right', 'top']].set_visible(False)
ax2[0].spines[['left', 'bottom']].set_linewidth(1)
ax2[0].tick_params(width=1)
ax2[0].set_yticks(ticks=[0,5,10,15,20], labels=[0,5,10,15,20], fontsize=6)
ax2[0].set_xticks(ticks=[], labels=[])

ax2[1].plot(M2.t/ms, M2[0].V/mV+39.2, color=cvolt, linewidth=1)
ax2[1].fill_between(M2.t/ms, 0, M2[0].V/mV+39.2, color=cvolt, alpha=0.7, linewidth=1)
ax2[1].set_xlabel('Time (ms)', fontsize=8)
ax2[1].set_ylabel('EPSP (mV)', fontsize=8)
ax2[1].set_xlim([150, 200])
ax2[1].spines[['right', 'top']].set_visible(False)
ax2[1].spines[['left', 'bottom']].set_linewidth(1)
ax2[1].tick_params(width=1)
ax2[1].set_yticks(ticks=[0,2,4,6], labels=[0,2,4,6], fontsize=6)
ax2[1].set_xticks(ticks=[160,170,180,190,200], labels=[10,20,30,40,50], fontsize=6)

fig2.align_ylabels()
# plt.show()
fig2.savefig('EPSP_new.tif', format="tiff", pil_kwargs={"compression": "tiff_lzw"}, dpi=600)
