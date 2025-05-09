'''
network.py - Script for the simulation of a neuronal network
    based on biophysical Hodgkin-Huxley-type excitatory neurons.

Written by Valerio Barabino - valerio.barabino@edu.unige.it
NNT Group, DIBRIS, University of Genoa (UNIGE), Italy
last edit: 08-05-2025

This software is provided as-is, without any warranty.
You are free to modify or share this file, provided that the above
copyright notice is kept intact.
'''

from brian2 import *
import numpy as np
from time import time
prefs.codegen.target = "cython"

Ne = 100
El = -39.2 * mV                     # Nernst potential of leaky ions
EK = -80 * mV                       # Nernst potential of potassium
ENa = 70 * mV                       # Nernst potential of sodium
VT = -30.4*mV                       # alters firing threshold of neurons
E_ampa = 0*mV                       # reverse synaptic potential
g_ampa = 0.35*nS                    # conductance increment when spike on pre
tau_ampa = 2*ms                     # synaptic time constant AMPA
E_nmda = 0 * mV                     # Nernst potential of synaptic channels
taud_nmda = 100 * ms                # decay time constant of nmda conductance
taur_nmda = 2 * ms                  # rise time constant of nmda conductance
tau_d = 800 * ms                    # recovery time constant of synaptic depression
Vmax = 25 * um / second

eqs = '''
dV/dt = (noise + -gl*(V-El) - g_na*(m**3)*h*(V-ENa) - g_kd*(n**4)*(V-EK) - I_syn)/Cm : volt
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
I_syn = I_ampa+I_nmda : amp
I_ampa = g_ampa*(V-E_ampa)*s_ampa : amp
ds_ampa/dt = -s_ampa/tau_ampa : 1
I_nmda = g_nmda*(V-E_nmda)*s_nmda_tot/(1+exp(-0.062*V/mV)/3.57) : amp
s_nmda_tot : 1
x : meter
y : meter
area : meter**2
Cm = (1*uF*cm**-2) * area : farad
g_na = (50*mS*cm**-2) * area : siemens
g_kd = (5*mS*cm**-2) * area : siemens
gl = (0.3*mS*cm**-2) * area : siemens
'''

eqs_synE_model = '''
s_nmda_tot_post = w * s_nmda * x_d : 1 (summed)
ds_nmda/dt = -s_nmda/(taud_nmda)+x_nmda*(1-s_nmda)/taur_nmda : 1 (clock-driven)
dx_nmda/dt = -x_nmda/(taur_nmda) : 1 (clock-driven)
dx_d/dt = (1-x_d)/tau_d :1 (clock-driven)
w : 1
'''

eqs_synE_onpre = '''
s_ampa += w * x_d
x_nmda += 1
x_d *= (1-fD)
'''

sources, targets, x_coords, y_coords = ...  # TO BE ADDED
fD_array = 0.0075
g_nmda = 0.0275*nS
sigma = 5.35*mV

seed(1893)

cellsExc = NeuronGroup(Ne, model=eqs, threshold='V>0*mV', refractory=2*ms, method='exponential_euler')
cellsExc.area = np.random.uniform(low=170, high=270, size=Ne)*um**2
cellsExc.V = El
cellsExc.x = x_coords
cellsExc.y = y_coords

syn_EE = Synapses(cellsExc, cellsExc, model=eqs_synE_model, on_pre=eqs_synE_onpre, method='euler')
# syn_EE.connect(p=prob_conn/100, condition='i!=j')
syn_EE.connect(i=sources, j=targets)
syn_EE.w = cellsExc.area[syn_EE.j]/(300*um**2)
syn_EE.delay = '(sqrt((x_pre - x_post)**2 + (y_pre - y_post)**2))/Vmax'

spikesE = SpikeMonitor(cellsExc)

dur = 65*second
start = time()
run(dur, report='text')
print(f'Finished iteration in {time()-start:.2f} seconds')

sp_t = spikesE.t[spikesE.t > 5*second]-5*second
sp_i = spikesE.i[spikesE.t > 5*second]
