README – In Silico Network Model
================================

General Description
-------------------
This project implements and extends a **computational model consisting of 100 modified Hodgkin–Huxley neurons**, originally developed by **Doorn et al. (2023)**.  
The model architecture, synaptic mechanisms, and overall network structure are directly **adapted from their publicly available implementation**.

DOI: https://doi.org/10.1016/j.stemcr.2023.06.003
Original model repository: https://gitlab.utwente.nl/m7706783/mea-model

------------------------------------------------------------

Overview
--------
The current implementation reproduces and extends the **conductance-based network model** from Doorn et al. (2023), providing a framework for:
- simulation of mixed excitatory populations;
- manipulation of noise amplitude, synaptic conductances, and synaptic plasticity;
- analysis of firing and bursting dynamics across multiple network realizations.

The project combines **neuronal simulations** (Brian2-based) with **post-simulation analysis** including burst detection, spike time histogram fitting, and principal component analysis (PCA).

------------------------------------------------------------

Project Structure
-----------------

```text
Project/
│
├── Analysis/
│   ├── burst_detection.py
│   ├── fittingSTH.py
│   ├── principal_component_analysis.ipynb
│   └── saveSTH.py
│
└── Model/
    ├── Topologies/
    │   ├── sources_*.npy
    │   └── targets_*.npy
    │
    ├── Figures/
    │   ├── degree_distribution.py
    │   ├── delay_distribution.py
    │   ├── EPSP_EPSC_simulations.py
    │   ├── FI_curve_noise.py
    │   ├── generate_concurrent_topologies.py
    │   ├── Suppl_FI_curve_Idc.py
    │   ├── Suppl_Integration_step_Idc.ipynb
    │   ├── Suppl_Integration_step_noise.ipynb
    │   ├── Suppl_Integration_step_plot_error.ipynb
    │   ├── Suppl_Multiple_Networks_gampa.ipynb
    │   ├── Suppl_Multiple_Networks_Idc.ipynb
    │   ├── Suppl_Multiple_Networks_Idc_no_noise.ipynb
    │   ├── Suppl_pacemakers.ipynb
    │   └── Syn_STD.py
    │
    └── Main_network_model.ipynp
```
------------------------------------------------------------

Model Directory
---------------

**Main_network_model.ipynb**  
Main notebook defining and simulating the full neural network model.  
It includes:
- initialization of simulation parameters;
- loading of network topologies;
- execution of network simulations and saving of resulting activity.

**Topologies/**  
Contains `.npy` files defining the network connectivity:  
- `sources_*.npy` → indices of presynaptic neurons  
- `targets_*.npy` → indices of postsynaptic neurons  
Each source/target pair represents a different network topology (e.g., random, small-world, scale-free and concurrent).

**Figures/**  
Contains all scripts and notebooks used to **generate both main and supplementary figures**.  
This includes:
- parameter sensitivity analyses (e.g., g_AMPA, I_DC, noise),
- numerical and stability tests,
- studies of F–I curves and short-term synaptic depression (STD),
- figure generation for validation and supplementary results.  
Both `.py` and `.ipynb` files are provided for automated and interactive reproduction of results.

------------------------------------------------------------

Analysis Directory
------------------
Contains tools for post-simulation analysis:
- `burst_detection.py` – automatic burst detection and quantification;
- `fittingSTH.py` – fitting of Spike Time Histograms (STH) using analytical models;
- `principal_component_analysis.ipynb` – PCA for dimensionality reduction and visualization of network activity patterns;
- `saveSTH.py` – structured saving of STH analysis results.
