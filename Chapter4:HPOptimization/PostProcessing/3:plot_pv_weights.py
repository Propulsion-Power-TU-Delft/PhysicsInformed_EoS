#!/usr/bin/env python3
###############################################################################################
#       #      _____ __  _____      ____        __        __  ____                   #        #  
#       #     / ___// / / /__ \    / __ \____ _/ /_____ _/  |/  (_)___  ___  _____   #        #  
#       #     \__ \/ / / /__/ /   / / / / __ `/ __/ __ `/ /|_/ / / __ \/ _ \/ ___/   #        #      
#       #    ___/ / /_/ // __/   / /_/ / /_/ / /_/ /_/ / /  / / / / / /  __/ /       #        #  
#       #   /____/\____//____/  /_____/\__,_/\__/\__,_/_/  /_/_/_/ /_/\___/_/        #        #
#       #                                                                            #        #
###############################################################################################

############################ FILE NAME: 3:plot_pv_weights.py ##################################
#=============================================================================================#
# author: Evert Bunschoten                                                                    |
#    :PhD Candidate ,                                                                         |
#    :Flight Power and Propulsion                                                             |
#    :TU Delft,                                                                               |
#    :The Netherlands                                                                         |
#                                                                                             |
#                                                                                             |
# Description:                                                                                |
# Plot the coefficients of the three progress variables.                                      |
#                                                                                             |  
# Version: 2.0.0                                                                              |
#                                                                                             |
#=============================================================================================#
import numpy as np 
import matplotlib.pyplot as plt
from su2dataminer.config import Config_FGM

# Load SU2 DataMiner configurations
Config_wrp = Config_FGM("../WRP.cfg")
Config_pca = Config_FGM("../PCA.cfg")
Config_optimized = Config_FGM("../OPT.cfg")

# Retrieve the coefficients for each progress variable
species_names = Config_wrp.gas.species_names
pv_sp_wrp = Config_wrp.GetProgressVariableSpecies()
pv_w_wrp = Config_wrp.GetProgressVariableWeights()

pv_w_wrp_all = np.zeros(len(species_names))
for iSp, Sp in enumerate(pv_sp_wrp):
    pv_w_wrp_all[Config_wrp.gas.species_index(Sp)] = pv_w_wrp[iSp]
idx_significant_wrp = np.abs(pv_w_wrp_all) > 1e-4

pv_sp_pca = Config_pca.GetProgressVariableSpecies()
pv_w_pca = Config_pca.GetProgressVariableWeights()
pv_w_pca_all = np.zeros(len(species_names))
for iSp, Sp in enumerate(pv_sp_pca):
    pv_w_pca_all[Config_pca.gas.species_index(Sp)] = pv_w_pca[iSp]
idx_significant_pca = np.abs(pv_w_pca_all) > 1e-4

pv_sp_optimized = Config_optimized.GetProgressVariableSpecies()
pv_w_optimized = Config_optimized.GetProgressVariableWeights()
pv_w_optimized_all = np.zeros(len(species_names))
for iSp, Sp in enumerate(pv_sp_optimized):
    pv_w_optimized_all[Config_optimized.gas.species_index(Sp)] = pv_w_optimized[iSp]
idx_significant_optimized = np.abs(pv_w_optimized_all) > 1e-4
   
# Plot only the coefficients with a significant value.  
idx_all_significant = [any([d, p, o]) for d,p,o in zip(idx_significant_wrp,idx_significant_pca,idx_significant_optimized)]
x_all = np.arange(len(species_names))
x_bars = np.arange(len(x_all[idx_all_significant]))*1.5

N=3
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.cubehelix(np.linspace(0,1,N+1)))
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

color_wrp = colors[0]
color_pca = colors[1]
color_optim = colors[2]
fsize = 24
figformat='pdf'
fig = plt.figure(figsize=[12,7])
ax = plt.axes()
w = 1.0
w_bar = w / (N) 

for i in range(0,len(x_bars),2):
    ax.axvspan(x_bars[i]-0.75, x_bars[i]+0.75, alpha=0.05, color='k')
ax.bar(x=x_bars+ (0.5 + 0)*w_bar - 0.5*w,height=pv_w_wrp_all[idx_all_significant],width=w_bar,color=color_wrp,zorder=3,label=r'WRP')
ax.bar(x=x_bars+ (0.5 + 1)*w_bar - 0.5*w,height=pv_w_pca_all[idx_all_significant],width=w_bar,color=color_pca,zorder=3,label=r'PCA')
ax.bar(x=x_bars+ (0.5 + 2)*w_bar - 0.5*w,height=pv_w_optimized_all[idx_all_significant],width=w_bar,color=color_optim,zorder=3,label=r"SPV")

ax.set_xticks(x_bars)
ax.set_xticklabels(np.array([r""+s for s in species_names])[idx_all_significant])
ax.tick_params(which='both',labelsize=fsize)
ax.set_xlabel("Progress variable species",fontsize=fsize)
ax.set_ylabel("Species weight value (ai)[-]",fontsize=fsize)
ax.grid(zorder=0)
ax.set_title("Progress variable weights visualization",fontsize=fsize)
ax.legend(fontsize=fsize,ncols=N,loc="upper center",bbox_to_anchor=[0.5, -0.13],fancybox=True,shadow=True)
plt.show()

