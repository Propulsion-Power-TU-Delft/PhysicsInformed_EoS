#!/usr/bin/env python3
###############################################################################################
#       #      _____ __  _____      ____        __        __  ____                   #        #  
#       #     / ___// / / /__ \    / __ \____ _/ /_____ _/  |/  (_)___  ___  _____   #        #  
#       #     \__ \/ / / /__/ /   / / / / __ `/ __/ __ `/ /|_/ / / __ \/ _ \/ ___/   #        #      
#       #    ___/ / /_/ // __/   / /_/ / /_/ / /_/ /_/ / /  / / / / / /  __/ /       #        #  
#       #   /____/\____//____/  /_____/\__,_/\__/\__,_/_/  /_/_/_/ /_/\___/_/        #        #
#       #                                                                            #        #
###############################################################################################

######################## FILE NAME: 1:compare_merit_functions.py ##############################
#=============================================================================================#
# author: Evert Bunschoten                                                                    |
#    :PhD Candidate ,                                                                         |
#    :Flight Power and Propulsion                                                             |
#    :TU Delft,                                                                               |
#    :The Netherlands                                                                         |
#                                                                                             |
#                                                                                             |
# Description:                                                                                |
# Determine whether each of the progress variable definitions (WRP, PCA, and OPT), is         |
# monotonic throughout the flamelet data state space.                                         |
#                                                                                             |  
# Version: 2.0.0                                                                              |
#                                                                                             |
#=============================================================================================#
import numpy as np
import matplotlib.pyplot as plt 
from su2dataminer.config import Config_FGM
from su2dataminer.process_data import PVOptimizer

# Load SU2 DataMiner configurations
Config_default = Config_FGM("../WRP.cfg")
Config_pca = Config_FGM("../PCA.cfg")
Config_optim = Config_FGM("../OPT.cfg")

PVO = PVOptimizer(Config_optim)
PVO._CollectFlameletData()

# Retrieve progress variable definitions
pv_species = Config_optim.GetProgressVariableSpecies()
x_default = np.zeros(len(pv_species))
x_pca = np.zeros(len(pv_species))
x_optim = Config_optim.GetProgressVariableWeights()

pv_species_default = Config_default.GetProgressVariableSpecies()
pv_species_pca = Config_pca.GetProgressVariableSpecies()
pv_species_optim = Config_optim.GetProgressVariableSpecies()
for ipv, pv in enumerate(pv_species):
    if pv in pv_species_default:
        x_default[ipv] = Config_default.GetProgressVariableWeights()[pv_species_default.index(pv)]
    if pv in pv_species_pca:
        x_pca[ipv] = Config_pca.GetProgressVariableWeights()[pv_species_pca.index(pv)]

# Calculate maximum value of the derivative of the progress vector w.r.t. the progress variable
penalty_val_default = PVO.penalty_function(x_default)
penalty_val_pca = PVO.penalty_function(x_pca)
penalty_val_optim = PVO.penalty_function(Config_optim.GetProgressVariableWeights())

# Calculate the monotonicity penalty function value.
monotonicity_penalty_default = PVO.monotonicity_penalty(x_default)
monotonicity_penalty_pca = PVO.monotonicity_penalty(x_pca)
monotonicity_penalty_optim = PVO.monotonicity_penalty(x_optim)

# Output results
print("Penalty value default pv definition: %.6e Monotonicity penalty: %.6e" % (penalty_val_default-monotonicity_penalty_default, monotonicity_penalty_default))
print("Penalty value pca pv definition: %.6e Monotonicity penalty: %.6e" % (penalty_val_pca-monotonicity_penalty_pca, monotonicity_penalty_pca))
print("Penalty value optimized pv definition: %.6e Monotonicity penalty: %.6e" % (penalty_val_optim, monotonicity_penalty_optim))

N=3
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.cubehelix(np.linspace(0,1,N)))
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_Default = colors[0]
color_PCA = colors[1]
color_Optimized = colors[2]

fig = plt.figure(figsize=[10,7])
ax = plt.axes()
ax.bar(x=[0], height=[penalty_val_default-monotonicity_penalty_default],color=color_Default,width=0.8,zorder=3)
ax.bar(x=[1], height=[penalty_val_pca-monotonicity_penalty_pca],color=color_PCA,width=0.8,zorder=3)
ax.bar(x=[2], height=[penalty_val_optim - monotonicity_penalty_optim],width=0.8,zorder=3,color=color_Optimized)

ax.set_yscale('log')
ax.grid()
ax.set_xticks([0,1,2])
ax.set_xticklabels(["WRP", "PCA", "OPT"])
ax.set_ylabel("Maximum gradient w.r.t. pv",fontsize=30)
ax.tick_params(which='both',labelsize=30)
plt.show()