#!/usr/bin/env python3
###############################################################################################
#       #      _____ __  _____      ____        __        __  ____                   #        #  
#       #     / ___// / / /__ \    / __ \____ _/ /_____ _/  |/  (_)___  ___  _____   #        #  
#       #     \__ \/ / / /__/ /   / / / / __ `/ __/ __ `/ /|_/ / / __ \/ _ \/ ___/   #        #      
#       #    ___/ / /_/ // __/   / /_/ / /_/ / /_/ /_/ / /  / / / / / /  __/ /       #        #  
#       #   /____/\____//____/  /_____/\__,_/\__/\__,_/_/  /_/_/_/ /_/\___/_/        #        #
#       #                                                                            #        #
###############################################################################################

####################### FILE NAME: 7:plot_hp_optim_convergence.py #############################
#=============================================================================================#
# author: Evert Bunschoten                                                                    |
#    :PhD Candidate ,                                                                         |
#    :Flight Power and Propulsion                                                             |
#    :TU Delft,                                                                               |
#    :The Netherlands                                                                         |
#                                                                                             |
#                                                                                             |
# Description:                                                                                |
# Plot the convergence trends of the Pareto fronts produced during the hyperparameter         |
# optimization.                                                                               |
#                                                                                             |  
# Version: 2.0.0                                                                              |
#                                                                                             |
#=============================================================================================#
import numpy as np 
import matplotlib.pyplot as plt 
from su2dataminer.config import Config_FGM

c = Config_FGM("../HP_Optimization.cfg")
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.cubehelix(np.linspace(0,1,c.GetNMLPOutputGroups()+1)))
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
markers=['o','s','^','+','d']
fig, axs = plt.subplots(nrows=1, ncols=2,figsize=[12,6])

# For each group, load the hypervolume and general distance history.
for iGroup in range(c.GetNMLPOutputGroups()):
    hypervolume = np.load("../Architectures_Group%i_OptimLRAPhi/pareto_hypervolume_history.npy" % (iGroup+1))
    general_distance = np.load("../Architectures_Group%i_OptimLRAPhi/pareto_general_distance_history.npy" % (iGroup+1))
    gens = np.arange(len(hypervolume))+1.0

    axs[0].plot(gens, general_distance, marker=markers[iGroup],linestyle='-',color=colors[iGroup],markerfacecolor='none',markersize=12,linewidth=3,markeredgewidth=2)
    axs[1].plot(gens, hypervolume, marker=markers[iGroup],linestyle='-',color=colors[iGroup],markerfacecolor='none',markersize=12,linewidth=3,markeredgewidth=2,label="Group %i" % (iGroup+1))
axs[0].grid()
axs[0].tick_params(which='both',labelsize=20)
axs[0].set_xlabel("Generation",fontsize=20)
axs[0].set_ylabel("General distance",fontsize=20)
axs[0].set_xticks([1, 5, 10, 15, 20])
axs[0].set_xticklabels(["1","5","10","15","20"])
axs[1].grid()
axs[1].tick_params(which='both',labelsize=20)
axs[1].set_xlabel("Generation",fontsize=20)
axs[1].set_ylabel("Hyper-volume",fontsize=20)
axs[1].set_xticks([1, 5, 10, 15, 20])
axs[1].set_xticklabels(["1","5","10","15","20"])
axs[1].legend(fontsize=20,ncol=1,bbox_to_anchor=(1, 0.5),loc='center left',fancybox=True,shadow=True)
fig.suptitle("Hyperparameter optimization convergence history",fontsize=20)
plt.show()
