#!/usr/bin/env python3
###############################################################################################
#       #      _____ __  _____      ____        __        __  ____                   #        #  
#       #     / ___// / / /__ \    / __ \____ _/ /_____ _/  |/  (_)___  ___  _____   #        #  
#       #     \__ \/ / / /__/ /   / / / / __ `/ __/ __ `/ /|_/ / / __ \/ _ \/ ___/   #        #      
#       #    ___/ / /_/ // __/   / /_/ / /_/ / /_/ /_/ / /  / / / / / /  __/ /       #        #  
#       #   /____/\____//____/  /_____/\__,_/\__/\__,_/_/  /_/_/_/ /_/\___/_/        #        #
#       #                                                                            #        #
###############################################################################################

########################## FILE NAME: 5:plot_optim_history.py #################################
#=============================================================================================#
# author: Evert Bunschoten                                                                    |
#    :PhD Candidate ,                                                                         |
#    :Flight Power and Propulsion                                                             |
#    :TU Delft,                                                                               |
#    :The Netherlands                                                                         |
#                                                                                             |
#                                                                                             |
# Description:                                                                                |
# Plot the convergence trends of the progress variable optimization process.                  |
#                                                                                             |  
# Version: 2.0.0                                                                              |
#                                                                                             |
#=============================================================================================#
import numpy as np 
import matplotlib.pyplot as plt 

N=7
H_file_optim = "../PV_Optimization/Optim_PV_convergence_history.csv"

fsize = 20
with open(H_file_optim,'r') as fid:
    vars = fid.readline().strip().split(',')
H_optim = np.loadtxt(H_file_optim,delimiter=',',skiprows=1)

f = H_optim[:,0]
f_rel = f
f_rel_scaled = 100*(f_rel - f_rel[-1])/f_rel[-1]
fig, axs = plt.subplots(nrows=2,ncols=1,figsize=[10,10])
ax = axs[0]
ax.plot(f_rel_scaled,'k',linewidth=3)
ax.set_yscale('log')
ax.tick_params(which='both',labelsize=fsize)
ax.set_xticklabels([])
ax.set_ylabel("Objective function, f",fontsize=fsize)
ax.set_title("Progress variable optimization convergence",fontsize=fsize)
ax.grid()
ax = axs[1]
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.cubehelix(np.linspace(0,1,N+1)[:-1]))
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
line_styles = ['-', '--', '-.', ':']
for i in range(8):
    v = vars[i+1].strip("alpha")
    v = "a" + v
    ax.plot(H_optim[:,i+1], linestyle=line_styles[i%len(line_styles)],linewidth=3,label=v)

ax.grid(which='both')
ax.tick_params(which='both',labelsize=fsize)
ax.set_xlabel("Iteration",fontsize=fsize)
ax.set_ylabel("Species weight value, a",fontsize=fsize)
ax.legend(fontsize=fsize,ncol=4,bbox_to_anchor=(0.5, -0.2),loc='upper center',fancybox=True,shadow=True)
plt.show()