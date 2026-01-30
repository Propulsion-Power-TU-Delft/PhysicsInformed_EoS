#!/usr/bin/env python3

############################## FILE NAME: generate_mesh.py ####################################
#=============================================================================================#
# author: Evert Bunschoten                                                                    |
#    :PhD Candidate ,                                                                         |
#    :Flight Power and Propulsion                                                             |
#    :TU Delft,                                                                               |
#    :The Netherlands                                                                         |
#                                                                                             |
#                                                                                             |
# Description:                                                                                |
# Evaluate the average iteration time for adjoint and direct simulations with and without     |
# using a data-driven fluid model.                                                            |
#                                                                                             |  
#                                                                                             |
#=============================================================================================#
import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt 
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams["font.family"] = "Times New Roman"

# Load convergence history files from direct and adjoint simulations
hist_file_direct_FGM = "Direct/history_fluid_0.csv"
hist_file_direct_noFGM = "Direct_IGL/history_fluid_0.csv"
hist_file_adjoint_FGM = "Adjoint/history_adj_fluid_0.csv"
hist_file_adjoint_noFGM = "Adjoint_IGL/history_fluid_0.csv"

H_direct_FGM = np.loadtxt(hist_file_direct_FGM,delimiter=',',skiprows=1)
H_adjoint_FGM = np.loadtxt(hist_file_adjoint_FGM,delimiter=',',skiprows=1)
H_direct_noFGM = np.loadtxt(hist_file_direct_noFGM,delimiter=',',skiprows=1)
H_adjoint_noFGM = np.loadtxt(hist_file_adjoint_noFGM,delimiter=',',skiprows=1)

# Evaluate the average iteration time and standard devaiation of the last 100 iterations
t_direct_FGM = H_direct_FGM[-100:,0]
t_adjoint_FGM = H_adjoint_FGM[-100:,0]
t_direct_noFGM = H_direct_noFGM[-100:,0]
t_adjoint_noFGM = H_adjoint_noFGM[-100:,0]

t_avg_direct_FGM, std_direct_FGM = np.average(t_direct_FGM), np.std(t_direct_FGM)
t_avg_adjoint_FGM, std_adjoint_FGM = np.average(t_adjoint_FGM), np.std(t_adjoint_FGM)
t_avg_direct_noFGM, std_direct_noFGM = np.average(t_direct_noFGM), np.std(t_direct_noFGM)
t_avg_adjoint_noFGM, std_adjoint_noFGM = np.average(t_adjoint_noFGM), np.std(t_adjoint_noFGM)

print("Adjoint speedup FGM: %.2f" % (t_avg_adjoint_FGM/t_avg_direct_FGM))
print("Adjoint speedup IG: %.2f" % (t_avg_adjoint_noFGM/t_avg_direct_noFGM))

n_dv = 58
t_FD_FGM = (n_dv+1) * t_avg_direct_FGM 
t_AD_FGM = t_avg_direct_FGM + t_avg_adjoint_FGM
t_FD_noFGM = (n_dv+1) * t_avg_direct_noFGM 
t_AD_noFGM = t_avg_direct_noFGM + t_avg_adjoint_noFGM
print(t_FD_FGM/t_AD_FGM, t_FD_noFGM/t_AD_noFGM)
