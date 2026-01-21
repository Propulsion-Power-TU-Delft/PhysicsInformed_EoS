#!/usr/bin/env python3
###############################################################################################
#       #      _____ __  _____      ____        __        __  ____                   #        #  
#       #     / ___// / / /__ \    / __ \____ _/ /_____ _/  |/  (_)___  ___  _____   #        #  
#       #     \__ \/ / / /__/ /   / / / / __ `/ __/ __ `/ /|_/ / / __ \/ _ \/ ___/   #        #      
#       #    ___/ / /_/ // __/   / /_/ / /_/ / /_/ /_/ / /  / / / / / /  __/ /       #        #  
#       #   /____/\____//____/  /_____/\__,_/\__/\__,_/_/  /_/_/_/ /_/\___/_/        #        #
#       #                                                                            #        #
###############################################################################################

########################## FILE NAME: 6:evaluate_MLP_output.py ################################
#=============================================================================================#
# author: Evert Bunschoten                                                                    |
#    :PhD Candidate ,                                                                         |
#    :Flight Power and Propulsion                                                             |
#    :TU Delft,                                                                               |
#    :The Netherlands                                                                         |
#                                                                                             |
#                                                                                             |
# Description:                                                                                |
# Evaluate the temperature along flamelet solutions using ML-FGM networks trained on the      |
# flamelet data parameterized by each progress variable.                                      |
#                                                                                             |  
# Version: 2.0.0                                                                              |
#                                                                                             |
#=============================================================================================#
import os
import numpy as np
import matplotlib.pyplot as plt 
from su2dataminer.config import Config_FGM 
from su2dataminer.manifold import Train_FGM_PINN

config_WRP = Config_FGM("../WRP.cfg")
config_PCA = Config_FGM("../PCA.cfg")
config_OPT = Config_FGM("../OPT.cfg")
configs = [config_WRP, config_PCA, config_OPT]

# Transformation functions used to improve the visualization of the training data in lean conditions
def transform_phi(phi_dim:np.ndarray[float]):
    phi_norm_all = np.zeros(phi_dim.shape)
    phi_norm_all[phi_dim < 1.0] = 0.5*(phi_dim[phi_dim < 1.0])
    phi_norm_all[phi_dim >= 1.0] = 0.5*((phi_dim[phi_dim >= 1.0]-1.0)/(20.0 - 1)) + 0.5
    return phi_norm_all
 
def inv_transform_phi(phi_norm:np.ndarray[float]):
    phi_dim = np.zeros(phi_norm.shape)
    phi_dim[phi_norm >= 0.5] = (phi_norm[phi_norm >= 0.5] - 0.5) * (20.0 - 1)/0.5 + 1.0
    phi_dim[phi_norm < 0.5] = phi_norm[phi_norm < 0.5] / 0.5
    return phi_dim 


labels = ["WRP","PCA","OPT"]
flamelet_dir = config_WRP.GetOutputDir()
phis = os.listdir(flamelet_dir + "freeflame_data/")

plot_var = "Temperature"
xlabel="c*"
ylabel="T*"

error_interps = []
error_vals = []
fig, axs = plt.subplots(nrows=1,ncols=len(configs),figsize=[20,10])
iGroup = 0

# Clip the maximum error value to 2% for visualization purposes
e_min = 0
e_max = 2.0
for iconfig, config_PINN in enumerate(configs):
  
    MLP_PINN = Train_FGM_PINN(config_PINN, iGroup)

    
    MLP_PINN.SetAlphaExpo(config_PINN.GetAlphaExpo(iGroup))
    MLP_PINN.SetLRDecay(config_PINN.GetLRDecay(iGroup))
    MLP_PINN.SetBatchExpo(config_PINN.GetBatchExpo(iGroup))
    MLP_PINN.SetHiddenLayers(config_PINN.GetHiddenLayerArchitecture(iGroup))
    # Load the weights and biases of the trained MLP
    weights_dir = "../TrainedMLPs/%s/Group0/" % labels[iconfig]
    w = [np.load(weights_dir + "/W_0.npy"),\
         np.load(weights_dir + "/W_1.npy"),\
         np.load(weights_dir + "/W_2.npy"),\
         np.load(weights_dir + "/W_3.npy"),\
         np.load(weights_dir + "/W_4.npy"),\
         np.load(weights_dir + "/W_5.npy"),\
         np.load(weights_dir + "/W_6.npy"),\
         np.load(weights_dir + "/W_7.npy")]
    
    b = [np.load(weights_dir + "/b_0.npy"),\
         np.load(weights_dir + "/b_1.npy"),\
         np.load(weights_dir + "/b_2.npy"),\
         np.load(weights_dir + "/b_3.npy"),\
         np.load(weights_dir + "/b_4.npy"),\
         np.load(weights_dir + "/b_5.npy"),\
         np.load(weights_dir + "/b_6.npy"),\
         np.load(weights_dir + "/b_7.npy")]
    MLP_PINN.SetWeightsBiases(w, b)
    MLP_PINN.InitializeWeights_and_Biases()
    MLP_PINN.SetActivationFunction(config_PINN.GetActivationFunction(iGroup))
    MLP_PINN.SetTrainVariables(config_PINN.GetMLPOutputGroup(iGroup))
    MLP_PINN.SetTrainFileHeader(config_PINN.GetOutputDir()+"/"+config_PINN.GetConcatenationFileHeader())
    MLP_PINN.SetBoundaryDataFile(config_PINN.GetOutputDir()+"/boundary_data_%s_full.csv" % labels[iconfig])
    MLP_PINN.SetSaveDir(os.getcwd())
    MLP_PINN.SetModelIndex(100)

    MLP_PINN.Preprocessing()
    var_out = config_PINN.GetMLPOutputGroup(iGroup)[0]

    pv_scaled_flamelets_all = []
    phi_flamelets_all = []
    error_flamelets_all = []

    for p in phis:
        # For each equivalence ratio for which there is flamelet data, extract the adiabatic flamelet solution
        # with the lowest reactant temperature.
        flamelets = os.listdir(flamelet_dir + "freeflame_data/" + p)
        flamelets.sort()
        if flamelets:
            f = flamelets[0]
            with open(flamelet_dir + "freeflame_data/" + p + "/" + f, 'r') as fid:
                vars = fid.readline().strip().split(',')
            F = np.loadtxt(flamelet_dir + "freeflame_data/" + p + "/" + f,delimiter=',',skiprows=1)

            # Calculate the progress variable, total enthalpy, and mixture fraction along the flamelet solution.
            pv = config_PINN.ComputeProgressVariable(vars, F)
            h = F[:,vars.index("EnthalpyTot")]
            Z = F[:, vars.index("MixtureFraction")]
            T = F[:, vars.index("Temperature")]
            
            config_PINN.gas.set_mixture_fraction(Z[0],config_PINN.GetFuelString(), config_PINN.GetOxidizerString())
            val_phi = config_PINN.gas.equivalence_ratio(config_PINN.GetFuelString(), config_PINN.GetOxidizerString())
            
            cv_flamelet = np.vstack((pv,h,Z)).T 
            ref_data = F[:, vars.index(var_out)]

            # Evaluate the MLP output along the flamelet
            pred_data = MLP_PINN.EvaluateMLP(cv_flamelet)[:,0]
            ref_max, ref_min = max(ref_data), min(ref_data)

            # Calculate the absolute percentage error between the flamelet data and the MLP output
            pred_error = 100*np.abs((pred_data - ref_data) / (np.abs(ref_data)+1e-4*(ref_max - ref_min)))

            # Normalize the progress variable for visualization
            pvmin, pvmax = pv[0], pv[-1]
            pv_norm = (pv - pvmin)/(pvmax - pvmin)
            
            pv_scaled_flamelets_all.append(pv_norm)
            phi_flamelets_all.append(val_phi*np.ones(pv_norm.shape))
            error_flamelets_all.append(pred_error)
            
    pv_scaled_flamelets_all = np.hstack(tuple(pv for pv in pv_scaled_flamelets_all))
    phi_flamelets_all = np.hstack(tuple(p for p in phi_flamelets_all))
    error_flamelets_all = np.hstack(tuple(e for e in error_flamelets_all))

    phi_norm_all = transform_phi(phi_flamelets_all)
    cs = axs[iconfig].tricontourf(pv_scaled_flamelets_all,phi_norm_all,np.clip((error_flamelets_all), e_min, e_max), levels=np.linspace(e_min, e_max, 20))
    
cbar = fig.colorbar(cs,ax=axs, ticks=np.linspace(e_min, e_max, 6),shrink=0.95)
cbar.ax.set_title("e",fontsize=24)
cbar.ax.tick_params(which='both',labelsize=24)
xmin = 0.0
xmax = 0.0
for i in range(len(configs)):
    xlim = axs[i].get_xlim()
    xmin = min(xmin, xlim[0])
    xmax = max(xmax, xlim[1])

fsize = 20
for i in range(len(configs)):
    axs[i].set_xlim(xmin, xmax)
    axs[i].set_xlabel("c*",fontsize=fsize)
    axs[i].tick_params(which='both',labelsize=fsize)

phi_to_plot = np.array([0.25, 0.5, 1.0, 3.0, 10.0, 20.0])
phi_norm_to_plot = transform_phi(phi_to_plot)
for i in range(len(configs)):
    axs[i].set_yticks(phi_norm_to_plot)
    axs[i].set_yticklabels(phi_to_plot)
    if i > 0:
        axs[i].set_yticks([])
        axs[i].set_yticklabels([])

    axs[i].set_title(labels[i],fontsize=fsize)
axs[0].set_ylabel("Equivalence ratio (p)[-]", fontsize=fsize)

plt.show()
