#!/usr/bin/env python3
###############################################################################################
#       #      _____ __  _____      ____        __        __  ____                   #        #  
#       #     / ___// / / /__ \    / __ \____ _/ /_____ _/  |/  (_)___  ___  _____   #        #  
#       #     \__ \/ / / /__/ /   / / / / __ `/ __/ __ `/ /|_/ / / __ \/ _ \/ ___/   #        #      
#       #    ___/ / /_/ // __/   / /_/ / /_/ / /_/ /_/ / /  / / / / / /  __/ /       #        #  
#       #   /____/\____//____/  /_____/\__,_/\__/\__,_/_/  /_/_/_/ /_/\___/_/        #        #
#       #                                                                            #        #
###############################################################################################

###################### FILE NAME: 1:compare_consistency_accuracy.py ###########################
#=============================================================================================#
# author: Evert Bunschoten                                                                    |
#    :PhD Candidate ,                                                                         |
#    :Flight Power and Propulsion                                                             |
#    :TU Delft,                                                                               |
#    :The Netherlands                                                                         |
#                                                                                             |
#                                                                                             |
# Description:                                                                                |
# Generate fluid data used for training the artificial neural networks of the EEoS-MTNN and   |
# EEoS-PINN models.                                                                           |
#                                                                                             |  
# Version: 2.0.0                                                                              |
#                                                                                             |
#=============================================================================================#
import numpy as np 
import matplotlib.pyplot as plt 
import CoolProp.CoolProp as CP
from Common.Properties import EntropicVars
from su2dataminer.config import Config_NICFD
from su2dataminer.manifold import Train_Entropic_Segregated, Train_Entropic_PINN

N=4
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.cubehelix(np.linspace(0,1,N)))
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Load SU2 DataMiner configurations and network weights and biases
config_MTNN_low = Config_NICFD("../Direct_low.cfg")
config_MTNN_high = Config_NICFD("../Direct_high.cfg")
config_PINN = Config_NICFD("../PINN.cfg")

w_MTNN_low, b_MTNN_low = config_MTNN_low.GetWeightsBiases()


w_MTNN_high,b_MTNN_high = config_MTNN_high.GetWeightsBiases()

w_PINN, b_PINN = config_PINN.GetWeightsBiases()


# Define MTNN's for low and high density ranges
MTNN_low = Train_Entropic_Segregated()
MTNN_low.SetActivationFunction("swish")
MTNN_low.SetScaler("minmax")
MTNN_low.SetHiddenLayers([11,14])
MTNN_low.SetTrainFileHeader(config_MTNN_low.GetOutputDir()+"/"+config_MTNN_low.GetConcatenationFileHeader())
MTNN_low.SetAlphaExpo(-3.1146)
MTNN_low.SetLRDecay(0.98732)

MTNN_low.SetWeightsBiases(w_MTNN_low, b_MTNN_low)
MTNN_low.GetTrainData()
MTNN_low.DefineMLP()

MTNN_high = Train_Entropic_Segregated()
MTNN_high.SetActivationFunction("gelu")
MTNN_high.SetScaler("minmax")
MTNN_high.SetHiddenLayers([13, 15,12])
MTNN_high.SetTrainFileHeader(config_MTNN_high.GetOutputDir()+"/"+config_MTNN_high.GetConcatenationFileHeader())
MTNN_high.SetAlphaExpo(-3.1146)
MTNN_high.SetLRDecay(0.98732)

MTNN_high.SetWeightsBiases(w_MTNN_high, b_MTNN_high)
MTNN_high.GetTrainData()
MTNN_high.DefineMLP()

# Define PINN
PINN = Train_Entropic_PINN()
PINN.SetActivationFunction("exponential")
PINN.SetScaler("minmax")
PINN.SetHiddenLayers([12,12])
PINN.SetStateVars([s.name for s in EntropicVars][:-1])
PINN.SetTrainFileHeader(config_PINN.GetOutputDir()+"/"+config_PINN.GetConcatenationFileHeader())
PINN.SetWeightsBiases(w_PINN, b_PINN)
PINN.GetTrainData()
PINN.InitializeWeights_and_Biases()
PINN.CollectVariables()

# Load solution of the flow calculation performed with HEoS
solution_file_CoolProp = "CoolProp_solution.csv"
with open(solution_file_CoolProp,'r') as fid:
    vars = fid.readline().strip().split(',')
    vars = [v.strip("\"") for v in vars]
CoolProp_solution = np.loadtxt(solution_file_CoolProp,delimiter=',',skiprows=1)
rho_coolprop = CoolProp_solution[:, vars.index("Density")]
e_coolprop = (CoolProp_solution[:, vars.index("Energy")] / rho_coolprop) - 0.5 * np.sqrt(np.power(CoolProp_solution[:, vars.index("Velocity:0")],2) + np.power(CoolProp_solution[:, vars.index("Velocity:1")],2))
rhoe_all = np.vstack((rho_coolprop, e_coolprop)).T 

def EntropicEOS(vals_state:np.ndarray[float]):
    # Entropic equation of state used to calculate the thermodynamic state based on the Jacobian and Hessian of the entropy potential.

    dsdrho_e = vals_state[:, EntropicVars.dsdrho_e.value]
    dsde_rho = vals_state[:, EntropicVars.dsde_rho.value]
    d2sdrho2 = vals_state[:, EntropicVars.d2sdrho2.value]
    d2sdedrho = vals_state[:, EntropicVars.d2sdedrho.value]
    d2sde2 = vals_state[:, EntropicVars.d2sde2.value]
    rho = vals_state[:, EntropicVars.Density.value]
    s = vals_state[:, EntropicVars.s.value]

    T = np.power(dsde_rho, -1)
    rho2 = rho*rho
    P = -rho2 * T * dsdrho_e
    dTde_rho = -T*T * d2sde2 
    dTdrho_e = -T*T * d2sdedrho 

    dPde_rho = -rho2 * (dTde_rho * dsdrho_e + T * d2sdedrho)
    dPdrho_e = -2 * rho * T * dsdrho_e - rho2 * (dTdrho_e * dsdrho_e + T * d2sdrho2)
    dhdrho_e = -P * (1.0/rho2) + dPdrho_e / rho
    dhde_rho = 1 + dPde_rho / rho

    dhdrho_P = dhdrho_e - dhde_rho * (1 / dPde_rho) * dPdrho_e
    dhdP_rho = dhde_rho * (1 / dPde_rho)
    dsdrho_P = dsdrho_e - dPdrho_e * (1 / dPde_rho) * dsde_rho
    dsdP_rho = dsde_rho / dPde_rho

    drhode_p = -dPde_rho/dPdrho_e
    dTde_p = dTde_rho + dTdrho_e*drhode_p
    dhde_p = dhde_rho + drhode_p*dhdrho_e
    Cp = dhde_p / dTde_p
    
    c2 = dPdrho_e - dsdrho_e * dPde_rho / dsde_rho

    vals_state[:, EntropicVars.T.value] = T 
    vals_state[:, EntropicVars.p.value] = P 
    vals_state[:, EntropicVars.dTde_rho.value] = dTde_rho
    vals_state[:, EntropicVars.dTdrho_e.value] = dTdrho_e
    vals_state[:, EntropicVars.dpde_rho.value] = dPde_rho
    vals_state[:, EntropicVars.dpdrho_e.value] = dPdrho_e
    vals_state[:, EntropicVars.dhdrho_p.value] = dhdrho_P
    vals_state[:, EntropicVars.dhdp_rho.value] = dhdP_rho
    vals_state[:, EntropicVars.dsdrho_p.value] = dsdrho_P
    vals_state[:, EntropicVars.dsdp_rho.value] = dsdP_rho
    vals_state[:, EntropicVars.cp.value] = Cp
    vals_state[:, EntropicVars.c2.value] = c2
    
    return vals_state
    
def EvaluateState_MTNN(vals_rho:np.ndarray[float], vals_e:np.ndarray[float], is_low:bool=False):
    # Calculate the thermodynamic state based on the MTNN output
    rhoe = np.vstack((vals_rho, vals_e)).T 

    state_pred_MTNN = np.zeros([len(rhoe), EntropicVars.N_STATE_VARS.value])
    state_pred_MTNN[:, EntropicVars.Density.value] = vals_rho
    state_pred_MTNN[:, EntropicVars.Energy.value] = vals_e
    if is_low:
        s_Jac_Hess = MTNN_low.EvaluateMLP(rhoe)
    else:
        s_Jac_Hess = MTNN_high.EvaluateMLP(rhoe)

    state_pred_MTNN[:, EntropicVars.s.value] = s_Jac_Hess[:, 0]
    state_pred_MTNN[:, EntropicVars.dsdrho_e.value] = s_Jac_Hess[:, 1]
    state_pred_MTNN[:, EntropicVars.dsde_rho.value] = s_Jac_Hess[:, 2]
    state_pred_MTNN[:, EntropicVars.d2sdrho2.value] = s_Jac_Hess[:, 3]
    state_pred_MTNN[:, EntropicVars.d2sdedrho.value] = s_Jac_Hess[:, 4]
    state_pred_MTNN[:, EntropicVars.d2sde2.value] = s_Jac_Hess[:, 5]
    return EntropicEOS(state_pred_MTNN)

def EvaluateState_PINN(vals_rho:np.ndarray[float], vals_e:np.ndarray[float]):
    # Calculate the thermodynamic state based on the PINN Jacobian and Hessian
    rhoe = np.vstack((vals_rho, vals_e)).T 
    return PINN.EvaluateMLP(rhoe)


def EvaluateState_CoolProp(vals_rho, vals_e):
    state_pred_CP = np.zeros([len(vals_rho), EntropicVars.N_STATE_VARS.value])
    fluid = CP.AbstractState("HEOS","MM")
    for i in range(len(vals_rho)):
        fluid.update(CP.DmassUmass_INPUTS, vals_rho[i], vals_e[i])
        state_pred_CP[i,EntropicVars.s.value] = fluid.smass()
        state_pred_CP[i,EntropicVars.dsde_rho.value] = fluid.first_partial_deriv(CP.iSmass, CP.iUmass, CP.iDmass)
        state_pred_CP[i,EntropicVars.dsdrho_e.value] = fluid.first_partial_deriv(CP.iSmass, CP.iDmass, CP.iUmass)
        state_pred_CP[i,EntropicVars.d2sde2.value] = fluid.second_partial_deriv(CP.iSmass, CP.iUmass, CP.iDmass, CP.iUmass, CP.iDmass)
        state_pred_CP[i,EntropicVars.d2sdedrho.value] = fluid.second_partial_deriv(CP.iSmass, CP.iUmass, CP.iDmass, CP.iDmass, CP.iUmass)
        state_pred_CP[i,EntropicVars.d2sdrho2.value] = fluid.second_partial_deriv(CP.iSmass, CP.iDmass, CP.iUmass, CP.iDmass, CP.iUmass)
        state_pred_CP[i,EntropicVars.Density.value] = fluid.rhomass()
        state_pred_CP[i,EntropicVars.Energy.value] = fluid.umass()
        state_pred_CP[i,EntropicVars.T.value] = fluid.T()
        state_pred_CP[i,EntropicVars.p.value] = fluid.p()
        state_pred_CP[i,EntropicVars.c2.value] = fluid.speed_sound()**2
        state_pred_CP[i,EntropicVars.dTde_rho.value] = fluid.first_partial_deriv(CP.iT, CP.iUmass, CP.iDmass)
        state_pred_CP[i,EntropicVars.dTdrho_e.value] = fluid.first_partial_deriv(CP.iT, CP.iDmass, CP.iUmass)
        state_pred_CP[i,EntropicVars.dpde_rho.value] = fluid.first_partial_deriv(CP.iP, CP.iUmass, CP.iDmass)
        state_pred_CP[i,EntropicVars.dpdrho_e.value] = fluid.first_partial_deriv(CP.iP, CP.iDmass, CP.iUmass)
        state_pred_CP[i,EntropicVars.dhde_rho.value] = fluid.first_partial_deriv(CP.iHmass, CP.iUmass, CP.iDmass)
        state_pred_CP[i,EntropicVars.dhdrho_e.value] = fluid.first_partial_deriv(CP.iHmass, CP.iDmass, CP.iUmass)
        state_pred_CP[i,EntropicVars.dhdp_rho.value] = fluid.first_partial_deriv(CP.iHmass, CP.iP, CP.iDmass)
        state_pred_CP[i,EntropicVars.dhdrho_p.value] = fluid.first_partial_deriv(CP.iHmass, CP.iDmass, CP.iP)
        state_pred_CP[i,EntropicVars.dsdp_rho.value] = fluid.first_partial_deriv(CP.iSmass, CP.iP, CP.iDmass)
        state_pred_CP[i,EntropicVars.dsdrho_p.value] = fluid.first_partial_deriv(CP.iSmass, CP.iDmass, CP.iP)
        state_pred_CP[i,EntropicVars.cp.value] = fluid.cpmass()
    return state_pred_CP 

ix_low = rhoe_all[:,0] < +1.0064969062805176e+01
ix_high = np.invert(ix_low)
rhoe_low = rhoe_all[ix_low,:]
rhoe_high = rhoe_all[ix_high,:]
state_CoolProp_ref = EvaluateState_CoolProp(rhoe_all[:,0],rhoe_all[:,1])
state_MTNN_low = EvaluateState_MTNN(rhoe_low[:,0], rhoe_low[:,1], True)
state_MTNN_high = EvaluateState_MTNN(rhoe_high[:,0], rhoe_high[:,1], False)
state_MTNN = np.zeros(np.shape(state_CoolProp_ref))
state_MTNN[ix_low,:] = state_MTNN_low
state_MTNN[ix_high,:] = state_MTNN_high

state_PINN = EvaluateState_PINN(rhoe_all[:,0], rhoe_all[:,1])

def CalcStateError(state_ROM:np.ndarray[float], state_ref:np.ndarray[float]):
    return np.mean(np.abs((state_ROM - state_ref)) / (np.abs(state_ref) + 1e-12),axis=0)


state_error_MTNN = CalcStateError(state_MTNN, state_CoolProp_ref)
state_error_PINN = CalcStateError(state_PINN, state_CoolProp_ref)

vars_of_interest = [EntropicVars.p, EntropicVars.T, EntropicVars.c2, EntropicVars.cp, EntropicVars.s, EntropicVars.dsdrho_e, EntropicVars.dsde_rho, EntropicVars.d2sdrho2, EntropicVars.d2sdedrho, EntropicVars.d2sde2]
x_bars = np.arange(len(vars_of_interest))
print("TD var, MAPE MTNN, MAPE PINN")
for q in vars_of_interest:
    print("%s : %.6e, %.6e" % (q.name, state_error_MTNN[q.value], state_error_PINN[q.value]))




fac_delta = 5e-9
fac_delta= 1e-5
rhoe_plus = rhoe_all * (1 + fac_delta)
rhoe_minus= rhoe_all * (1 - fac_delta)

delta_rho_plus = (rhoe_plus[:,0]-rhoe_all[:,0])
delta_e_plus = (rhoe_plus[:,1]-rhoe_all[:,1])
state_CoolProp_rho_plus = EvaluateState_CoolProp(rhoe_plus[:,0],rhoe_all[:,1])
state_CoolProp_e_plus = EvaluateState_CoolProp(rhoe_all[:,0],rhoe_plus[:,1])

rhoe_plus_low = rhoe_plus[ix_low,:]
rhoe_plus_high = rhoe_plus[ix_high,:]

state_MTNN_rho_plus_low = EvaluateState_MTNN(rhoe_plus_low[:,0], rhoe_low[:,1],True)
state_MTNN_rho_plus_high = EvaluateState_MTNN(rhoe_plus_high[:,0], rhoe_high[:,1],False)
state_MTNN_rho_plus = np.zeros(np.shape(state_CoolProp_ref))
state_MTNN_rho_plus[ix_low,:] = state_MTNN_rho_plus_low
state_MTNN_rho_plus[ix_high,:] = state_MTNN_rho_plus_high

state_MTNN_e_plus_low = EvaluateState_MTNN(rhoe_low[:,0], rhoe_plus_low[:,1],True)
state_MTNN_e_plus_high = EvaluateState_MTNN(rhoe_high[:,0], rhoe_plus_high[:,1],False)
state_MTNN_e_plus = np.zeros(np.shape(state_CoolProp_ref))
state_MTNN_e_plus[ix_low,:] = state_MTNN_e_plus_low
state_MTNN_e_plus[ix_high,:] = state_MTNN_e_plus_high

state_PINN_rho_plus = EvaluateState_PINN(rhoe_plus[:,0], rhoe_all[:,1])
state_PINN_e_plus = EvaluateState_PINN(rhoe_all[:,0], rhoe_plus[:,1])

dstate_drho_CoolProp = (state_CoolProp_rho_plus - state_CoolProp_ref) / delta_rho_plus[:,np.newaxis]
dstate_de_CoolProp = (state_CoolProp_e_plus - state_CoolProp_ref) / delta_e_plus[:,np.newaxis]

dstate_drho_MTNN = (state_MTNN_rho_plus - state_MTNN)/delta_rho_plus[:,np.newaxis]
dstate_de_MTNN = (state_MTNN_e_plus - state_MTNN)/delta_e_plus[:,np.newaxis]

dstate_drho_PINN = (state_PINN_rho_plus - state_PINN)/delta_rho_plus[:,np.newaxis]
dstate_de_PINN = (state_PINN_e_plus - state_PINN)/delta_e_plus[:,np.newaxis]

vars_FD = [EntropicVars.p,EntropicVars.T, EntropicVars.s, EntropicVars.dsdrho_e, EntropicVars.dsde_rho]
vars_drho = [EntropicVars.dpdrho_e, EntropicVars.dTdrho_e, EntropicVars.dsdrho_e, EntropicVars.d2sdrho2, EntropicVars.d2sdedrho]
vars_de = [EntropicVars.dpde_rho, EntropicVars.dTde_rho, EntropicVars.dsde_rho, EntropicVars.d2sdedrho, EntropicVars.d2sde2]



dstate_drho_CoolProp_FD = dstate_drho_CoolProp[:, [q.value for q in vars_FD]]
dstate_drho_CoolProp_ref = state_CoolProp_ref[:, [q.value for q in vars_drho]]
dstate_de_CoolProp_FD = dstate_de_CoolProp[:, [q.value for q in vars_FD]]
dstate_de_CoolProp_ref = state_CoolProp_ref[:, [q.value for q in vars_de]]

dstate_drho_MTNN_FD = dstate_drho_MTNN[:, [q.value for q in vars_FD]]
dstate_drho_MTNN_ref = state_MTNN[:, [q.value for q in vars_drho]]
dstate_de_MTNN_FD = dstate_de_MTNN[:, [q.value for q in vars_FD]]
dstate_de_MTNN_ref = state_MTNN[:, [q.value for q in vars_de]]

dstate_drho_PINN_FD = dstate_drho_PINN[:, [q.value for q in vars_FD]]
dstate_drho_PINN_ref = state_PINN[:, [q.value for q in vars_drho]]
dstate_de_PINN_FD = dstate_de_PINN[:, [q.value for q in vars_FD]]
dstate_de_PINN_ref = state_PINN[:, [q.value for q in vars_de]]

truncation_error_drho = np.mean(np.abs((dstate_drho_CoolProp_FD - dstate_drho_CoolProp_ref)) / (np.abs(dstate_drho_CoolProp_ref) + 1e-12),axis=0)
truncation_error_de = np.mean(np.abs((dstate_de_CoolProp_FD - dstate_de_CoolProp_ref)) / (np.abs(dstate_de_CoolProp_ref) + 1e-12),axis=0)

MAPE_const_drho_MTNN = np.mean(100*np.abs((dstate_drho_MTNN_FD - dstate_drho_MTNN_ref)) / (np.abs(dstate_drho_MTNN_ref) + 1e-12),axis=0)
MAPE_const_drho_PINN = np.mean(100*np.abs((dstate_drho_PINN_FD - dstate_drho_PINN_ref)) / (np.abs(dstate_drho_PINN_ref) + 1e-12),axis=0)
MAPE_const_de_MTNN = np.mean(100*np.abs((dstate_de_MTNN_FD - dstate_de_MTNN_ref)) / (np.abs(dstate_de_MTNN_ref) + 1e-12),axis=0)
MAPE_const_de_PINN = np.mean(100*np.abs((dstate_de_PINN_FD - dstate_de_PINN_ref)) / (np.abs(dstate_de_PINN_ref) + 1e-12),axis=0)


color_MTNN = colors[1]
color_PINN = colors[2]

fsize= 20
fig,axs = plt.subplots(ncols=1,nrows=2,figsize=[10,10])
ax = axs[0]
width_bar = 0.3
MAPE_state_MTNN = [state_error_MTNN[q.value] for q in vars_of_interest]
MAPE_state_PINN = [state_error_PINN[q.value] for q in vars_of_interest]
ax.bar(x=x_bars-0.5*width_bar, height=MAPE_state_MTNN,width=width_bar,color=color_MTNN,zorder=3,label="MTNN")
ax.bar(x=x_bars+0.5*width_bar, height=MAPE_state_PINN,width=width_bar,color=color_PINN,zorder=3,label="PINN")
ax.set_ylabel("Relative state error",fontsize=fsize)
ax.set_title("Prediction error according to MTNN and PINN",fontsize=fsize)
ax.set_xticks(x_bars)
ax.grid()
ax.set_xticklabels([q.name for q in vars_of_interest])
ax.set_yscale('log')
ax.tick_params(which='both',labelsize=fsize)

ax = axs[1]
width_bar = 0.3

MAPE_const_MTNN = [MAPE_const_drho_MTNN[0],MAPE_const_drho_MTNN[1],MAPE_const_de_MTNN[0],MAPE_const_de_MTNN[1],MAPE_const_drho_MTNN[2],MAPE_const_de_MTNN[2],MAPE_const_drho_MTNN[3],MAPE_const_de_MTNN[3],MAPE_const_de_MTNN[4]]
MAPE_const_PINN = [MAPE_const_drho_PINN[0],MAPE_const_drho_PINN[1],MAPE_const_de_PINN[0],MAPE_const_de_PINN[1],MAPE_const_drho_PINN[2],MAPE_const_de_PINN[2],MAPE_const_drho_PINN[3],MAPE_const_de_PINN[3],MAPE_const_de_PINN[4]]
truncation_error = [truncation_error_drho[0],truncation_error_drho[1],truncation_error_de[0],truncation_error_de[1],truncation_error_drho[2],truncation_error_de[2],truncation_error_drho[3],truncation_error_de[3],truncation_error_de[4]]

x_labels = ["dpdrho_e", "dTdrho_e", "dpde_rho", "dTde_rho", "dsdrho_e","dsde_rho","d2sdrho2","d2sdedrho","d2sde2"]
x_bars = np.arange(len(MAPE_const_MTNN))
ax.bar(x=x_bars-0.5*width_bar, height=MAPE_const_MTNN,width=width_bar,color=color_MTNN,zorder=3,label="MTNN")
ax.bar(x=x_bars+0.5*width_bar, height=MAPE_const_PINN,width=width_bar,color=color_PINN,zorder=3,label="PINN")
ax.set_ylabel("Relative consistency error",fontsize=fsize)
ax.set_title("Consistency error according to MTNN and PINN",fontsize=fsize)
ax.grid()
ax.set_xticks(x_bars)
ax.set_xticklabels(x_labels)
ax.set_yscale('log')
ax.tick_params(which='both',labelsize=fsize)
ax.legend(fontsize=fsize,ncol=2,bbox_to_anchor=(0.5, -0.12),loc='upper center',fancybox=True,shadow=True)
#fig.savefig("Images/MAPE_prediction_consistency.eps",format='eps',bbox_inches='tight')
plt.show()

