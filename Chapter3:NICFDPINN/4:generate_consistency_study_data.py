#!/usr/bin/env python3
###############################################################################################
#       #      _____ __  _____      ____        __        __  ____                   #        #  
#       #     / ___// / / /__ \    / __ \____ _/ /_____ _/  |/  (_)___  ___  _____   #        #  
#       #     \__ \/ / / /__/ /   / / / / __ `/ __/ __ `/ /|_/ / / __ \/ _ \/ ___/   #        #      
#       #    ___/ / /_/ // __/   / /_/ / /_/ / /_/ /_/ / /  / / / / / /  __/ /       #        #  
#       #   /____/\____//____/  /_____/\__,_/\__/\__,_/_/  /_/_/_/ /_/\___/_/        #        #
#       #                                                                            #        #
###############################################################################################

############################ FILE NAME: 3:train_MLP_PINN.py ###################################
#=============================================================================================#
# author: Evert Bunschoten                                                                    |
#    :PhD Candidate ,                                                                         |
#    :Flight Power and Propulsion                                                             |
#    :TU Delft,                                                                               |
#    :The Netherlands                                                                         |
#                                                                                             |
#                                                                                             |
# Description:                                                                                |
# Verification of the correctness of the entropy-based equation of state                      |                                                                                          |  
# Version: 2.0.0                                                                              |
#                                                                                             |
#=============================================================================================#
import numpy as np
import CoolProp.CoolProp as CP
from su2dataminer.config import Config_NICFD
from Common.Properties import EntropicVars


def EntropicEOS(rho:np.ndarray[float],e:np.ndarray[float], s:np.ndarray[float], dsdrhoe:np.ndarray[float], d2sdrho2e2:np.ndarray[float]):
    """Evaluate the thermodynamic state variables with the entropy-based equation of state

    :param rho: fluid density [kg/m3]
    :type rho: np.ndarray[float]
    :param e: fluid static energy [J/kg]
    :type e: np.ndarray[float]
    :param s: entropy [J/kg]
    :type s: np.ndarray[float]
    :param dsdrhoe: entropy Jacobian with respect to density and static energy
    :type dsdrhoe: np.ndarray[float]
    :param d2sdrho2e2: entropy Hessian with respect to density and static energy
    :type d2sdrho2e2: np.ndarray[float]
    :return: thermodynamic state variables
    :rtype: np.ndarray[float]
    """
    dsdrho_e = dsdrhoe[0]
    dsde_rho = dsdrhoe[1]
    d2sdrho2 = d2sdrho2e2[0][0]
    d2sdedrho = d2sdrho2e2[0][1]
    d2sde2 = d2sdrho2e2[1][1]
    T = np.power(dsde_rho, -1)
    rho2 = rho*rho
    P = -rho2 * T * dsdrho_e
    blue_term = (dsdrho_e * (2 - rho * T * d2sdedrho) + rho*d2sdrho2)
    green_term = (-T * d2sde2 * dsdrho_e + d2sdedrho)
    c2 = -rho *T * (blue_term - rho * green_term * (dsdrho_e / dsde_rho))

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
    Y_state = np.vstack((rho, e, T, P, c2, s, dsdrho_e, dsde_rho, d2sdrho2, d2sdedrho, d2sde2, dTdrho_e, dTde_rho, dPdrho_e, dPde_rho, dhdrho_e, dhde_rho, dhdP_rho, dhdrho_P, dsdP_rho, dsdrho_P, Cp)).transpose()
    return Y_state

# Load SU2 DataMienr configuration and read fluid data
Config = Config_NICFD("PINN.cfg")
fluid_data_file = Config.GetOutputDir()+"/"+Config.GetConcatenationFileHeader()+"_full.csv"
with open(fluid_data_file,'r') as fid:
    vars = fid.readline().strip().split(',')
fluid_data = np.loadtxt(fluid_data_file,delimiter=',',skiprows=1)
rhoe = fluid_data[:, [vars.index(EntropicVars.Density.name),vars.index(EntropicVars.Energy.name)]]


# For each thermodynamic state, evaluate the thermodynamic state properties with CoolProp and with the EEoS and compare the values.
fluid = CP.AbstractState(Config.GetEquationOfState(), Config.GetFluid())
state_vector_EEoS = np.zeros([np.shape(rhoe)[0], EntropicVars.N_STATE_VARS.value+1])
state_vector_CoolProp = np.zeros([np.shape(rhoe)[0], EntropicVars.N_STATE_VARS.value+1])

for i in range(len(rhoe)):
    rho = rhoe[i,0]
    e = rhoe[i,1]
    try:
        fluid.update(CP.DmassUmass_INPUTS, rho, e)
        state_vector_CoolProp[i, EntropicVars.s.value] = fluid.smass()
        state_vector_CoolProp[i, EntropicVars.dsde_rho.value] = fluid.first_partial_deriv(CP.iSmass, CP.iUmass, CP.iDmass)
        state_vector_CoolProp[i, EntropicVars.dsdrho_e.value] = fluid.first_partial_deriv(CP.iSmass, CP.iDmass, CP.iUmass)
        state_vector_CoolProp[i, EntropicVars.d2sde2.value] = fluid.second_partial_deriv(CP.iSmass, CP.iUmass, CP.iDmass, CP.iUmass, CP.iDmass)
        state_vector_CoolProp[i, EntropicVars.d2sdedrho.value] = fluid.second_partial_deriv(CP.iSmass, CP.iUmass, CP.iDmass, CP.iDmass, CP.iUmass)
        state_vector_CoolProp[i, EntropicVars.d2sdrho2.value] = fluid.second_partial_deriv(CP.iSmass, CP.iDmass, CP.iUmass, CP.iDmass, CP.iUmass)
        state_vector_CoolProp[i, EntropicVars.Density.value] = fluid.rhomass()
        state_vector_CoolProp[i, EntropicVars.Energy.value] = fluid.umass()
        state_vector_CoolProp[i, EntropicVars.T.value] = fluid.T()
        state_vector_CoolProp[i, EntropicVars.p.value] = fluid.p()
        state_vector_CoolProp[i, EntropicVars.c2.value] = fluid.speed_sound()**2
        state_vector_CoolProp[i, EntropicVars.dTde_rho.value] = fluid.first_partial_deriv(CP.iT, CP.iUmass, CP.iDmass)
        state_vector_CoolProp[i, EntropicVars.dTdrho_e.value] = fluid.first_partial_deriv(CP.iT, CP.iDmass, CP.iUmass)
        state_vector_CoolProp[i, EntropicVars.dpde_rho.value] = fluid.first_partial_deriv(CP.iP, CP.iUmass, CP.iDmass)
        state_vector_CoolProp[i, EntropicVars.dpdrho_e.value] = fluid.first_partial_deriv(CP.iP, CP.iDmass, CP.iUmass)
        state_vector_CoolProp[i, EntropicVars.dhde_rho.value] = fluid.first_partial_deriv(CP.iHmass, CP.iUmass, CP.iDmass)
        state_vector_CoolProp[i, EntropicVars.dhdrho_e.value] = fluid.first_partial_deriv(CP.iHmass, CP.iDmass, CP.iUmass)
        state_vector_CoolProp[i, EntropicVars.dhdp_rho.value] = fluid.first_partial_deriv(CP.iHmass, CP.iP, CP.iDmass)
        state_vector_CoolProp[i, EntropicVars.dhdrho_p.value] = fluid.first_partial_deriv(CP.iHmass, CP.iDmass, CP.iP)
        state_vector_CoolProp[i, EntropicVars.dsdp_rho.value] = fluid.first_partial_deriv(CP.iSmass, CP.iP, CP.iDmass)
        state_vector_CoolProp[i, EntropicVars.dsdrho_p.value] = fluid.first_partial_deriv(CP.iSmass, CP.iDmass, CP.iP)
        state_vector_CoolProp[i, EntropicVars.cp.value] = fluid.cpmass()

        s = state_vector_CoolProp[i, EntropicVars.s.value]
        dsdrhoe = [state_vector_CoolProp[i, EntropicVars.dsdrho_e.value],\
                state_vector_CoolProp[i, EntropicVars.dsde_rho.value]]

        d2sdrhode = [[state_vector_CoolProp[i, EntropicVars.d2sdrho2.value],state_vector_CoolProp[i, EntropicVars.d2sdedrho.value]],\
                    [state_vector_CoolProp[i, EntropicVars.d2sdedrho.value],state_vector_CoolProp[i, EntropicVars.d2sde2.value]]]

        state_vector_EEoS[i, :-1] = EntropicEOS(fluid.rhomass(),fluid.umass(),s,dsdrhoe, d2sdrhode)
        
        Gamma_EEoS = 1 + (fluid.rhomass() / fluid.speed_sound())*fluid.first_partial_deriv(CP.ispeed_sound, CP.iDmass, CP.iSmass)
        Gamma_CoolProp = fluid.fundamental_derivative_of_gas_dynamics()

        state_vector_CoolProp[i,-1]=Gamma_CoolProp
        state_vector_EEoS[i,-1]=Gamma_EEoS
    except:
        pass

# Calculate the root-mean-square relative errors 
rmsr_error_consistency = np.sqrt(np.average(np.power((state_vector_EEoS - state_vector_CoolProp)/np.abs(state_vector_CoolProp+1e-6),2),axis=0))

print("Entropic equation of state consistency errors:")
print("Temperature: %.3f" % (np.log10(rmsr_error_consistency[EntropicVars.T.value])))
print("Pressure: %.3f" % (np.log10(rmsr_error_consistency[EntropicVars.p.value])))
print("Speed of sound: %.3f" % (np.log10(rmsr_error_consistency[EntropicVars.c2.value])))
