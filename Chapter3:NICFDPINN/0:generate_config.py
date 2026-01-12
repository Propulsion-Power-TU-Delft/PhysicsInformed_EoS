#!/usr/bin/env python3
###############################################################################################
#       #      _____ __  _____      ____        __        __  ____                   #        #  
#       #     / ___// / / /__ \    / __ \____ _/ /_____ _/  |/  (_)___  ___  _____   #        #  
#       #     \__ \/ / / /__/ /   / / / / __ `/ __/ __ `/ /|_/ / / __ \/ _ \/ ___/   #        #      
#       #    ___/ / /_/ // __/   / /_/ / /_/ / /_/ /_/ / /  / / / / / /  __/ /       #        #  
#       #   /____/\____//____/  /_____/\__,_/\__/\__,_/_/  /_/_/_/ /_/\___/_/        #        #
#       #                                                                            #        #
###############################################################################################

############################ FILE NAME: 0:generate_config.py ##################################
#=============================================================================================#
# author: Evert Bunschoten                                                                    |
#    :PhD Candidate ,                                                                         |
#    :Flight Power and Propulsion                                                             |
#    :TU Delft,                                                                               |
#    :The Netherlands                                                                         |
#                                                                                             |
#                                                                                             |
# Description:                                                                                |
# Define SU2 DataMiner configurations for generating the neural networks used in the EEoS-PINN|
# and EEoS-MTNN models.                                                                       |
#                                                                                             |  
# Version: 2.0.0                                                                              |
#                                                                                             |
#=============================================================================================#

import os 
from su2dataminer.config import Config_NICFD
Config = Config_NICFD()

# Density bounds: 1.0e-1 - 3.6e+2, where the EEoS-MTNN uses separate networks for the density
# ranges 1.0e-1 - 1.006e+1 and 1.006e+1 - 3.6e+2. 
rho_low = 0.1
rho_high = 360.0
rho_split = +1.0064969312093510e+01
Np_x = 500

# Static energy bounds: 2.5e+5 - 5.5e+5 for both models.
e_low = 2.5e5
e_high = 5.5e5
Np_y = 500

# Define fluid name and equation of state used to generate the reference fluid data.
Config.SetFluid("MM")               # Siloxane MM.
Config.SetEquationOfState("HEOS")   # Helmholtz-equation of state.
Config.UsePTGrid(False)             # Use density-energy data grid.
Config.UseAutoRange(False)          # User-defined bounds for density and static energy.

# Define discretization of fluid data grid for static energy.
Config.SetEnergyBounds(E_lower=e_low, E_upper=e_high)
Config.SetNpDensity(Np_rho=Np_x)
Config.SetNpEnergy(Np_Energy=Np_y)

# Define configurations for the EEoS-MTNN model for the upper density range
# (Direct_high) and the lower density range (Direct_lower). Fluid data are 
# stored in the folder "Direct" that is created under the current working 
# directory.
Config.SetConfigName("Direct_high")
Config.SetDensityBounds(Rho_lower=rho_split, Rho_upper=rho_high)
if not (os.path.isdir(os.getcwd()+"/Direct/")):
    os.mkdir(os.getcwd()+"/Direct/")
Config.SetOutputDir(os.getcwd()+"/Direct/")
Config.SetConcatenationFileHeader("fluid_data_high")
Config.SaveConfig()

Config.SetConfigName("Direct_low")
Config.SetDensityBounds(Rho_lower=rho_low, Rho_upper=rho_split)
Config.SetConcatenationFileHeader("fluid_data_low")
Config.SaveConfig()

# Define configuration for the EEoS-PINN model. Fluid data are 
# stored in the folder "PhysicsInformed" that is created under the current working 
# directory.
Config.SetConfigName("PINN")
Config.SetDensityBounds(Rho_lower=rho_low, Rho_upper=rho_high)
Config.SetNpDensity(Np_x)
Config.SetConcatenationFileHeader("fluid_data_PINN")
if not (os.path.isdir(os.getcwd()+"/PhysicsInformed/")):
    os.mkdir(os.getcwd()+"/PhysicsInformed/")
Config.SetOutputDir(os.getcwd()+"/PhysicsInformed/")
Config.SaveConfig()