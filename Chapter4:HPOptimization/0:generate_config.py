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
# Define SU2 DataMiner configurations for training ML-FGM networks with three different       |
# progress variable definitions.                                                              |
#                                                                                             |  
# Version: 2.0.0                                                                              |
#                                                                                             |
#=============================================================================================#

import os 
from su2dataminer.config import Config_FGM

# Equivalence ratio range: 0.25-20.0
phi_min = 0.25 
phi_max = 20.0
Np_mix = 40 

# Reactant temperature range: 300K-860K
Tu_min = 300.0
Tu_max = 860.0
Np_temp = 30

# Common settings for manifold
Config = Config_FGM()
# # Reactants: hydrogen-air
Config.SetFuelDefinition(fuel_species=["H2"],fuel_weights=[1.0])

# # Use Zhang syngas-nox mechanism
Config.SetReactionMechanism('stanford.yaml')

# # Equivalence ratio between 0.25 and 20
Config.SetMixtureBounds(phi_min, phi_max)
Config.SetNpMix(Np_mix)

# # Reactant temperature between 300 and 860K
Config.SetUnbTempBounds(Tu_min, Tu_max)
Config.SetNpTemp(Np_temp)

# Consider adiabatic flamelets, burner-stabilized flamelets, and chemical equilibrium data
Config.RunBurnerFlames(True)
Config.RunFreeFlames(True)
Config.RunEquilibrium(True)

# Multi-component transport model for preferential diffusion.
Config.SetTransportModel('multicomponent')

# Set average Lewis numbers at equivalence ratio 0.5 and reactant temperature 300 K.
Config.SetAverageLewisNumbers(0.5, 300)

# No passive look-ups and species
Config.SetPassiveSpecies([])
Config.SetLookUpVariables(["Heat_Release"])

# MLP outputs are set according to physics-informed quantities:
# Temperature, mean molecular weight, progress variable source term,
# and the preferential diffusion scalars for progress variable, mixture 
# fraction, and specific enthalpy.
Config.ClearOutputGroups()
Config.AddOutputGroup(["Temperature"])
Config.AddOutputGroup(["Cp"])
Config.AddOutputGroup(["ViscosityDyn"])
Config.AddOutputGroup(["MolarWeightMix"])
Config.AddOutputGroup(["Conductivity"])
Config.AddOutputGroup(["DiffusionCoefficient"])
Config.AddOutputGroup(["Beta_ProgVar"])
Config.AddOutputGroup(["Beta_Enth_Thermal"])
Config.AddOutputGroup(["Beta_Enth"])
Config.AddOutputGroup(["Beta_MixFrac"])
Config.AddOutputGroup(["ProdRateTot_PV"])
Config.AddOutputGroup(["Heat_Release"])


Config.SetOutputDir(os.getcwd()+"/flamelet_data/")

Config.SetConfigName("WRP")
Config.SetConcatenationFileHeader("flamelet_data_WRP")
Config.SaveConfig()

Config.SetConfigName("PCA")
Config.SetConcatenationFileHeader("flamelet_data_PCA")
Config.SaveConfig()

Config.SetConfigName("OPT")
Config.SetConcatenationFileHeader("flamelet_data_OPT")
Config.SaveConfig()
