#!/usr/bin/env python3
###############################################################################################
#       #      _____ __  _____      ____        __        __  ____                   #        #  
#       #     / ___// / / /__ \    / __ \____ _/ /_____ _/  |/  (_)___  ___  _____   #        #  
#       #     \__ \/ / / /__/ /   / / / / __ `/ __/ __ `/ /|_/ / / __ \/ _ \/ ___/   #        #      
#       #    ___/ / /_/ // __/   / /_/ / /_/ / /_/ /_/ / /  / / / / / /  __/ /       #        #  
#       #   /____/\____//____/  /_____/\__,_/\__/\__,_/_/  /_/_/_/ /_/\___/_/        #        #
#       #                                                                            #        #
###############################################################################################

############################# FILE NAME: 6:group_variables.py #################################
#=============================================================================================#
# author: Evert Bunschoten                                                                    |
#    :PhD Candidate ,                                                                         |
#    :Flight Power and Propulsion                                                             |
#    :TU Delft,                                                                               |
#    :The Netherlands                                                                         |
#                                                                                             |
#                                                                                             |
# Description:                                                                                |
# Extract ML-FGM training data from flamelets and group the thermochemical state variables to |
# reduce the number of networks in the ML-FGM model.                                          |
#                                                                                             |
# Version: 2.0.0                                                                              |
#                                                                                             |
#=============================================================================================#
from su2dataminer.config import Config_FGM
from su2dataminer.process_data import FlameletConcatenator,GroupOutputs

config = Config_FGM("HP_Optimization.cfg")

# Extract training data from flamelet solutions
FC = FlameletConcatenator(config)

# Reduce the number of samples in the training data by half to reduce the training time
FC.SetNFlameletNodes(2**(config.GetBatchExpo(0)-1))
FC.SetBoundaryFileName("boundary_data_hpoptim")
FC.IgnoreMixtureBounds(True)
FC.ConcatenateFlameletData()
FC.CollectBoundaryData()

# Group thermochemical state variables 
G = GroupOutputs(config)
G.EvaluateGroups()
G.PlotCorrelationMatrix()
G.UpdateConfig()
config.SaveConfig()