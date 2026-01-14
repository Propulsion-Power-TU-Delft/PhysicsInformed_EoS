#!/usr/bin/env python3
###############################################################################################
#       #      _____ __  _____      ____        __        __  ____                   #        #  
#       #     / ___// / / /__ \    / __ \____ _/ /_____ _/  |/  (_)___  ___  _____   #        #  
#       #     \__ \/ / / /__/ /   / / / / __ `/ __/ __ `/ /|_/ / / __ \/ _ \/ ___/   #        #      
#       #    ___/ / /_/ // __/   / /_/ / /_/ / /_/ /_/ / /  / / / / / /  __/ /       #        #  
#       #   /____/\____//____/  /_____/\__,_/\__/\__,_/_/  /_/_/_/ /_/\___/_/        #        #
#       #                                                                            #        #
###############################################################################################

############################## FILE NAME: 2:optimize_pv.py ####################################
#=============================================================================================#
# author: Evert Bunschoten                                                                    |
#    :PhD Candidate ,                                                                         |
#    :Flight Power and Propulsion                                                             |
#    :TU Delft,                                                                               |
#    :The Netherlands                                                                         |
#                                                                                             |
#                                                                                             |
# Description:                                                                                |
# Calculate the coefficients of the progress variable using an optimization-based approach    |
# and through principal component analysis.                                                   |
#                                                                                             |  
# Version: 2.0.0                                                                              |
#                                                                                             |
#=============================================================================================#
import os 
from su2dataminer.config import Config_FGM
from su2dataminer.process_data import PVOptimizer, PVOptimizer_PCA

# Calculate progress variable coefficients through optimization.
config = Config_FGM("OPT.cfg")
pvo = PVOptimizer(config)
pvo.SetOutputDir(os.getcwd()+"/PV_Optimization")
pvo.OptimizePV()
config.SetProgressVariableDefinition(pvo.GetOptimizedSpecies(), pvo.GetOptimizedWeights())
config.PrintBanner()
config.SaveConfig()

# Calculate progress variable coefficients through principal component analysis.
config = Config_FGM("PCA.cfg")
pvo_pca = PVOptimizer_PCA(config)
pvo_pca.OptimizePV()
config.SetProgressVariableDefinition(pvo_pca.GetOptimizedSpecies(), pvo_pca.GetOptimizedWeights())
config.PrintBanner()
config.SaveConfig()
