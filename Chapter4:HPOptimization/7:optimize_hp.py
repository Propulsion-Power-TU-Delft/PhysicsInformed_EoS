#!/usr/bin/env python3
###############################################################################################
#       #      _____ __  _____      ____        __        __  ____                   #        #  
#       #     / ___// / / /__ \    / __ \____ _/ /_____ _/  |/  (_)___  ___  _____   #        #  
#       #     \__ \/ / / /__/ /   / / / / __ `/ __/ __ `/ /|_/ / / __ \/ _ \/ ___/   #        #      
#       #    ___/ / /_/ // __/   / /_/ / /_/ / /_/ /_/ / /  / / / / / /  __/ /       #        #  
#       #   /____/\____//____/  /_____/\__,_/\__/\__,_/_/  /_/_/_/ /_/\___/_/        #        #
#       #                                                                            #        #
###############################################################################################

############################### FILE NAME: 7:optimize_hp.py ###################################
#=============================================================================================#
# author: Evert Bunschoten                                                                    |
#    :PhD Candidate ,                                                                         |
#    :Flight Power and Propulsion                                                             |
#    :TU Delft,                                                                               |
#    :The Netherlands                                                                         |
#                                                                                             |
#                                                                                             |
# Description:                                                                                |
# Optimize the learning rate, activation function, and hidden layer architectures of the      |
# networks in each group by calculating the Pareto front of accuracy and computational cost.  |
#                                                                                             |
# Version: 2.0.0                                                                              |
#                                                                                             |
#=============================================================================================#
import sys 
from su2dataminer.config import Config_FGM
from su2dataminer.manifold import MLPOptimizer_FGM

config = Config_FGM("HP_Optimization.cfg")
try:
    N_proc = int(sys.argv[-1])
except:
    N_proc = 1

for iGroup in range(config.GetNMLPOutputGroups()):
    MLPO = MLPOptimizer_FGM(config)
    MLPO.SetNWorkers(N_proc)
    MLPO.SetOutputGroup(iGroup)
    MLPO.SetNGenerations(20)
    MLPO.Optimize_ActivationFunction(True)
    MLPO.Optimize_Architecture_HP(True)
    MLPO.Optimize_LearningRate_HP(True)
    MLPO.Optimize_Batch_HP(False)
    MLPO.SetBatch_Expo(6)
    MLPO.Optimize_Pareto(True)
    MLPO.optimizeHP()