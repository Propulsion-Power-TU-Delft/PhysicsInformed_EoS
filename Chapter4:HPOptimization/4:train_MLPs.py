#!/usr/bin/env python3
###############################################################################################
#       #      _____ __  _____      ____        __        __  ____                   #        #  
#       #     / ___// / / /__ \    / __ \____ _/ /_____ _/  |/  (_)___  ___  _____   #        #  
#       #     \__ \/ / / /__/ /   / / / / __ `/ __/ __ `/ /|_/ / / __ \/ _ \/ ___/   #        #      
#       #    ___/ / /_/ // __/   / /_/ / /_/ / /_/ /_/ / /  / / / / / /  __/ /       #        #  
#       #   /____/\____//____/  /_____/\__,_/\__/\__,_/_/  /_/_/_/ /_/\___/_/        #        #
#       #                                                                            #        #
###############################################################################################

############################### FILE NAME: 4:train_MLPs.py ####################################
#=============================================================================================#
# author: Evert Bunschoten                                                                    |
#    :PhD Candidate ,                                                                         |
#    :Flight Power and Propulsion                                                             |
#    :TU Delft,                                                                               |
#    :The Netherlands                                                                         |
#                                                                                             |
#                                                                                             |
# Description:                                                                                |
# Train the ML-FGM networks for each progress variable definition.                            |
#                                                                                             |
# Version: 2.0.0                                                                              |
#                                                                                             |
#=============================================================================================#

import os 
from su2dataminer.config import Config_FGM
from su2dataminer.manifold import TrainMLP_FGM

pv_defs = ["PCA","OPT","WRP"]
for pv in pv_defs:

    Config = Config_FGM("%s.cfg" % pv)

    for iGroup in range(Config.GetNMLPOutputGroups()):
        Eval = TrainMLP_FGM(Config)
        Eval.SetVerbose(1)
        Eval.SetOutputGroup(iGroup)
        Eval.SetSaveDir(os.getcwd()+"/Architectures_%s" % pv)
        Eval.SetBoundaryDataFile("%s/boundary_data_%s_full.csv" % (Config.GetOutputDir(), pv))
        Eval.CommenceTraining()
        Eval.TrainPostprocessing()
        Config.UpdateMLPHyperParams(Eval)
        Config.SaveConfig()
