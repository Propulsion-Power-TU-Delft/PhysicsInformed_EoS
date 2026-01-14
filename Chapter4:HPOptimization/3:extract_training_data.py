#!/usr/bin/env python3
###############################################################################################
#       #      _____ __  _____      ____        __        __  ____                   #        #  
#       #     / ___// / / /__ \    / __ \____ _/ /_____ _/  |/  (_)___  ___  _____   #        #  
#       #     \__ \/ / / /__/ /   / / / / __ `/ __/ __ `/ /|_/ / / __ \/ _ \/ ___/   #        #      
#       #    ___/ / /_/ // __/   / /_/ / /_/ / /_/ /_/ / /  / / / / / /  __/ /       #        #  
#       #   /____/\____//____/  /_____/\__,_/\__/\__,_/_/  /_/_/_/ /_/\___/_/        #        #
#       #                                                                            #        #
###############################################################################################

######################### FILE NAME: 3:extract_training_data.py ###############################
#=============================================================================================#
# author: Evert Bunschoten                                                                    |
#    :PhD Candidate ,                                                                         |
#    :Flight Power and Propulsion                                                             |
#    :TU Delft,                                                                               |
#    :The Netherlands                                                                         |
#                                                                                             |
#                                                                                             |
# Description:                                                                                |
# Extract ML-FGM training data from flamelet solutions for each progress variable definition. |
#                                                                                             |  
# Version: 2.0.0                                                                              |
#                                                                                             |
#=============================================================================================#
from su2dataminer.config import Config_FGM
from su2dataminer.process_data import FlameletConcatenator

config_files = ["OPT.cfg", "PCA.cfg", "WRP.cfg"]
for config_file_name in config_files:
    Config = Config_FGM(config_file_name)
    FC = FlameletConcatenator(Config)
    FC.SetNFlameletNodes(2**Config.GetBatchExpo())
    FC.IgnoreMixtureBounds(True)
    FC.SetBoundaryFileName("boundary_data_%s" % (config_file_name.split(".")[0]))
    FC.ConcatenateFlameletData()
    FC.CollectBoundaryData()

