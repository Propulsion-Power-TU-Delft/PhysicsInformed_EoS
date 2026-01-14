#!/usr/bin/env python3
###############################################################################################
#       #      _____ __  _____      ____        __        __  ____                   #        #  
#       #     / ___// / / /__ \    / __ \____ _/ /_____ _/  |/  (_)___  ___  _____   #        #  
#       #     \__ \/ / / /__/ /   / / / / __ `/ __/ __ `/ /|_/ / / __ \/ _ \/ ___/   #        #      
#       #    ___/ / /_/ // __/   / /_/ / /_/ / /_/ /_/ / /  / / / / / /  __/ /       #        #  
#       #   /____/\____//____/  /_____/\__,_/\__/\__,_/_/  /_/_/_/ /_/\___/_/        #        #
#       #                                                                            #        #
###############################################################################################

######################## FILE NAME: 1:generate_flamelet_data.py ###############################
#=============================================================================================#
# author: Evert Bunschoten                                                                    |
#    :PhD Candidate ,                                                                         |
#    :Flight Power and Propulsion                                                             |
#    :TU Delft,                                                                               |
#    :The Netherlands                                                                         |
#                                                                                             |
#                                                                                             |
# Description:                                                                                |
# Generate flamelet solutions used for calculating the coefficients of the progress variables |
# and from which to extract the ML-FGM training data.                                         |
#                                                                                             |  
# Version: 2.0.0                                                                              |
#                                                                                             |
#=============================================================================================#
import sys
from su2dataminer.config import Config_FGM
from su2dataminer.generate_data import ComputeFlameletData,ComputeBoundaryData

try:
    N_proc = int(sys.argv[-1])
except:
    N_proc = 1

config = Config_FGM("WRP.cfg")
run_parallel=(N_proc>1)
ComputeFlameletData(config, run_parallel=run_parallel, N_processors=N_proc)
ComputeBoundaryData(config,run_parallel,N_proc)
