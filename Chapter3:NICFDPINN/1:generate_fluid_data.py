#!/usr/bin/env python3
###############################################################################################
#       #      _____ __  _____      ____        __        __  ____                   #        #  
#       #     / ___// / / /__ \    / __ \____ _/ /_____ _/  |/  (_)___  ___  _____   #        #  
#       #     \__ \/ / / /__/ /   / / / / __ `/ __/ __ `/ /|_/ / / __ \/ _ \/ ___/   #        #      
#       #    ___/ / /_/ // __/   / /_/ / /_/ / /_/ /_/ / /  / / / / / /  __/ /       #        #  
#       #   /____/\____//____/  /_____/\__,_/\__/\__,_/_/  /_/_/_/ /_/\___/_/        #        #
#       #                                                                            #        #
###############################################################################################

########################## FILE NAME: 1:generate_fluid_data.py ################################
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


from su2dataminer.config import Config_NICFD
from su2dataminer.generate_data import DataGenerator_CoolProp

# Load SU2 DataMiner configurations.
Config_direct_high = Config_NICFD("Direct_high.cfg")
Config_direct_low = Config_NICFD("Direct_low.cfg")
Config_PINN = Config_NICFD("PINN.cfg")

for Config in [Config_PINN, Config_direct_high, Config_direct_low]:
    D = DataGenerator_CoolProp(Config)

    D.UseAutoRange(False)

    # Define data grid.
    D.PreprocessData()

    # For every node in the data grid, compute thermodynamic state.
    D.ComputeData()
    
    # Save all fluid data in storage directory for each configuration.
    D.SaveData()
