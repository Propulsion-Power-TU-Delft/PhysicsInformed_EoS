#!/usr/bin/env python3
###############################################################################################
#       #      _____ __  _____      ____        __        __  ____                   #        #  
#       #     / ___// / / /__ \    / __ \____ _/ /_____ _/  |/  (_)___  ___  _____   #        #  
#       #     \__ \/ / / /__/ /   / / / / __ `/ __/ __ `/ /|_/ / / __ \/ _ \/ ___/   #        #      
#       #    ___/ / /_/ // __/   / /_/ / /_/ / /_/ /_/ / /  / / / / / /  __/ /       #        #  
#       #   /____/\____//____/  /_____/\__,_/\__/\__,_/_/  /_/_/_/ /_/\___/_/        #        #
#       #                                                                            #        #
###############################################################################################

######################### FILE NAME: 2:train_MLP_segregated.py ################################
#=============================================================================================#
# author: Evert Bunschoten                                                                    |
#    :PhD Candidate ,                                                                         |
#    :Flight Power and Propulsion                                                             |
#    :TU Delft,                                                                               |
#    :The Netherlands                                                                         |
#                                                                                             |
#                                                                                             |
# Description:                                                                                |
# Train the two artificial neural networks used in the EEoS-MTNN model.                       |
#                                                                                             |  
# Version: 2.0.0                                                                              |
#                                                                                             |
#=============================================================================================#

from su2dataminer.config import Config_NICFD
from su2dataminer.manifold import TrainMLP_NICFD_Segregated

# Set network training hyper-parameters, hidden layer architectures, and hidden layer activation
# functions of the MLPs.

Config_low = Config_NICFD("Direct_low.cfg")
Config_low.SetAlphaExpo(-3.1146e+00)            # Initial learning rate exponent (base 10).
Config_low.SetLRDecay(+9.8732e-01)              # Learning rate decay parameter.
Config_low.SetBatchExpo(5)                      # Mini-batch size exponent (base 2).
Config_low.SetActivationFunction("swish")       # Hidden layer activation function.
Config_low.SetHiddenLayerArchitecture([11,14])  # Hidden layer architecture.
Config_low.SaveConfig()

Config_high = Config_NICFD("Direct_high.cfg")
Config_high.SetAlphaExpo(-3.0721e+00)               # Initial learning rate exponent (base 10).
Config_high.SetLRDecay(+9.925e-01)                  # Learning rate decay parameter.
Config_high.SetBatchExpo(5)                         # Mini-batch size exponent (base 2).
Config_high.SetActivationFunction("gelu")           # Hidden layer activation function.
Config_high.SetHiddenLayerArchitecture([13,15,12])  # Hidden layer architecture.
Config_high.SaveConfig()


# Train MLP for lower density range.
Eval = TrainMLP_NICFD_Segregated(Config_low)
Eval.SetScaler("minmax")
Eval.SetVerbose(1)

# Start the training process.
Eval.CommenceTraining()
Eval.TrainPostprocessing()
Config_low.UpdateMLPHyperParams(Eval)
Config_low.SaveConfig()

# Train MLP for upper density range.
Eval = TrainMLP_NICFD_Segregated(Config_high)
Eval.SetScaler("minmax")
Eval.SetVerbose(1)

# Start the training process.
Eval.CommenceTraining()
Eval.TrainPostprocessing()
Config_high.UpdateMLPHyperParams(Eval)
Config_high.SaveConfig()