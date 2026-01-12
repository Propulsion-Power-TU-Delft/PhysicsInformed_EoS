#!/usr/bin/env python3
###############################################################################################
#       #      _____ __  _____      ____        __        __  ____                   #        #  
#       #     / ___// / / /__ \    / __ \____ _/ /_____ _/  |/  (_)___  ___  _____   #        #  
#       #     \__ \/ / / /__/ /   / / / / __ `/ __/ __ `/ /|_/ / / __ \/ _ \/ ___/   #        #      
#       #    ___/ / /_/ // __/   / /_/ / /_/ / /_/ /_/ / /  / / / / / /  __/ /       #        #  
#       #   /____/\____//____/  /_____/\__,_/\__/\__,_/_/  /_/_/_/ /_/\___/_/        #        #
#       #                                                                            #        #
###############################################################################################

############################ FILE NAME: 3:train_MLP_PINN.py ###################################
#=============================================================================================#
# author: Evert Bunschoten                                                                    |
#    :PhD Candidate ,                                                                         |
#    :Flight Power and Propulsion                                                             |
#    :TU Delft,                                                                               |
#    :The Netherlands                                                                         |
#                                                                                             |
#                                                                                             |
# Description:                                                                                |
# Train the physics-informed neural network used to evaluate the fluid thermodynamic state in |
# the EEoS-PINN model.                                                                        |
#                                                                                             |  
# Version: 2.0.0                                                                              |
#                                                                                             |
#=============================================================================================#

from su2dataminer.config import Config_NICFD
from su2dataminer.manifold import TrainMLP_NICFD


# Set network training hyper-parameters, hidden layer architectures, and hidden layer activation
# functions of the PINN.
Config = Config_NICFD("PINN.cfg")
Config.SetActivationFunction("exponential") # Hidden layer activation function.
Config.SetAlphaExpo(-3.0)                   # Initial learning rate exponent (base 10).
Config.SetLRDecay(+9.8787e-01)              # Learning rate decay parameter.
Config.SetBatchExpo(6)                      # Mini-batch size exponent (base 2).
Config.SetHiddenLayerArchitecture([12,12])  # Hidden layer architecture.
Config.SaveConfig()


Eval = TrainMLP_NICFD(Config)
Eval.SetScaler("minmax")
Eval.SetVerbose(1)
# Start the training process.
Eval.CommenceTraining()
Eval.TrainPostprocessing()
Config.UpdateMLPHyperParams(Eval)
Config.SaveConfig()