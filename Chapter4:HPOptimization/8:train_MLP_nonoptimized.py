###############################################################################################
#       #      _____ __  _____      ____        __        __  ____                   #        #
#       #     / ___// / / /__ \    / __ \____ _/ /_____ _/  |/  (_)___  ___  _____   #        #
#       #     \__ \/ / / /__/ /   / / / / __ `/ __/ __ `/ /|_/ / / __ \/ _ \/ ___/   #        #
#       #    ___/ / /_/ // __/   / /_/ / /_/ / /_/ /_/ / /  / / / / / /  __/ /       #        #
#       #   /____/\____//____/  /_____/\__,_/\__/\__,_/_/  /_/_/_/ /_/\___/_/        #        #
#       #                                                                            #        #
###############################################################################################

######################### FILE NAME: 8:train_MLP_nonoptimized.py ##############################
#=============================================================================================#
# author: Evert Bunschoten                                                                    |
#    :PhD Candidate ,                                                                         |
#    :Flight Power and Propulsion                                                             |
#    :TU Delft,                                                                               |
#    :The Netherlands                                                                         |
#                                                                                             |
#                                                                                             |
# Description:                                                                                |
#  Train the ML-FGM networks with the hyper-parameters selected through trial and error.      |
#                                                                                             |
# Version: 2.0.0                                                                              |
#                                                                                             |
#=============================================================================================#
import os
from su2dataminer.config import Config_FGM
from su2dataminer.manifold import TrainMLP_FGM

config_optim = Config_FGM("HP_Optimization.cfg")
config_optim.SetConfigName("Unoptimized")
N_H = [16,20,28,34,30,24,20] 
r_l_0 = -2.80
lr_decay = 0.9896
batch_expo = 6
activation_f = "gelu"

for iGroup in range(config_optim.GetNMLPOutputGroups()):
    trainer = TrainMLP_FGM(config_optim, iGroup)
    trainer.SetHiddenLayers(N_H)
    trainer.SetAlphaExpo(r_l_0)
    trainer.SetLRDecay(lr_decay)
    trainer.SetBatchExpo(batch_expo)
    trainer.SetActivationFunction(activation_f)
    base_dir = "%s/Architectures_UnOptimized/" % (os.getcwd())
    if not os.path.isdir(base_dir + "Group%i" %(iGroup+1)):
        os.mkdir(base_dir + "Group%i" %(iGroup+1))
    trainer.SetSaveDir(base_dir + "Group%i" %(iGroup+1))
    trainer.CommenceTraining()
    config_optim.UpdateMLPHyperParams(trainer)
    config_optim.SaveConfig()