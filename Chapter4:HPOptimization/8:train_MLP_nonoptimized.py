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

for iGroup in range(1, config_optim.GetNMLPOutputGroups()):
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