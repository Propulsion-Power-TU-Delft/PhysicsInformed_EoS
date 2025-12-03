
from su2dataminer.config import Config_FGM
from su2dataminer.manifold import TrainMLP_FGM

Config = Config_FGM("config_PIML.cfg")

# Train MLPs while using the physics-informed penalty functions.
Eval_PINN = TrainMLP_FGM(Config)
Eval_PINN.SetVerbose(1)
for iGroup in range(Config.GetNMLPOutputGroups()):
    Eval_PINN.SetOutputGroup(iGroup)
    Eval_PINN.EnableBCLoss(True)
    Eval_PINN.CommenceTraining()
    Eval_PINN.TrainPostprocessing()
    Config.UpdateMLPHyperParams(Eval_PINN)
    Config.SaveConfig()

# Train another set of MLPs using data-fitting only.
Config.SetConfigName("config_DF")
Config.SaveConfig()
Eval_DF = TrainMLP_FGM(Config)

Eval_DF.SetVerbose(1)
for iGroup in range(Config.GetNMLPOutputGroups()):
    Eval_DF.SetOutputGroup(iGroup)
    Eval_DF.EnableBCLoss(False)
    Eval_DF.CommenceTraining()
    Eval_DF.TrainPostprocessing()
    Config.UpdateMLPHyperParams(Eval_DF)
    Config.SaveConfig()

