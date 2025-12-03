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
