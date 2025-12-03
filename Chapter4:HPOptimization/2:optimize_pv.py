import os 
from su2dataminer.config import Config_FGM
from su2dataminer.process_data import PVOptimizer, PVOptimizer_PCA

config = Config_FGM("OPT.cfg")
pvo = PVOptimizer(config)
pvo.SetOutputDir(os.getcwd()+"/PV_Optimization")
pvo.OptimizePV()
config.SetProgressVariableDefinition(pvo.GetOptimizedSpecies(), pvo.GetOptimizedWeights())
config.SaveConfig()


config = Config_FGM("PCA.cfg")
pvo_pca = PVOptimizer_PCA(config)
pvo_pca.OptimizePV()
config.SetProgressVariableDefinition(pvo_pca.GetOptimizedSpecies(), pvo_pca.GetOptimizedWeights())
config.PrintBanner()
config.SaveConfig()
