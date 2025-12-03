from su2dataminer.config import Config_FGM
from su2dataminer.manifold import MLPOptimizer_FGM

config = Config_FGM("HP_Optimization.cfg")
Nproc = 24 

for iGroup in range(config.GetNMLPOutputGroups()):
    MLPO = MLPOptimizer_FGM(config)
    MLPO.SetNWorkers(24)
    MLPO.SetOutputGroup(iGroup)
    MLPO.SetNGenerations(20)
    MLPO.Optimize_ActivationFunction(True)
    MLPO.Optimize_Architecture_HP(True)
    MLPO.Optimize_LearningRate_HP(True)
    MLPO.Optimize_Batch_HP(False)
    MLPO.SetBatch_Expo(6)
    MLPO.Optimize_Pareto(True)
    MLPO.optimizeHP()