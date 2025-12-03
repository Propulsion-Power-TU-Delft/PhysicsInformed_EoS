from su2dataminer.config import Config_FGM
from su2dataminer.process_data import FlameletConcatenator,GroupOutputs

config = Config_FGM("HP_Optimization.cfg")
FC = FlameletConcatenator(config)
FC.SetNFlameletNodes(2**(config.GetBatchExpo(0)-1))
FC.SetBoundaryFileName("boundary_data_hpoptim")
FC.IgnoreMixtureBounds(True)
FC.ConcatenateFlameletData()
FC.CollectBoundaryData()

G = GroupOutputs(config)
G.EvaluateGroups()
G.PlotCorrelationMatrix()
G.UpdateConfig()
config.SaveConfig()