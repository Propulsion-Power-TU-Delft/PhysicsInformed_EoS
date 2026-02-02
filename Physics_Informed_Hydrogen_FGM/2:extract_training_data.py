# Extract training data sets from flamelet solutions.
from su2dataminer.config import Config_FGM
from su2dataminer.process_data import FlameletConcatenator

# Load SU2 DataMiner configuration
Config = Config_FGM("config_PIML.cfg")

# Specify training data set options.
FC = FlameletConcatenator(Config)
FC.SetNFlameletNodes(2**Config.GetBatchExpo())
FC.IgnoreMixtureBounds(True)
FC.IncludeEquilibrium(True)
FC.ConcatenateFlameletData()
FC.CollectBoundaryData()

# Construct an additional flamelet data set which excludes chemical equilibrium data.
FC_domain = FlameletConcatenator(Config)
FC_domain.SetOutputFileName("flamelet_domain_data")
FC_domain.SetNFlameletNodes(2**Config.GetBatchExpo())
FC_domain.IncludeEquilibrium(False)
FC_domain.IncludeFreeFlames(True)
FC_domain.IncludeBurnerFlames(True)
FC_domain.ConcatenateFlameletData()
