
from su2dataminer.config import Config_FGM
from su2dataminer.process_data import FlameletConcatenator

config_files = ["OPT.cfg", "PCA.cfg", "WRP.cfg"]
for config_file_name in config_files:
    Config = Config_FGM(config_file_name)
    FC = FlameletConcatenator(Config)
    FC.SetNFlameletNodes(2**Config.GetBatchExpo())
    FC.IgnoreMixtureBounds(True)
    FC.SetBoundaryFileName("boundary_data_%s" % (config_file_name.split(".")[0]))
    FC.ConcatenateFlameletData()
    FC.CollectBoundaryData()

