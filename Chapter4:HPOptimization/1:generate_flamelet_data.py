from su2dataminer.config import Config_FGM
from su2dataminer.generate_data import ComputeFlameletData,ComputeBoundaryData


N_proc = 4
config = Config_FGM("WRP.cfg")
run_parallel=(N_proc>1)
ComputeFlameletData(config, run_parallel=run_parallel, N_processors=N_proc)
ComputeBoundaryData(config,run_parallel,N_proc)
