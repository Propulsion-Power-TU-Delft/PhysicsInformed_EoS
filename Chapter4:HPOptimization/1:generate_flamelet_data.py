import sys
from su2dataminer.config import Config_FGM
from su2dataminer.generate_data import ComputeFlameletData,ComputeBoundaryData

try:
    N_proc = int(sys.argv[-1])
except:
    N_proc = 1
    
config = Config_FGM("WRP.cfg")
run_parallel=(N_proc>1)
ComputeFlameletData(config, run_parallel=run_parallel, N_processors=N_proc)
ComputeBoundaryData(config,run_parallel,N_proc)
