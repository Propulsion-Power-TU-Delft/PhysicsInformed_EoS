# Generate flamelet data used for to train the ML-FGM networks.

from su2dataminer.config import Config_FGM
from su2dataminer.generate_data import ComputeFlameletData,ComputeBoundaryData

# Number of cores to use for the flamelet data calculation.
N_proc = 4

# Load SU2 DataMiner configuration
config = Config_FGM("config_PIML.cfg")
run_parallel=(N_proc>1)

# Compute flamelet and chemical equilibrium solutions within the specified 
# fraction ranges.
ComputeFlameletData(config, run_parallel=run_parallel, N_processors=N_proc)

# Compute additional equilibrium solutions throughout the mixture range
# for the physics-informed penalty terms.
ComputeBoundaryData(config,run_parallel,N_proc)
