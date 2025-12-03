# Generate SU2 DataMiner configurations for comparing the accuracy and consistency 
# between using data-fitting and physics-informed training.

import os 
from su2dataminer.config import Config_FGM

# Equivalence ratio range: 0.25-20.0
phi_min = 0.25 
phi_max = 20.0
Np_mix = 40 

# Reactant temperature range: 300K-860K
Tu_min = 300.0
Tu_max = 860.0
Np_temp = 30

# Common settings for manifold
Config = Config_FGM()
Config.SetOutputDir(os.getcwd() + "/flamelet_data/")
# # Reactants: hydrogen-air
Config.SetFuelDefinition(fuel_species=["H2"],fuel_weights=[1.0])

# # Use Zhang syngas-nox mechanism
Config.SetReactionMechanism('stanford.yaml')

# # Equivalence ratio between 0.25 and 20
Config.SetMixtureBounds(phi_min, phi_max)
Config.SetNpMix(Np_mix)

# # Reactant temperature between 300 and 860K
Config.SetUnbTempBounds(Tu_min, Tu_max)
Config.SetNpTemp(Np_temp)

# Consider adiabatic flamelets, burner-stabilized flamelets, and chemical equilibrium data
Config.RunBurnerFlames(True)
Config.RunFreeFlames(True)
Config.RunEquilibrium(True)

# Multi-component transport model for preferential diffusion.
Config.SetTransportModel('multicomponent')

# Set average Lewis numbers at equivalence ratio 0.5 and reactant temperature 300 K.
Config.SetAverageLewisNumbers(0.5, 300)

# No passive look-ups and species
Config.SetPassiveSpecies([])
Config.SetLookUpVariables([])

# Define progress variable
pv_sp = ["H2","H","O","O2","OH","H2O","HO2","H2O2"]
pv_w = [-2.70893555e+00, -8.53623742e+00, -3.22050644e-01, -7.83324996e-01,-1.88069610e+00,  2.92465664e+00,  2.30540678e+00, -6.35595883e-03]
Config.SetProgressVariableDefinition(pv_sp, pv_w)

# MLP outputs are set according to physics-informed quantities:
# Temperature, mean molecular weight, progress variable source term,
# and the preferential diffusion scalars for progress variable, mixture 
# fraction, and specific enthalpy.
Config.ClearOutputGroups()
Config.AddOutputGroup(["Temperature"])
Config.AddOutputGroup(["Beta_ProgVar"])
Config.AddOutputGroup(["Beta_MixFrac"])
Config.AddOutputGroup(["Beta_Enth"])
Config.AddOutputGroup(["MolarWeightMix"])
Config.AddOutputGroup(["ProdRateTot_PV"])
Config.AddOutputGroup(["Cp"])
Config.AddOutputGroup(["ViscosityDyn"])
Config.AddOutputGroup(["Conductivity"])
Config.AddOutputGroup(["DiffusionCoefficient"])
Config.AddOutputGroup(["Beta_Enth_Thermal"])

Config.SetConfigName("config_PIML")
Config.SetConcatenationFileHeader("flamelet_data")

Config.PrintBanner()

Config.SaveConfig()
