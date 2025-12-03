import numpy as np 
import matplotlib.pyplot as plt 
from su2dataminer.config import Config_FGM

Config = Config_FGM("../config_PIML.cfg")
N=3
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.cubehelix(np.linspace(0,1,N+1)))
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def FiniteDifferenceDerivative(y:np.ndarray[float],x:np.ndarray[float]):
    """Second-order accurate finite-difference approximation.
    """
    if y.shape != x.shape:
        raise Exception("x and y must be the same shape.")
    Np = len(x)
    dydx = np.zeros(Np)
    for i in range(1, Np-1):
        y_m = y[i-1]
        y_p = y[i+1]
        y_0 = y[i]
        x_m = x[i-1]
        x_p = x[i+1]
        x_0 = x[i]
        dx_1 = x_p - x_0 
        dx_2 = x_0 - x_m 
        dx2_1 = dx_1*dx_1 
        dx2_2 = dx_2*dx_2
        dydx[i] = (dx2_2 * y_p + (dx2_1 - dx2_2)*y_0 - dx2_1*y_m)/(dx_1*dx_2*(dx_1+dx_2))
    dx_1 = x[1] - x[0]
    dx_2 = x[2] - x[0]
    dx2_1 = dx_1*dx_1 
    dx2_2 = dx_2*dx_2 
    y_0 = y[0]
    y_p = y[1]
    y_pp = y[2]
    dydx[0] = (dx2_1 * y_pp + (dx2_2 - dx2_1)*y_0 - dx2_2*y_p)/(dx_1*dx_2*(dx_1 - dx_2))

    dx_1 = x[-2] - x[-1]
    dx_2 = x[-3] - x[-1]
    dx2_1 = dx_1*dx_1 
    dx2_2 = dx_2*dx_2 
    y_0 = y[-1]
    y_p = y[-2]
    y_pp = y[-3]
    dydx[-1] = (dx2_1 * y_pp + (dx2_2 - dx2_1)*y_0 - dx2_2*y_p)/(dx_1*dx_2*(dx_1 - dx_2))
    return dydx 

# Evaluate the verification error of the consistency relations for 200 mixtures.
Np = 200
mixfrac_range = np.linspace(0, 1, Np)
q = 0
error_dTdh = 0.0
error_dbetah2dh = 0.0

# For each mixture, evaluate the verification error for the enthalpy consistency relations.
for z in mixfrac_range:
    T_range = np.linspace(300, 800, Np)

    vals_enth = np.zeros(Np)
    vals_cp = np.zeros(Np)
    vals_beta_pv = np.zeros(Np)
    vals_beta_h1 = np.zeros(Np)
    vals_beta_h2 = np.zeros(Np)
    vals_beta_z = np.zeros(Np)
    for iT, T in enumerate(T_range):

        # Set the mixture fraction and temperature of the reactants.
        Config.gas.set_mixture_fraction(z, Config.GetFuelString(), Config.GetOxidizerString())
        Config.gas.TP=T,101325

        # Retrieve thermochemical state and calculate preferential diffusion scalars.
        Y_gas = Config.gas.Y[:,np.newaxis]
        vals_enth[iT] = Config.gas.enthalpy_mass 
        vals_cp[iT] = Config.gas.cp_mass 
        z_i = Config.GetMixtureFractionCoefficients()
        z_c = Config.GetMixtureFractionConstant()
        Le_sp = Config.GetConstSpecieLewisNumbers()
        pv_w, pv_sp = Config.GetProgressVariableWeights(), Config.GetProgressVariableSpecies()
        cp = Config.gas.cp_mass 
        cp_i = Config.gas.partial_molar_cp/Config.gas.molecular_weights
        cp_c = cp_i[Config.gas.species_index("N2")]
        h_i = Config.gas.partial_molar_enthalpies/Config.gas.molecular_weights
        h_c = h_i[Config.gas.species_index("N2")]

        beta_z = 0
        beta_pv = 0
        beta_h1 = cp
        beta_h2 = 0
        for iSp in range(Config.gas.n_species):
            beta_z += (z_i[iSp] - z_c)*Y_gas[iSp]/Le_sp[iSp]
            beta_h1 -= (cp_i[iSp] - cp_c)*Y_gas[iSp]/Le_sp[iSp]
            beta_h2 += (h_i[iSp] - h_c)*Y_gas[iSp]/Le_sp[iSp]
        for sp, w in zip(pv_sp, pv_w):
            beta_pv += w*Y_gas[Config.gas.species_index(sp)]/Le_sp[Config.gas.species_index(sp)]
        vals_beta_pv[iT] = beta_pv[0] 
        vals_beta_h1[iT] = beta_h1[0] 
        vals_beta_h2[iT] = beta_h2[0] 
        vals_beta_z[iT] = beta_z[0]

    # Calculate temperature-enthalpy derivative according to finite-differences.
    dTdh_FD = FiniteDifferenceDerivative(T_range, vals_enth)

    # Temperature-enthalpy derivative according to consistency relation.
    dTdh_A = 1 / vals_cp

    # Update mean square error value
    error_dTdh += np.average(np.power(dTdh_A - dTdh_FD,2))

    # Calculate beta_h2-enthalpy derivative according to finite-differneces.
    dbetah2_dh_FD = FiniteDifferenceDerivative(vals_beta_h2, vals_enth)

    # Beta_h2-enthalpy derivative according to consistency relation.
    dbetah2_dh_A = 1 - vals_beta_h1 / vals_cp 

    # Update mean square error value
    error_dbetah2dh += np.average(np.power(dbetah2_dh_A - dbetah2_dh_FD,2))
    q += 1 

# Calculate progress variable-mixutre fraction derivative according to consistency relation.
Config.gas.set_mixture_fraction(0, Config.GetFuelString(), Config.GetOxidizerString())
Y_ox = Config.gas.Y
beta_z_ox = 0
beta_pv_ox = 0
for iSp in range(Config.gas.n_species):
    beta_z_ox += (z_i[iSp] - z_c)*Y_ox[iSp]/Le_sp[iSp]
pv_ox = 0
for sp, w in zip(pv_sp, pv_w):
    beta_pv_ox += w*Y_ox[Config.gas.species_index(sp)]/Le_sp[Config.gas.species_index(sp)]
    pv_ox += w*Y_ox[Config.gas.species_index(sp)]

Config.gas.set_mixture_fraction(1, Config.GetFuelString(), Config.GetOxidizerString())
Y_fuel = Config.gas.Y
beta_z_fuel = 0
beta_pv_fuel = 0
pv_fuel =0
for iSp in range(Config.gas.n_species):
    beta_z_fuel += (z_i[iSp] - z_c)*Y_fuel[iSp]/Le_sp[iSp]
for sp, w in zip(pv_sp, pv_w):
    beta_pv_fuel += w*Y_fuel[Config.gas.species_index(sp)]/Le_sp[Config.gas.species_index(sp)]
    pv_fuel += w*Y_fuel[Config.gas.species_index(sp)]
dbetapvdZ_A = (beta_pv_fuel - beta_pv_ox)
dbetazdZ_A = (beta_z_fuel - beta_z_ox)
dpvdZ_A = (pv_fuel - pv_ox)

val_enth = 0.0
vals_beta_pv = np.zeros(Np)
vals_beta_z = np.zeros(Np)
vals_pv = np.zeros(Np)
for iz, z in enumerate(mixfrac_range):
    
    Config.gas.set_mixture_fraction(z, Config.GetFuelString(), Config.GetOxidizerString())
    Config.gas.HP=val_enth,101325
    Y_gas = Config.gas.Y[:,np.newaxis]
    z_i = Config.GetMixtureFractionCoefficients()
    z_c = Config.GetMixtureFractionConstant()
    Le_sp = Config.GetConstSpecieLewisNumbers()
    pv_w, pv_sp = Config.GetProgressVariableWeights(), Config.GetProgressVariableSpecies()
    cp = Config.gas.cp_mass 
    cp_i = Config.gas.partial_molar_cp/Config.gas.molecular_weights
    cp_c = cp_i[Config.gas.species_index("N2")]
    h_i = Config.gas.partial_molar_enthalpies/Config.gas.molecular_weights
    h_c = h_i[Config.gas.species_index("N2")]

    beta_z = 0
    beta_pv = 0
    for iSp in range(Config.gas.n_species):
        beta_z += (z_i[iSp] - z_c)*Y_gas[iSp]/Le_sp[iSp]
    for sp, w in zip(pv_sp, pv_w):
        beta_pv += w*Y_gas[Config.gas.species_index(sp)]/Le_sp[Config.gas.species_index(sp)]
    vals_beta_pv[iz] = beta_pv[0] 
    vals_beta_z[iz] = beta_z[0]
    vals_pv[iz] =  Config.ComputeProgressVariable(variables=None,flamelet_data=None,Y_flamelet=Y_gas)[0]


dbetapvdZ_FD = FiniteDifferenceDerivative(vals_beta_pv, mixfrac_range)
error_dbetapvdZ = np.average(np.power(dbetapvdZ_A - dbetapvdZ_FD,2))

dbetazdZ_FD = FiniteDifferenceDerivative(vals_beta_z, mixfrac_range)
error_dbetazdZ = np.average(np.power(dbetazdZ_A - dbetazdZ_FD,2))

dpvdZ_FD = FiniteDifferenceDerivative(vals_pv, mixfrac_range)
error_dpvdZ = np.average(np.power(dpvdZ_A - dpvdZ_FD, 2))

print("Beta_h2-h: %.3f" % (np.log10(error_dbetah2dh/q)))
print("T - h: %.3f" % (np.log10(error_dTdh/q)))
print("progressvariable - mixture fraction: %.3f" % (np.log10(error_dpvdZ)))
print("Beta_pv - Z: %.3f" % (np.log10(error_dbetapvdZ)))
print("Beta_Z - Z: %.3f" % (np.log10(error_dbetazdZ)))
