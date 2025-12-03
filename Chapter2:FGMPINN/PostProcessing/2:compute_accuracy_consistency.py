import numpy as np 
import os 
import csv
import matplotlib.pyplot as plt 
from su2dataminer.config import Config_FGM 
from su2dataminer.manifold import Train_FGM_PINN

config_PINN = Config_FGM("../config_PIML.cfg")
config_DNN = Config_FGM("../config_DF.cfg")
scale_enth = config_PINN._scaler_function_vals_in[0][1][1]
scale_z = config_PINN._scaler_function_vals_in[0][2][1]

def RMSE(y_true, y_pred):
    return np.sqrt(np.average(np.power(y_pred - y_true, 2)))

def EvaluateMLPs(cv_vals:np.ndarray[float]):
    for iGroup in range(6):
        MLP_PINN = Train_FGM_PINN(config_PINN, iGroup)
        w,b = config_PINN.GetWeightsBiases(iGroup)

        MLP_PINN.SetAlphaExpo(config_PINN.GetAlphaExpo(iGroup))
        MLP_PINN.SetLRDecay(config_PINN.GetLRDecay(iGroup))
        MLP_PINN.SetBatchExpo(config_PINN.GetBatchExpo(iGroup))
        MLP_PINN.SetHiddenLayers(config_PINN.GetHiddenLayerArchitecture(iGroup))
        MLP_PINN.SetWeightsBiases(w, b)
        MLP_PINN.InitializeWeights_and_Biases()
        MLP_PINN.SetActivationFunction(config_PINN.GetActivationFunction(iGroup))
        MLP_PINN.SetTrainVariables(config_PINN.GetMLPOutputGroup(iGroup))
        MLP_PINN.SetTrainFileHeader(config_PINN.GetOutputDir()+"/"+config_PINN.GetConcatenationFileHeader())
        MLP_PINN.GetTrainData()
        MLP_PINN.SetSaveDir(os.getcwd())
        MLP_PINN.SetModelIndex(100)

        MLP_PINN.Preprocessing()
        scale_out = config_PINN._scaler_function_vals_out[iGroup][0][1]
        MLP_output_PINN = MLP_PINN.EvaluateMLP(cv_vals)

        MLP_DNN = Train_FGM_PINN(config_DNN, iGroup)
        w,b = config_DNN.GetWeightsBiases(iGroup)
        
        MLP_DNN.SetAlphaExpo(config_DNN.GetAlphaExpo(iGroup))
        MLP_DNN.SetLRDecay(config_DNN.GetLRDecay(iGroup))
        MLP_DNN.SetBatchExpo(config_DNN.GetBatchExpo(iGroup))
        MLP_DNN.SetHiddenLayers(config_DNN.GetHiddenLayerArchitecture(iGroup))
        MLP_DNN.SetWeightsBiases(w, b)
        MLP_DNN.InitializeWeights_and_Biases()
        MLP_DNN.SetActivationFunction(config_DNN.GetActivationFunction(iGroup))
        MLP_DNN.SetTrainVariables(config_DNN.GetMLPOutputGroup(iGroup))
        MLP_DNN.SetTrainFileHeader(config_DNN.GetOutputDir()+"/"+config_DNN.GetConcatenationFileHeader())
        MLP_DNN.GetTrainData()
        MLP_DNN.SetSaveDir(os.getcwd())
        MLP_DNN.SetModelIndex(100)

        MLP_DNN.Preprocessing()
        MLP_output_DNN = MLP_DNN.EvaluateMLP(cv_vals)

        vars_MLP = config_PINN.GetMLPOutputGroup(iGroup)
        if iGroup==0:
            outputs_PINN = MLP_output_PINN
            outputs_DNN = MLP_output_DNN
            scale_vars_out = [scale_out]
            vars_out = vars_MLP.copy()
        else:
            outputs_PINN = np.hstack((outputs_PINN, MLP_output_PINN))
            outputs_DNN = np.hstack((outputs_DNN, MLP_output_DNN))
            vars_out += vars_MLP 
            scale_vars_out.append(scale_out)
    return vars_out, outputs_PINN, outputs_DNN, scale_vars_out

gas = config_PINN.gas

Np_T = 400
Np_phi = 200

T_range = np.linspace(config_PINN.GetUnbTempBounds()[0], config_PINN.GetUnbTempBounds()[1],Np_T)

phi_vals = np.linspace(0.2, 5.0, Np_phi)

eq_data = np.zeros([len(T_range), len(phi_vals), 11])

for iPhi, phi in enumerate(phi_vals):
    gas.set_equivalence_ratio(phi, config_PINN.GetFuelString(), config_PINN.GetOxidizerString())
    
    for iT, T in enumerate(T_range):
        gas.TP = T, 101325
        pv = config_PINN.ComputeProgressVariable(variables=None, flamelet_data=None, Y_flamelet=gas.Y[:,np.newaxis])[0]
        h = gas.enthalpy_mass
        z = gas.mixture_fraction(config_PINN.GetFuelString(), config_PINN.GetOxidizerString())
        cp = gas.cp_mass 

        beta_pv = 0
        beta_h1 = cp 
        beta_h2 = 0 
        beta_z = 0
        cp_i = gas.partial_molar_cp.T/gas.molecular_weights
        h_i = gas.partial_molar_enthalpies.T/gas.molecular_weights
        cp_c = cp_i[gas.species_index("N2")]
        h_c = h_i[gas.species_index("N2")]
        z_c = config_PINN.GetMixtureFractionCoeff_Carrier()
        for pvSp, pvW in zip(config_PINN.GetProgressVariableSpecies(), config_PINN.GetProgressVariableWeights()):
            beta_pv += pvW * gas.Y[gas.species_index(pvSp)] / config_PINN.GetConstSpecieLewisNumbers()[gas.species_index(pvSp)]
        
        for iSp in range(gas.n_species):
            beta_z += (config_PINN.GetMixtureFractionCoefficients()[iSp] - z_c)* gas.Y[iSp]/config_PINN.GetConstSpecieLewisNumbers()[iSp]
            beta_h1 -= (cp_i[iSp] - cp_c)*gas.Y[iSp] / config_PINN.GetConstSpecieLewisNumbers()[iSp]
            beta_h2 += (h_i[iSp] - h_c)*gas.Y[iSp] / config_PINN.GetConstSpecieLewisNumbers()[iSp]
        
        X = gas.X 
        mean_molar_weight = np.dot(gas.molecular_weights, X)
        eq_data[iT, iPhi, :] = np.array([pv, h, z, T, cp, beta_pv,beta_h1,beta_h2,beta_z,mean_molar_weight,0.0])
vars_ref = ["ProgressVariable","EnthalpyTot","MixtureFraction","Temperature","Cp","Beta_ProgVar","Beta_Enth_Thermal","Beta_Enth","Beta_MixFrac","MolarWeightMix", "ProdRateTot_PV"]

eq_data_flat = np.vstack(tuple(e for e in eq_data))
vars_MLP, output_PINN_flat, output_DNN_flat,scale_fac_out = EvaluateMLPs(eq_data_flat[:, :3])
output_PINN = np.reshape(output_PINN_flat, [eq_data.shape[0], eq_data.shape[1], len(vars_MLP)])
output_DNN = np.reshape(output_DNN_flat, [eq_data.shape[0], eq_data.shape[1], len(vars_MLP)])


flamelet_domain_data_file = config_DNN.GetOutputDir()+"/flamelet_domain_data_full.csv"
with open(flamelet_domain_data_file,'r') as fid:
    vars_domain_data = fid.readline().strip().split(',')
flamelet_domain_data = np.loadtxt(flamelet_domain_data_file,delimiter=',',skiprows=1)
_, output_PINN_flamelet_domain, output_DNN_flamelet_domain, _ = EvaluateMLPs(flamelet_domain_data[:, :3])

vals_pv = eq_data[:,:,0]
vals_h = eq_data[:,:,1]
vals_z = eq_data[:,:,2]
T_ref = eq_data[:,:, 3]
T_PINN = output_PINN[:,:,vars_MLP.index("Temperature")]
T_DNN = output_DNN[:,:,vars_MLP.index("Temperature")]




# Calculate errors for mean molecular weight
def ComparePredJacobians(var_out, ref_jacobian:np.ndarray[float]):
    y_out_ref = eq_data[:,:, vars_ref.index(var_out)]
    y_out_PIML = output_PINN[:,:,vars_MLP.index(var_out)]
    y_out_jacobian_h_PIML = np.gradient(y_out_PIML)[0]/np.gradient(vals_h)[0]
    y_out_DF = output_DNN[:,:,vars_MLP.index(var_out)]
    y_out_jacobian_h_DF = np.gradient(y_out_DF)[0]/np.gradient(vals_h)[0]

    y_out_jacobian_h_ref = ref_jacobian

    scale_var = scale_fac_out[vars_MLP.index(var_out)]
    scale_fac_y_out = scale_var/(scale_enth)

    rmse_jacobian_y_out_PIML = RMSE(y_out_jacobian_h_ref, y_out_jacobian_h_PIML) / scale_fac_y_out
    rmse_jacobian_y_out_DF = RMSE(y_out_jacobian_h_ref, y_out_jacobian_h_DF) / scale_fac_y_out

    rmse_pred_y_out_PIML = RMSE(y_out_PIML, y_out_ref) / scale_var
    rmse_pred_y_out_DF = RMSE(y_out_DF, y_out_ref) / scale_var

    y_out_domain_ref = flamelet_domain_data[:, vars_domain_data.index(var_out)]
    y_out_domain_PIML = output_PINN_flamelet_domain[:, vars_MLP.index(var_out)]
    y_out_domain_DF = output_DNN_flamelet_domain[:, vars_MLP.index(var_out)]
    rmse_pred_domain_y_out_PIML = RMSE(y_out_domain_ref, y_out_domain_PIML) / scale_var
    rmse_pred_domain_y_out_DF = RMSE(y_out_domain_ref, y_out_domain_DF) / scale_var

    print("RMSE output %s PIML (equilibrium): %.5e" % (var_out, rmse_pred_y_out_PIML))
    print("RMSE output %s DF (equilibrium): %.5e" % (var_out, rmse_pred_y_out_DF))
    print("RMSE output %s PIML (non-equilibrium): %.5e" % (var_out, rmse_pred_domain_y_out_PIML))
    print("RMSE output %s DF (non-equilibrium): %.5e" % (var_out, rmse_pred_domain_y_out_DF))

    print("RMSE jacobian %s PIML: %.5e" % (var_out, rmse_jacobian_y_out_PIML))
    print("RMSE jacobian %s DF: %.5e" % (var_out, rmse_jacobian_y_out_DF))
    print()
    return rmse_pred_y_out_PIML, rmse_pred_y_out_DF, rmse_jacobian_y_out_PIML, rmse_jacobian_y_out_DF, rmse_pred_domain_y_out_PIML, rmse_pred_domain_y_out_DF

var_out = "MolarWeightMix"
WM_jacobian_h_ref = np.zeros([Np_T, Np_phi])
rmse_pred_WM_PIML, rmse_pred_WM_DF, rmse_J_WM_PIML, rmse_J_WM_DF, rmse_WM_domain_PIML, rmse_WM_domain_DF = ComparePredJacobians(var_out, WM_jacobian_h_ref)

# Calculate errors for Beta pv:
var_out = "Beta_ProgVar"
Beta_pv_jacobian_h_ref = np.zeros([Np_T,Np_phi])
rmse_pred_Beta_pv_PIML, rmse_pred_Beta_pv_DF, rmse_J_Beta_pv_PIML, rmse_J_Beta_pv_DF, rmse_Beta_pv_domain_PIML, rmse_Beta_pv_domain_DF = ComparePredJacobians(var_out, Beta_pv_jacobian_h_ref)


# Calculate errors for Beta Z:
var_out = "Beta_MixFrac"
Beta_z_jacobian_h_ref = np.zeros([Np_T,Np_phi])
rmse_pred_Beta_z_PIML, rmse_pred_Beta_z_DF, rmse_J_Beta_z_PIML, rmse_J_Beta_z_DF, rmse_Beta_z_domain_PIML, rmse_Beta_z_domain_DF = ComparePredJacobians(var_out, Beta_z_jacobian_h_ref)

# Calculate errors for Temperature:
var_out = "Temperature"
T_ref = eq_data[:,:, vars_ref.index(var_out)]
Cp_ref = eq_data[:,:,vars_ref.index("Cp")]
T_jacobian_h_ref = 1/Cp_ref 
rmse_pred_T_PIML, rmse_pred_T_DF, rmse_J_T_PIML, rmse_J_T_DF, rmse_T_domain_PIML, rmse_T_domain_DF = ComparePredJacobians(var_out, T_jacobian_h_ref)


# Calculate errors for Beta_h2:
var_out = "Beta_Enth"
Beta_h1_ref = eq_data[:,:, vars_ref.index("Beta_Enth_Thermal")]
Beta_h2_jacobian_h_ref = 1 - Beta_h1_ref / Cp_ref
rmse_pred_Beta_h2_PIML, rmse_pred_Beta_h2_DF, rmse_J_Beta_h2_PIML, rmse_J_Beta_h2_DF, rmse_Beta_h2_domain_PIML, rmse_Beta_h2_domain_DF = ComparePredJacobians(var_out, Beta_h2_jacobian_h_ref)

# Calculate errors for source pv:
var_out = "ProdRateTot_PV"
ppv_ref = eq_data[:, :,vars_ref.index(var_out)]
ppv_PIML = output_PINN[:,:,vars_MLP.index(var_out)]
ppv_DF = output_DNN[:,:,vars_MLP.index(var_out)]
z_max, z_min = np.max(vals_z), np.min(vals_z)
pv_max, pv_min = np.max(vals_pv), np.min(vals_pv)
scale_var = scale_fac_out[vars_MLP.index(var_out)]

ppv_Jacobian_h_PIML = np.gradient(ppv_PIML)[0]/np.gradient(vals_h)[0]
ppv_Jacobian_h_DF = np.gradient(ppv_DF)[0]/np.gradient(vals_h)[0]
ppv_Jacobian_z_PIML = np.gradient(ppv_PIML)[1]/np.gradient(vals_z)[1]
ppv_Jacobian_z_DF = np.gradient(ppv_DF)[1]/np.gradient(vals_z)[1]

ppv_Jacobian_PIML = np.abs(ppv_Jacobian_h_PIML  * (scale_enth / scale_var)) + np.abs(ppv_Jacobian_z_PIML * (scale_z / scale_var))
ppv_Jacobian_DF = np.abs(ppv_Jacobian_h_DF  * (scale_enth / scale_var)) + np.abs(ppv_Jacobian_z_DF * (scale_z / scale_var))
ppv_Jacobian_ref = np.zeros([Np_T, Np_phi])

rmse_jacobian_ppv_PIML = RMSE(ppv_Jacobian_ref, ppv_Jacobian_PIML)
rmse_jacobian_ppv_DF = RMSE(ppv_Jacobian_ref, ppv_Jacobian_DF)
rmse_pred_ppv_PIML = RMSE(ppv_ref, ppv_PIML) / scale_var 
rmse_pred_ppv_DF = RMSE(ppv_ref, ppv_DF) / scale_var 

ppv_domain_ref = flamelet_domain_data[:, vars_domain_data.index(var_out)]
ppv_domain_PIML = output_PINN_flamelet_domain[:, vars_MLP.index(var_out)]
ppv_domain_DF = output_DNN_flamelet_domain[:, vars_MLP.index(var_out)]
rmse_pred_domain_ppv_PIML = RMSE(ppv_domain_ref, ppv_domain_PIML) / scale_var
rmse_pred_domain_ppv_DF = RMSE(ppv_domain_ref, ppv_domain_DF) / scale_var

print("RMSE output %s PIML (equilibrium): %.5e" % (var_out,rmse_pred_ppv_PIML))
print("RMSE output %s DF (equilibrium): %.5e" % (var_out, rmse_pred_ppv_DF))
print("RMSE output %s PIML (non-equilibrium): %.5e" % (var_out,rmse_pred_domain_ppv_PIML))
print("RMSE output %s DF (non-equilibrium): %.5e" % (var_out, rmse_pred_domain_ppv_DF))

print("RMSE jacobian %s PIML: %.5e" % (var_out, rmse_jacobian_ppv_PIML))
print("RMSE jacobian %s DF: %.5e" % (var_out, rmse_jacobian_ppv_DF))
print()

# Plot results
vars_to_compare = [r"$\dot{\omega}_c$", r"$T$", r"$\beta_c$", r"$\beta_{h_t,2}$",r"$\beta_Z$",r"$W_M$"]
rms_direct_flamelet_list = [rmse_pred_domain_ppv_DF, rmse_T_domain_DF, rmse_Beta_pv_domain_DF, rmse_Beta_h2_domain_DF, rmse_Beta_z_domain_DF, rmse_WM_domain_DF]
rms_PIML_flamelet_list = [rmse_pred_domain_ppv_PIML, rmse_T_domain_PIML, rmse_Beta_pv_domain_PIML, rmse_Beta_h2_domain_PIML, rmse_Beta_z_domain_PIML, rmse_WM_domain_PIML]
rms_direct_boundary_list = [rmse_pred_ppv_DF, rmse_pred_T_DF, rmse_pred_Beta_pv_DF, rmse_pred_Beta_h2_DF, rmse_pred_Beta_z_DF, rmse_pred_WM_DF]
rms_PIML_boundary_list = [rmse_pred_ppv_PIML, rmse_pred_T_PIML, rmse_pred_Beta_pv_PIML, rmse_pred_Beta_h2_PIML, rmse_pred_Beta_z_PIML, rmse_pred_WM_PIML]
Jacobian_error_direct = [rmse_jacobian_ppv_DF, rmse_J_T_DF,rmse_J_Beta_pv_DF, rmse_J_Beta_h2_DF, rmse_J_Beta_z_DF, rmse_J_WM_DF]
Jacobian_error_PIML = [rmse_jacobian_ppv_PIML, rmse_J_T_PIML,rmse_J_Beta_pv_PIML, rmse_J_Beta_h2_PIML, rmse_J_Beta_z_PIML, rmse_J_WM_PIML]

N=4
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.cubehelix(np.linspace(0,1,N)))
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_ref = colors[0]
color_direct = colors[1]
color_PI = colors[2]

fig_format = 'eps'
x_range = np.array([s for s in range(len(vars_to_compare))])
fig = plt.figure(figsize=[10,7])
ax = plt.axes()
ax.bar(x=x_range-0.125,height=rms_direct_flamelet_list,width=0.25,color=color_direct,label="DF",zorder=3)
ax.bar(x=x_range+0.125,height=rms_PIML_flamelet_list,width=0.25,color=color_PI,label="PIML",zorder=3)
ax.set_xticks(x_range)
ax.set_xticklabels(vars_to_compare)
ax.set_ylim([1e-6, 1e0])
ax.set_yscale('log')
ax.set_ylabel("Validation set loss (Lval)",fontsize=20)
ax.set_title("Training method validation loss comparison for domain data",fontsize=20)
ax.tick_params(which='both',labelsize=20)
ax.grid(which='both')
ax.legend(fontsize=20,ncol=2,bbox_to_anchor=(0.5, -0.12),loc='upper center',fancybox=True,shadow=True)
fig.savefig("Images/rms_bar_chart_flamelet."+fig_format,format=fig_format,bbox_inches='tight')
plt.show()

fig = plt.figure(figsize=[10,7])
ax = plt.axes()
ax.bar(x=x_range-0.25,height=rms_direct_boundary_list,width=0.167,color=color_direct,label="DF,Ly",zorder=3)
ax.bar(x=x_range-0.083333, height=Jacobian_error_direct,width=0.167, color=color_direct,hatch="/",edgecolor='black',label="DF, LJ",zorder=3)
ax.bar(x=x_range+0.083333,height=rms_PIML_boundary_list,width=0.167,color=color_PI,label="PIML, Ly",zorder=3)
ax.bar(x=x_range+0.25,height=Jacobian_error_PIML,width=0.167,color=color_PI,hatch="/",edgecolor='black',label="PIML, LJ",zorder=3)
ax.set_xticks(x_range)
ax.set_xticklabels([])
ax.set_ylim([1e-6,1e0])
ax.set_yticks([1e-5,1e-4,1e-3,1e-2, 1e-1])
ax.set_yscale('log')
ax.set_yticklabels([105,104,103,102,101])
ax.set_ylabel("Validation set loss (Lval)",fontsize=20)
ax.set_title("Training method validation loss comparison for boundary data",fontsize=20)
ax.tick_params(which='both',labelsize=20)
ax.grid(which='both')
ax.legend(fontsize=20,ncol=2,bbox_to_anchor=(0.5, -0.12),loc='upper center',fancybox=True,shadow=True)
fig.savefig("Images/rms_bar_chart_boundary_Jacobians.pdf",format='pdf',bbox_inches='tight')
plt.show()