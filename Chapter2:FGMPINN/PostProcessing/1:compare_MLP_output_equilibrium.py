import numpy as np 
import os 
import csv
import matplotlib.pyplot as plt 
from su2dataminer.config import Config_FGM 
from su2dataminer.manifold import Train_FGM_PINN

def FiniteDifferenceDerivative(y, x):
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
        if (dx_1==0) or (dx_2==0):
            dydx[i] = 0.0
        else:
            dydx[i] = (dx2_2 * y_p + (dx2_1 - dx2_2)*y_0 - dx2_1*y_m)/(dx_1*dx_2*(dx_1+dx_2))
    dx_1 = x[1] - x[0]
    dx_2 = x[2] - x[0]
    dx2_1 = dx_1*dx_1 
    dx2_2 = dx_2*dx_2 
    y_0 = y[0]
    y_p = y[1]
    y_pp = y[2]
    if (dx_1==0) or (dx_2==0):
        dydx[0] = 0.0
    else:
        dydx[0] = (dx2_1 * y_pp + (dx2_2 - dx2_1)*y_0 - dx2_2*y_p)/(dx_1*dx_2*(dx_1 - dx_2))

    dx_1 = x[-2] - x[-1]
    dx_2 = x[-3] - x[-1]
    dx2_1 = dx_1*dx_1 
    dx2_2 = dx_2*dx_2 
    y_0 = y[-1]
    y_p = y[-2]
    y_pp = y[-3]
    if (dx_1==0) or (dx_2==0):
        dydx[-1] = 0.0
    else:
        dydx[-1] = (dx2_1 * y_pp + (dx2_2 - dx2_1)*y_0 - dx2_2*y_p)/(dx_1*dx_2*(dx_1 - dx_2))
    return dydx 

N=4
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.cubehelix(np.linspace(0,1,N)))
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

color_ref = colors[0]
color_direct = colors[1]
color_PI = colors[2]


config_PINN = Config_FGM("../config_PIML.cfg")
config_DNN = Config_FGM("../config_DF.cfg")


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
            
            vars_out = vars_MLP.copy()
        else:
            outputs_PINN = np.hstack((outputs_PINN, MLP_output_PINN))
            outputs_DNN = np.hstack((outputs_DNN, MLP_output_DNN))
            vars_out += vars_MLP 
    return vars_out, outputs_PINN, outputs_DNN

fsize=25
T_unb_range = np.linspace(300, 860, 300)


phis = [0.2, 0.5, 1.0, 3.0, 5.0]

fig_1,axs1 = plt.subplots(ncols=3,nrows=1,figsize=[12,7])

fig_2,axs2 = plt.subplots(ncols=2,nrows=1,figsize=[10,7])

fig_3 = plt.figure(figsize=[10,6])
ax3 = plt.axes()

label_PINN = "PIML"
label_DNN="DF"
label_ref="Ref"

for p in phis:
    config_PINN.gas.set_equivalence_ratio(p, config_PINN.GetFuelString(), config_PINN.GetOxidizerString())
    val_Z = config_PINN.gas.mixture_fraction(config_PINN.GetFuelString(), config_PINN.GetOxidizerString())
    val_pv = config_PINN.ComputeProgressVariable(variables=None, flamelet_data=None, Y_flamelet=config_PINN.gas.Y[:,np.newaxis])[0]
    cvs_phi = np.zeros([len(T_unb_range), 3])
    beta_pv = 0
    beta_Z = sum(config_PINN.gas.Y*config_PINN.GetMixtureFractionCoefficients()/config_PINN.GetConstSpecieLewisNumbers())
    for wSp, Sp in zip(config_PINN.GetProgressVariableWeights(), config_PINN.GetProgressVariableSpecies()):
        iSp = config_PINN.gas.species_index(Sp)
        Le_sp = config_PINN.GetConstSpecieLewisNumbers()[iSp]
        beta_pv += wSp * config_PINN.gas.Y[iSp] / Le_sp 
    
    ref_vars = ["Cp", "Beta_ProgVar","Beta_MixFrac","MolarWeightMix","ProdRateTot_PV","Beta_h1"]
    ref_data = np.zeros([len(T_unb_range), len(ref_vars)])
    for i,T in enumerate(T_unb_range):
        config_PINN.gas.TP=T,101325
        cp_ref = config_PINN.gas.cp_mass
        cp_i = config_PINN.gas.partial_molar_cp / config_PINN.gas.molecular_weights
        cp_N2 = cp_i[config_PINN.gas.species_index("N2")]
        beta_h1 = cp_ref - np.sum((cp_i - cp_N2) * config_PINN.gas.Y / config_PINN.GetConstSpecieLewisNumbers())
        cvs_phi[i,0] = val_pv
        cvs_phi[i,1] = config_PINN.gas.enthalpy_mass 
        cvs_phi[i,2] = val_Z 
        ref_data[i, ref_vars.index("Cp")] = cp_ref 
        ref_data[i, ref_vars.index("Beta_ProgVar")] = beta_pv 
        ref_data[i, ref_vars.index("Beta_MixFrac")] = beta_Z 
        ref_data[i, ref_vars.index("MolarWeightMix")] = np.sum(config_PINN.gas.molecular_weights*config_PINN.gas.X)# config_PINN.gas.mean_molar_weights = np.dot(molar_weights.T, X)
        ref_data[i, ref_vars.index("ProdRateTot_PV")] = 0.0
        ref_data[i, ref_vars.index("Beta_h1")] = beta_h1 

    vars_MLP, MLP_output_PINN, MLP_output_DNN = EvaluateMLPs(cvs_phi)
    dTdh_ref = 1 / ref_data[:, ref_vars.index("Cp")]
    dTdh_PINN = FiniteDifferenceDerivative(MLP_output_PINN[:, vars_MLP.index("Temperature")], cvs_phi[:,1])
    dTdh_DNN = FiniteDifferenceDerivative(MLP_output_DNN[:, vars_MLP.index("Temperature")], cvs_phi[:,1])
    dBetah2_dh_PINN = FiniteDifferenceDerivative(MLP_output_PINN[:, vars_MLP.index("Beta_Enth")], cvs_phi[:,1])
    dBetah2_dh_DNN = FiniteDifferenceDerivative(MLP_output_DNN[:, vars_MLP.index("Beta_Enth")], cvs_phi[:,1])
    dBetah2_dh_ref = 1 - ref_data[:, ref_vars.index("Beta_h1")] / ref_data[:, ref_vars.index("Cp")]

    beta_pv_ref = ref_data[:, ref_vars.index("Beta_ProgVar")]
    beta_pv_PINN = MLP_output_PINN[:, vars_MLP.index("Beta_ProgVar")]
    beta_pv_DNN = MLP_output_DNN[:, vars_MLP.index("Beta_ProgVar")]
    
    beta_Z_ref = ref_data[:, ref_vars.index("Beta_MixFrac")]
    beta_Z_PINN = MLP_output_PINN[:, vars_MLP.index("Beta_MixFrac")]
    beta_Z_DNN = MLP_output_DNN[:, vars_MLP.index("Beta_MixFrac")]
    
    Wm_ref = ref_data[:, ref_vars.index("MolarWeightMix")]
    Wm_PINN = MLP_output_PINN[:, vars_MLP.index("MolarWeightMix")]
    Wm_DNN = MLP_output_DNN[:, vars_MLP.index("MolarWeightMix")]

    ppv_Ref = ref_data[:, ref_vars.index("ProdRateTot_PV")]
    ppv_PINN = MLP_output_PINN[:, vars_MLP.index("ProdRateTot_PV")]
    ppv_DNN = MLP_output_DNN[:, vars_MLP.index("ProdRateTot_PV")]
    

    axs1[1].plot(T_unb_range, beta_pv_ref,color=color_ref,linewidth=5.0,label=label_ref)
    axs1[1].plot(T_unb_range, beta_pv_DNN,color=color_direct,linestyle='--',linewidth=3.0,label=label_DNN)
    axs1[1].plot(T_unb_range, beta_pv_PINN,color=color_PI,linestyle='-.',linewidth=3.0,label=label_PINN)
    axs1[1].text(x=np.average(T_unb_range),y=np.average(beta_pv_ref),s=("phi=%.1f" % p),verticalalignment='bottom',horizontalalignment='center',fontsize=fsize)

    axs1[2].plot(T_unb_range, beta_Z_ref,color=color_ref,linewidth=5.0,label=label_ref)
    axs1[2].plot(T_unb_range, beta_Z_DNN,color=color_direct,linestyle='--',linewidth=3.0,label=label_DNN)
    axs1[2].plot(T_unb_range, beta_Z_PINN,color=color_PI,linestyle='-.',linewidth=3.0,label=label_PINN)
    axs1[2].text(x=np.average(T_unb_range),y=np.average(beta_Z_ref),s=("phi=%.1f" % p),verticalalignment='bottom',horizontalalignment='center',fontsize=fsize)

    axs1[0].plot(T_unb_range, Wm_ref,color=color_ref,linewidth=5.0,label=label_ref)
    axs1[0].plot(T_unb_range, Wm_DNN,color=color_direct,linestyle='--',linewidth=3.0,label=label_DNN)
    axs1[0].plot(T_unb_range, Wm_PINN,color=color_PI,linestyle='-.',linewidth=3.0,label=label_PINN)
    axs1[0].text(x=np.average(T_unb_range),y=np.average(Wm_ref),s=("phi=%.1f" % p),verticalalignment='bottom',horizontalalignment='center',fontsize=fsize)

    axs2[0].plot(T_unb_range, dTdh_ref,color=color_ref,linewidth=5.0,label=label_ref)
    axs2[0].plot(T_unb_range, dTdh_DNN,color=color_direct,linestyle='--',linewidth=3.0,label=label_DNN)
    axs2[0].plot(T_unb_range, dTdh_PINN,color=color_PI,linestyle='-.',linewidth=3.0,label=label_PINN)
    axs2[0].text(x=np.average(T_unb_range),y=np.average(dTdh_ref),s=("phi=%.1f" % p),verticalalignment='bottom',horizontalalignment='center',fontsize=fsize)

    axs2[1].plot(T_unb_range, dBetah2_dh_ref,color=color_ref,linewidth=5.0,label=label_ref)
    axs2[1].plot(T_unb_range, dBetah2_dh_DNN,color=color_direct,linestyle='--',linewidth=3.0,label=label_DNN)
    axs2[1].plot(T_unb_range, dBetah2_dh_PINN,color=color_PI,linestyle='-.',linewidth=3.0,label=label_PINN)
    axs2[1].text(x=np.average(T_unb_range),y=np.average(dBetah2_dh_ref),s=("phi=%.1f" % p),verticalalignment='bottom',horizontalalignment='center',fontsize=fsize)

    ax3.plot(T_unb_range, ppv_Ref,color=color_ref,linewidth=5.0,label=label_ref)
    ax3.plot(T_unb_range, ppv_DNN,color=color_direct,linestyle='--',linewidth=3.0,label=label_DNN)
    ax3.plot(T_unb_range, ppv_PINN,color=color_PI,linestyle='-.',linewidth=3.0,label=label_PINN)

    label_ref = None 
    label_DNN=None 
    label_PINN=None

axs1[1].set_ylabel("Beta_pv",fontsize=fsize)
axs1[0].set_xlabel("Temperature",fontsize=fsize)
axs1[2].set_ylabel("Beta_z",fontsize=fsize)
axs1[1].set_xlabel("Temperature",fontsize=fsize)
axs1[0].set_ylabel("WM",fontsize=fsize)
axs1[2].set_xlabel("Temperature",fontsize=fsize)
axs1[0].tick_params(which='both',labelsize=fsize)
axs1[1].tick_params(which='both',labelsize=fsize)
axs1[2].tick_params(which='both',labelsize=fsize)
axs1[0].grid()
axs1[1].grid()
axs1[2].grid()
axs1[1].legend(fontsize=24,ncols=3,loc='upper center', bbox_to_anchor=(0.5, -0.14))
axs1[0].set_title("WM",fontsize=fsize)
axs1[1].set_title("Beta_PV",fontsize=fsize)
axs1[2].set_title("Beta_Z",fontsize=fsize)

fig_1.savefig("Images/Bpv_WM_BZ_comparison.eps",format='eps',bbox_inches='tight')

axs2[0].set_ylabel("dTdh",fontsize=fsize)
axs2[1].set_ylabel("dBetah2dh",fontsize=fsize)
axs2[0].set_xlabel("Temperature",fontsize=fsize)
axs2[1].set_xlabel("Temperature",fontsize=fsize)
axs2[0].tick_params(which='both',labelsize=fsize)
axs2[1].tick_params(which='both',labelsize=fsize)
axs2[0].grid()
axs2[1].grid()
axs2[0].set_title("JTh",fontsize=fsize)
axs2[1].set_title("JBh2",fontsize=fsize)
axs2[0].legend(fontsize=24,ncols=3,loc='upper center', bbox_to_anchor=(1.0, -0.14))
fig_2.savefig("Images/JBh2_JT_comparison.eps",format='eps',bbox_inches='tight')

ax3.set_ylabel("ppv",fontsize=fsize)
ax3.set_xlabel("Temperature",fontsize=fsize)
ax3.tick_params(which='both',labelsize=fsize)
ax3.grid()
ax3.legend(fontsize=24,ncols=3,loc='upper center', bbox_to_anchor=(0.5, -0.14))
fig_3.savefig("Images/PPV_comparison.eps",format='eps',bbox_inches='tight')
plt.show()