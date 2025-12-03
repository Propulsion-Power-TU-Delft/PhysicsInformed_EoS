import os
import numpy as np
import matplotlib.pyplot as plt 
from scipy.spatial import cKDTree as KDTree
from su2dataminer.config import Config_FGM 
from su2dataminer.manifold import Train_FGM_PINN

config_WRP = Config_FGM("../WRP.cfg")
config_PCA = Config_FGM("../PCA.cfg")
config_SPV = Config_FGM("../OPT.cfg")
N=3
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.cubehelix(np.linspace(0,1,N+1)))
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def transform_phi(phi_dim:np.ndarray[float]):
    phi_norm_all = np.zeros(phi_dim.shape)
    phi_norm_all[phi_dim < 1.0] = 0.5*(phi_dim[phi_dim < 1.0])
    phi_norm_all[phi_dim >= 1.0] = 0.5*((phi_dim[phi_dim >= 1.0]-1.0)/(20.0 - 1)) + 0.5
    return phi_norm_all 
def inv_transform_phi(phi_norm:np.ndarray[float]):
    phi_dim = np.zeros(phi_norm.shape)
    phi_dim[phi_norm >= 0.5] = (phi_norm[phi_norm >= 0.5] - 0.5) * (20.0 - 1)/0.5 + 1.0
    phi_dim[phi_norm < 0.5] = phi_norm[phi_norm < 0.5] / 0.5

    return phi_dim 

marker_frq = 0.08
fsize = 20
color_ref = 'm'
marker_ref = 's'
msize_ref = 14
wline_ref = 4
mcolor_ref = 'none'

color_default = colors[0]
marker_default = 's'
msize_default = 12
wline_default = 3
mcolor_default="none"

color_pca = colors[1]
marker_pca = '^'
msize_pca = 11
wline_pca = 3
mcolor_pca="none"


color_optim = colors[2]
marker_optim = 'o'
msize_optim = 9
wline_optim = 2
mcolor_optim=color_optim

configs = [config_WRP, config_PCA, config_SPV]
markers = [marker_default, marker_pca, marker_optim]
msizes = [msize_default,msize_pca,msize_optim]
colors = [color_default,color_pca,color_optim]
wlines = [wline_default,wline_pca,wline_optim]
labels = ["WRP","PCA","OPT"]
flamelet_dir = config_WRP.GetOutputDir()
phis = os.listdir(flamelet_dir + "freeflame_data/")

flamelets_to_plot = ["/phi_0.328947/freeflamelet_phi0.328947_Tu300.0.csv","/phi_1.0/freeflamelet_phi1.0_Tu300.0.csv","/phi_2.9/freeflamelet_phi2.9_Tu300.0.csv"]
#phis=[0.33, 1.0, 2.9]
plot_var = "Temperature"
xlabel="c*"
ylabel="T*"

error_interps = []
error_vals = []
fig, axs = plt.subplots(nrows=1,ncols=len(configs),figsize=[20,10])
fig2, axs2 = plt.subplots(nrows=1,ncols=len(configs),figsize=[20,10])
iGroup = 0
e_min = 0
e_max = 2.0
grad_max = 5e3
for iconfig, config_PINN in enumerate(configs):
    with open(config_PINN.GetOutputDir()+"/"+config_PINN.GetConcatenationFileHeader()+"_full.csv",'r') as fid:
        vars_train = fid.readline().strip().split(',')
    train_data_full = np.loadtxt(config_PINN.GetOutputDir()+"/"+config_PINN.GetConcatenationFileHeader()+"_full.csv",delimiter=',',skiprows=1)[::10,:]
    cv_train_full = train_data_full[:, :3]

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
    #MLP_PINN.GetTrainData()
    MLP_PINN.SetSaveDir(os.getcwd())
    MLP_PINN.SetModelIndex(100)

    MLP_PINN.Preprocessing()
    var_out = config_PINN.GetMLPOutputGroup(iGroup)[0]

    pv_scaled_flamelets_all = []
    phi_flamelets_all = []
    phi_flamelets_red = []
    error_flamelets_all = []
    pv_scaled_flamelets_red = []
    dTdpv = []
    for p in phis:
        flamelets = os.listdir(flamelet_dir + "freeflame_data/" + p)
        flamelets.sort()
        if flamelets:
            f = flamelets[0]
            with open(flamelet_dir + "freeflame_data/" + p + "/" + f, 'r') as fid:
                vars = fid.readline().strip().split(',')
            F = np.loadtxt(flamelet_dir + "freeflame_data/" + p + "/" + f,delimiter=',',skiprows=1)
            pv = config_PINN.ComputeProgressVariable(vars, F)
            h = F[:,vars.index("EnthalpyTot")]
            Z = F[:, vars.index("MixtureFraction")]
            T = F[:, vars.index("Temperature")]
            
            config_PINN.gas.set_mixture_fraction(Z[0],config_PINN.GetFuelString(), config_PINN.GetOxidizerString())
            val_phi = config_PINN.gas.equivalence_ratio(config_PINN.GetFuelString(), config_PINN.GetOxidizerString())
            
            cv_flamelet = np.vstack((pv,h,Z)).T 
            ref_data = F[:, vars.index(var_out)]

            pred_data = MLP_PINN.EvaluateMLP(cv_flamelet)[:,0]
            ref_max, ref_min = max(ref_data), min(ref_data)
            pred_error = 100*np.abs((pred_data - ref_data) / (np.abs(ref_data)+1e-4*(ref_max - ref_min)))

            pvmin, pvmax = pv[0], pv[-1]
            pv_norm = (pv - pvmin)/(pvmax - pvmin)
            dT_dpv = (T[1:] - T[:-1]) / (np.abs(pv_norm[1:] - pv_norm[:-1]) + 1e-4)
            
            pv_scaled_flamelets_all.append(pv_norm)
            pv_scaled_flamelets_red.append(pv_norm[1:])
            dTdpv.append(np.abs(dT_dpv))
            phi_flamelets_all.append(val_phi*np.ones(pv_norm.shape))
            phi_flamelets_red.append(val_phi*np.ones(pv_norm[1:].shape))
            error_flamelets_all.append(pred_error)
            #axs[1].plot(val_phi, max(pred_error), marker=markers[iconfig],linestyle='none',color=colors[iconfig],label=labels[iconfig],markerfacecolor='none')
        
      #
    pv_scaled_flamelets_all = np.hstack(tuple(pv for pv in pv_scaled_flamelets_all))
    pv_scaled_flamelets_red = np.hstack(tuple(pv for pv in pv_scaled_flamelets_red))
    dTdpv = np.hstack(tuple(t for t in dTdpv))
    phi_flamelets_all = np.hstack(tuple(p for p in phi_flamelets_all))
    phi_flamelets_red = np.hstack(tuple(p for p in phi_flamelets_red))
    error_flamelets_all = np.hstack(tuple(e for e in error_flamelets_all))

    # phi_norm_all = np.zeros(phi_flamelets_all.shape)
    # phi_norm_all[phi_flamelets_all < 1.0] = (phi_flamelets_all[phi_flamelets_all < 1.0]/(2))
    # phi_norm_all[phi_flamelets_all >= 1.0] = 0.5*((phi_flamelets_all[phi_flamelets_all >= 1.0]-1.0)/19) + 0.5
    phi_norm_all = transform_phi(phi_flamelets_all)
    cs = axs[iconfig].tricontourf(pv_scaled_flamelets_all,phi_norm_all,np.clip((error_flamelets_all), e_min, e_max), levels=np.linspace(e_min, e_max, 20))
    cs2 = axs2[iconfig].tricontourf(pv_scaled_flamelets_red,phi_flamelets_red,np.clip(dTdpv, 0, grad_max),levels=np.linspace(0, grad_max,20))
          

    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.plot3D(cv[:,0],cv[:,1], error_flamelets_all,'k.')
    # plt.show()
cbar = fig.colorbar(cs,ax=axs, ticks=np.linspace(e_min, e_max, 6),shrink=0.95)
cbar.ax.set_title("e",fontsize=24)
cbar.ax.tick_params(which='both',labelsize=24)

cbar2 = fig2.colorbar(cs2,ax=axs2, ticks=[0, grad_max],shrink=0.95)
cbar2.ax.set_title("dTdc",fontsize=24)
cbar2.ax.tick_params(which='both',labelsize=24)
xmin = 0.0
xmax = 0.0
for i in range(len(configs)):
    xlim = axs[i].get_xlim()
    xmin = min(xmin, xlim[0])
    xmax = max(xmax, xlim[1])

fsize = 20
for i in range(len(configs)):
    axs[i].set_xlim(xmin, xmax)
    axs[i].set_xlabel("c*",fontsize=fsize)
    axs[i].tick_params(which='both',labelsize=fsize)
    axs2[i].set_xlim(xmin, xmax)
    axs2[i].set_xlabel("c*",fontsize=fsize)
    axs2[i].tick_params(which='both',labelsize=fsize)

phi_to_plot = np.array([0.25, 0.5, 1.0, 3.0, 10.0, 20.0])
phi_norm_to_plot = transform_phi(phi_to_plot)
for i in range(len(configs)):
    axs[i].set_yticks(phi_norm_to_plot)
    axs[i].set_yticklabels(phi_to_plot)

    
    if i > 0:

        axs[i].set_yticks([])
        axs[i].set_yticklabels([])
        axs2[i].set_yticks([])
        axs2[i].set_yticklabels([])

    axs[i].set_title(labels[i],fontsize=fsize)
    axs2[i].set_title(labels[i],fontsize=fsize)
axs[0].set_ylabel("Equivalence ratio (p)[-]", fontsize=fsize)
axs2[0].set_ylabel("Equivalence ratio (p)[-]", fontsize=fsize)


#fig.savefig("Images/evaluation_error.eps",format='eps',bbox_inches='tight')
#fig2.savefig("Images/dTdpv.eps",format='eps',bbox_inches='tight')
plt.show()
