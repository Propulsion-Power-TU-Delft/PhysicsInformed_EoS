# Perform apriori analysis of DC solutions with flamelet data used to train the ML-FGM networks. 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.spatial import cKDTree
from su2dataminer.config import Config_FGM
from Common.Interpolators import Invdisttree

# Load SU2 DataMiner configuration
config_name = "../config_PIML.cfg"
config = Config_FGM(config_name)

# Load full flamelet data set used for apriori analysis
flamelet_data_file = config.GetOutputDir()+"/"+config.GetConcatenationFileHeader()+"_full.csv"
with open(flamelet_data_file,'r') as fid:
    vars_FGM = fid.readline().strip().split(',')
FGM_data = np.loadtxt(flamelet_data_file,delimiter=',',skiprows=1)

# Extract FGM controlling variables and normalize using standard scaling.
scaler_FGM = StandardScaler()
CV_data_FGM = FGM_data[:,[0,1,2]]
CV_data_FGM_scaled = scaler_FGM.fit_transform(CV_data_FGM)

# Create an inverse-distance weighted KD tree to interpolate thermochemical state information from the flamelet data set.
FGM_data_interpolator = Invdisttree(CV_data_FGM_scaled, FGM_data)

def AprioryAnalysis(DC_data_file:str):
    """Interpolate thermochemical state information from flamelet data based on the DC controlling variable solution.

    :param DC_data_file: detailed chemistry csv solution file
    :type DC_data_file: str
    :return: interpolated thermochemistry data
    :rtype: np.ndarray
    """

    # Extract DC solution
    with open(DC_data_file,'r') as fid:
        vars_DC = fid.readline().strip().split(',')
        vars_DC = [v.strip("\"") for v in vars_DC]
    DC_data = np.loadtxt(DC_data_file,delimiter=',',skiprows=1)

    # Read FGM controlling variable solution from DC data and normalize.
    CV_data_DC = DC_data[:, [vars_DC.index("ProgressVariable"),vars_DC.index("EnthalpyTot"),vars_DC.index("MixtureFraction")]]
    CV_data_DC_scaled = scaler_FGM.transform(CV_data_DC)

    # Interpolate thermochemical state data from flamelet data set from the five nearest neighbors.
    apriory_data = FGM_data_interpolator(CV_data_DC_scaled, nnear=5,p=2)
    return apriory_data

def InterpolateSU2FGMSolution(DC_vars:list[str], DC_sol:np.ndarray[float], FGM_vars:list[str], FGM_sol:np.ndarray[float]):
    """Find the nearest neighbor of the SU2 simulation result with respect to the DC solution.

    :param DC_vars: DC solution variables
    :type DC_vars: list[str]
    :param DC_sol: DC solution
    :type DC_sol: np.ndarray[float]
    :param FGM_vars: FGM solution variables
    :type FGM_vars: list[str]
    :param FGM_sol: FGM solution
    :type FGM_sol: np.ndarray[float]
    :return: nearest neighbor nodes of the SU2 FGM solution
    :rtype: np.ndarray[int]
    """
    xy_DC = DC_sol[:, [DC_vars.index("Points:0"), DC_vars.index("Points:1")]]
    xy_FGM = FGM_sol[:, [FGM_vars.index("Points:0"), FGM_vars.index("Points:1")]]
    xy_FGM[:, 0] *= -1
    tree = cKDTree(xy_FGM)
    _, ix_nearest = tree.query(xy_DC,k=1)
    return ix_nearest

    
# DC and FGM solution ASCII files.
DC_data_files = ["../DC_solutions/DomainData_phi_050_v0565_Tw450.csv",\
                 "../DC_solutions/DomainData_phi_050_v1000_Tw450.csv",\
                 "../DC_solutions/DomainData_phi_050_v2000_Tw450.csv"]
FGM_data_files = ["../SU2_Simulations/DomainData_v0565_Tw450.csv",\
                  "../SU2_Simulations/DomainData_v1000_Tw450.csv",\
                  "../SU2_Simulations/DomainData_v2000_Tw450.csv"]

# For each inflow velocity, perform the apriori analysis.
u_inlet = ["0565", "1000","2000"]
FGM_interp_data = []
DC_data = []
DC_vars = []
FGM_data = []
for f, g, u in zip(DC_data_files, FGM_data_files, u_inlet):

    # Calculate AP data from DC solution
    AP_data = AprioryAnalysis(f)
    FGM_interp_data.append(AP_data)

    # Find the nearest neighbors of the FGM solution and reorder such that FGM and DC solution data can be plotted over the same x-variable
    with open(f,'r') as fid:
        vars_DC = fid.readline().strip().split(',')
        vars_DC = [v.strip("\"") for v in vars_DC]
    DC_vars.append(vars_DC)
    DC = np.loadtxt(f,delimiter=',',skiprows=1)
    DC_data.append(DC)
    with open(g,'r') as fid:
        vars_FGM_sol = fid.readline().strip().split(',')
        vars_FGM_sol = [v.strip("\"") for v in vars_FGM_sol]
    FGM_sol_data = np.loadtxt(g, delimiter=',',skiprows=1)

    ix_FGM = InterpolateSU2FGMSolution(vars_DC, DC, vars_FGM_sol, FGM_sol_data)
    FGM_data.append(FGM_sol_data[ix_FGM, :])
      
    
fsize = 20
marker_frq = 5
msize = 6
alpha_fgm = 0.5
N=3
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.cubehelix(np.linspace(0,1,N+1)))
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_DC= colors[0]
color_FGM = colors[1]
color_NN = colors[2]

def PlotData(ax:plt.subplot, i_v:int, varname:str, ylabel:str):
    """Plot DC solution, FGM solution, and AP data for a single thermochemical state variable.

    :param ax: plot axes
    :type ax: plt.subplot
    :param i_v: solution index (0,1, or 2)
    :type i_v: int
    :param varname: thermochemical state variable
    :type varname: str
    :param ylabel: plot label 
    :type ylabel: str
    :return: updated plot axes
    """

    # Extract DC, FGM, and AP plot data
    try:
        y_data_DC = DC_data[i_v][:, DC_vars[i_v].index(varname)]
    except:
        y_data_DC = DC_data[i_v][:, DC_vars[i_v].index(varname.lower())]
    y_data_FGM = FGM_data[i_v][:, vars_FGM_sol.index(varname)]
    y_data_AP = FGM_interp_data[i_v][:, vars_FGM.index(varname)]
    
    # Extract progress variable of DC, FGM, and AP solutions
    pv_DC = DC_data[i_v][:, DC_vars[i_v].index("ProgressVariable")]
    pv_FGM = FGM_data[i_v][:, vars_FGM_sol.index("ProgressVariable")]
    pv_AP = FGM_interp_data[i_v][:, vars_FGM.index("ProgressVariable")]
    
    # Scale plot data for visualization
    y_max, y_min = np.max(y_data_DC), np.min(y_data_DC)
    DC_data_scaled = (y_data_DC - y_min)/(y_max - y_min)
    FGM_data_scaled = (y_data_FGM - y_min)/(y_max - y_min)
    AP_data_scaled = (y_data_AP - y_min)/(y_max - y_min)

    pv_min, pv_max = min(pv_DC), max(pv_DC)
    pv_DC_scaled = (pv_DC - pv_min)/(pv_max - pv_min)
    pv_FGM_scaled = (pv_FGM -pv_min)/(pv_max - pv_min)
    pv_AP_scaled = (pv_AP -pv_min)/(pv_max - pv_min)
    
    # Plot scaled thermochemical state data against the normalized progress variable
    ax.plot(pv_AP_scaled, AP_data_scaled,linestyle='none',marker='.',color=color_NN,markevery=marker_frq,alpha=alpha_fgm,markersize=1.2*msize,label="AP")
    ax.plot(pv_FGM_scaled, FGM_data_scaled,linestyle='none',marker='.',color=color_FGM,markevery=marker_frq,alpha=alpha_fgm,markersize=1.2*msize,label="FGM")
    ax.plot(pv_DC_scaled, DC_data_scaled,linestyle='none',marker='.',color=color_DC,markevery=marker_frq,alpha=alpha_fgm,markersize=1.2*msize,label="DC")
    
    ylim_ax = ax.get_ylim()
    ax.axvspan((-0.15 - pv_min)/(pv_max - pv_min),(0.25 - pv_min)/(pv_max - pv_min),  color='r',alpha=0.15)
    ax.axvspan((max(pv_FGM) - pv_min)/(pv_max - pv_min),(0.32 - pv_min)/(pv_max - pv_min), color='b',alpha=0.15)
    ax.axvspan(0.0, (-0.15 - pv_min)/(pv_max - pv_min), color='k',alpha=0.15)
    ax.set_xlabel(r"$c^*$",fontsize=fsize)
    ax.set_ylabel(ylabel,fontsize=fsize)
    ax.tick_params(which='both',labelsize=fsize)
    ax.grid()
    
    return ylim_ax

# Prepare subplots
fig = plt.figure(figsize=[14, 10])
ax_layout = [(2,4,1),(2,4,2),(2,4,3),(2,4,5),(2,4,6), (2,4,7)]
axs = []
for nrows,ncols,p in ax_layout:
    axs.append(fig.add_subplot(nrows,ncols,p))

# Plot DC, FGM, and AP solutions for temperature and the pv source term
ylim_min_0 = ylim_min_1 = ylim_min_2 = 1e3
ylim_max_0 = ylim_max_1 = ylim_max_2 = -1e3
for i_v in range(len(DC_data)):
    ylim_ax_1 = PlotData(axs[i_v], i_v, "Temperature", r"$T^*$")
    ylim_ax_2 = PlotData(axs[i_v+3], i_v, "ProdRateTot_PV", r"$\dot{\omega}_c*$")
    # ylim_ax_1 = PlotData(axs[i_v], i_v, "EnthalpyTot", r"$h^*$")
    # ylim_ax_2 = PlotData(axs[i_v+3], i_v, "MixtureFraction", r"$Z^*$")
    ylim_min_1 = min(ylim_min_1, ylim_ax_1[0])
    ylim_min_2 = min(ylim_min_2, ylim_ax_2[0])
    ylim_max_1 = max(ylim_max_1, ylim_ax_1[1])
    ylim_max_2 = max(ylim_max_2, ylim_ax_2[1])
 
# Create shaded regions for the preheating, flame, and reaction zone
titles = [r"$u=0.565ms^{-1}$",r"$u=1.00ms^{-1}$",r"$u=2.00ms^{-1}$"]
for i in range(3):
    axs[i].set_ylim(ylim_min_1, ylim_max_1)
    axs[i+3].set_ylim(ylim_min_2, ylim_max_2)
    axs[i+3].text(s="2", x=0.5,y=1.6,color='r',horizontalalignment='center',fontsize=20,verticalalignment='top')
    axs[i+3].text(s="3", x=0.95,y=1.6,color='b',horizontalalignment='center',fontsize=20,verticalalignment='top')
    axs[i+3].text(s="1", x=0.05,y=1.6,color='k',horizontalalignment='center',fontsize=20,verticalalignment='top')
    axs[i].set_title(titles[i], fontsize=fsize)
    axs[i].set_xlabel("")
    axs[i].set_xticklabels([])
    if i > 0:
        axs[i].set_ylabel("")
        axs[i].set_yticklabels([])
        axs[i+3].set_ylabel("")
        axs[i+3].set_yticklabels([])
        
fig.tight_layout()
lgnd = axs[-3].legend(fontsize=fsize,ncol=3,loc="upper center",bbox_to_anchor=[0.5,-0.2],shadow=True,fancybox=True)
for h in lgnd.legend_handles:
        h._markersize=20
        h._alpha = 1 
plt.show()
   
