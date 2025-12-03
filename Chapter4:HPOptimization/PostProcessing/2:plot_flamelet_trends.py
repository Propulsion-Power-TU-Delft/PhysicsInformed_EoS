import numpy as np 
import matplotlib.pyplot as plt
from su2dataminer.config import Config_FGM

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
        dydx[i] = (dx2_2 * y_p + (dx2_1 - dx2_2)*y_0 - dx2_1*y_m)/(dx_1*dx_2*(dx_1+dx_2) + 1e-10)
    dx_1 = x[1] - x[0]
    dx_2 = x[2] - x[0]
    dx2_1 = dx_1*dx_1 
    dx2_2 = dx_2*dx_2 
    y_0 = y[0]
    y_p = y[1]
    y_pp = y[2]
    dydx[0] = (dx2_1 * y_pp + (dx2_2 - dx2_1)*y_0 - dx2_2*y_p)/(dx_1*dx_2*(dx_1 - dx_2) + 1e-10)

    dx_1 = x[-2] - x[-1]
    dx_2 = x[-3] - x[-1]
    dx2_1 = dx_1*dx_1 
    dx2_2 = dx_2*dx_2 
    y_0 = y[-1]
    y_p = y[-2]
    y_pp = y[-3]
    dydx[-1] = (dx2_1 * y_pp + (dx2_2 - dx2_1)*y_0 - dx2_2*y_p)/(dx_1*dx_2*(dx_1 - dx_2) + 1e-10)
    return dydx 


N=3
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.cubehelix(np.linspace(0,1,N+1)))
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
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

color_pca2 = colors[1]
marker_pca2 = 'P'
msize_pca2 = 10
wline_pca2 = 1
mcolor_pca2=color_pca2

color_optim = colors[2]
marker_optim = 'o'
msize_optim = 9
wline_optim = 2
mcolor_optim=color_optim

Config_default = Config_FGM("../WRP.cfg")
Config_pca = Config_FGM("../PCA.cfg")
Config_optimized = Config_FGM("../OPT.cfg")

flamelet_dir = Config_default.GetOutputDir()
flamelets_to_plot = ["/phi_0.328947/freeflamelet_phi0.328947_Tu300.0.csv","/phi_1.0/freeflamelet_phi1.0_Tu300.0.csv","/phi_2.9/freeflamelet_phi2.9_Tu300.0.csv"]
phis=[0.33, 1.0, 2.9]
plot_var = "Temperature"
xlabel="c*"
ylabel="T*"
xlims = [[0.7, 1.050],[0.65, 1.050],[0.9,1.050]]
ylims = [[0.7, 1.050],[0.5, 1.050],[0.7, 1.050]]
ylims2 = [[-0.2, 2.0],[-1.0, 4.0],[-15.0, 40.0]]
fig,axs= plt.subplots(nrows=2,ncols=3,figsize=[12,6])
#fig_dpv,axs_dpv= plt.subplots(nrows=1,ncols=3,figsize=[10,6])
marker_frq = 0.08
fsize=20

for i in range(len(flamelets_to_plot)):
    with open(flamelet_dir + "freeflame_data/"+flamelets_to_plot[i],'r') as fid:
        vars_f = fid.readline().strip().split(',')
    Fd = np.loadtxt(flamelet_dir + "freeflame_data/"+ flamelets_to_plot[i],delimiter=',',skiprows=1)
    plot_data = Fd[:, vars_f.index(plot_var)]
    plot_data_norm = (plot_data - min(plot_data))/(max(plot_data) - min(plot_data))

    pv_default = Config_default.ComputeProgressVariable(vars_f, Fd)
    pv_default_min,pv_default_max = min(pv_default),max(pv_default)
    pv_default_norm = (pv_default - pv_default_min)/(pv_default_max-pv_default_min)

    pv_pca = Config_pca.ComputeProgressVariable(vars_f, Fd)
    pv_pca_min,pv_pca_max = min(pv_pca),max(pv_pca)
    pv_pca_norm = (pv_pca - pv_pca_min)/(pv_pca_max-pv_pca_min)

    pv_optimized = Config_optimized.ComputeProgressVariable(vars_f, Fd)
    pv_optimized_min,pv_optimized_max = min(pv_optimized),max(pv_optimized)
    pv_optimized_norm = (pv_optimized - pv_optimized_min)/(pv_optimized_max-pv_optimized_min)



    dy_dpv_default = FiniteDifferenceDerivative(y=plot_data_norm,x=pv_default_norm)
    dy_dpv_pca = FiniteDifferenceDerivative(y=plot_data_norm,x=pv_pca_norm)
    dy_dpv_optimized = FiniteDifferenceDerivative(y=plot_data_norm,x=pv_optimized_norm)
    
    # axs[0,i].set_xlim(xlims[i][0],xlims[i][1])
    # axs[1,i].set_xlim(xlims[i][0],xlims[i][1])
    # axs[0,i].set_ylim(ylims[i][0],ylims[i][1])
    # axs[1,i].set_ylim(ylims2[i][0],ylims2[i][1])

    axs[0,i].plot(pv_default_norm, plot_data_norm,color=color_default,linewidth=wline_default,marker=marker_default,markersize=msize_default,markevery=marker_frq,markerfacecolor=mcolor_default, label="WRP")
    axs[1,i].plot(pv_default_norm, dy_dpv_default,color=color_default,linewidth=wline_default,marker=marker_default,markersize=msize_default,markevery=marker_frq,markerfacecolor=mcolor_default)
    

    axs[0,i].plot(pv_pca_norm, plot_data_norm,color=color_pca,linewidth=wline_pca,marker=marker_pca,markersize=msize_pca,markevery=marker_frq,markerfacecolor=mcolor_pca,label="PCA")
    axs[1,i].plot(pv_pca_norm, dy_dpv_pca,color=color_pca,linewidth=wline_pca,marker=marker_pca,markersize=msize_pca,markevery=marker_frq,markerfacecolor=mcolor_pca)
 
    axs[0,i].plot(pv_optimized_norm, plot_data_norm,color=color_optim,linewidth=wline_optim,marker=marker_optim,markersize=msize_optim,markevery=marker_frq,markerfacecolor=mcolor_optim,label="OPT")
    axs[1,i].plot(pv_optimized_norm, dy_dpv_optimized,color=color_optim,linewidth=wline_optim,marker=marker_optim,markersize=msize_optim,markevery=marker_frq,markerfacecolor=mcolor_optim)
    
    axs[0,i].grid()
    axs[1,i].grid()
    axs[0,i].tick_params(which='both',labelsize=fsize)
    axs[0,i].set_xticklabels([])
    axs[1,i].tick_params(which='both',labelsize=fsize)
    

for i in range(len(flamelets_to_plot)):
    axs[0, i].set_title("phi=%.2f" % phis[i], fontsize=fsize)
    axs[1, i].set_xlabel(xlabel,fontsize=fsize)
axs[0,0].set_ylabel(ylabel,fontsize=fsize)
axs[1,0].set_ylabel("d%s/d%s" % (ylabel,xlabel),fontsize=fsize)
plt.subplots_adjust(wspace=0.25, hspace=0.08)
handles, labels = axs[0,0].get_legend_handles_labels()
lgnd = fig.legend(handles, labels, fontsize=fsize,ncol=4,bbox_to_anchor=(0.5, 0.005),loc='upper center',fancybox=True,shadow=True)

#fig.savefig("./Images/flamelet_trends_Tstar.eps",format='eps',bbox_inches='tight')
plt.show()
