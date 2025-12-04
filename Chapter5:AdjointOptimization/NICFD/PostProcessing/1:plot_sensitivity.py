import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import gmsh 
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar, Bounds,root

#---------------------------------------------------------------------------------------------#
# Importing general packages
#---------------------------------------------------------------------------------------------#
import sys
import os
import time
import copy
#---------------------------------------------------------------------------------------------#
# Importing ParaBlade classes and functions
#---------------------------------------------------------------------------------------------#
from common.common_utils import *
from common.config_utils import *
from src.blade_geom_2D import BladeGeom2D
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams["font.family"] = "Times New Roman"
N=3
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.cubehelix(np.linspace(0,1,N+1)))
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

surface_sens_filename="../Base/DOT/surface_sensitivity.csv"
IN = read_user_input("../ORCHID_stator_base_ParaBlade.cfg")
blade = BladeGeom2D(IN)
blade.make_blade()


with open(surface_sens_filename,'r') as fid:
    vars_sens_file = fid.readline().strip().split(",")
    vars_sens_file = [v.strip("\"") for v in vars_sens_file]
sens_surf_AD = np.loadtxt(surface_sens_filename,delimiter=',',skiprows=1)
x_surf, y_surf = sens_surf_AD[:, vars_sens_file.index("x")],sens_surf_AD[:, vars_sens_file.index("y")]
iPoint_surf = sens_surf_AD[:,0]

u_range_coarse = np.linspace(0, 1, 100)
xy_blade_coarse = blade.get_coordinates(u_range_coarse).T

vals_u_surf = np.zeros(len(sens_surf_AD))
def GetDist(delta_u:float, u_ref:float, x:float, y:float):
    U = (u_ref + delta_u) % 1.0
    xy_blade = blade.get_coordinates(U).T
    dist = (xy_blade[0,0] - x)**2 + (xy_blade[0,1] - y)**2
    return dist

for i_surf in range(len(sens_surf_AD)):
    x, y = x_surf[i_surf], y_surf[i_surf]
    dist = np.power(xy_blade_coarse[:,0]-x,2)+np.power(xy_blade_coarse[:,1]-y,2)
    u_coarse = u_range_coarse[np.argmin(dist)]
    
    bounds = (-1./10, 1./10)
    #res = minimize_scalar(GetDist, args=(u_coarse, x,y),tol=1e-20)
    res = root(GetDist, x0=0.0,args=(u_coarse, x,y),tol=1e-30)
    vals_u_surf[i_surf] = (res.x + u_coarse) % 1.0

xy_fitted = blade.get_coordinates(vals_u_surf).T
sens_cad_all = blade.get_sensitivity(vals_u_surf)

sens_AD = sens_surf_AD[:, [vars_sens_file.index("Sensitivity_x"),vars_sens_file.index("Sensitivity_y")]]

color_AD = colors[2]
color_FD = colors[1]
color_FD_CP = colors[0]
cdir = os.getcwd() + "../"
x = 0.0

x_vars = []
sens_vars = []
var_labels = []
for dv_name, dv_val in zip(blade.DVs_values.keys(), blade.DVs_values.values()):
    for iDv in range(len(dv_val)):
        dv_name_full = "%s_%i" % (dv_name, iDv)
        try:
            sens_cad = sens_cad_all[dv_name_full].T

            IN_n = copy.deepcopy(IN)
            IN_n[dv_name][iDv] *= 1.0001
            delta_dv = IN_n[dv_name][iDv] - IN[dv_name][iDv]

            x_blade_n = sens_surf_AD[:, vars_sens_file.index("x")] + sens_cad[:,0]*delta_dv 
            y_blade_n = sens_surf_AD[:, vars_sens_file.index("y")] + sens_cad[:,1]*delta_dv 
            
            sens_dot = np.sum(sens_cad*sens_AD)
            if abs(sens_dot) > 5:
                x_vars.append(x)
                sens_vars.append(sens_dot)
                var_labels.append(dv_name_full)
                x += 1
        except:
            pass


ix_sort = np.argsort(np.abs(np.array(sens_vars)))[::-1]
x_vars = np.array([i for i in range(len(sens_vars))])
sens_vars = [sens_vars[i] for i in ix_sort]
var_labels = [var_labels[i] for i in ix_sort]

sens_vars_FD = []
sens_vars_FD_CP = []

for v in var_labels:
    
    sens_AD = float(np.genfromtxt("%s/plus_%s/sens_AD.txt" % (cdir, v)))

    history_perturbed = np.loadtxt("%s/plus_%s/DIRECT/history_direct_JST.csv" % (cdir, v),delimiter=',',skiprows=1)
    val_f_perturbed = history_perturbed[-1,-1]
    with open("%s/plus_%s/FD_step.txt" % (cdir, v), 'r') as fid:
        line = fid.readline().strip()
        val_f_ref = float(line.split(" ")[-1])
        line = fid.readline().strip()
        FD_step = float(line.split(" ")[-1])
    sens_FD = (val_f_perturbed-val_f_ref)/FD_step
    sens_vars_FD.append(sens_FD)

    history_base_CoolProp = np.loadtxt("%s/Base/CoolProp/history_direct_JST_CoolProp.csv" % (cdir),delimiter=',',skiprows=1)
    val_f_ref_CP = history_base_CoolProp[-1,-1]
    try:
        history_perturbed = np.loadtxt("%s/plus_%s/CoolProp/history_direct_JST_CoolProp.csv" % (cdir, v),delimiter=',',skiprows=1)
        val_f_perturbed_CP = history_perturbed[-1,-1]
        sens_FD_CP = (val_f_perturbed_CP - val_f_ref_CP)/FD_step
        sens_vars_FD_CP.append(sens_FD_CP)
    except:
        sens_vars_FD_CP.append(0)

    
#print(" && ".join("cd plus_%s/CoolProp/ && ln -sf restart_JST_CoolProp.dat solution_JST.dat && cd ../.." % v for v in var_labels))
var_labels = [v.replace("thickness_upper", "U") for v in var_labels]
var_labels = [v.replace("thickness_lower", "L") for v in var_labels]
var_labels[var_labels.index("chord_axial_0")] = "c"
var_labels[var_labels.index("y_in_0")] = "y_\mathrm{LE}"
var_labels[var_labels.index("radius_out_0")] = "R_\mathrm{TE}"
for ivar, v in enumerate(var_labels):
    v_split = v.split("_")
    v_new = "_".join("{%s}" % q for q in v_split)
    var_labels[ivar] = v_new
var_labels = [r"$%s$" % (v) for v in var_labels]

w = 0.8
w_bar = w / (N) 
fig, axs = plt.subplots(ncols=1,nrows=2,figsize=[12,8])
ax = axs[0]
ax.bar(x=x_vars+ (0.5 + 2)*w_bar - 0.5*w, height=sens_vars,width=w_bar,zorder=3,color=color_AD,label="PINN AD")
ax.bar(x=x_vars+ (0.5 + 1)*w_bar - 0.5*w, height=sens_vars_FD,width=w_bar,zorder=3,color=color_FD,label="PINN FD")
ax.bar(x=x_vars+ (0.5 + 0)*w_bar - 0.5*w, height=sens_vars_FD_CP,width=w_bar,zorder=3,color=color_FD_CP,label="CP FD")
ax.set_xticks(x_vars, var_labels)
ax.tick_params(which='both',labelsize=20)
ax.grid()
ax.set_ylabel(r"$\nabla_p f$",fontsize=20)
ax.set_xticklabels([])
ax.legend(fontsize=20, ncol=1, loc="center left",bbox_to_anchor=(1, 0.5),fancybox=True,shadow=True)


diff_FD_AD = np.abs(np.array(sens_vars) - np.array(sens_vars_FD))
diff_FD_CP = np.abs(np.array(sens_vars) - np.array(sens_vars_FD_CP))
rel_diff_FD_AD = 100*np.abs((np.array(sens_vars) - np.array(sens_vars_FD))/np.array(sens_vars_FD))
rel_diff_FD_CP = 100*np.abs((np.array(sens_vars) - np.array(sens_vars_FD_CP))/np.array(sens_vars_FD_CP))

for L, sens_AD, sens_FD_EEoS, sens_FD_CP, diff_EEoS, diff_HEoS in zip(var_labels, sens_vars, sens_vars_FD, sens_vars_FD_CP, rel_diff_FD_AD, rel_diff_FD_CP):
    print("%s & $\SI{%+.4f}{}$ & $\SI{%+.4f}{}$ &$\SI{%+.4f}{}$ & $\SI{%+.4f}{}$ & $\SI{%+.4f}{}$\\\\" % (L, sens_AD, sens_FD_EEoS, sens_FD_CP, diff_EEoS, diff_HEoS))


w_bar = w / (N) 
ax = axs[1]
ax.bar(x=x_vars+ (0.5 + 1)*w_bar - 0.5*w, height=rel_diff_FD_AD,width=w_bar,zorder=3,color=color_FD,label="PINN FD")
ax.bar(x=x_vars+ (0.5 + 0)*w_bar - 0.5*w, height=rel_diff_FD_CP,width=w_bar,zorder=3,color=color_FD_CP,label="CP FD")
ax.set_xticks(x_vars, var_labels)
ax.tick_params(which='both',labelsize=20)
ax.grid()
ax.set_ylabel(r"$\epsilon_{\nabla_p f}[\%]$",fontsize=20)
ax.set_xlabel(r"Design parameter $p$",fontsize=20)
fig.tight_layout()
plt.show()