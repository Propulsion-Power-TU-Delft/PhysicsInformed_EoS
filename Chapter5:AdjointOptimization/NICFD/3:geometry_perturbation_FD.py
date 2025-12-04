import numpy as np
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

surface_sens_filename="Base/DOT/surface_sensitivity.csv"
IN = read_user_input("ORCHID_stator_base_ParaBlade.cfg")
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
    res = root(GetDist, x0=0.0,args=(u_coarse, x,y),tol=1e-30)
    vals_u_surf[i_surf] = (res.x + u_coarse) % 1.0

xy_fitted = blade.get_coordinates(vals_u_surf).T
sens_cad_all = blade.get_sensitivity(vals_u_surf)

sens_AD = sens_surf_AD[:, [vars_sens_file.index("Sensitivity_x"),vars_sens_file.index("Sensitivity_y")]]

val_obj_ref = np.loadtxt("Base/DIRECT/history_direct_JST_rr.csv",delimiter=',',skiprows=1)[-1,-1]

cdir = os.getcwd()
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
            
            if not os.path.isdir("%s/plus_%s" % (os.getcwd(), dv_name_full)):
                os.mkdir("%s/plus_%s" % (os.getcwd(), dv_name_full))
            
            fid = open("%s/plus_%s/MoveSurface.dat" % (os.getcwd(), dv_name_full), "w+")
            for i in range(len(x_blade_n)):
                fid.write("%i\t%+.16e\t%+.16e\t%+.16e\n" % (iPoint_surf[i], x_blade_n[i],y_blade_n[i],0.0))
            fid.close()

            
            sens_dot = np.sum(sens_cad*sens_AD)
            blade.update_DVs_values(IN)
            blade.make_blade()
            print("%s : %+.6e" % (dv_name_full, sens_dot))
            with open("%s/plus_%s/sens_AD.txt" % (os.getcwd(), dv_name_full), "w") as fid:
                fid.write("%+.16e" % sens_dot)
            with open("%s/plus_%s/FD_step.txt" % (os.getcwd(), dv_name_full), "w") as fid:
                fid.write("val_f_ref: %+.16e\n" % val_obj_ref)
                fid.write("FD step: %+.16e\n" % delta_dv)
                
            if not os.path.isdir("%s/plus_%s/DEFORM" % (os.getcwd(), dv_name_full)):
                os.mkdir("%s/plus_%s/DEFORM" % (os.getcwd(), dv_name_full))
            if not os.path.isdir("%s/plus_%s/DIRECT" % (os.getcwd(), dv_name_full)):
                os.mkdir("%s/plus_%s/DIRECT" % (os.getcwd(), dv_name_full))
            if not os.path.isdir("%s/plus_%s/CoolProp" % (os.getcwd(), dv_name_full)):
                os.mkdir("%s/plus_%s/CoolProp" % (os.getcwd(), dv_name_full))
            
            os.chdir("%s/plus_%s/DEFORM" % (cdir, dv_name_full))
            os.system("ln -sf ../../config_DEFORM_FD.cfg ./config_DEFORM.cfg")
            os.system("ln -sf ../../ORCHID_mesh.su2 ./su2mesh.su2")
            os.system("SU2_DEF config_DEFORM.cfg")
            os.chdir("%s/plus_%s/DIRECT" % (cdir, dv_name_full))
            os.system("cp --remove-destination ../../config_DIRECT_JST.cfg ./config_JST.cfg")
            os.system("cp --remove-destination ../../Base/DIRECT/restart_JST.dat ./solution_JST.dat")
            os.system("cp --remove-destination ../../MLP_PINN.mlp ./")
            os.system("cp --remove-destination ../DEFORM/mesh_deformed.su2 ./su2mesh.su2")
            os.chdir("%s/plus_%s/CoolProp" % (cdir, dv_name_full))
            os.system("cp --remove-destination ../../config_DIRECT_COOLPROP.cfg ./config_JST_direct.cfg")
            os.system("cp --remove-destination ../../Base/CoolProp/restart_JST.dat ./solution_JST.dat")
            os.system("cp --remove-destination ../DEFORM/mesh_deformed.su2 ./su2mesh.su2")
            os.chdir(cdir)
        except:
            pass

