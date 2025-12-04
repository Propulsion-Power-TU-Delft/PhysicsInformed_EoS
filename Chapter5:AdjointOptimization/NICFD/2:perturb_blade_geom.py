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

xy_orig = blade.get_coordinates(np.linspace(0, 1, 300)).T
sens_cad_all = blade.get_sensitivity(vals_u_surf)

sens_AD = sens_surf_AD[:, [vars_sens_file.index("Sensitivity_x"),vars_sens_file.index("Sensitivity_y")]]

cad_sens = {}
sens_mag = 0.0
for dv_name, dv_val in zip(blade.DVs_values.keys(), blade.DVs_values.values()):
    for iDv in range(len(dv_val)):
        dv_name_full = "%s_%i" % (dv_name, iDv)
        try:
            sens_cad = sens_cad_all[dv_name_full].T
            sens_dot = np.sum(sens_cad*sens_AD)
            cad_sens[dv_name_full] = sens_dot 
            sens_mag += sens_dot**2
        except:
            pass

IN_n = copy.deepcopy(IN)
step_size = 2e-5
sens_mag = np.sqrt(sens_mag)

x_blade_n = sens_surf_AD[:, vars_sens_file.index("x")].copy()
y_blade_n = sens_surf_AD[:, vars_sens_file.index("y")].copy()
delta_f = 0.0
for dv_name, dv_val in zip(blade.DVs_values.keys(), blade.DVs_values.values()):
    for iDv in range(len(dv_val)):
        dv_name_full = "%s_%i" % (dv_name, iDv)
        try:
            sens_cad = sens_cad_all[dv_name_full].T
            delta_cad = step_size * cad_sens[dv_name_full] / sens_mag
            IN_n[dv_name][iDv] -= step_size * cad_sens[dv_name_full] / sens_mag
            delta_f -= step_size * cad_sens[dv_name_full] / sens_mag
            x_blade_n -= sens_cad[:,0]*delta_cad
            y_blade_n -= sens_cad[:,1]*delta_cad
        except:
            pass

blade.update_DVs_values(IN_n)
blade.make_blade()

fid = open("ORCHID_stator_updated_ParaBlade.cfg","w+")
write_config_file(fid, IN_n)
fid.close()

xy_updated = blade.get_coordinates(vals_u_surf).T


fid = open("%s/Base_Updated/MoveSurface.dat" % (os.getcwd()), "w+")
for i in range(len(x_blade_n)):
    fid.write("%i\t%+.16e\t%+.16e\t%+.16e\n" % (iPoint_surf[i], x_blade_n[i],y_blade_n[i],0.0))
fid.close()
