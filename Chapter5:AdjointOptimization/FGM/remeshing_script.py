#!/usr/bin/env python3

############################ FILE NAME: remeshing_script.py ###################################
#=============================================================================================#
# author: Evert Bunschoten                                                                    |
#    :PhD Candidate ,                                                                         |
#    :Flight Power and Propulsion                                                             |
#    :TU Delft,                                                                               |
#    :The Netherlands                                                                         |
#                                                                                             |
#                                                                                             |
# Description:                                                                                |
# Regenerate the mesh around the deformed boundaries of the burner and heat exchanger and     |
# copy the deformed FFD box to the new mesh.                                                  |
#                                                                                             |  
#                                                                                             |
#=============================================================================================#
import numpy as np 
from Mesh.generate_mesh import MeshZone
from check_mesh_quality import checkMeshQuality 
import copy_ffdb_n as cp_ffdb

deformed_mesh_filename = "mesh_out.su2"
remesh_header = "remesh"

# Read deformed surface coordinates of the burner geometry.
surface_filename_burner = "surface_deformed_1.csv" 
xy_ref = np.loadtxt(surface_filename_burner,delimiter=',',skiprows=1)[:,1:]
i_sorted = [0]
xy_sorted = np.array([xy_ref[0,:]])
xy = np.delete(xy_ref,i_sorted,axis=0)
while len(xy) > 1:
    dist = np.sum(np.power(xy - xy_sorted[-1,:], 2),axis=1)
    i_min = np.argmin(dist)
    xy_sorted = np.vstack((xy_sorted, xy[i_min,:]))
    xy = np.delete(xy, i_min,axis=0)
xy_sorted = np.vstack((xy_sorted, xy[0,:]))
xy_surf_deformed_burner = xy_sorted.copy()
ix_sort = np.argsort(xy_surf_deformed_burner[:,0])[::-1]
xy_surf_deformed_burner = xy_surf_deformed_burner[ix_sort, :]

# Read deformed surface coordinates of the heat exchanger geometry.
surface_filename_hex = "surface_deformed_2.csv" 
xy_ref = np.loadtxt(surface_filename_hex,delimiter=',',skiprows=1)[:,1:]
i_sorted = [0]
xy_sorted = np.array([xy_ref[0,:]])
xy = np.delete(xy_ref,i_sorted,axis=0)
while len(xy) > 1:
    dist = np.sum(np.power(xy - xy_sorted[-1,:], 2),axis=1)
    i_min = np.argmin(dist)
    xy_sorted = np.vstack((xy_sorted, xy[i_min,:]))
    xy = np.delete(xy, i_min,axis=0)
xy_sorted = np.vstack((xy_sorted, xy[0,:]))
xy_surf_deformed_hex = xy_sorted.copy()
ix_sort = np.argsort(xy_surf_deformed_hex[:,0])[::-1]
xy_surf_deformed_hex = xy_surf_deformed_hex[ix_sort, :]

# Check which zones need remeshing based on the maximum cell skewedness.
remesh_zones = [False, False, False]
for iZone in range(3):
    q = max(checkMeshQuality(deformed_mesh_filename, iZone))
    if q >= 0.65:
        remesh_zones[iZone] = True 

# Read information from deformed mesh.
with open(deformed_mesh_filename,'r') as fid:
    lines_deformed_mesh = fid.readlines()

# Remesh each zone of insufficient quality
if any(remesh_zones):
    print("Quality of deformed mesh insufficient, remeshing")

    # Regenerate mesh in zones with bad quality.
    for iZone, remesh in enumerate(remesh_zones):
        if remesh:
            MeshZone(iZone, xy_surf_deformed_hex=xy_surf_deformed_hex,xy_surf_deformed_burner=xy_surf_deformed_burner)
    lines_zone_split = []

    # Collect zone information from deformed mesh.
    zones_sep_lines = []
    iline_ffdbox_start = -1
    for iline, line in enumerate(lines_deformed_mesh):
        if "IZONE" in line:
            zones_sep_lines.append(iline)
        if "FFD_NBOX" in line:
            iline_ffdbox_start = iline
    zones_sep_lines.append(iline_ffdbox_start)

    lines_fluid_mesh_deformed = lines_deformed_mesh[zones_sep_lines[0]+1:zones_sep_lines[1]]
    lines_burner_mesh_deformed = lines_deformed_mesh[zones_sep_lines[1]+1:zones_sep_lines[2]]
    lines_hex_mesh_deformed = lines_deformed_mesh[zones_sep_lines[2]+1:zones_sep_lines[3]]

    # Replace zone information with regenerated mesh for zones with bad mesh quality.
    if remesh_zones[0]:
        print("Writing regenerated mesh for fluid zone")
        with open("fluid_mesh.su2", "r") as fid:
            lines_fluid_mesh = fid.readlines()
    else:
        lines_fluid_mesh = lines_fluid_mesh_deformed
    
    if remesh_zones[1]:
        print("Writing regenerated mesh for burner zone")
        with open("burner_mesh.su2", "r") as fid:
            lines_burner_mesh = fid.readlines()
    else:
        lines_burner_mesh = lines_burner_mesh_deformed
    
    if remesh_zones[2]:
        print("Writing regenerated mesh for hex zone")
        with open("hex_mesh.su2", "r") as fid:
            lines_hex_mesh = fid.readlines()
    else:
        lines_hex_mesh = lines_hex_mesh_deformed

    # Write zone information to new mesh file.
    output_mesh_filename = "%s.su2" % remesh_header
    fid_combined_mesh = open(output_mesh_filename, "w+")
    fid_combined_mesh.write("NZONE=3\n")
    fid_combined_mesh.write("IZONE=1\n")
    fid_combined_mesh.writelines(lines_fluid_mesh)
    fid_combined_mesh.write("IZONE=2\n")
    fid_combined_mesh.writelines(lines_burner_mesh)
    fid_combined_mesh.write("IZONE=3\n")
    fid_combined_mesh.writelines(lines_hex_mesh)
    fid_combined_mesh.close()

    # Copy deformed FFD box information from deformed mesh to regenerated mesh.
    cp_ffdb.copy_ffdb(deformed_mesh_filename, output_mesh_filename, \
                      [["cht_hex_fluid_solid","cht_hex_solid_fluid"],\
                       ["cht_burner_fluid_solid","cht_burner_solid_fluid"]])
else:
    print("Quality of deformed mesh sufficient, copying mesh content")
    
    with open("%s_ffd_box.su2" % remesh_header,"w+") as fid:
        fid.writelines(lines_deformed_mesh)
