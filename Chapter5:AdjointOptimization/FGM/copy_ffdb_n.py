# +------------------------------------------------------------------------------------------------+
# | FFDB copy script                                                                               |
# |                                                                                                |
# | This script copies an FFD box from one SU2 mesh to another SU2 mesh with no FFD box.           |
# | The mesh file with no FDD box will be overwritten.                                             |
# |                                                                                                |
# | Usage terminal:                                                                                |
# |     $ python copy_ffdb.py <file_name from> <file_name to>                                      |
# |                                                                                                |
# | Usage in python script:                                                                        |
# |     import copy_ffdb as cp_ffdb                                                                |
# |     cp_ffdb.copy_ffdb('<file_name from> <file_name to>', '<file_name from> <file_name to>')    |
# |                                                                                                |
# | Author:      Daniel Mayer                                                                      |
# | Institution: Robert Bosch LLC, North America                                                   |
# |                                                                                                |
# | If you received this script from the original author, you are allowed to USE it.               |
# | Redistribution in unedited or edited form is prohibited without the written consent            |
# | from the original author.                                                                      |
# +------------------------------------------------------------------------------------------------+

import sys
import numpy as np

def copy_ffdb(mesh_file_ffdb, mesh_file_nodes, marker_names:list[list[str]]):

    print('')
    print('Copying FFD info from ' + mesh_file_ffdb + ' to ' + mesh_file_nodes +'...')
    print('')

    # check if output file already includes an ffd box
    with open(mesh_file_nodes, 'r') as in_str:
        for line in in_str:
            if 'FFD_NBOX' in line:
                print('')
                print('  ' + mesh_file_nodes + ' already includes an FFD box. Aborting...')
                print('')
                return
    print('')
    print('  Reading points from mesh file ' + mesh_file_nodes)
    print('')

    with open(mesh_file_nodes, 'r') as in_str:
        lines = in_str.readlines()

    x_zones,y_zones, z_zones,ix_nodes_zones = [],[],[],[]
    with open(mesh_file_nodes, 'r') as in_str:
        lines = in_str.readlines()
        for iline, line in enumerate(lines):
            if "NPOIN" in line:
                n_mesh_points = int(line.strip().replace(" ","").split("=")[-1])
                print('  Points found.')
                print('  Number of points: ' + str(n_mesh_points))
                x_mesh = np.ndarray(shape = (n_mesh_points,1))
                y_mesh = np.ndarray(shape = (n_mesh_points,1))
                z_mesh = np.ndarray(shape = (n_mesh_points,1))
                index_mesh_point = np.ndarray(shape=(n_mesh_points,1), dtype=np.uint64)
                for iPoint in range(n_mesh_points):
                    x_mesh[iPoint] = float(lines[iline + 1 + iPoint].split()[0])
                    y_mesh[iPoint] = float(lines[iline + 1 + iPoint].split()[1])
                    z_mesh[iPoint] = 0.0
                    index_mesh_point[iPoint] = int(lines[iline + 1 + iPoint].split()[2])
                x_zones.append(x_mesh)
                y_zones.append(y_mesh)
                z_zones.append(z_mesh)
                ix_nodes_zones.append(index_mesh_point)
    
    # read markers that should be deformed from target file
    zones_box_markers = []
    x_box_markers, y_box_markers, z_box_markers, ix_box_markers = [],[],[],[]
    for m in marker_names:
        zones_box_markers.append([0]*len(m))
        x_box_markers.append([0] * len(m))
        y_box_markers.append([0] * len(m))
        z_box_markers.append([0] * len(m))
        ix_box_markers.append([0] * len(m))

    with open(mesh_file_nodes, 'r') as in_str:
        lines_remesh = in_str.readlines()
        for iline, line in enumerate(lines_remesh):
            line_stripped = line.strip().replace(" ","")
            if "IZONE" in line_stripped:
                i_zone = int(line_stripped.split("=")[-1])-1
                index_mesh_point = ix_nodes_zones[i_zone]
                x_mesh = x_zones[i_zone]
                y_mesh = y_zones[i_zone]
                z_mesh = z_zones[i_zone]

            for ibox, box_markers in enumerate(marker_names):
                for imarker, marker in enumerate(box_markers):
                    if (marker in line_stripped):
                        zones_box_markers[ibox][imarker] = i_zone
                        next_line = lines_remesh[iline+1].strip().replace(" ","")
                        n_marker_points = int(next_line.split("=")[-1])
                        index_marker_point   = np.ndarray(shape=(n_marker_points,1), dtype=np.uint64)
                        for ip in range(n_marker_points):
                            index_marker_point[ip] = int(lines_remesh[iline + 2 + ip].split()[1])

                        x_marker, y_marker, z_marker, ix_marker = np.ndarray(shape=(n_marker_points,1)),np.ndarray(shape=(n_marker_points,1)),np.ndarray(shape=(n_marker_points,1)),np.ndarray(shape=(n_marker_points,1),dtype=np.uint64)
                        for i_marker_point in range (n_marker_points):

                            ix_in_mesh = np.where(index_mesh_point == index_marker_point[i_marker_point])
                            
                            x_marker[i_marker_point] = x_mesh[ix_in_mesh]
                            y_marker[i_marker_point] = y_mesh[ix_in_mesh]
                            z_marker[i_marker_point] = z_mesh[ix_in_mesh]
                            ix_marker[i_marker_point] = index_marker_point[i_marker_point]
                        x_box_markers[ibox][imarker] = x_marker 
                        y_box_markers[ibox][imarker] = y_marker 
                        z_box_markers[ibox][imarker] = z_marker 
                        ix_box_markers[ibox][imarker] = ix_marker

    n_FFD_box = len(marker_names)
    FFD_box_tags = []
    FFD_level = []
    FFD_degree_I = []
    FFD_degree_J = []
    FFD_parents = []
    FFD_children = []
    n_corner_poins_FFD = []
    corner_points_FFD = []
    n_control_points_FFD = []
    deformed_control_points_FFD = []
    FFD_delta_x = []
    FFD_delta_y = []
    FFD_x_offset = []
    FFD_y_offset = []

    with open(mesh_file_ffdb,'r') as fid:
        lines = fid.readlines()
        iBox = 0
        for iline, line in enumerate(lines):
            line_p = line.strip().replace(" ","")
            if "FFD_TAG" in line_p:
                FFD_box_tags.append(line_p.split("=")[-1])
            if "FFD_LEVEL" in line_p:
                FFD_level.append(int(line_p.split("=")[-1]))
            if "FFD_DEGREE_I" in line_p:
                FFD_degree_I.append(int(line_p.split("=")[-1]))
            if "FFD_DEGREE_J" in line_p:
                FFD_degree_J.append(int(line_p.split("=")[-1]))
            if "FFD_PARENTS" in line_p:
                FFD_parents.append(int(line_p.split("=")[-1]))
            if "FFD_CHILDREN" in line_p:
                FFD_children.append(int(line_p.split("=")[-1]))
            if "FFD_CORNER_POINTS" in line_p:
                n = int(line_p.split("=")[-1])
                n_corner_poins_FFD.append(n)

                FFD_cp_x = []
                FFD_cp_y = []
                FFD_cp_z = []
                for jLine in range(n):
                    line_cp = lines[iline + 1 + jLine].split()
                    x_cp = float(line_cp[0])
                    y_cp = float(line_cp[1])
                    z_cp = (-1**jLine)*0.5 
                    FFD_cp_x.append(x_cp)
                    FFD_cp_y.append(y_cp)
                    FFD_cp_z.append(z_cp)
                ffd_x0 = FFD_cp_x[0]
                ffd_x1 = FFD_cp_x[1]
                ffd_y0 = FFD_cp_y[0]
                ffd_y1 = FFD_cp_y[2]
                delta_x_ffd = ffd_x1 - ffd_x0 
                delta_y_ffd = ffd_y1 - ffd_y0 
                FFD_delta_x.append(delta_x_ffd)
                FFD_delta_y.append(delta_y_ffd)
                FFD_x_offset.append(ffd_x0)
                FFD_y_offset.append(ffd_y0)

                corner_points_FFD.append(lines[iline+1:iline+1+n])

            if "FFD_CONTROL_POINTS" in line_p:
                np_cp = int(line_p.split("=")[-1])
                n_control_points_FFD.append(np_cp)
                deformed_control_points_FFD.append(lines[iline+1:iline+1+np_cp])

    FFD_box_string = "FFD_NBOX= %i\nFFD_NLEVEL=1\n" % n_FFD_box
    
    for iBox in range(n_FFD_box):
        FFD_box_string += "FFD_TAG= %s\n" % FFD_box_tags[iBox]
        FFD_box_string += "FFD_LEVEL= %i\n" % FFD_level[iBox]
        FFD_box_string += "FFD_DEGREE_I= %i\n" % FFD_degree_I[iBox]
        FFD_box_string += "FFD_DEGREE_J= %i\nFFD_BLENDING= BEZIER\n" % FFD_degree_J[iBox]
        FFD_box_string += "FFD_PARENTS= %i\nFFD_CHILDREN= %i\n" % (FFD_parents[iBox],FFD_children[iBox])
        FFD_box_string += "FFD_CORNER_POINTS= %i\n" % n_corner_poins_FFD[iBox]
        FFD_box_string += "".join(cp for cp in corner_points_FFD[iBox])
        FFD_box_string += "FFD_CONTROL_POINTS= %i\n" % n_control_points_FFD[iBox]
        FFD_box_string += "".join(cp for cp in deformed_control_points_FFD[iBox])
        n_surface_points = 0
        for iMarker in range(len(marker_names[iBox])):
            n_surface_points += len(x_box_markers[iBox][iMarker])
        FFD_box_string += "FFD_SURFACE_POINTS= %i\n" % n_surface_points
        for iMarker in range(len(marker_names[iBox])):
            x_marker = x_box_markers[iBox][iMarker]
            y_marker = y_box_markers[iBox][iMarker]
            ix_marker = ix_box_markers[iBox][iMarker]
            x_marker_p = (x_marker - FFD_x_offset[iBox])/FFD_delta_x[iBox]
            y_marker_p = (y_marker - FFD_y_offset[iBox])/FFD_delta_y[iBox]
            z_marker_p = 0.5*np.ones(np.shape(x_marker_p))
            FFD_box_string += "\n".join("%s %i %+.16e %+.16e %+.16e" % (marker_names[iBox][iMarker], ix, xp, yp,zp) for ix,xp,yp,zp in zip(ix_marker, x_marker_p, y_marker_p, z_marker_p))
            FFD_box_string += "\n"
    
    mesh_file_nodes_noext = mesh_file_nodes.split(".")[0]
    with open("%s_ffd_box.su2" % mesh_file_nodes_noext ,"w+") as fid:
        fid.writelines(lines_remesh)
        
        fid.write("\n"+FFD_box_string)


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("")
        print("Usage: python copy_ffdb.py <file_name from> <file_name to>")
        print("")
        sys.exit(1)

    file_name_from = sys.argv[1]
    file_name_to   = sys.argv[2]

    copy_ffdb(file_name_from, file_name_to, [["cht_hex_fluid_solid","cht_hex_solid_fluid"],["cht_burner_fluid_solid","cht_burner_solid_fluid"]])
