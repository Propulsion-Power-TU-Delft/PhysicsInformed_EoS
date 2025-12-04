import numpy as np 
import gmsh 
from check_mesh_quality import checkMeshQuality 
import copy_ffdb_n as cp_ffdb
scale = 1.0

x_inlet = 0.0
y_ref = 0.0
h_cocenter_inlet = scale * 7e-4
h_center_inlet = scale * 7e-4 
t_inlet_edges = scale * 1e-4
l_inlet_edges = scale * 1e-3 
t_bl = scale * 5e-5
x_center_burner  = x_inlet + scale * 3e-3
r_burner = scale * 5e-4
x_center_hex = scale *6e-3
r_hex = r_burner
r_hex_inner = scale *2.5e-4
x_outlet = scale * 1.2e-2

mesh_size_max = 4e-5 
mesh_size_min = 1.5e-5

deformed_mesh_filename = "mesh_out.su2"
remesh_header = "remesh"


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

def MeshZone(zone_id:int=0):
    mesh_fluid = (zone_id==0)
    mesh_burner = (zone_id ==1)
    mesh_hex = (zone_id ==2)

    
    
    gmsh.initialize() 
    gmsh.model.add("CHT_burner_mesh")
    gmsh.option.setNumber("Mesh.SaveAll", 0)
    factory = gmsh.model.geo
    mesher = gmsh.model.mesh


    inlet_center_pt_1 = factory.addPoint(x_inlet, 0.0,0)
    inlet_center_pt_2 = factory.addPoint(x_inlet, 0.5*h_center_inlet,0)
    inlet_upper_cocenter_pt_1 = factory.addPoint(x_inlet, 0.5*h_center_inlet + t_inlet_edges,0)
    inlet_upper_cocenter_pt_2 = factory.addPoint(x_inlet, 0.5*h_center_inlet + t_inlet_edges + 0.5*h_cocenter_inlet,0)

    upper_cocenter_line = factory.addLine(inlet_upper_cocenter_pt_1, inlet_upper_cocenter_pt_2)
    center_inlet_line = factory.addLine(inlet_center_pt_1, inlet_center_pt_2)

    upper_mixing_edge_pts = [inlet_center_pt_2, factory.addPoint(x_inlet+l_inlet_edges, 0.5*h_center_inlet,0),factory.addPoint(x_inlet+l_inlet_edges, 0.5*h_center_inlet + t_inlet_edges,0),inlet_upper_cocenter_pt_1]
    upper_mixing_edge_center_pt = factory.addPoint(x_inlet+l_inlet_edges, 0.5*h_center_inlet + 0.5*t_inlet_edges,0)
    upper_mixing_edge_lines = [factory.addLine(upper_mixing_edge_pts[0], upper_mixing_edge_pts[1]), factory.addCircleArc(upper_mixing_edge_pts[1], upper_mixing_edge_center_pt, upper_mixing_edge_pts[2]), factory.addLine(upper_mixing_edge_pts[2],upper_mixing_edge_pts[3])]

    # upper_burner_pts = [factory.addPoint(x_center_burner - r_burner, 0.5*h_center_inlet + t_inlet_edges + 0.5*h_cocenter_inlet,0),\
    #                         factory.addPoint(x_center_burner, 0.5*h_center_inlet + t_inlet_edges + 0.5*h_cocenter_inlet,0),\
    #                         factory.addPoint(x_center_burner + r_burner, 0.5*h_center_inlet + t_inlet_edges + 0.5*h_cocenter_inlet,0)]
    # upper_burner_bl_pts = [factory.addPoint(x_center_burner - r_burner-t_bl,0.5*h_center_inlet + t_inlet_edges + 0.5*h_cocenter_inlet,0), \
    #                         upper_burner_pts[1], \
    #                         factory.addPoint(x_center_burner + r_burner+t_bl,0.5*h_center_inlet + t_inlet_edges + 0.5*h_cocenter_inlet,0)]

    y_center_burner = 0.5*h_center_inlet + t_inlet_edges + 0.5*h_cocenter_inlet
    x_burner_pts_centered = xy_surf_deformed_burner[:,0] - x_center_burner
    y_burner_pts_centered = xy_surf_deformed_burner[:,1] - y_center_burner
    theta_burner_pts = np.arctan2(y_burner_pts_centered, x_burner_pts_centered)
    x_burner_bl_pts = xy_surf_deformed_burner[:,0] + t_bl * np.cos(theta_burner_pts)
    y_burner_bl_pts = xy_surf_deformed_burner[:,1] + t_bl * np.sin(theta_burner_pts)
    
    upper_burner_pts = []
    upper_burner_bl_pts = []
    for i in range(len(xy_surf_deformed_burner)):
        upper_burner_pts.append(factory.addPoint(xy_surf_deformed_burner[i,0],xy_surf_deformed_burner[i,1],0))
        upper_burner_bl_pts.append(factory.addPoint(x_burner_bl_pts[i],y_burner_bl_pts[i],0))
    upper_burner_line = factory.addSpline(upper_burner_pts)
    upper_burner_bl_line = factory.addSpline(upper_burner_bl_pts)

    y_center_hex = 0.0
    x_hex_pts_centered = xy_surf_deformed_hex[:,0] - x_center_hex
    y_hex_pts_centered = xy_surf_deformed_hex[:,1] - y_center_hex
    theta_hex_pts = np.arctan2(y_hex_pts_centered, x_hex_pts_centered)
    x_hex_bl_pts = xy_surf_deformed_hex[:,0] + t_bl * np.cos(theta_hex_pts)
    y_hex_bl_pts = xy_surf_deformed_hex[:,1] + t_bl * np.sin(theta_hex_pts)
    
    hex_pts = []
    hex_bl_pts = []
    for i in range(len(xy_surf_deformed_hex)):
        hex_pts.append(factory.addPoint(xy_surf_deformed_hex[i,0],xy_surf_deformed_hex[i,1],0))
        hex_bl_pts.append(factory.addPoint(x_hex_bl_pts[i],y_hex_bl_pts[i],0))
    hex_upper_curve = factory.addSpline(hex_pts)
    hex_bl_upper_curve = factory.addSpline(hex_bl_pts)

    

    #upper_burner_line = factory.addCircleArc(upper_burner_pts[0],upper_burner_pts[1],upper_burner_pts[2])
    #upper_burner_bl_line = factory.addCircleArc(upper_burner_bl_pts[0],upper_burner_bl_pts[1],upper_burner_bl_pts[2])
    upper_burner_bl_connecting_lines = [factory.addLine(upper_burner_bl_pts[0], upper_burner_pts[0]),factory.addLine(upper_burner_bl_pts[-1], upper_burner_pts[-1])]
    
    upper_burner_bl_curvloop = factory.addCurveLoop([upper_burner_line, -upper_burner_bl_connecting_lines[1], -upper_burner_bl_line, upper_burner_bl_connecting_lines[0]])
    upper_burner_bl_plane = factory.addPlaneSurface([upper_burner_bl_curvloop])
    
    upper_burner_center_pt = factory.addPoint(x_center_burner, y_center_burner, 0)
    upper_burner_inner_pts = [factory.addPoint(x_center_burner - r_hex_inner, 0.5*h_center_inlet + t_inlet_edges + 0.5*h_cocenter_inlet, 0.0), upper_burner_center_pt, factory.addPoint(x_center_burner + r_hex_inner, 0.5*h_center_inlet + t_inlet_edges + 0.5*h_cocenter_inlet, 0.0)]
    upper_burner_inner_curve = factory.addCircleArc(upper_burner_inner_pts[0], upper_burner_inner_pts[1], upper_burner_inner_pts[2])
    upper_burner_inner_connecting_lines = [factory.addLine(upper_burner_pts[0], upper_burner_inner_pts[2]),\
                                        factory.addLine(upper_burner_pts[-1], upper_burner_inner_pts[0])]
    
    upper_burner_curvloop = factory.addCurveLoop([upper_burner_line, upper_burner_inner_connecting_lines[1], upper_burner_inner_curve, -upper_burner_inner_connecting_lines[0]])
    upper_burner_solid = factory.addPlaneSurface([upper_burner_curvloop])
    
    #hex_pts = [factory.addPoint(x_center_hex - r_hex, 0.0,0),factory.addPoint(x_center_hex, 0.0,0),factory.addPoint(x_center_hex+r_hex, 0.0,0)]
    #hex_bl_pts = [factory.addPoint(x_center_hex - r_hex - t_bl, 0.0,0),hex_pts[1],factory.addPoint(x_center_hex+r_hex + t_bl, 0.0,0)]
    hex_center_pt = factory.addPoint(x_center_hex, 0, 0)
    hex_inner_pts = [factory.addPoint(x_center_hex - r_hex_inner, 0.0,0),hex_center_pt,factory.addPoint(x_center_hex+r_hex_inner, 0.0,0)]

    #hex_upper_curve = factory.addCircleArc(hex_pts[2],hex_pts[1],hex_pts[0])
    #hex_bl_upper_curve = factory.addCircleArc(hex_bl_pts[2],hex_pts[1],hex_bl_pts[0])
    hex_inner_upper_curve = factory.addCircleArc(hex_inner_pts[2], hex_center_pt, hex_inner_pts[0])
    hex_inner_connecting_lines = [factory.addLine(hex_pts[0], hex_inner_pts[2]), factory.addLine(hex_pts[-1], hex_inner_pts[0])]
    
    hex_bl_connecting_lines = [factory.addLine(hex_bl_pts[-1], hex_pts[-1]),factory.addLine(hex_bl_pts[0], hex_pts[0])]
    
    hex_bl_curvloops = [factory.addCurveLoop([hex_upper_curve, -hex_bl_connecting_lines[0], -hex_bl_upper_curve, hex_bl_connecting_lines[1]])]
    hex_bl_planes = [factory.addPlaneSurface([hex_bl_curvloops[0]])]
    
    hex_curvloop = factory.addCurveLoop([hex_upper_curve, hex_inner_connecting_lines[1], -hex_inner_upper_curve, -hex_inner_connecting_lines[0]])
    hex_solid = factory.addPlaneSurface([hex_curvloop])
    
    

    outlet_pts = [factory.addPoint(x_outlet, 0,0),factory.addPoint(x_outlet, 0.5*h_center_inlet + t_inlet_edges + 0.5*h_cocenter_inlet,0)]
    outlet_line = factory.addLine(outlet_pts[0],outlet_pts[1])

    side_lines = [factory.addLine(inlet_center_pt_1, hex_bl_pts[-1]), \
                factory.addLine(hex_bl_pts[0], outlet_pts[0]), \
                factory.addLine(outlet_pts[1], upper_burner_bl_pts[0]), \
                factory.addLine(upper_burner_bl_pts[-1], inlet_upper_cocenter_pt_2)]
    pt_refinement = factory.addPoint(0.5*(x_center_burner+x_center_hex), 0.0, 0)
    
    fluid_curvloop = factory.addCurveLoop([center_inlet_line, upper_mixing_edge_lines[0],upper_mixing_edge_lines[1],upper_mixing_edge_lines[2], upper_cocenter_line, \
                                        -side_lines[-1], -upper_burner_bl_line, -side_lines[-2], -outlet_line, -side_lines[-3],\
                                            hex_bl_upper_curve, -side_lines[0]])
    fluid_plane = factory.addPlaneSurface([fluid_curvloop])
    
    
    # fluid domain 
    if mesh_fluid:
        factory.addPhysicalGroup(1, [upper_cocenter_line], name="inlet_cocenter")
        factory.addPhysicalGroup(1, [center_inlet_line], name="inlet_center")
        factory.addPhysicalGroup(1, upper_mixing_edge_lines + side_lines + upper_burner_bl_connecting_lines + hex_bl_connecting_lines, name="symmetry")
        factory.addPhysicalGroup(1, [outlet_line],name="outlet")
        factory.addPhysicalGroup(1, [upper_burner_line], name="cht_burner_fluid_solid")
        factory.addPhysicalGroup(1, [hex_upper_curve],name="cht_hex_fluid_solid")
        factory.addPhysicalGroup(2, [fluid_plane] + hex_bl_planes + [upper_burner_bl_plane],name="fluid")

    # solid domain
    if mesh_burner:
        factory.addPhysicalGroup(1, [upper_burner_line], name="cht_burner_solid_fluid")
        factory.addPhysicalGroup(1, upper_burner_inner_connecting_lines, name="burner_sym")
        factory.addPhysicalGroup(1, [upper_burner_inner_curve],name="inside_burner")
        factory.addPhysicalGroup(2, [upper_burner_solid], name="solid_1")

    if mesh_hex:
            factory.addPhysicalGroup(1, [hex_inner_upper_curve],name="inside_hex")
            factory.addPhysicalGroup(1, hex_inner_connecting_lines, name="hex_sym")
            factory.addPhysicalGroup(1, [hex_upper_curve],name="cht_hex_solid_fluid")
            factory.addPhysicalGroup(2, [hex_solid],name='solid_2')
    factory.synchronize()
    
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size_max)

    dist_field = mesher.field.add("Distance")
    mesher.field.setNumbers(dist_field, "PointsList", [pt_refinement])
    mesher.field.setNumber(dist_field, "Sampling", 150)
    threshold_field = mesher.field.add("Threshold")
    mesher.field.setNumber(threshold_field, "InField", dist_field)
    mesher.field.setNumber(threshold_field, "SizeMin", mesh_size_min)
    mesher.field.setNumber(threshold_field, "SizeMax", mesh_size_max)

    mesher.field.setNumber(threshold_field, "DistMin", 2*r_burner)
    mesher.field.setNumber(threshold_field, "DistMax", 3*r_burner)
    # out_field = mesher.field.add("Min")
    # mesher.field.setNumbers(out_field, "FieldsList", [threshold_field])
    # mesher.field.setAsBackgroundMesh(out_field)

    dist_field2 = mesher.field.add("Distance")
    mesher.field.setNumbers(dist_field2, "PointsList", [upper_mixing_edge_pts[2]])
    mesher.field.setNumber(dist_field2, "Sampling", 150)
    threshold_field2 = mesher.field.add("Threshold")
    mesher.field.setNumber(threshold_field2, "InField", dist_field2)
    mesher.field.setNumber(threshold_field2, "SizeMin", mesh_size_min)
    mesher.field.setNumber(threshold_field2, "SizeMax", mesh_size_max)

    mesher.field.setNumber(threshold_field2, "DistMin", 2*t_inlet_edges)
    mesher.field.setNumber(threshold_field2, "DistMax", 3*t_inlet_edges)

    out_field = mesher.field.add("Min")
    mesher.field.setNumbers(out_field, "FieldsList", [threshold_field, threshold_field2])
    mesher.field.setAsBackgroundMesh(out_field)


    Np_rim = 150#int(np.pi*2*r_hex / mesh_size_min)#100
    Np_bl = 20#7*int(t_bl / mesh_size_min)#15
    coef_bl = 0.9
    mesher.setTransfiniteCurve(upper_burner_line, Np_rim, coef=1.0)
    mesher.setTransfiniteCurve(upper_burner_bl_line, Np_rim, coef=1.0)
    mesher.setTransfiniteCurve(upper_burner_bl_connecting_lines[0], Np_bl, coef=coef_bl)
    mesher.setTransfiniteCurve(upper_burner_bl_connecting_lines[1], Np_bl, coef=coef_bl)
    mesher.setTransfiniteSurface(upper_burner_bl_plane, cornerTags=[upper_burner_bl_pts[0], upper_burner_pts[0], upper_burner_pts[-1], upper_burner_bl_pts[-1]])
    mesher.setRecombine(2, upper_burner_bl_plane)

    mesher.setTransfiniteCurve(hex_upper_curve, Np_rim, coef=1.0)
    mesher.setTransfiniteCurve(hex_inner_upper_curve, Np_rim, coef=1.0)
    mesher.setTransfiniteCurve(upper_burner_inner_curve, Np_rim, coef=1.0)
    mesher.setTransfiniteCurve(hex_bl_upper_curve, Np_rim, coef=1.0)
    mesher.setTransfiniteCurve(hex_bl_connecting_lines[0], Np_bl,coef=coef_bl)
    mesher.setTransfiniteCurve(hex_bl_connecting_lines[1], Np_bl,coef=coef_bl)
    mesher.setTransfiniteSurface(hex_bl_planes[0], cornerTags=[hex_bl_pts[0], hex_pts[0], hex_pts[-1], hex_bl_pts[-1]])
    mesher.setRecombine(2, hex_bl_planes[0])
    mesher.setTransfiniteCurve(upper_burner_inner_connecting_lines[0], 27,coef=1.0)
    mesher.setTransfiniteCurve(upper_burner_inner_connecting_lines[1], 27,coef=1.0)
    mesher.setTransfiniteCurve(hex_inner_connecting_lines[0], 27,coef=1.0)
    mesher.setTransfiniteCurve(hex_inner_connecting_lines[1], 27,coef=1.0)

    mesher.generate(2)
    if mesh_fluid:
        gmsh.write("fluid_mesh.su2")
    if mesh_burner:
        gmsh.write("burner_mesh.su2")
    if mesh_hex:
        gmsh.write("hex_mesh.su2")
    gmsh.finalize()
    return 

# Check which zones need remeshing based on the maximum cell skewedness.
remesh_zones = [False, False, False]
for iZone in range(3):
    q = max(checkMeshQuality(deformed_mesh_filename, iZone))
    if q >= 0.65:
        remesh_zones[iZone] = True 

# Read information from deformed mesh.
with open(deformed_mesh_filename,'r') as fid:
    lines_deformed_mesh = fid.readlines()

if any(remesh_zones):
    print("Quality of deformed mesh insufficient, remeshing")

    # Regenerate mesh in zones with bad quality.
    for iZone, remesh in enumerate(remesh_zones):
        if remesh:
            MeshZone(iZone)
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
