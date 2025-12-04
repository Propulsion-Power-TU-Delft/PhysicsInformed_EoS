import numpy as np
import gmsh 
from scipy.optimize import minimize_scalar


#---------------------------------------------------------------------------------------------#
# Importing ParaBlade classes and functions
#---------------------------------------------------------------------------------------------#
from common.common_utils import *
from common.config_utils import *
from src.blade_geom_2D import BladeGeom2D

class BladeMesh2D:
    design_params:dict = {}
    parablade_cfg_file:str = None 
    blade_geom:BladeGeom2D = None 
    n_coords = 301
    u_blade:np.ndarray[float]=None
    n_blades:int = None 
    mesh_size_max = 1.5e-4
    mesh_size_min = 1e-4
    delta_r_inlet = 1e-2
    delta_r_outlet = 5e-3
    fac_mesh_size_max:float = 20
    t_bl = 1e-4
    XY_blade:np.ndarray[float]=None 
    XY_perio_1:np.ndarray[float] = None 
    XY_perio_2:np.ndarray[float] = None 
    mesh_filename:str="ORCHID_mesh.su2"
    def __init__(self, blade_in_file:str):
        self.parablade_cfg_file = blade_in_file
        self.GenerateBladeGeom()
        return 
    
    def SetMeshSizeMax(self, ds_max:float):
        self.mesh_size_max = ds_max 
        return 
    def SetMeshSizeMin(self, ds_min:float):
        self.mesh_size_min = ds_min 
        return 
    def SetBL_Thickness(self, t_bl:float):
        self.t_bl = t_bl 
        return 
    def SetNp_Blade(self, np_blade:int):
        self.n_coords = np_blade
        return 
    def SetMeshFileName(self, filename:str):
        self.mesh_filename=filename 
        return 
    def GetPeriodicAngle(self):
        return (180 / np.pi) * self.delta_theta

    def GenerateBladeGeom(self):
        IN = read_user_input(self.parablade_cfg_file)
        self.blade_geom = BladeGeom2D(IN)
        self.blade_geom.make_blade()
        self.n_blades = int(IN["N_BLADES"][0])
        self.u_blade = np.linspace(0, 1, self.n_coords)
        self.XY_blade = self.blade_geom.get_coordinates(self.u_blade).T
        self.MakePeriodicBoundary()
        return 
    
    def UpdateBladeGeom(self, varname:str, value_new:float, idx:int=0):
        self.blade_geom.IN[varname][idx] = value_new
        self.blade_geom.update_DVs_values(self.blade_geom.IN)
        return 
    def GetParamValue(self, varname:str):
        return self.blade_geom.IN[varname]
    def GetBladeSensitivity(self):
        sens = self.blade_geom.get_sensitivity(self.u_blade)
        return sens
    def MakePeriodicBoundary(self):
        XY_blade = self.blade_geom.get_coordinates(self.u_blade).T
        self.delta_theta = 2*np.pi / self.n_blades
        r_blade =np.linalg.norm(XY_blade,axis=1)
        r_max = max(r_blade) + self.delta_r_inlet
        r_min =min(r_blade) - self.delta_r_outlet
        theta_blade = np.arctan2(XY_blade[:,1],XY_blade[:,0])
        rtheta_blade = r_blade * theta_blade
        def getdist(rtheta, r):
            dist_1 = np.power((rtheta_blade - rtheta), 2) + np.power((r_blade - r), 2)
            dist_2 = np.power((rtheta_blade - rtheta + r * self.delta_theta), 2) + np.power((r_blade - r), 2)

            return (np.abs(np.min(dist_1) - np.min(dist_2)))

        r_range = np.linspace(r_min, r_max,200)
        rtheta_perio = np.zeros(np.shape(r_range))
        rtheta_at_r = rtheta_blade[np.argmin(np.power(r_min - r_blade, 2))]
        bounds = (rtheta_at_r - r_min*self.delta_theta, rtheta_at_r + r_min*self.delta_theta)

        res = minimize_scalar(getdist, bounds=bounds, args=(r_min),method='bounded')
        rtheta_perio[0] = res.x
        rtheta_max = np.max(rtheta_blade)
        rtheta_min = np.min(rtheta_blade)
        delta_rtheta = 0.05*np.abs(rtheta_max - rtheta_min)
        for ir, r in enumerate(r_range[1:]):
            bounds = (rtheta_perio[ir] - delta_rtheta, rtheta_perio[ir] + delta_rtheta)
            res = minimize_scalar(getdist, bounds=bounds, args=(r),method='bounded')
            rtheta_perio[ir+1] = res.x


        theta_perio_1 = rtheta_perio / r_range 
        theta_perio_2 = theta_perio_1 - self.delta_theta
        x_perio_1 = r_range * np.cos(theta_perio_1)
        y_perio_1 = r_range  * np.sin(theta_perio_1)

        x_perio_2 = r_range * np.cos(theta_perio_2)
        y_perio_2 = r_range  * np.sin(theta_perio_2)
        
        self.XY_perio_1 = np.vstack((x_perio_1, y_perio_1)).T
        self.XY_perio_2 = np.vstack((x_perio_2, y_perio_2)).T
        return 
    
    def make_mesh(self):
        self.blade_geom.make_blade()
        self.n_blades = int(self.blade_geom.IN["N_BLADES"][0])
        self.u_blade = np.linspace(0, 1, self.n_coords)
        u_TE = np.linspace(0.4, 0.6, 100)
        u_1 = np.linspace(0.0, u_TE[0], int(0.5*self.n_coords)+1)
        u_2 = np.linspace(u_TE[-1], 1.0,int(0.5*self.n_coords)+1)
        self.u_blade = np.hstack((u_1[:-1], u_TE, u_2[1:]))
        self.XY_blade = self.blade_geom.get_coordinates(self.u_blade).T

        lengths_sections = np.linalg.norm(self.XY_blade[1:, :] - self.XY_blade[:-1],axis=1)
        ds_min = min(lengths_sections)
        self.mesh_size_min = ds_min
        self.mesh_size_max = self.fac_mesh_size_max * self.mesh_size_min
        self.t_bl = min(lengths_sections)
        self.MakePeriodicBoundary()
        #self.GenerateBladeGeom()
        norm_blade = self.blade_geom.get_normals(self.u_blade).T
        XY_bl = self.XY_blade - self.t_bl*norm_blade 
        theta_points = np.arctan2(self.XY_blade[:,1], self.XY_blade[:,0])
        gmsh.initialize()
        gmsh.option.setNumber("Mesh.SaveAll", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMax", self.mesh_size_max)
        #gmsh.option.setNumber("Mesh.Algorithm", 8)
        gmsh.model.add("FLUID")
        factory = gmsh.model.geo
        mesher = gmsh.model.mesh

        iPoints_wall = []
        iPoints_bl = []
        lengths_sections = []

        for i in range(len(self.XY_blade)-1):
            j = (i + 1)
            iPoints_wall.append(factory.addPoint(self.XY_blade[i, 0],self.XY_blade[i, 1],0))
            iPoints_bl.append(factory.addPoint(XY_bl[i, 0],XY_bl[i, 1],0))
        iPoints_wall.append(iPoints_wall[0])
        iPoints_bl.append(iPoints_bl[0])
        bl_connecting_curves = []
        bl_curves = []
        wall_curves = []

        np_per_section = 10
        wall_pts_sections = [iPoints_wall[i:(i+np_per_section)] for i in range(0, len(iPoints_wall), np_per_section-1)]
        bl_pts_sections = [iPoints_bl[i:(i+np_per_section)] for i in range(0, len(iPoints_wall), np_per_section-1)]
        delta_xy = self.XY_blade[1:, :] - self.XY_blade[:-1, :]
        ds = np.linalg.norm(delta_xy, axis=1)
        lengths_sections = [np.sum(ds[i:(i+np_per_section)])for i in range(0, len(iPoints_wall), np_per_section-1)]
        
        for w, b in zip(wall_pts_sections, bl_pts_sections):
            
            bl_curves.append(factory.addSpline(b))
            wall_curves.append(factory.addSpline(w))
            bl_connecting_line = factory.addLine(w[0], b[0])
            bl_connecting_curves.append(bl_connecting_line)
        bl_curvloops = []
        bl_surfs = []
        
        for iPoint in range(len(wall_curves)):
            jPoint = (iPoint+1) % len(wall_curves)
            bl_curvloops.append(factory.addCurveLoop([-wall_curves[iPoint], -bl_connecting_curves[jPoint], bl_curves[iPoint], bl_connecting_curves[iPoint]]))
        

        for c in bl_curvloops:
            bl_surfs.append(factory.addPlaneSurface([c]))

        
        iPoint_rotationaxis = factory.addPoint(0,0,0)

        iPoints_perio_1 = []
        iPoints_perio_2 = []
        for i in range(len(self.XY_perio_1)):
            iPoints_perio_1.append(factory.addPoint(self.XY_perio_1[i,0],self.XY_perio_1[i,1],0))
            iPoints_perio_2.append(factory.addPoint(self.XY_perio_2[i,0],self.XY_perio_2[i,1],0))

        periodic_curve_1 = factory.addSpline(iPoints_perio_1)
        periodic_curve_2 = factory.addSpline(iPoints_perio_2)
        outlet_curve = factory.addCircleArc(iPoints_perio_1[0],iPoint_rotationaxis,iPoints_perio_2[0])
        inlet_curve = factory.addCircleArc(iPoints_perio_1[-1],iPoint_rotationaxis,iPoints_perio_2[-1])

        blade_curvloop = factory.addCurveLoop([c for c in bl_curves])
        fluid_curvloop = factory.addCurveLoop([-inlet_curve, periodic_curve_2, outlet_curve, -periodic_curve_1])
        fluid_domain = factory.addPlaneSurface([fluid_curvloop, -blade_curvloop])

        factory.addPhysicalGroup(2, [fluid_domain] + [b for b in bl_surfs],name="fluid")
        factory.addPhysicalGroup(1, [inlet_curve],name="inflow")
        factory.addPhysicalGroup(1, [outlet_curve],name="outflow")
        factory.addPhysicalGroup(1, [w for w in wall_curves],name="wall")
        factory.addPhysicalGroup(1, [periodic_curve_1], name="periodic2")
        factory.addPhysicalGroup(1, [periodic_curve_2], name="periodic1")

        ghost_ref_pts = self.apply_refinement("mach_grad_contours.csv",factory)
        factory.synchronize()

        

        dist_field_2 = mesher.field.add("Distance")
        mesher.field.setNumbers(dist_field_2, "PointsList", ghost_ref_pts)
        threshold_field_2 = mesher.field.add("Threshold")
        mesher.field.setNumber(threshold_field_2, "InField", dist_field_2)
        mesher.field.setNumber(threshold_field_2, "SizeMin", self.mesh_size_min)
        mesher.field.setNumber(threshold_field_2, "SizeMax", self.mesh_size_max)


        mesher.field.setNumber(threshold_field_2, "DistMin", 1.5e-4)
        mesher.field.setNumber(threshold_field_2, "DistMax", 2e-3)

        dist_field_perio = mesher.field.add("Distance")
        mesher.field.setNumbers(dist_field_perio, "PointsList", iPoints_perio_1 + iPoints_perio_2)
        threshold_field_perio = mesher.field.add("Threshold")
        mesher.field.setNumber(threshold_field_perio, "InField", dist_field_perio)
        mesher.field.setNumber(threshold_field_perio, "SizeMin", 0.3*self.mesh_size_max)
        mesher.field.setNumber(threshold_field_perio, "SizeMax", self.mesh_size_max)
        mesher.field.setNumber(threshold_field_perio, "DistMin", 1.5e-4)
        mesher.field.setNumber(threshold_field_perio, "DistMax", 5e-4)

        out_field = mesher.field.add("Min")
        mesher.field.setNumbers(out_field, "FieldsList", [threshold_field_perio, threshold_field_2])
        mesher.field.setAsBackgroundMesh(out_field)


        for i_segment in range(len(wall_curves)):
            l_segment = lengths_sections[i_segment]
            numNodes = int(l_segment / self.mesh_size_min)+1
            w = wall_curves[i_segment]
            bl = bl_curves[i_segment]
            mesher.setTransfiniteCurve(w, numNodes=numNodes)
            mesher.setTransfiniteCurve(bl, numNodes=numNodes)
        for c in bl_connecting_curves:
            mesher.setTransfiniteCurve(c, numNodes=20, coef=1.2)

        for iPoint, b in enumerate(bl_surfs):
            jPoint = (iPoint + 1) % len(wall_pts_sections)
            mesher.setTransfiniteSurface(b, cornerTags=[wall_pts_sections[iPoint][0], wall_pts_sections[jPoint][0], bl_pts_sections[jPoint][0], bl_pts_sections[iPoint][0]])
            mesher.setRecombine(2, b)

        mesher.setRecombine(2, fluid_domain)
        gmsh.option.setNumber("Mesh.Algorithm", 8)
        mesher.generate(2)
        gmsh.fltk.run()
        gmsh.write(self.mesh_filename)
        gmsh.finalize()
        return 

    def apply_refinement(self, refinement_pts_file:str, factory:gmsh.model.geo):
        xyz_ref_pts = np.loadtxt(refinement_pts_file,delimiter=',',skiprows=1)[::3, :2]
        
        ref_pts = []
        for i in range(len(xyz_ref_pts)):
            pt = factory.addPoint(xyz_ref_pts[i,0], xyz_ref_pts[i,1],0.0)
            ref_pts.append(pt)
        
        ghost_ref_pts = ref_pts.copy()
        for i in range(1,self.n_blades):
            bl_points_plus_delta = factory.copy([(0, iPoint) for iPoint in ref_pts])
            factory.rotate(bl_points_plus_delta, 0,0,0,0,0,1, i*self.delta_theta)
            ghost_ref_pts += [b[1] for b in bl_points_plus_delta]
        return ghost_ref_pts

b = BladeMesh2D("ORCHID_stator_base_ParaBlade.cfg")
b.make_mesh()
