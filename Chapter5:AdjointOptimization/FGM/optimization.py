#!/usr/bin/env python3

############################### FILE NAME: optimization.py ####################################
#=============================================================================================#
# author: Evert Bunschoten                                                                    |
#    :PhD Candidate ,                                                                         |
#    :Flight Power and Propulsion                                                             |
#    :TU Delft,                                                                               |
#    :The Netherlands                                                                         |
#                                                                                             |
#                                                                                             |
# Description:                                                                                |
# Shape optimization of a partially premixed hydrogen burner with an equality constraint      |
# applied to the outflow temperature.                                                         |
#                                                                                             |  
#                                                                                             |
#=============================================================================================#
from FADO import *

# Design variables ----------------------------------------------------- #

nDV = 58
dx = 1e-8 
lb = np.ones(nDV)
ub = np.ones(nDV)
for i in range(nDV):
    lb[i] = -1.5e-4
    ub[i] = 1.5e-4
ffd = InputVariable(0.0,PreStringHandler("DV_VALUE="),size=nDV, lb=-1.5e-4, ub=1.5e-4)
OBJFUNC_NAMES=["avg_p_inlet_scaled", "avg_temp_outlet_scaled"]
OBJFUNC_NAME="avg_p_inlet"
DV_NAMES=["avg_p_inlet", "avg_temp_outlet"]
DV_NAME="avg_p_inlet"
DV_file_extensions = ["custom", "custom"]
DV_file_extension="custom"
# Parameters ----------------------------------------------------------- #

# switch from direct to adjoint mode and adapt settings.
enable_direct = Parameter([""], LabelReplacer("%__DIRECT__"))
enable_adjoint = Parameter([""], LabelReplacer("%__ADJOINT__"))

# Set convergence targets and objective functions
obj_func_setters = []
for f in OBJFUNC_NAMES:
   obj_func_setters.append(Parameter([f], LabelReplacer("__OBJ_FUNCTION_NAME__")))
conv_target_direct = Parameter(["-12.43"], LabelReplacer("__CONV_TARGET__"))
adj_conv_targets = [Parameter(["-13.2"],LabelReplacer("__CONV_TARGET__")),\
                    Parameter(["-11.2"],LabelReplacer("__CONV_TARGET__"))]

# SU2 configuration file names and mesh file name
ncores=12
configMaster="master.cfg"
config_files = ["master.cfg", "fluid.cfg", "solid_burner.cfg","solid_hex.cfg"]
meshName="mesh_ffd_box.su2"

# SU2 commands for each step in the gradient evaluation process

def_command = "SU2_DEF %s " % configMaster
cfd_command = "mpirun -n %i SU2_CFD %s" % (ncores, configMaster)

cfd_ad_command = "mpirun -n %i SU2_CFD_AD %s" % (ncores, configMaster)
dot_ad_command = "SU2_DOT_AD %s" % configMaster

max_tries = 1

# Step 1: mesh deformation
deform = ExternalRun("DEFORM",def_command,True)
deform.setMaxTries(max_tries)
for c in config_files:
   deform.addConfig(c)
deform.addData(meshName)
deform.addExpected("mesh_out.su2")
deform.addParameter(obj_func_setters[0])
deform.addParameter(conv_target_direct)

# Step 2: conditional remeshing around deformed boundaries
remesh = ExternalRun("REMESHING","python ../../remeshing_script.py", True)
remesh.setMaxTries(max_tries)
remesh.addData("DEFORM/mesh_out.su2",destination="mesh_out.su2")
remesh.addData("DEFORM/surface_deformed_1.csv",destination="surface_deformed_1.csv")
remesh.addData("DEFORM/surface_deformed_2.csv",destination="surface_deformed_2.csv")
remesh.addExpected("remesh_ffd_box.su2")

# Step 3: evaluate objective and constraint function values
direct = ExternalRun("DIRECT",cfd_command,True)
direct.setMaxTries(max_tries)
direct.addData("REMESHING/remesh_ffd_box.su2",destination=meshName)
for ic, c in enumerate(config_files):
   direct.addConfig(c)
   direct.addData(("restart_r_%i.dat" % ic), destination=("solution_%i.dat" % ic))
for i in range(1,6):
   direct.addData("MLP_Group%i.mlp" % i)
direct.addData("MLP_NULL.mlp")
direct.addExpected("restart_0.dat")
direct.addExpected("restart_1.dat")
direct.addExpected("restart_2.dat")
direct.addParameter(enable_direct)
direct.addParameter(obj_func_setters[0])
direct.addParameter(conv_target_direct)

# Step 4: adjoint calculations for objective and constraint function
adjoints = []
for f, q, e,t in zip(OBJFUNC_NAMES, obj_func_setters, DV_file_extensions, adj_conv_targets):
  adjoint = ExternalRun("ADJOINT_%s" % f,cfd_ad_command,True)
  adjoint.setMaxTries(max_tries)
  for ic, c in enumerate(config_files):
    adjoint.addConfig(c)
    adjoint.addData(("DIRECT/restart_%i.dat" % ic), destination=("solution_%i.dat" % ic))
  adjoint.addData("REMESHING/remesh_ffd_box.su2", destination=meshName)
  for i in range(1,6):
    adjoint.addData("MLP_Group%i.mlp" % i)
  adjoint.addData("MLP_NULL.mlp")
  adjoint.addData("restart_adj_"+f+"_0.dat",destination="solution_adj_"+DV_file_extension+"_0.dat")
  adjoint.addData("restart_adj_"+f+"_1.dat",destination="solution_adj_"+DV_file_extension+"_1.dat")
  adjoint.addData("restart_adj_"+f+"_2.dat",destination="solution_adj_"+DV_file_extension+"_2.dat")
  adjoint.addExpected("restart_adj_"+DV_file_extension+"_0.dat")
  adjoint.addExpected("restart_adj_"+DV_file_extension+"_1.dat")
  adjoint.addExpected("restart_adj_"+DV_file_extension+"_2.dat")
  adjoint.addParameter(enable_adjoint)
  adjoint.addParameter(q)
  adjoint.addParameter(t)
  adjoints.append(adjoint)
  

# Step 5: project gradients to FFD box nodes
dots = []
norm_grads = []
functions = []
for f, q, e in zip(OBJFUNC_NAMES, obj_func_setters, DV_file_extensions):
  dot = ExternalRun("DOT_%s" % f,dot_ad_command,True)
  dot.setMaxTries(max_tries)
  for ic, c in enumerate(config_files):
    dot.addConfig(c)
    dot.addData(("ADJOINT_%s/restart_adj_%s_%i.dat" % (f, e, ic)), destination=("solution_adj_%s_%i.dat" % (e, ic)))
  dot.addData("REMESHING/remesh_ffd_box.su2", destination=meshName)
  dot.addExpected("of_grad.csv")
  dot.addParameter(q) # necessary for correct file extension
  dot.addParameter(conv_target_direct)
  dots.append(dot)

for f, dot, adjoint in zip(OBJFUNC_NAMES, dots, adjoints):
  # Functions ------------------------------------------------------------ #
  func = Function(f, "DIRECT/history_fluid_0.csv",LabeledTableReader("\"%s\"" % f))
  func.addInputVariable(ffd, "DOT_%s/of_grad.csv" % f,TableReader(None,0,(1,0))) 
  func.addValueEvalStep(deform)
  func.addValueEvalStep(remesh)
  func.addValueEvalStep(direct)
  func.addGradientEvalStep(adjoint)
  func.addGradientEvalStep(dot)
  func.setDefaultValue(1.0)
  functions.append(func)

restart = False
slsqp_hist_file = "hist_slsqp.csv"
# Driver --------------------------------------------------------------- #

# Define objective and constraint function
driver = ScipyDriver()
driver.addObjective("min", functions[OBJFUNC_NAMES.index("avg_p_inlet_scaled")],scale=1e-8)
driver.addEquality(functions[OBJFUNC_NAMES.index("avg_temp_outlet_scaled")],target=1.0,scale=1e-8)

driver.setWorkingDirectory("DOE")
driver.setEvaluationMode(True,2.0)
driver.setStorageMode(True,"DSN_r1_")

# Set up optimization
import scipy.optimize

if restart:
  his = open("doe.csv",'a+',1)
else:
  his = open("doe.csv",'w+',1)

driver.setHistorian(his)
driver.preprocess()

# Read the design variable values of the last successful function evaluation in case of a restart
if restart:
  H = np.loadtxt(slsqp_hist_file,delimiter=',',skiprows=1)
  x = H[-1, 2:]
else:
  x = driver.getInitial()
  with open(slsqp_hist_file,"w+") as fid:
    fid.write("f,h," + ",".join("x_%i" % i for i in range(nDV)) + "\n")


def Callback(x:np.ndarray):
  """Optimization callback function which saves the objective, constraint, and design variable information at each iteration of the optimization

  :param x: design variable values
  :type x: np.ndarray
  """
  print("Evaluating callback...")
  f = driver.fun(x) 
  h = driver.getConstraints()[0]["fun"](x)
  print("Done!")
  H = np.loadtxt(slsqp_hist_file,delimiter=',',skiprows=1)
  n_iter = len(H)
  with open(slsqp_hist_file,"a+") as fid:
    fid.write("%.6e,%.6e,%s\n" % (f, h, ",".join("%+.6e" % s for s in x)))
  
  # Every 10 iterations, update the restart files
  if (n_iter % 10) == 0:
    os.system("cp DOE/DIRECT/restart_0.dat restart_r_0.dat")
    os.system("cp DOE/DIRECT/restart_1.dat restart_r_1.dat")
    os.system("cp DOE/DIRECT/restart_2.dat restart_r_2.dat")
    for f in OBJFUNC_NAMES:
      os.system("cp DOE/ADJOINT_%s/restart_adj_custom_0.dat restart_r_adj_%s_0.dat" % (f, f))
      os.system("cp DOE/ADJOINT_%s/restart_adj_custom_1.dat restart_r_adj_%s_1.dat" % (f, f))
      os.system("cp DOE/ADJOINT_%s/restart_adj_custom_2.dat restart_r_adj_%s_2.dat" % (f, f))
      
  return
 
options = {'disp': True, 'maxiter': 100, 'ftol' : 1e-32}
optimum = scipy.optimize.minimize(driver.fun, x, method="SLSQP", jac=driver.grad,\
          constraints=driver.getConstraints(), bounds=driver.getBounds(), options=options,tol=1e-16,callback=Callback)

x_optim = optimum.x 
jac_optimum = optimum.jac 
lambda_optimum = optimum.multipliers 

np.save("DV_optim", x_optim)
np.save("DV_Jac_optim", jac_optimum)
np.save("lambda_optim", lambda_optimum)
his.close()
