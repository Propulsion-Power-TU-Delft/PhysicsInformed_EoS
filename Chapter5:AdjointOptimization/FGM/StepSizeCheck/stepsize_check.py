# FADO script: Finite Differences vs adjoint run
import sys 
import csv
import math
import pandas as pd
import matplotlib.pyplot as plt 
from FADO import *

# Design variables ----------------------------------------------------- #
nDV = 58
dx = 1e-8 
lb = np.ones(nDV)
ub = np.ones(nDV)
for i in range(nDV):
    lb[i] = -1e-4
    ub[i] = 1e-4
ffd = InputVariable(0.0,PreStringHandler("DV_VALUE="),size=nDV, lb=-1e-4, ub=1e-4)
OBJFUNC_NAMES=["avg_p_inlet_scaled", "avg_temp_outlet_scaled"]
OBJFUNC_NAME="avg_p_inlet"
DV_NAMES=["avg_p_inlet", "avg_temp_outlet"]
DV_NAME="avg_p_inlet"
DV_file_extensions = ["custom", "custom"]
DV_file_extension="custom"
# Parameters ----------------------------------------------------------- #

# The master config `configMaster.cfg` serves as an SU2 adjoint regression test.
# For a correct gradient validation we need to exchange some options

# switch from direct to adjoint mode and adapt settings.
enable_direct = Parameter([""], LabelReplacer("%__DIRECT__"))
enable_adjoint = Parameter([""], LabelReplacer("%__ADJOINT__"))
obj_func_setters = []
for f in OBJFUNC_NAMES:
   obj_func_setters.append(Parameter([f], LabelReplacer("__OBJ_FUNCTION_NAME__")))
conv_target_direct = Parameter(["-12.43"], LabelReplacer("__CONV_TARGET__"))
adj_conv_targets = [Parameter(["-13.2"],LabelReplacer("__CONV_TARGET__")),\
                    Parameter(["-11.2"],LabelReplacer("__CONV_TARGET__"))]
# Evaluations ---------------------------------------------------------- #

# The master config `configMaster.cfg` serves as an SU2 adjoint regression test.
# For a correct gradient validation we need to exchange some options

# switch from direct to adjoint mode and adapt settings.
enable_direct = Parameter([""], LabelReplacer("%__DIRECT__"))
enable_adjoint = Parameter([""], LabelReplacer("%__ADJOINT__"))
obj_func_setters = []
for f in OBJFUNC_NAMES:
   obj_func_setters.append(Parameter([f], LabelReplacer("__OBJ_FUNCTION_NAME__")))
conv_target_direct = Parameter(["-12.43"], LabelReplacer("__CONV_TARGET__"))
adj_conv_targets = [Parameter(["-13.2"],LabelReplacer("__CONV_TARGET__")),\
                    Parameter(["-11.2"],LabelReplacer("__CONV_TARGET__"))]
# Evaluations ---------------------------------------------------------- #

# Define a few often used variables
ncores="12"
configMaster="master.cfg"
config_files = ["master.cfg", "fluid.cfg", "solid_burner.cfg","solid_hex.cfg"]
meshName="mesh_ffd_box.su2"

# Note that correct SU2 version needs to be in PATH

def_command = "SU2_DEF " + configMaster
cfd_command = "mpirun -n " + ncores + " SU2_CFD " + configMaster

cfd_ad_command = "mpirun -n " + ncores + " SU2_CFD_AD " + configMaster
dot_ad_command = "SU2_DOT_AD " + configMaster

max_tries = 1

# mesh deformation
deform = ExternalRun("DEFORM",def_command,True) # True means sym links are used for addData
deform.setMaxTries(max_tries)
for c in config_files:
   deform.addConfig(c)
deform.addData(meshName)
deform.addExpected("mesh_out.su2")
deform.addParameter(obj_func_setters[0])
deform.addParameter(conv_target_direct)

# direct run
direct = ExternalRun("DIRECT",cfd_command,True)
direct.setMaxTries(max_tries)
direct.addData("DEFORM/mesh_out.su2",destination=meshName)
for ic, c in enumerate(config_files):
   direct.addConfig(c)
   direct.addData(("restart_%i.dat" % ic), destination=("solution_%i.dat" % ic))
for i in range(1,6):
   direct.addData("MLP_Group%i.mlp" % i)
direct.addData("MLP_NULL.mlp")
direct.addExpected("restart_0.dat")
direct.addExpected("restart_1.dat")
direct.addExpected("restart_2.dat")
direct.addParameter(enable_direct)
direct.addParameter(obj_func_setters[0])
direct.addParameter(conv_target_direct)

# adjoint run
adjoints = []
for f, q, e,t in zip(OBJFUNC_NAMES, obj_func_setters, DV_file_extensions, adj_conv_targets):
  adjoint = ExternalRun("ADJOINT_%s" % f,cfd_ad_command,True)
  adjoint.setMaxTries(max_tries)
  for ic, c in enumerate(config_files):
    adjoint.addConfig(c)
    adjoint.addData(("DIRECT/restart_%i.dat" % ic), destination=("solution_%i.dat" % ic))
  adjoint.addData("DEFORM/mesh_out.su2", destination=meshName)
  for i in range(1,6):
    adjoint.addData("MLP_Group%i.mlp" % i)
  adjoint.addData("MLP_NULL.mlp")
  adjoint.addExpected("restart_adj_"+DV_file_extension+"_0.dat")
  adjoint.addExpected("restart_adj_"+DV_file_extension+"_1.dat")
  adjoint.addExpected("restart_adj_"+DV_file_extension+"_2.dat")
  adjoint.addParameter(enable_adjoint)
  adjoint.addParameter(q)
  adjoint.addParameter(t)

  adjoints.append(adjoint)
  

# gradient projection
dots = []
functions = []
for f, q, e in zip(OBJFUNC_NAMES, obj_func_setters, DV_file_extensions):
  dot = ExternalRun("DOT_%s" % f,dot_ad_command,True)
  dot.setMaxTries(max_tries)
  for ic, c in enumerate(config_files):
    dot.addConfig(c)
    dot.addData(("ADJOINT_%s/restart_adj_%s_%i.dat" % (f, e, ic)), destination=("solution_adj_%s_%i.dat" % (e, ic)))
  dot.addData("DEFORM/mesh_out.su2", destination=meshName)
  dot.addExpected("of_grad.csv")
  dot.addParameter(q) # necessary for correct file extension
  dot.addParameter(conv_target_direct)
  dots.append(dot)

for f, dot, adjoint in zip(OBJFUNC_NAMES, dots, adjoints):
  # Functions ------------------------------------------------------------ #
  func = Function(f, "DIRECT/history_fluid_0.csv",LabeledTableReader("\"%s\"" % f))
  func.addInputVariable(ffd, "DOT_%s/of_grad.csv" % f,TableReader(None,0,(1,0))) 
  func.addValueEvalStep(deform)
  func.addValueEvalStep(direct)
  func.addGradientEvalStep(adjoint)
  func.addGradientEvalStep(dot)
  func.setDefaultValue(1.0)
  functions.append(func)

# Driver --------------------------------------------------------------- #

# i_objfunc = 1
# The input variable is the constraint tolerance which is not used for our purpose of finite differences
driver = ExteriorPenaltyDriver(0.005)
driver.addObjective("min", functions[OBJFUNC_NAMES.index("avg_p_inlet_scaled")], 1e-7)
driver.addEquality(functions[OBJFUNC_NAMES.index("avg_temp_outlet_scaled")], 1.0)


step_sizes = np.logspace(-14, -4, 15,base=10)
DAvals_temp = np.loadtxt("../Baseline/Adjoint_Tout/of_grad.csv",skiprows=1)
DAvals_p = np.loadtxt("../Baseline/Adjoint_Pdrop/of_grad.csv",skiprows=1)
avg_grad = 0.5*(DAvals_temp + DAvals_p)
ivar = np.argmax(avg_grad)

doe_file = "doe_%i.csv" % ivar

driver.setWorkingDirectory("DOE_%i" % ivar)
driver.preprocessVariables()
driver.setStorageMode(True,"DSN_%i_" % ivar)

his = open(doe_file,"w",1)
driver.setHistorian(his)

# # print("Computing baseline primal")
x = driver.getInitial()
driver.fun(x) # baseline evaluation

for s in step_sizes:
  x = driver.getInitial()
  x[ivar] += s 
  driver.fun(x)

FDvals = np.loadtxt(doe_file,delimiter=',',skiprows=1)

FDvals_temp = FDvals[:,2]
FDvals_p = FDvals[:,1]
dTemp_dx_FD = (FDvals_temp[1:] - FDvals_temp[0])/step_sizes 
dPdrop_dx_FD = (FDvals_p[1:] - FDvals_p[0])/step_sizes 

dTemp_dx_AD = DAvals_temp[ivar]
dPdrop_dx_AD = DAvals_p[ivar]

diff_dTempdX = 100*(dTemp_dx_FD - dTemp_dx_AD)/dTemp_dx_AD
diff_dPdropdX = 100*(dPdrop_dx_FD - dPdrop_dx_AD)/dPdrop_dx_AD
print(diff_dTempdX)
print(diff_dPdropdX)

stepsize_study_data = np.vstack((step_sizes,diff_dTempdX,diff_dPdropdX)).T 
with open("Stepsize_diff.csv","w") as fid:
   fid.write("Step size, diff_dToutdX, diff_dPdropdX\n")
   csvwriter = csv.writer(fid,delimiter=',')
   csvwriter.writerows(stepsize_study_data)
fsize=20
fig = plt.figure(figsize=[10,10])
ax = plt.axes()
ax.plot(step_sizes, np.abs(diff_dTempdX),label="difference temperature objective")
ax.plot(step_sizes, np.abs(diff_dPdropdX),label="difference pressure objective")
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid()
ax.legend(fontsize=fsize)
ax.tick_params(which='both',labelsize=fsize)
ax.set_xlabel("FD step size [m]",fontsize=fsize)
ax.set_ylabel("Difference between AD and FD",fontsize=fsize)
fig.savefig("Stepsize_check.eps",format='eps',bbox_inches='tight')
plt.show()