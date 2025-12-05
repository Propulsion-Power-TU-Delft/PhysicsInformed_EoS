# FADO script: Finite Differences vs adjoint run
import sys 
import math
import pandas as pd
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

driver.setWorkingDirectory("DOE")
driver.preprocessVariables()
driver.setStorageMode(True,"DSN_")

his = open("doe.csv","w",1)
driver.setHistorian(his)

# print("Computing baseline primal")
x = driver.getInitial()
driver.fun(x) # baseline evaluation

# # Compute discrete adjoint gradient
# print("Computing discrete adjoint gradient")
# driver.grad(x)

# ix_to_analyze = [39,  3, 34, 19, 52, 49, 17, 45, 12,  2, 10,  8]
# # Simulation Runs ------------------------------------------------------ #

# Primal simulation for each deformed DV
for iLoop in range(nDV):
    print("Computing deformed primal ", iLoop, "/", nDV-1)
    x = driver.getInitial()
    x[iLoop] = dx # DV_VALUE, FD-step
    driver.fun(x)
    
#end

# # # Undeformed/initial primal last in order to have the correct solution in
# # # the WorkindDirectory for the following adjoint

# his.close()

# # For results run `python postprocess.py` to get screen output
# # of the differences between primal and adjoint simulation.
def printGradVal(FDgrad, DAgrad):
  """
  Print Gradient Comparison to screen between DA and FD.

  Input:
  FDgrad: array with the Finite Difference gradient
  DAgrad: array with the Discrete Adjoint gradient
  """
  # Check that both arrays have the same length and deduce a size parameter
  assert(DAgrad.size == FDgrad.size)

  # absolute difference
  absoluteDiff = DAgrad - FDgrad
  # relative difference in percent
  relDiffPercent = (DAgrad - FDgrad)/abs(DAgrad) * 100
  # sign change
  sign = lambda x : math.copysign(1, x)
  signChange = [sign(DA) != sign(FD) for DA,FD in zip(DAgrad,FDgrad)]

  print('+-----------+-------------------+-------------------+-------------------+-------------------+-------------+')
  print('| DV number |       DA gradient |       FD gradient |     absolute diff | relative diff [%] | sign change |')
  print('+-----------+-------------------+-------------------+-------------------+-------------------+-------------+')

  for iDV in range(0, DAgrad.size, 1):
      print('|{0:10d} |{1:18.10f} |{2:18.10f} |{3:18.10f} |{4:18.10f} |{5:12} |'.format(iDV, DAgrad[iDV], FDgrad[iDV], absoluteDiff[iDV], relDiffPercent[iDV], signChange[iDV]))

  print('+-----------+-------------------+-------------------+-------------------+-------------------+-------------+')

# Load Discrete Adjont gradient
DAvals_temp = pd.read_csv("../Baseline/Adjoint_Tout/of_grad.csv")
DAvals_p = pd.read_csv("../Baseline/Adjoint_Pdrop/of_grad.csv")

DAstring_specVar ='CUSTOM_OBJFUNC gradient '

DAgrad_temp = DAvals_temp[DAstring_specVar].values
DAgrad_p = DAvals_p[DAstring_specVar].values
#nDV = DAgrad_p.size

# Load primal values and create FD gradient
FDvals = np.loadtxt("doe.csv",delimiter=',',skiprows=1)

FDvals_temp = FDvals[:,2]
FDvals_p = FDvals[:,1]

# FDvals = np.zeros([nDV,2])
# FDvals_temp = np.zeros(nDV)
# FDvals_p = np.zeros(nDV)

# for i in range(1,nDV+1):
#   hist_file = "DSN_%03d/DIRECT/history_fluid_0.csv" % (i)
#   with open(hist_file,'r') as fid:
#       vars = fid.readline().strip().replace(" ","").split(',')
#       vars = [v.strip("\"") for v in vars]
#   H = np.loadtxt(hist_file,delimiter=',',skiprows=1)
#   FDvals_temp[i-1] = H[-1, vars.index("avg_temp_outlet_scaled")]
#   FDvals_p[i-1] = H[-1, vars.index("avg_p_inlet_scaled")]
  
# Note that the FDvals have the baseline value written in its last position
FDgrad_temp = (FDvals_temp[1:] - FDvals_temp[0]) / dx
FDgrad_p = (FDvals_p[1:] - FDvals_p[0]) / dx

diff_grad_temp = ((DAgrad_temp - FDgrad_temp)/FDgrad_temp)
diff_grad_p = ((DAgrad_p - FDgrad_p)/FDgrad_p)
diff_tot = np.abs(diff_grad_temp) + np.abs(diff_grad_p)
ix_sort = np.argsort(diff_tot)

ix = np.arange(nDV)
np.random.shuffle(ix)

with open("dvparams.txt",'r') as fid:
   dvparams_lines = fid.readlines()
print(ix[:12])
for j,i in enumerate(ix[:12]):
  #print(dvparams_lines[i])
  print("%i & $\SI{%+.3e}{}$ & $\SI{%+.3e}{}$ \\\\"% ((j+1), 100*diff_grad_p[i], 100*diff_grad_temp[i]))
   #print("%s : %+.3e, %+.3e" % (dvparams_lines[i].strip(), 100*diff_grad_p[i], 100*diff_grad_temp[i]))


# # Legend
# print("absolute diff = DAgrad - FDgrad")
# print("relative diff = (DAgrad - FDgrad) / abs(DAgrad) * 100")
# print("Average outlet temperature:")
# printGradVal(FDgrad_temp, DAgrad_temp)
# print("Pressure drop:")
# printGradVal(FDgrad_p, DAgrad_p)