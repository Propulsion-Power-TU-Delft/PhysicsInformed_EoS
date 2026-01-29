import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams["font.family"] = "Times New Roman"
N=3
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.cubehelix(np.linspace(0,1,N+1)))
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


hist_file_HEoS = "../ComputationTime/CoolProp/history_direct_JST_CoolProp.csv"
hist_file_EEoS_direct = "../ComputationTime/PINN/history_direct_JST_rr.csv"
hist_file_EEoS_adjoint = "../ComputationTime/PINNAdjoint/history_adjoint_dev_vel.csv"

with open(hist_file_HEoS,'r') as fid:
    vars_HEoS = fid.readline().strip().replace(" ","").split(',')
    vars_HEoS = [v.strip("\"") for v in vars_HEoS]
H_HEoS = np.loadtxt(hist_file_HEoS,delimiter=',',skiprows=1)

with open(hist_file_EEoS_direct,'r') as fid:
    vars_EEoS = fid.readline().strip().replace(" ","").split(',')
    vars_EEoS = [v.strip("\"") for v in vars_EEoS]
H_EEoS_direct = np.loadtxt(hist_file_EEoS_direct,delimiter=',',skiprows=1)

with open(hist_file_EEoS_adjoint,'r') as fid:
    vars_EEoS_adjoint = fid.readline().replace(" ","").strip().split(',')
    vars_EEoS_adjoint = [v.strip("\"") for v in vars_EEoS_adjoint]
H_EEoS_adjoint = np.loadtxt(hist_file_EEoS_adjoint,delimiter=',',skiprows=1)

M_HEoS = 3.8 
M_EEoS = 3.8
M_adjoint_nopeacc = 132.9
M_adjoint=14.0

M_EEoS_factor = M_EEoS / M_HEoS
M_Adjoint_factor = M_adjoint / M_HEoS
N_avg = 100

t_HEoS = np.average(H_HEoS[-N_avg:,vars_HEoS.index("Time(sec)")])
t_EEoS_direct = np.average(H_EEoS_direct[-N_avg:,vars_EEoS.index("Time(sec)")])
t_EEoS_adjoint = np.average(H_EEoS_adjoint[-N_avg:,vars_EEoS_adjoint.index("Time(sec)")])

print("HEoS: %.3e" % (t_HEoS / t_HEoS))
print("EEoS direct: %.3e" % (t_EEoS_direct/t_HEoS))
print("EEoS adjoint: %.3e" % (t_EEoS_adjoint/t_HEoS))

w = 1.0
w_bar = w / (N) 
fig, axs= plt.subplots(ncols=1,nrows=2,figsize=[10,12])
ax = axs[0]
x = [(0.5 + 0)*w_bar - 0.5*w,(0.5 + 1)*w_bar - 0.5*w,(0.5 + 2)*w_bar - 0.5*w]
ax.bar(x=x[0], height=t_HEoS / t_HEoS,width=0.9*w_bar,zorder=3,color=colors[0],label="HEoS")
ax.bar(x=x[1], height=t_HEoS / t_EEoS_direct,width=0.9*w_bar,zorder=3,color=colors[1],label="EEoS")
ax.bar(x=x[2], height=t_HEoS / t_EEoS_adjoint,width=0.9*w_bar,zorder=3,color=colors[2],label="EEoS AD")
ax.grid()
ax.set_xticks(x)
ax.set_xticklabels([])
ax.set_ylabel(r"Speedup factor $\left(\frac{\overline{t}_\mathrm{HEoS}}{\overline{t}}\right)[-]$",fontsize=20)
ax.tick_params(which='both',labelsize=20)
ax.set_title(r"EEoS and Adjoint Speed-Up", fontsize=20)

ax = axs[1]
ax.bar(x=x[0],height=1.0, width=0.9*w_bar,zorder=3,color=colors[0],label="HEoS")
ax.bar(x=x[1],height=M_EEoS_factor, width=0.9*w_bar,zorder=3,color=colors[1],label="EEoS")
ax.bar(x=x[2],height=M_Adjoint_factor, width=0.9*w_bar,zorder=3,color=colors[2],label="EEoS AD")
ax.grid()
ax.set_xticks(x)
ax.set_xticklabels([r"HEoS",r"EEoS",r"EEoS AD"])
ax.set_ylabel(r"Memory footprint $\left(\frac{\overline{M}}{\overline{M}_\mathrm{HEoS}}\right)[-]$",fontsize=20)
ax.tick_params(which='both',labelsize=20)


plt.show()