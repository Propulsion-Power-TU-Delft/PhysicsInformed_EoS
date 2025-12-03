import numpy as np
import matplotlib.pyplot as plt 

from su2dataminer.config import Config_FGM

Config = Config_FGM("../WRP.cfg")

val_scores_default = []
val_scores_pca = []
val_scores_optim = []
val_scores_niu = []
for iGroup in range(Config.GetNMLPOutputGroups()):
    try:
        with open(("../TrainedMLPs/WRP/Group%i/MLP_performance.txt" % (iGroup)), 'r') as fid:
            lines = fid.readlines()
            val_score_default = float(lines[1].strip().split(":")[-1])
    except:
        val_score_default = 1e-2

    try:
        #with open(("../Architectures_PCA_n/Group%i/MLP_performance.txt" % (iGroup)), 'r') as fid:
        with open(("../TrainedMLPs/PCA/Group%i/MLP_performance.txt" % (iGroup)), 'r') as fid:
            lines = fid.readlines()
            val_score_pca = float(lines[1].strip().split(":")[-1])
    except:
        val_score_pca = 1e-2
    try:
        with open(("../TrainedMLPs/OPT/Group%i/MLP_performance.txt" % (iGroup)), 'r') as fid:
            lines = fid.readlines()
            val_score_optim = float(lines[1].strip().split(":")[-1])
    except:
        val_score_optim = 1e-2
    val_scores_default.append(val_score_default)
    val_scores_pca.append(val_score_pca)
    val_scores_optim.append(val_score_optim)

x_range = np.linspace(0, len(val_scores_optim), len(val_scores_optim))*1.5
bar_width=0.25
fsize = 20

N=3
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.cubehelix(np.linspace(0,1,N+1)))
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_Default = colors[0]
color_PCA = colors[1]
color_Optimized = colors[2]

w = 1.0
w_bar = w / (N) 

fig = plt.figure(figsize=[10,7])
ax = plt.axes()
for i in range(0,len(x_range),2):
    ax.axvspan(x_range[i]-0.6, x_range[i]+0.6, alpha=0.06, color='k')
ax.bar(x=x_range+ (0.5 + 0)*w_bar - 0.5*w, height=val_scores_default,width=w_bar,color=color_Default,label='WRP',zorder=2)
ax.bar(x=x_range+ (0.5 + 1)*w_bar - 0.5*w, height=val_scores_pca,width=w_bar,color=color_PCA,label='PCA',zorder=2)
#ax.bar(x=x_range+ (0.5 + 2)*w_bar - 0.5*w, height=val_scores_niu,width=bar_width,color=color_Niu,label='Niu',zorder=2)
ax.bar(x=x_range+ (0.5 + 2)*w_bar - 0.5*w, height=val_scores_optim,width=w_bar,color=color_Optimized,label='SPV',zorder=2)
ax.set_yscale('log')
ax.grid(which='both')
ax.set_xticks(x_range)
ax.set_xticklabels([Config.GetMLPOutputGroup(i)[0] for i in range(Config.GetNMLPOutputGroups())])
ylabels = ax.get_yticklabels()
print(len(ylabels))
ax.set_yticklabels([109, 108, 107,106,105,104,103,102,101, 100, 101])
ax.set_ylabel("Validation set loss value (Lval)",fontsize=fsize)
ax.legend(fontsize=fsize, ncol=N, loc="upper center",bbox_to_anchor=(0.5, -0.11),fancybox=True,shadow=True)
ax.tick_params(which='both',labelsize=fsize)
ax.set_title("MLP validation loss for each pv definition",fontsize=fsize)
#fig.savefig("Images/pv_def_loss.pdf",format='pdf',bbox_inches='tight')
plt.show()