import numpy as np 
import matplotlib.pyplot as plt 
from paretoset import paretoset
from su2dataminer.config import Config_FGM

iGroup = 0

c = Config_FGM("../HP_Optimization.cfg")
output_dir = c.GetOutputDir()

plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.cubehelix(np.linspace(0,1,c.GetNMLPOutputGroups()+1)))
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
markers = ['o','s','^','d','*']
linestyles = ['-','-.','--','-','-']
alpha_expos = []
lr_decays = []
i_phis = []
val_scores = []
cost_params = []
hidden_layers = []
number_perceptrons = []
aspect_ratio = []
NH_max = 0
for iGroup in range(c.GetNMLPOutputGroups()):
    hist_file ="%s/Architectures_Group%i_OptimLRAPhi/history_optim_Group%i_LRAPhi.csv" % (output_dir, iGroup+1, iGroup+1)
    with open(hist_file,'r') as fid:
        vars = fid.readline().strip().split(',')
    H = np.loadtxt(hist_file,delimiter=',',skiprows=1)
    generation = H[:,0]
    alpha_expo = H[:,1]
    lr_decay = H[:,2]
    i_phi = H[:,3]
    NH = 2 + np.sum(H[:, [6,8,10,12,14,16,18,20]],axis=1)
    hidden_layer_architecture = np.hstack((H[:,4:6], H[:, 6:22:2]*H[:, 7:23:2]))
    sq = np.zeros(len(H))
    AR = np.zeros(len(H))
    sym = np.zeros(len(H))
    avg_Nh = np.zeros(len(H))

    NP = np.sum(hidden_layer_architecture, axis=1)
    
    N_av = NP / NH 

    val_score = H[:, -2]
    cost_param = H[:,-1]

    mask = paretoset(np.vstack((val_score, cost_param)).T, sense=["min","min"])

    pareto_val_score = val_score[mask]
    pareto_cost_param = cost_param[mask]

    pareto_alpha_expo = alpha_expo[mask]
    pareto_lr_decay = lr_decay[mask]
    pareto_phi = i_phi[mask]
    pareto_NH = NH[mask]
    pareto_NP = avg_Nh[mask]
    pareto_aspect_ratio = AR[mask]

    alpha_expos.append(pareto_alpha_expo)
    lr_decays.append(pareto_lr_decay)
    i_phis.append(pareto_phi)
    val_scores.append(pareto_val_score)
    cost_params.append(pareto_cost_param)
    hidden_layers.append(pareto_NH)
    number_perceptrons.append(pareto_NP)
    aspect_ratio.append(pareto_aspect_ratio)
fsize=  20
fig = plt.figure(figsize=[12,10])
ax = plt.axes()
for iGroup in range(c.GetNMLPOutputGroups()):
    ix_sort = np.argsort(val_scores[iGroup])
    ax.plot(cost_params[iGroup][ix_sort],val_scores[iGroup][ix_sort],marker=markers[iGroup],markersize=12,markerfacecolor='none',linewidth=3,markeredgewidth=2,linestyle=linestyles[iGroup],label='Group %i' % (iGroup+1))
ax.grid(which='both')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylabel("Validation error",fontsize=fsize)
ax.set_xlabel("Cost parameter",fontsize=fsize)
ax.tick_params(which='both',labelsize=fsize)
ax.set_title("Cost-accuracy Pareto fronts",fontsize=fsize)
ax.legend(fontsize=fsize,ncol=1,bbox_to_anchor=(1, 0.5),loc='center left',fancybox=True,shadow=True)
plt.show()

fig = plt.figure(figsize=[12,7])
ax = plt.axes()
print("Alpha expo:")
mean_alpha_expo = 0.0
for iGroup in range(c.GetNMLPOutputGroups()):
    x1 = np.log10(cost_params[iGroup])
    x0 = np.ones(np.shape(cost_params[iGroup]))
    A = np.vstack((x1,x0)).T
    y = alpha_expos[iGroup]
    a = np.dot((np.dot(np.linalg.inv(np.dot(A.T,A)),A.T)),y[:,np.newaxis])
    color=colors[iGroup]

    mu = np.average(alpha_expos[iGroup])
    mean_alpha_expo += mu / c.GetNMLPOutputGroups()
    std = np.std(alpha_expos[iGroup])
    x_ls = np.linspace(mu - std, mu + std, 10)
    y_ls = np.power(10, (x_ls - a[1])/a[0])
    cc_alpha_expo = np.corrcoef(np.log10(cost_params[iGroup]), alpha_expos[iGroup])
    print("Group %i : %+.3e" % (iGroup+1, cc_alpha_expo[0,1]))
    ax.plot(cost_params[iGroup],np.power(10,alpha_expos[iGroup]),marker=markers[iGroup],markersize=8,linewidth=2,color=color,markerfacecolor='none',markeredgewidth=2,linestyle='none')
    ax.plot(y_ls, np.power(10,x_ls), color=color,marker=markers[iGroup], markerfacecolor='none',linewidth=3,markersize=12,linestyle='--',markeredgewidth=2,label='Group %i' % (iGroup+1))
    
ax.grid(which='both')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel("Cost parameter",fontsize=fsize)
ax.set_ylabel("Initial learning rate",fontsize=fsize)
ax.tick_params(which='both',labelsize=fsize)
ax.set_title("Initial learning rate value",fontsize=fsize)
ax.legend(fontsize=fsize,ncol=1,bbox_to_anchor=(1, 0.5),loc='center left',fancybox=True,shadow=True)

plt.show()

fig = plt.figure(figsize=[12,7])
ax = plt.axes()
N_phis = np.zeros([8, c.GetNMLPOutputGroups()])
for iGroup in range(c.GetNMLPOutputGroups()):
    for iphi in range(8):
        n_instances = np.sum(np.argwhere(i_phis[iGroup]==iphi))
        N_phis[iphi, iGroup] = n_instances
N_phis = np.delete(N_phis, [0, 4],axis=0)
N_phis /= (np.sum(N_phis,axis=0)+1e-3)
w = 1.0
w_bar = w / c.GetNMLPOutputGroups()
x_bars = np.arange(6)*1.2
for iGroup in range(c.GetNMLPOutputGroups()):
    ax.bar(x=x_bars+ (0.5 + iGroup)*w_bar - 0.5*w,height=100*N_phis[:, iGroup],width=w_bar,zorder=3,label='Group %i' % (iGroup+1))
ax.grid()
ax.set_ylabel("Occurance[%]",fontsize=fsize)
ax.set_xlabel("Hidden layer activation function",fontsize=fsize)
ax.set_xticks([x for x in x_bars])
ax.set_xticklabels(["elu","relu","tanh","gelu","sigmoid","swish"])
ax.tick_params(which='both',labelsize=fsize)
ax.set_title("Hidden layer activation function",fontsize=fsize)
ax.legend(fontsize=fsize,ncol=1,bbox_to_anchor=(1, 0.5),loc='center left',fancybox=True,shadow=True)
plt.show()

fig,axs = plt.subplots(ncols=2,nrows=1,figsize=[14,8])
ax = axs[1]
N_H = np.arange(2,11)
w = 0.8
w_bar = w / (c.GetNMLPOutputGroups()) 
for iGroup in range(c.GetNMLPOutputGroups()):
    freq_Nh = np.zeros(len(N_H))
    for j in range(len(N_H)):
        freq_Nh[j] = np.sum(hidden_layers[iGroup] == N_H[j])
    freq_Nh /= np.sum(freq_Nh)
    ax.bar(x=N_H+ (0.5 + iGroup)*w_bar - 0.5*w, width=w_bar,height=100*freq_Nh,label='Group %i' % (iGroup+1),zorder=3)
ax.set_xticks([int(i) for i in N_H])
ax.set_xticklabels([int(i) for i in N_H])
ax.grid(which='both')
ax.set_xlabel("Number of hidden layers",fontsize=fsize)
ax.set_ylabel("Frequency",fontsize=fsize)
ax.tick_params(which='both',labelsize=fsize)
ax.set_title("Number of hidden layers for each group",fontsize=fsize)
ax.legend(fontsize=fsize,ncol=1,bbox_to_anchor=(1, 0.5),loc='center left',fancybox=True,shadow=True)
plt.show()

ax = axs[0]
N_H = np.arange(2,11)
w = 0.8
w_bar = w / (c.GetNMLPOutputGroups()) 
for iGroup in range(c.GetNMLPOutputGroups()):
    freq_Nh = np.zeros(len(N_H))
    for j in range(len(N_H)):
        freq_Nh[j] = np.sum(hidden_layers[iGroup] == N_H[j])
    freq_Nh /= np.sum(freq_Nh)
    ax.plot(hidden_layers[iGroup],val_scores[iGroup],marker=markers[iGroup],markersize=12,linewidth=3,markerfacecolor='none',markeredgewidth=2,linestyle='none',label='Group %i' % (iGroup+1))
ax.set_xticks([int(i) for i in N_H])
ax.set_xticklabels([int(i) for i in N_H])
ax.grid(which='both')
ax.set_yscale('log')
ax.set_ylabel("Validation loss",fontsize=fsize)
ax.set_xlabel("Number of hidden layers",fontsize=fsize)
ax.tick_params(which='both',labelsize=fsize)
ax.set_title("Number of hidden layers for each group",fontsize=fsize)
plt.show()

fig = plt.figure(figsize=[12,7])
ax = plt.axes()
print("Cross correlation between validation loss and aspect ratio:")

for iGroup in range(c.GetNMLPOutputGroups()):
    x1 = np.log10(val_scores[iGroup])
    x0 = np.ones(np.shape(val_scores[iGroup]))
    A = np.vstack((x1,x0)).T
    y = aspect_ratio[iGroup]
    a = np.dot((np.dot(np.linalg.inv(np.dot(A.T,A)),A.T)),y[:,np.newaxis])
    std = np.std(aspect_ratio[iGroup])
    x_ls = np.linspace(mu - std, mu + std, 10)
    y_ls = np.power(10, (x_ls - a[1])/a[0])
    cc_aspect = np.corrcoef(np.log10(val_scores[iGroup]),aspect_ratio[iGroup])
    print("Group %i : %+.6e" % (iGroup+1, cc_aspect[0,1]))
    ax.plot(val_scores[iGroup],aspect_ratio[iGroup],color=colors[iGroup],marker=markers[iGroup],markersize=12,linewidth=3,markerfacecolor='none',markeredgewidth=2,linestyle='none',label='Group %i' % (iGroup+1))
    #ax.plot(y_ls, x_ls,linestyle='--',color=colors[iGroup])

ax.grid(which='both')
ax.set_xscale('log')
ax.set_ylabel("Aspect ratio",fontsize=fsize)
ax.set_xlabel("Validation score",fontsize=fsize)
ax.tick_params(which='both',labelsize=fsize)
ax.set_title("Aspect ratio of hidden layer archtiecture",fontsize=fsize)
ax.legend(fontsize=fsize,ncol=1,bbox_to_anchor=(1, 0.5),loc='center left',fancybox=True,shadow=True)
plt.show()


fig = plt.figure(figsize=[12,7])
ax = plt.axes()
ymax = -100
ymin = 100
print("LR decay:")
mean_lr_decay = 0.0
for iGroup in range(c.GetNMLPOutputGroups()):
    mu = np.median(lr_decays[iGroup])
    mean_lr_decay += mu/c.GetNMLPOutputGroups()
    ax.plot([min(cost_params[iGroup]),max(cost_params[iGroup])], [mu,mu],linewidth=3,marker=markers[iGroup],markeredgewidth=2,linestyle='--',markersize=12,markerfacecolor='none',label='Group %i' % (iGroup+1))
    ymax = max(ymax, np.average(lr_decays[iGroup]) + np.std(lr_decays[iGroup]))
    ymin = min(ymin, np.average(lr_decays[iGroup]) - np.std(lr_decays[iGroup]))
    x = np.log10(cost_params[iGroup])
    y = lr_decays[iGroup]
    x_norm = (x - min(x))/(max(x) - min(x))
    y_norm = (y - min(y))/(max(y) - min(y))

    std_lr_decay = np.std(lr_decays[iGroup])
    ix_outlier = np.argwhere(np.logical_or(lr_decays[iGroup] > mu+1*std_lr_decay, lr_decays[iGroup] < mu-1*std_lr_decay))
    cost = np.delete(cost_params[iGroup], ix_outlier)
    lr = np.delete(lr_decays[iGroup], ix_outlier)
    ax.plot(cost,lr,marker=markers[iGroup],color=colors[iGroup],markersize=8,linewidth=2,markeredgewidth=2,markerfacecolor='none',linestyle='none')
    
    cc_lr_decay = np.corrcoef(np.log10(cost), lr)[0,1]
    print("Group %i: %+.3e" % (iGroup+1, cc_lr_decay))
ax.grid(which='both')
ax.set_xscale('log')
ax.set_ylim([0.986, 0.994])
ax.set_xlabel("Cost parameter",fontsize=fsize)
ax.set_ylabel("Learning rate decay parameter",fontsize=fsize)
ax.tick_params(which='both',labelsize=fsize)
ax.set_title("Pareto learning rate decay parameter",fontsize=fsize)
ax.legend(fontsize=fsize,ncol=1,bbox_to_anchor=(1, 0.5),loc='center left',fancybox=True,shadow=True)
plt.show()
