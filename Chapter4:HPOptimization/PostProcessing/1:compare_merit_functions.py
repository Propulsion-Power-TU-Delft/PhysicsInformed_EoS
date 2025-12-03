import numpy as np
import matplotlib.pyplot as plt 
from su2dataminer.config import Config_FGM
from su2dataminer.process_data import PVOptimizer

N=5
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.cubehelix(np.linspace(0,1,N)))
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_Default = colors[0]
color_PCA = colors[1]
color_Niu = colors[2]
color_Optimized = colors[3]

Config_default = Config_FGM("../WRP.cfg")
Config_pca = Config_FGM("../PCA.cfg")
Config_optim = Config_FGM("../OPT.cfg")

Config_pca.PrintBanner()
PVO = PVOptimizer(Config_optim)
PVO._CollectFlameletData()

pv_species = Config_optim.GetProgressVariableSpecies()
x_default = np.zeros(len(pv_species))
x_pca = np.zeros(len(pv_species))
x_Niu = np.zeros(len(pv_species))
x_optim = Config_optim.GetProgressVariableWeights()

pv_species_default = Config_default.GetProgressVariableSpecies()
pv_species_pca = Config_pca.GetProgressVariableSpecies()
pv_species_optim = Config_optim.GetProgressVariableSpecies()
for ipv, pv in enumerate(pv_species):
    if pv in pv_species_default:
        x_default[ipv] = Config_default.GetProgressVariableWeights()[pv_species_default.index(pv)]
    if pv in pv_species_pca:
        x_pca[ipv] = Config_pca.GetProgressVariableWeights()[pv_species_pca.index(pv)]

penalty_val_default = PVO.penalty_function(x_default)
penalty_val_pca = PVO.penalty_function(x_pca)
penalty_val_optim = PVO.penalty_function(Config_optim.GetProgressVariableWeights())

monotonicity_penalty_default = PVO.monotonicity_penalty(x_default)
monotonicity_penalty_pca = PVO.monotonicity_penalty(x_pca)
monotonicity_penalty_Niu = PVO.monotonicity_penalty(x_Niu)
monotonicity_penalty_optim = PVO.monotonicity_penalty(x_optim)

print("Penalty value default pv definition: %.6e Monotonicity penalty: %.6e" % (penalty_val_default-monotonicity_penalty_default, monotonicity_penalty_default))
print("Penalty value pca pv definition: %.6e Monotonicity penalty: %.6e" % (penalty_val_pca-monotonicity_penalty_pca, monotonicity_penalty_pca))

print("Penalty value optimized pv definition: %.6e Monotonicity penalty: %.6e" % (penalty_val_optim, monotonicity_penalty_optim))


fig = plt.figure(figsize=[10,7])
ax = plt.axes()
ax.bar(x=[0], height=[penalty_val_default-monotonicity_penalty_default],color=color_Default,width=0.8,zorder=3)
ax.bar(x=[1], height=[penalty_val_pca-monotonicity_penalty_pca],color=color_PCA,width=0.8,zorder=3)

bar_SU2DataMiner = ax.bar(x=[3], height=[penalty_val_optim - monotonicity_penalty_optim],width=0.8,zorder=3,color=color_Optimized)

ax.set_yscale('log')
ax.grid()
ax.set_xticks([0,1,2,3])
ax.set_xticklabels(["WRP", "PCA", "Prufert","SPV"])
ax.set_ylabel("Maximum gradient w.r.t. pv",fontsize=30)
ax.tick_params(which='both',labelsize=30)
#fig.savefig("./Images/max_gradient.eps",format='eps',bbox_inches='tight')
plt.show()