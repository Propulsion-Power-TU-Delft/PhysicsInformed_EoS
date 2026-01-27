Baseline Simulations of the Partially Premixed Hydrogen Burner
==============================================================

Author: Evert Bunschoten

Requires: [SU2](https://github.com/su2code/SU2.git)

The scripts in this folder describe how to generate the baseline direct and adjoint solutions of the multi-zone, partially premixed hydrogen burner which is used as a case study in Chapter 5 of the doctoral dissertation of E.C.Bunschoten titled *Consistent data-driven equations of state for reacting and nonideal compressible fluids*. The baseline direct and adjoint solutions are used as the starting point of the simulations conducted during gradient validation and optimization.

All results were generated with SU2 version 8.3.0 "Harrier". After downloading the source code, SU2 was configured with the following build command:
```
meson.py build -Denable-autodiff=true -Denable-mlpcpp=true 
```

Step 1: Baseline Direct Solution
--------------------------------

The initial flow field is calculated with the SU2 direct solver using the four SU2 configuration files for the [fluid zone](fluid.cfg), the solid zone for the [burner](solid_burner.cfg), and for the [heat exchanger](solid_hex.cfg), and the [main](master.cfg) configuration file that describes the multi-zone problem. The flow field is initialized by running 500 iterations without a flame front. After that, the flame front is initialzed with an artifical spark located between the burner and heat exchanger. From that point onward, the flame develops and anchors on the burner and the direct solution converges. To calcualte the baseline direct solution, run the following command:

```
mpirun -n <NP> SU2_CFD master.cfg
```

The convergence of the solution process can be monitored by visualizing the data in the history files for the [fluid zone](history_fluid_0.csv), the [burner zone](history_burner_1.csv), and the [heat exchanger zone](history_hex_2.csv). Every 1000 iterations, the [direct solution](vol_solution.vtm) is updated and can be visualized in ParaView. The binary files headed by *restart_* are used to restart the SU2 solution process in subsequent simulations. All solution and convergence files can also be downloaded from the 4TU research data repository. 


Step 2: Adjoint Solution for Pressure Drop
------------------------------------------

Separate adjoint solutions are needed to calculate the sensitivities of the objective function and the constraint function during optimization. The folder *Adjoint_Pdrop* contains the SU2 configuration files needed to calculate the initial solution of the adjoints for the pressure drop over the computational domain. The folder also contains linked files to the direct baseline solution which are needed to initialize the adjoint solution process. The adjoint solution is calculated with the following terminal command:
```
mpirun -n <NP> SU2_CFD_AD master.cfg
```

Similarly to the direct solution, the convergence of the adjoint solution process can be monitored by visualizing the data in the history files for the [fluid zone](Adjoint_Pdrop/history_adj_fluid_0.csv), the [burner zone](Adjoint_Pdrop/history_adj_burner_1.csv), and the [heat exchanger zone](Adjoint_Pdrop/history_adj_hex_2.csv). Every 1000 iterations, the [adjoint solution](Adjoint_Pdrop/adj_vol_solution.vtm) is updated and can be visualized in ParaView. The binary files headed by *restart_adj_* are used to restart the SU2 solution process in subsequent simulations and to calculate the sensitivities. All solution and convergence files can also be downloaded from the 4TU research data repository. 

Step 3: Adjoint Solution for Outflow Temperature
------------------------------------------------

The folder *Adjoint_Tout* contains the SU2 configuration files needed to calculate the initial solution of the adjoints for the pressure drop over the computational domain. The folder also contains linked files to the direct baseline solution which are needed to initialize the adjoint solution process. The adjoint solution is calculated with the same terminal command used for the adjoint solution process for the pressure drop:
```
mpirun -n <NP> SU2_CFD_AD master.cfg
```

Similarly to the direct solution, the convergence of the adjoint solution process can be monitored by visualizing the data in the history files for the [fluid zone](Adjoint_Tout/history_adj_fluid_0.csv), the [burner zone](Adjoint_Tout/history_adj_burner_1.csv), and the [heat exchanger zone](Adjoint_Tout/history_adj_hex_2.csv). Every 1000 iterations, the [adjoint solution](Adjoint_Tout/adj_vol_solution.vtm) is updated and can be visualized in ParaView. The binary files headed by *restart_adj_* are used to restart the SU2 solution process in subsequent simulations and to calculate the sensitivities. All solution and convergence files can also be downloaded from the 4TU research data repository. 

