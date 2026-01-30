Computational Cost Evaluation for Adjoint and Direct Solver
===========================================================

Author: Evert Bunschoten

Requires: [SU2](https://github.com/su2code/SU2.git)

The scripts in this folder were used to evaluate the computational cost of the direct and the adjoint solver in SU2 when using the ML-FGM method or the ideal gas law to calculate the thermodynamic state properties. The results are documented in Chapter 5 of the doctoral dissertation of E.C.Bunschoten titled *Consistent Data-Driven Equations of State for Reacting and Nonideal Compressible Fluids*. The computational cost was quantified by the average iteration time during the solution process recorded by the solver and the runtime memory requirement of SU2. The solution data and measurements can be downladed from the 4TU research data repository.

All results were generated with SU2 version 8.3.0 "Harrier". After downloading the source code, SU2 was configured with the following build command:
```
meson.py build -Denable-autodiff=true -Denable-mlpcpp=true 
```

Step 1: Direct Solver Simulations
---------------------------------

The flow through the partially premixed hydrogen burner was simulated in this performance measurement. The computational cost of two fluid models were compared: the ML-FGM model documented in Chapter 2 of the dissertation and the ideal gas model which does not model thermochemistry. The SU2 configuration files, mesh files, and other prerequisits for the flow calulations are found in the folders *Direct* and *Direct_IGL* where "IGL" refers to "ideal gas law". 

The simulations were run on a single core using the command
```
SU2_CFD master.cfg
```
and run for 1000 iterations. The convergence history and iteration time are recorded in the file *history_fluid_0.csv*. The runtime memory requirement of the solver was evaluated by running the ```atop -p``` command during the solution process and recording the memory footprint reported by the program.

Step 2: Adjoint Solver Simulations
----------------------------------

The computational cost of the direct solver was compared to that of the adjoint solver for the two equation of state models. The configuration files and other prerequisits can be found in the folders *Adjoint* and *Adjoint_IGL* and the simulations were run using the command
```
SU2_CFD_AD master.cfg
```
for 1000 iterations. The runtime memory requirement of the solver was evaluated by running the ```atop -p``` command during the solution process and recording the memory footprint reported by the program.

Step 3: Compare Performance
---------------------------

The average iteration time of the direct and adjoint solution processes was calculated over the last 100 iterations of the solver. The calculation of the average iteration time and the speed-up of the adjoint solver were calculated using the script [compare_comp_cost.py](compare_comp_cost.py). Running this script will display the average iteration time, standard deviation, and speed-up in the terminal.


