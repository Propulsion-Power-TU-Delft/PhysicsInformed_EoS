Constrained Optimization of a Partially Premixed Hydrogen Burner
================================================================

Author: Evert Bunschoten

Requires: [SU2](https://github.com/su2code/SU2.git)

The scripts in this folder were used to perform the constrained optimization of the partially premixed hydrogen burner documented in Chapter 5 of the doctoral dissertation of E.C.Bunschoten titled *Consistent Data-Driven Equations of State for Reacting and Nonideal Compressible Fluids*. The objective function was the drop in the static pressure over the flow domain and an equality constraint was imposed on the temperature of the fluid at the domain outlet. The design surfaces were the burner to which the flame was anchored and the heat exchanger downstream of the flame. Both design surfaces were parameterized by FFD boxes. 
The thermochemistry of the flame was modeled using the ML-FGM model documented in Chapter 1 of the manuscript and the MLPs used to calculate the thermochemical state are described by the files in this folder with the ```.mlp``` extension.

All results were generated with SU2 version 8.3.0 "Harrier". After downloading the source code, SU2 was configured with the following build command:
```
meson.py build -Denable-autodiff=true -Denable-mlpcpp=true 
```

Step 1: Baseline Solution
-------------------------

The optimization is initialized from the converged solution of the baseline design. The results can be downloaded from the 4TU research data repository while the instructions for mesh generation can be found in the folder titled *Mesh*. 
The converged flow solution of the initial design is stored in the binary files headed by ```restart_*.dat```, where the number refers to the zone (0=fluid zone, 1=solid burner zone, 2=solid heat exchanger zone).

Step 2: Gradient Validation
---------------------------

To evalulate whether the sensitivities are correctly calculated with the adjoint solver, a gradient validation step was performed during the research. The scripts and results of this study can be downloaded separately from the 4TU research data repository. The binary files headed by ```restart_adj_*.dat``` contain the converged solutions of the adjoint solver.

Step 3: Optimization
--------------------

The optimization is initialized by running the script [optimization.py](optimization.py). The optimization is performed using the FADO framework within SU2 commonly used for optimization. The optimization algorithm is SLSQP and sensitivities are calculated with the discrete adjoint solver. During each design iteration, a folder is created titled ```DSN_*``` in which the solution of the direct solver is stored in the folder ```DIRECT``` and the solutions of the adjoint in ```ADJOINT```. The sensitivities are stored in the file ```of_grad.csv``` in the folder ```DOT```. 
The convergence history of the optimization is documented in the file [hist_slsqp.csv](hist_slsqp.csv), which was updated for every iteration of the SLSQP algorithm (excluding line searches) with the value of the objective function, the constraint function, and the values of the design parameters. The optimization results can be accessed by inflating the compressed zip file titled *OptimizationResults.zip* which can be accessed through the 4TU reserach data repository.