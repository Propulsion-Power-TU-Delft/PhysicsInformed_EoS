Gradient Validation Study for Partially Premixed Hydrogen Burner
================================================================

Author: Evert Bunschoten

Requires: [SU2](https://github.com/su2code/SU2.git)

The scripts in this folder were used to perform the gradient validation study of the partially premixed hydrogen burner documented in Chapter 5 of the doctoral dissertation of E.C.Bunschoten titled *Consistent Data-Driven Equations of State for Reacting and Nonideal Compressible Fluids*. The gradient validation study was performed by comparing the design sensitivities calculated with the SU2 discrete adjoint solver with those calculated with finite-difference approximation.

All results were generated with SU2 version 8.3.0 "Harrier". After downloading the source code, SU2 was configured with the following build command:
```
meson.py build -Denable-autodiff=true -Denable-mlpcpp=true 
```

Step 1: Baseline Direct Solution
--------------------------------

The solution of the initial flow field can be downloaded from the 4TU research data repository. The baseline solution is used as an initial guess for the direct and adjoint solution process. The files headed by ```restart_``` and ```restart_adj``` are the binary solution files that can be loaded into SU2. 

Step 2: Run Gradient Validation
-------------------------------

The script [1:gradient_validation.py](1:gradient_validation.py) was used to compute the design sensitivities with the discrete adjoint solver and finite-differences. The gradient validation is performed by first calculating the baseline adjoint solutions and calculating the sensitivities. The results of these calculations can be found in the folder ```DSN_001``` and the design sensitivities calculated with the adjoint solver are listed in the file ```of_grad.csv``` in the folders headed by ```DOT_```. 
After calculating the adjoint sensitivities, the sensitivities are calculated with finite-differences. Here, each design parameter is perturbed, the mesh is deformed, and the direct solution is calculated from which the values of the objective function and the constraint function are evaluated. The values of the objective and constraint function of each calculation are listed in the file ```doe.csv```, in which the first entry describes the function values of the initial design. 

Step 3: Visualizing the Results
-------------------------------

Figure 5.18 of the dissertation shows the adjoint and finite-difference sensitivities. This figure can be recreated by running the script [2:compare_sensitivity.py](2:compare_sensitivity.py). Additionally, the script calculates the mean absolute percentage error between the sensitivities calculated with finite-differences and the discrete adjoint. 