Sensitivity Analysis of an Annular ORC Stator Vane
==================================================

Author: Evert Bunschoten

Requires: [SU2](https://github.com/su2code/SU2.git) [ParaBlade](https://github.com/NAnand-TUD/parablade)

The scripts in this folder were used to perform the sensitivity analysis of the ORC stator as documented in Chapter 5 of the doctoral dissertation of E.C.Bunschoten titled *Consistent Data-Driven Equations of State for Reacting and Nonideal Compressible Fluids*. The CAD sensitivities were calculated in three ways: using the discrete adjoint method with the EEoS-PINN model, with finite-differences while using the EEoS-PINN model, and with finite-differences while using the CoolProp Helmholtz equation of state. 
The blade geometry is generated with ParaBlade. To reproduce the results documented in the dissertation, the branch *feature/conv_div_blade* is required from the TU Delft BitBucket repository. 
All flow and adjoint calculations were performed with SU2 version 8.3.0. To enable the use of the discrete adjoint solver, the EEoS-PINN model, and the HEoS fluid model, configure SU2 with the following options:
```
meson.py build -Denable-autodiff=true -Denable-coolprop=true -Denable-mlpcpp=true
```

Step 1: Generate Mesh
---------------------

The computational domain is discretized with Gmesh using the script [1:generate_mesh_gmsh.py](1:generate_mesh_gmsh.py). The blade geometry is made with ParaBlade reading the [configuration file](ORCHID_stator_base_ParaBlade.cfg) which lists the CAD parameters used to parameterized the blade geometry. The mesh contains triangular and rectangular elements with structured mesh sections around the blade surface for boundary layer refinement. Additional refinement is applied in the flow domain to more accurately resolve the shock and expansion waves. The refinement locations are listed in [this file](mach_grad_contours.csv) and are read from the file while generating the mesh. 
The completed mesh is titled [ORCHID_mesh.su2](ORCHID_mesh.su2) and it can also be downloaded from the 4TU repository instead of generating it locally. 

Step 2: Direct and Adjoint Solution
-----------------------------------

The direct and adjoint solutions calculated with the EEoS-PINN and HEoS fluid models were generated using the scripts in the folder 