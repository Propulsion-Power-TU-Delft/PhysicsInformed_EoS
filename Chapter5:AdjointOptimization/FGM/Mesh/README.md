Mesh Generation of the Partially Premixed Hydrogen Burner Case Study
====================================================================

Author: Evert Bunschoten

Requires: [SU2](https://github.com/su2code/SU2.git)

The scripts in this folder describe how to generate the mesh of the multi-zone, partially premixed hydrogen burner which is used as a case study in Chapter 5 of the doctoral dissertation of E.C.Bunschoten titled *Consistent data-driven equations of state for reacting and nonideal compressible fluids*. 

Step 1: Generate SU2 Mesh for each Zone
---------------------------------------

The computational domain of the partially premixed hydrogen burner contains three zones: the fluid zone where the transport equations for the FGM controlling variables and the Navier-Stokes equations are solved, and the two solid zones corresponding to the internally cooled burner and heat exchanger where the heat equation is solved. The first step in generating the mesh was to create separate SU2 meshes for each zone using the initial, semi-circular shape for the burner and heat exchanger.

Generating the initial three meshses is done by running the script [generate_mesh.py](generate_mesh.py) which uses Gmesh for discretizing the computational domains. Running this script will produce three SU2 mesh files titled *fluid_mesh.su2*, *burner_mesh.su2*, and *hex_mesh.su2*. 

Step 2: Place FFD Boxes and Create Multi-Zone Mesh
--------------------------------------------------

The next step is to combine the three single-zone meshes into a multi-zone mesh and write free-form deformation (FFD) box information to the mesh file. This can be done by running the following terminal command:
```
SU2_DEF master.cfg
```
SU2 will read the three single-zone mesh files and combine them into a multi-zone mesh and associate the FFD boxes with the CHT inerfaces of the burner and heat exchanger. 

