Methods for Chapter 2: Modeling hydrogen flames with physics-informed neural networks
=====================================================================================

Author: Evert Bunschoten

Requires: SU2, SU2 DataMiner

The scripts in this folder explain the research methodologies presented in Chapter 2 of the dissertation. 

Step 0: Install SU2 DataMiner and set up environment.
-----------------------------------------------------

The generation of flamelet data and the training of the networks in the ML-DDEoS model were done with SU2 DataMiner. Install SU2 DataMiner according to the installation instructions and install all the necessary python packages.

To run the FGM simulations, install SU2 and add the ```-Dwith-mlpcpp=True``` when running ```meson.py``` to enable the use of multi-layer perceptrons in SU2. 

Step 1: Generate configuration
------------------------------

Information about the flamelet manifold, the definition of the progress variable, and the network hyperparameters are stored in the SU2 DataMiner configuration class. To generate the configuration used throughout the research presented in this chapter, run the script [0:generate_config.py](0:generate_config.py). 

Running this script generates the configuration file named *config_PIML.cfg* and displays information on the flamelet manifold in the terminal.

Step 2: Generate flamelet data
------------------------------

The next step involves the generation of flamelet data used to train the networks of the ML-FGM model. In the script[1:generate_flamelet_data.py](1:generate_flamelet_data.py), the user can specify the number of cores used to calculate the flamelets. Information regarding the progress of the flamelet data calculation is displayed in the terminal during calculation. The reaction mechanism used to calculate the flamelet solutions is called [stanford.yaml](stanford.yaml) and can be loaded in Cantera for themochemical state calculations.

The flamelet solutions are stored in the folder *flamelet_data*, in which the solutions for each flamelet type are stored in appropriately named sub-folders for each equivalence ratio and mixture fraction. Individual flamelet solutions can be visually inspected by running 
```
PlotFlamelets.py --c config_PIML.cfg --m
```

Step 3: Mine ML-FGM training data
---------------------------------

After calculating the flamelet solutions, the ML-FGM training data sets are defined by running [2:extract_training_data.py](2:extract_training_data.py). This will calculate the values of the FGM controlling variables for each flamelet solution, extract the thermodynamic, thermophysical, and thermochemical state variables and source terms and interpolate 64 points along each flamelet solution to minimize observational biases in the training data set. 

80% of the randomly sampled flamelet data are used for training, 10% for testing, and the remaining 10% for validation. Each data set is stored in the *flamelet_data* folder with appropriate extensions. Additionally, training data sets are generated on which the physics-informed penalty functions are evaluated which contain only chemical equilibrium solutions. These data sets are titled *boundary_data*.

The data of the individual flamelets and the training data sets can be extracted from the compressed zip file named *flamelet_training_data.zip*.

Step 4: Train ML-FGM MLPs with PIML and DF
-------------------------------------------

The training data sets calculated in Step 3 can be used to train the ML-FGM networks using [3:train_MLP.py](3:train_MLP.py). Two sets of networks are generated to evaluate the thermochemical state variables given the solution of the FGM controlling variables. The first set of networks is trained using the loss function that includes the physics-informed penalty functions derived in Chapter 2 of the dissertation. The second set of networks does not include these penalty terms and is therefore trained with a typical data-fitting approach, for which a separate SU2 DataMiner configuration is created named *config_DF.cfg*. 

The progress of the training processes can be monitored from the terminal and by navigating to *flamelet_data/architectures_Group\** and visualizing the training convergence history trends. After the training of a network completes, the network weights and biases are stored in the SU2 DataMiner configuration.

The trained networks and convergence history can be found in the compressed zip file named *TrainedNetworks.zip*.

Step 5: SU2 Simulations
-----------------------

The folder *SU2_Simulations* contains the SU2 configuration files, mesh generation instructions, and ASCII files describing the ML-FGM networks used to evaluate the thermochemical state during CFD calculations. The mesh can be generated using GMESH by running the [mesh generation script](SU2_Simulations/Mesh/generate_mesh.geo). The files with the ```.cfg``` extensions are the SU2 version 8 configuration files containing the instructions for the flow solver. The files with the ```.mlp``` extension describe the multi-layer perceptrons used for thermochemical state calculations.

The flow simulations are run with the following command:
```
mpirun -n <NP> SU2_CFD <config_file>
```
where ```<NP>``` is the number of cores you want to use for the simulation and ```<config_file>``` the name of the SU2 configuration file. 

The SU2 simulation results can also be found in the compressed zip file titled *SU2_Simulations.zip* which can be downloaded from the 4TU research data repository. The flow simulation results are in multi-block vtk format and the file name contains the value of the inflow velocity. The folder also contains the solution binary files needed to restart the flow solution.

In Chapter 2 of the dissertation, the SU2 flow simulation results are compared with those generated with detailed chemistry (DC) analysis, generated with Ansys Fluent. The case files, diffusion model, and reaction mechanism files can be accessed by inflating the compressed zip file named *DC_solutions.zip*.

Post-processing
===============

The following steps explain the methods used for the post-processing of the simulation results and the analysis of the performance of the ML-FGM networks trained with DF and PIML. The post-processing results can also be downloaded from the 4TU research data repository.

Step 1: Compare the output of ML-FGM networks trained with DF and PIML in chemical equilibrium
----------------------------------------------------------------------------------------------

In the first part of the Results section of Chapter 2, the accuracy of the MLPs trained with DF and PIML are compared for thermochemical state variables in chemical equilibrium.
The methods used to produce the trends shown in Figures 2.9-2.11 are explained in [this script](1:compare_MLP_output_equilibrium.py). 

Running this script will load the weights and biases of the trained MLPs from the SU2 DataMiner configurations for DF and PIML and visualizes the output of the networks in chemical equilibrium.

Step 2: Verify correctness of consistency relations
---------------------------------------------------

The correctness of the consistency relations presented in Section 2.2.2 of the dissertation was verified by comparing the Jacobian terms of the consistency relations with finite-difference derivatives of thermochemical state variables in chemical equilibrium. 

The methods used to calculate the verification errors are present in [4:verify_Jacobians.py](4:verify_Jacobians.py), which calculates the average error between the expressions of the consistency relations and the thermochemical state derivatives in chemical equilibrium.

Step 3: Compare accuracy and consistency errors
-----------------------------------------------

The DF and PIML networks were compared in terms of accuracy and consistency. The methods used to calculate the charts shown in Figures 2.12 and 2.13 are presented in [2:compute_accuracy_consistency.py](2:compute_accuracy_consistency.py). Running the script will evaluate the regression error values of the thermochemical state variables for both sets of networks and calculate the consistency error values according to the derived consistency relations and plot the respective bar charts.

Step 4: Flame thickness calculation
-----------------------------------

Section 2.3.2 of the dissertation discusses the accuracy of the SU2 FGM simulation results. The flame thickness values were compared between the DC and FGM simulation results in which the flame thickness was calculated with Equation 2.81. The [ParaView state file](3:compute_flame_thickness.pvsm) applies Equation 2.81 to the simulation results.

Step 5: Apriori analysis
------------------------

An apriori analysis was conducted to help explain the differences observed between the SU2 FGM and DC solutions. The script [5:Apriori_analaysis.py](5:Apriori_analaysis.py) interpolates the thermochemical state data from the flamelet solutions based on the FGM controlling variables of the DC solutions. The results shown in Figures 2.18 and 2.21 of the dissertation are generated by running [5:Apriori_analaysis.py](5:Apriori_analaysis.py).

