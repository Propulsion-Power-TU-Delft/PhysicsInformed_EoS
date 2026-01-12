Methods for Chapter 3: Modeling nonideal compressible fluids with physics-informed neural networks
==================================================================================================

Author: Evert Bunschoten

Requires: [SU2](https://github.com/su2code/SU2.git), [SU2 DataMiner](https://github.com/EvertBunschoten/SU2_DataMiner.git)

The scripts in this folder explain the research methodologies presented in Chapter 3 of the dissertation. This file explains the function of each script and how to reproduce the results presented in the manuscript. The research data can also be downloaded from the 4TU Research data archive. 

Step 0: Install SU2, SU2 DataMiner and set up the environment
-------------------------------------------------------------

The training data for the ML-DDEoS methods and the training of the neural networks is done with [SU2 DataMiner](https://github.com/EvertBunschoten/SU2_DataMiner.git). Install SU2 DataMiner according to the installation instructions provided in the repository and install all the required python packages.

The simulation results are obtained with SU2 version 8.3.0 "Harrier". Before compiling SU2, a part of the source code has to be changed in order to be able to transform the network output of the EEoS-MTNN method. After cloning the SU2 source code, copy the file [CDataDrivenFluid.cpp](CDataDrivenFluid.cpp) to ```SU2_CFD/src/fluid/```. After that, compile SU2 with the following command line instructions:
```
meson.py build -Denable-mlpcpp=true -Denable-coolprop=true
./ninja -C build install
```

Step 1: Generate configuration
------------------------------

Information about the fluid and size of the training data set for the DDEoS models is stored in the SU2 DataMiner configuration. To generate the configuration used throughout the research presented in this chapter, run the script [0:generate_config.py](0:generate_config.py). 

This will create three SU2 DataMiner configurations. The configurations named *High.cfg* and *Low.cfg* contain the training data information for the EEoS-MTNN networks for the low and high density ranges, while the configuration named *PINN.cfg* describes the training data information for the EEoS-PINN network.

Step 2: Generate fluid data
---------------------------

The next step is to generate the training data for the three ML-DDEoS networks. This can be done by running the script [1:generate_fluid_data.py](1:generate_fluid_data.py). The training data sets for the EEoS-MTNN networks are titled *fluid_data_high* and *fluid_data_low* and are saved in the folder titled *Direct*. The training data for the EEoS-PINN method are stored in the folder *PhysicsInformed* and are headed by *fluid_data_PINN*.


Step 3: Train EEoS-MTNN networks
--------------------------------

The networks used to evaluate the Jacobian and Hessian components according to the EEoS-MTNN model are trained through data-fitting on the fluid data generated in Step 2 by running the script [2:train_MLP_segregated.py](2:train_MLP_segragated.py). This will first train the network for the low density range (`<`10 kg/m3) followed by the network for the high density range. The progress of the training process can be followed from the terminal or by visualizing the convergence trends stored in *Direct/Worker_0/Model_0/* and *Direct/Worker_0/Model_1/*. 

After completing the training process, the weights and biases of the network are stored in the SU2 DataMiner configurations and can also be accessed in *Direct/Worker_0/Model_0/* and *Direct/Worker_0/Model_1/*.

Step 4: Train EEoS-PINN network
-------------------------------

After generating the training data, the PIML method documented in Section 3.2.3 of the manuscript is used to train the network to evaluate the entropy potential. This is done by running the script [3:train_MLP_PINN.py](3:train_MLP_PINN.py). Information regarding the progress of the training process is printed in the terminal, but can also be accessed by visualizing the training convergence trends plotted in *PhysicsInformed/Worker_0/Model_0* during training. 

The PINN is initially trained for 1000 epochs through data-fitting, after which the training restarts in which the network is trained to evaluate the thermodynamic state variables following the entropy-based equation of state. 

Step 5: Verify correctness of EEoS
----------------------------------

The correctness of the entropy-based equation of state is verified by comparing the values of the thermodynamic state variables calculated with the entropy Jacobian and Hessian and those according to the Helmholtz equation of state. Running the script [4:generate_consistency_study_data.py](4:generate_consistency_study_data.py) calculates the mean square relative error between the thermodynamic state variables evaluated by both methods and prints the results in the terminal.

Step 6: Run SU2 simulations
---------------------------

The SU2 simulations can be run by navigating to the *SU2_Simulation* folder. The sub-folders contain the SU2 configuration files corresponding to the EEoS, HEoS, and CEoS simulations. The mesh and flow solution files can be downloaded by inflating the compressed file titled *SU2_Simulation_Results.zip*. To run the simulations yourself, run the following command in the terminal:
```
mpirun -n <NP> SU2_CFD <config file>
```
where `<NP>` is the number of processors and `<config_file>` the name of the configuration file. 

