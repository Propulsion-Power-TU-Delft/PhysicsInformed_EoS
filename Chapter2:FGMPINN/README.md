Chapter 2: Modeling hydrogen flames with physics-informed neural networks
=========================================================================

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

The next step involves the generation of flamelet data used to train the networks of the ML-FGM model. In the script[1:generate_flamelet_data.py](1:generate_flamelet_data.py), the user can specify the number of cores used to calculate the flamelets. Information regarding the progress of the flamelet data calculation is displayed in the terminal during calculation.

The flamelet solutions are stored in the folder *flamelet_data*, in which the solutions for each flamelet type are stored in appropriately named sub-folders for each equivalence ratio and mixture fraction. Individual flamelet solutions can be visually inspected by running 
```
PlotFlamelets.py --c config_PIML.cfg --m
```

Step 3: Mine ML-FGM training data
---------------------------------

After calculating the flamelet solutions, the ML-FGM training data sets are defined by running [2:extract_training_data.py](2:extract_training_data.py). This will calculate the values of the FGM controlling variables for each flamelet solution, extract the thermodynamic, thermophysical, and thermochemical state variables and source terms and interpolate 64 points along each flamelet solution to minimize observational biases in the training data set. 

80% of the randomly sampled flamelet data are used for training, 10% for testing, and the remaining 10% for validation. Each data set is stored in the *flamelet_data* folder with appropriate extensions. Additionally, training data sets are generated on which the physics-informed penalty functions are evaluated which contain only chemical equilibrium solutions. These data sets are titled *boundary_data*.

Step 4: Train ML-FGM MLPs with PIML and DF
-------------------------------------------

The training data sets calculated in Step 3 can be used to train the ML-FGM networks using [3:train_MLP.py](3:train_MLP.py). Two sets of networks are generated to evaluate the thermochemical state variables given the solution of the FGM controlling variables. The first set of networks is trained using the loss function that includes the physics-informed penalty functions derived in Chapter 2 of the dissertation. The second set of networks does not include these penalty terms and is therefore trained with a typical data-fitting approach, for which a separate SU2 DataMiner configuration is created named *config_DF.cfg*. 

The progress of the training processes can be monitored from the terminal and by navigating to *flamelet_data/architectures_Group\** and visualizing the training convergence history trends. After the training of a network completes, the network weights and biases are stored in the SU2 DataMiner configuration.

Step 5: SU2 Simulations
-----------------------

The folder *SU2_Simulations* contains the SU2 configuration files, mesh generation instructions, and ASCII files describing the ML-FGM 