Chapter 2: Modeling hydrogen flames with physics-informed neural networks
=========================================================================

Author: Evert Bunschoten

Requires: SU2, SU2 DataMiner

The scripts in this folder explain the post-processing methods used to produce the charts shown in in Chapter 2 of the dissertation. 

Step 1: Compare the output of ML-FGM networks trained with DF and PIML in chemical equilibrium
----------------------------------------------------------------------------------------------

In the first part of the Results section of Chapter 2, the accuracy of the MLPs trained with DF and PIML are compared for thermochemical state variables in chemical equilibrium.
The methods used to produce the trends shown in Figures 2.9-2.11 are explained in [this script](1:compare_MLP_output_equilibrium.py). 

Running this script will load the weights and biases of the trained MLPs from the SU2 DataMiner configurations for DF and PIML and visualizes the output of the networks in chemical equilibrium.

Step 2: 