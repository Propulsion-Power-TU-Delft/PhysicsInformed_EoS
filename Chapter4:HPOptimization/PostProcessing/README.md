Post-processing for Chapter 4: On the effects of dimensionality reduction and hyper-parmameter optimization on ML-FGM accuracy
==============================================================================================================================

Author: Evert Bunschoten

Requires: [SU2](https://github.com/su2code/SU2.git), [SU2 DataMiner](https://github.com/EvertBunschoten/SU2_DataMiner.git) version 3.0.0.

The scripts in this folder explain the post-processing methods used to derive the results and produce the most relevant plots presented in Chapter 4 of the doctoral dissertation of E.C.Bunschoten titled *Consistent data-driven equations of state for reacting and nonideal compressible fluids*. The first part focusses on the effect of the progress variable definition on ML-FGM accuracy and the second part focusses on the use of hyper-parameter optimization to derive relations between accuracy, computational cost, and ML-FGM hyperparameters.  

Post-processing for the progress variable study
===============================================

The following post-processing steps were used to derive the conclusions presented in Chapter 4 on the topic of the effect of the progress variable definition on ML-FGM accuracy.

Step 1: Determine monotonicity of each progress variable
--------------------------------------------------------

For the three progress variable definitions, determine whether they are monotonic. This is done by calculating the monotonicity penalty function value of the three progress variable definitions. Running the script [1:compare_merit_functions.py](1:compare_merit_functions.py) loads the three SU2 DataMiner configurations corresponding to the three progress variable definitions and calculates the monotonicity penalty function value. If the value of the monotonicity penalty function value is equal to zero, that indicates that the progress variable is monotonic throughout the flamelet data state space. Of the three progress variable definitions, only the optimized progress variable is monotonic.


Step 2: Plot flamelet trends and progress variable coefficients
---------------------------------------------------------------

The three progress variable definitions are first compared by plotting the temperature of three flamelet solutions along each progress variable and calculating the derivative of the temperature w.r.t. the progress variable to qualitatively assess the gradients in the data set. In the dissertation, this is comparison is made through Figure 4.9. Running the script [2:plot_flamelet_trends.py](2:plot_flamelet_trends.py) calculates the second-order accurate temperature derivatives w.r.t. the progress variables using finite-differences and plots the trends along each progress variable, thereby reproducing Figure 4.9.
The coefficients of the progress variables are visualized side-by-side in Figure 4.8 of the dissertation. This figure is reproduced by runing the script [3:plot_pv_weights.py](3:plot_pv_weights.py).

Step 3: Visualize ML-FGM training loss values for each progress variable
------------------------------------------------------------------------

One of the most important figures in Chapter 4 of the dissertation is Figure 4.10, which shows the validation set loss value of the ML-FGM networks trained on the flamelet data parameterized by each progress variable. It indicates how the definition of the progress variable affects the accuracy of the networks. To reproduce Figure 4.10, run the script [4:plot_training_loss.py](4:plot_training_loss.py). This will read the validation loss values from the information stored in the folder *TrainedMLPs* which content can be downloaded from the 4TU research data repository. 

Step 4: Calculate evaluation error for temperature for the MLPs trained on each progress variable
-------------------------------------------------------------------------------------------------

Another way in which the three progress variable definitions are compared in the dissertation is in terms of the distribution of the evaluation error. While Figure 4.10 shows that the OPT progress variable produces the most accurate MLPs overall, that does not necessarily mean that the evaluation error is the lowest everywhere in the thermodynamic state space. 
To this end, the evaluation error of the MLP used to evaluate the temperature is calculated for adiabatic flamelets throughout the thermodynamic state space for each progress variable and is visualized in Figure 4.11 of the dissertation. This figure can be reproduced by running the script [6:evaluate_MLP_output.py](6:evaluate_MLP_output.py). Here, the weights and biases values of the trained MLPs are read from the folder *TrainedMLPs* which content can be downloaded from the 4TU research data repository. 

Post-processing for the hyperparameter optimization study
=========================================================

Step 1: Plot hyperparameter optimization convergence trends 
-----------------------------------------------------------

All the networks trained during the hyperparameter optimization can be found be inflating the compressed files headed by *HP_optimization_group* downloaded from the 4TU research data repository. These folders also contain information regarding the convergence history of the optimization processes. The convergence trends can be plotted by running the script [7:plot_hp_optim_convergence.py](7:plot_hp_optim_convergence.py) after downloading and inflating the optimization results. Running the script will produce the convergence plots shown in Figure 4.13 of the dissertation. 

Step 2: Plot hyperparameters along the Pareto fronts
----------------------------------------------------

Figures 4.15, 4.16, 4.17, and 4.18 of the dissertation show the hyperparameters of the individuals along the Pareto fronts. To reproduce these figures, run the script [8:plot_Pareto_sets.py](8:plot_pareto_sets.py) after inflating the compressed files headed by *HP_optimization_group* downloaded from the 4TU research data repository. Running this script produces the figures and also reports the cross correlation values between the validation loss value, cost parameter value, and hyperparameters along each Pareto front. 

