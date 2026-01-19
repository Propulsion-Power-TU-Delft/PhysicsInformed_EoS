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


