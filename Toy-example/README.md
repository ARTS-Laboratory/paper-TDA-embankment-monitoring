# Toy-example:

The strategy is to create different synthetic point clouds (data sets) using simple slop with abnormalities such as humps and cavities.  
To create the slpos, the two main codes are used:
1- Meshed slop with hump.py: This code creates a hump at different locations with different diameters.
   By tuning the parameters, any arbitrary shape can be created.
2- meshed slope with cavity_2.py: This code creates a cavity at different locations with different diameters.
   By tuning the parameters, any arbitrary shape can be created.

 Note: After creating the meshed slope, the point clouds can be captured using the Cloudcompare software.
       Then, the point cloud can be saved in the .las format.

 The next step is to run the TDA over pointclouds.
 The TDA's main function is "TDA-teast". However, this function is developed and used to capture all 16 features along with the RANSCAN algorithm for more optimisation in terms of 
 the system memory usage. The value of sampling points "m" and iteration "k" can be adjusted as needed. Moreover, the homology dimension can be selected. (H0, H1, H2).

 Note: The features are also saved as a CSV file in the directory.

 ** Another code is developed: "TDA-features with graph-3" to demonstrate the overall and normalize the value of features over several different numbers of sampling points to capture 
 the convergence ratio of each feature. 

 ** To illustrate the changes in features with respect to each other, a grid plot was created using "Abnormalities_Features.py." This script accepts an input file from the Data folder to plot all features.
 

## Codes

### Abnormalities_Features
* Describe what the code demonstrate
* Describe what the code demonstrate
* Describe what the code demonstrate


### Another code
* Describe what the code demonstrate
* Describe what the code demonstrate
* Describe what the code demonstrate