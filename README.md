# Image-Registration
This repository contain 2 notebooks and a python file that attempt image registration on a data set referenced through a DataFrame.

The dependencies are highlighted in the notebooks:

1) Notebook1 - Implementation of an Area-based image registration.
2) Notebook2 - Implementation of a contour-based registration.

Although the results are not satisfactory the notebooks detail the different steps attempted.

The python file (Area_Based_Rgistration.py) can run from a terminal with one argument (an integer between 0 and 102 otherwise it exits). This value is the index of the instance refenreced by the dataframe to fetch an RGB image and the misaligned narrowband red image associated to it --> The function returns the transformation matrix, similarity measures (between the reference image and the misaligned image as well as between the reference image and the registered image).
