# FourierAnalysisForSignalReconstruction
This code serves as a basis for further developing skills in computational modelling and singal processing. 

# Software
Python 3.12.4

# Files 
'FourierAnalysis_SignalReconstruction.py' - main code
'README.md' - this file

# How to run
Download files (should automatically download as zip file), and then extract.

# Project Summary
A noisy multi-frequency signal is generated and plotted on a time domain where it is then converted to a frequency domain through the Fast Fourier Transform (FFT) algorithm. 
Dominanat frequencies are identified and highlighted for further analysis.
Basic modelling methods are introduced first through the construction of a feature matrix comprised of sine and cosine basis functions. Two approaches to computational modelling using a linear regression model are then utilised: a full fit model and a test/training split model. The full fit model is trained on all the data from a feature matrix, and then prompted to approximate which linear combinations of coefficients of each feature for a given sample match the target source signal. The test/training split model is trained on only the first 70% of features and tasked with predicting the last 30%, more closely mimicking a real-world scenario where new information is the objective. A root mean squared error calculation validates the predictions as meaningful. 
A manual reconstruction is also executed, where the corresponding amplitudes and phases of the sinusoids at the most dominant frequencies are considered through a Fourier series. 
The findings are illustrated visually throughout the code, where it is clear that the modelling techniques are much more precise than the manual reconstruction. 
The techniques employed in this project provide the skills for further and more complex data analysis and computational modelling.  
