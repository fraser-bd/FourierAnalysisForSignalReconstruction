# -*- coding: utf-8 -*-
"""


@author: 2636038
"""



import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

print()



# Signal Generation
freq_s = 1000                       # Sampling frequency, Hz
T = 1                               # Period in seconds
t = np.linspace(0, T, freq_s*T)     # Time array
N = len(t)                          # Number of samples


f1 = 25                             # Input frequencies: varied for complexity
f2 = 50
f3 = 75

signal1 = 1 * np.sin(2 * np.pi * f1 * t)            # Individual signals: scaling coefficients varied for complexity
signal2 = 2 * np.sin(2 * np.pi * f2 * t)
signal3 = 3 * np.sin(2 * np.pi * f3 * t)

 
signal = signal1 + signal2 + signal3                # Source signal

noise = 0.2 * np.random.randn(len(t))               # random.randn(n) produces a Gaussian distribution of n random numbers; Gaussian distribution is used for its realism and ease-of-use
signal_noisy = signal + noise

signal = signal_noisy                               # Source signal with noise. 



# Plot of Source Signal
plt.figure(figsize = (10,6), dpi = 100)

plt.plot(t, signal, label = 'Source Signal')

plt.xlim(np.min(t), np.max(t))

plt.xlabel('Time (s)', fontsize = 18)
plt.ylabel('Amplitude', fontsize = 18)

plt.title('Source Signal', fontsize = 20)

plt.grid()
plt.show()



# Computing the Fast Fourier Transform
signal_FFT = fft(signal)                # fft(signal) converts the signal from the time domain to the frequency domain through a Fourier transform, producing a complex array of the same length as the signal variable

dt = t[1] - t[0]                        # Sampling interval, assuming uniform sampling; can just take the interval between two arbitrary times

freqs = fftfreq(N, dt)                  # fftfreq(N, dt) returns the real frequencies corresponding to each value in signal_FFT
amplitude = np.abs(signal_FFT) / N      # Taking the absolute value of a complex element returns the magnitude of that element. Here, it's returning the strength of each frequency element in the signal; dividing by N scales the magnitude such that it reflects the amplitude of the original signal element

freqs_pos = freqs[:N//2]                # Taking only the positive frequencies for clarity ( [:N//2] ): the FFT of a sine wave is symmetric and so the negative frequencies can be discarded. Only the first half of the freqs variable is postitive so an integer division is done to return only the positive elements. 
amplitude_pos = 2 * amplitude[:N//2]    # Multiply by 2 to compensate for the discarded negative frequencies, restoring the whole amplitude to the signal. [:N//2] again to only consider the amplitudes corresponding to the positive frequencies



# Identifying Dominant Frequencies 
max_indices = np.argsort(amplitude_pos)[-3:][::-1]                                                  # np.argsort(a) sorts an array in ascending order and returns the index of each element. From visual inspection of the frequency-domain plot, 3 frequency peaks are dominant, and so np.argsort(amplitude_pos)[-3:][::-1] returns the indices of these three peaks in descending order. 
max_freqs = freqs_pos[max_indices]                                                                  # Returns the frequency at the indices found in the preceeding line
max_amp = amplitude_pos[max_indices]

print('The dominant frequencies are: ', max_freqs)
print('Their corresponding amplitudes are: ', max_amp)
print()

plt.figure(figsize = (10,6), dpi = 100)

plt.plot(freqs_pos, amplitude_pos)
plt.scatter(max_freqs, max_amp, zorder = 5, color = 'k', label = 'Dominant Frequencies')            # plt.scatter() is used to plot the dots on the frequency peaks. 'zorder = 5' allows the points to lay in front of all the other objects on the plot. 

plt.xlim(np.min(freqs_pos), np.max(freqs_pos))
plt.ylim(np.min(amplitude_pos), 3.25)

plt.xlabel('Frequency (Hz)', fontsize = 18)
plt.ylabel('Amplitude', fontsize = 18)

plt.title('FFT of Signal, Peaks Highlighted', fontsize = 20)

plt.legend(loc = 'center right')
plt.grid()
plt.show()



# Building matrix of features using sine and cosine basis functions since we know we're dealing with an almost pure sine wave. 
X = []                                          # Initialises a list for storing the sine and cosine functions for each frequency. 

for f in max_freqs:                             # This loop is essentially a Fourier series which only considers the dominant frequencies of the source signal, storing the sine and cosine components in X. 
    X.append(np.sin(2 * np.pi * f * t))
    X.append(np.cos(2 * np.pi * f * t))
    
X = np.array(X).T                               # X is now a 1000x6 array, where the rows are samples and the columns are features, in line with the ML convention. There are 6 columns since we have one sine and one cosine basis function for the 3 dominant frequencies. 



'''
------------------------------------------------------------------------------------------------------------------------------------------------------
From here we use two different methods of training a Linear Regression model to predict the signal: A full model fit and a training/test split model.
A full model fit is trained on all the features and approximates the signal for each corresponding sample. 
The training/test split model is trained on the first 70% of features, and predicts the remaining 30%.  
------------------------------------------------------------------------------------------------------------------------------------------------------
'''



# Full Model Fit
model_full = LinearRegression()                 # Calls the currently untrained Linear Regression model
model_full.fit(X, signal)                       # The .fit(X, signal) method trains the Linear Regression model, finding the optimal linear combination of features for a given sample to match the target, the signal variable. 
signal_model = model_full.predict(X)            # With the model now trained on all the data, the .predit(X) method prompts the model to predict what the value of 'signal' will be. 


print('Model coefficients: ', model_full.coef_)
print()



# Train/Test Split Model
split = int(0.7 * len(t))                       # 70% of the data will be used for training, with the remaining 30% left to test the model's predictions. 70% training, 30% testing is a standard split. 

X_train = X[:split]                             # Takes the first 70% of the data from X
X_test = X[split:]                              # Takes the last 30%

y_train = signal[:split]
y_test = signal[split:]

model_split = LinearRegression()                # Very similar process to the one outlined on lines 125-127. 
model_split.fit(X_train, y_train)
signal_pred = model_split.predict(X_test)       # Key difference: the model is now predicting features which it hasn't seen; contrary to previously where it was predicting for already known features. 

mse = mean_squared_error(y_test, signal_pred)   # Mean Squared Error is used to evaluate the split model's predictions. MSE is chosen since its one of the standard methods for comparing predicted values to true values. It penalises errors proportionally due to its squared characteristic (larger errors are penalised more; smaller errors less so.)
rmse = np.sqrt(mse)                             # Taking the square root of the MSE converts the value back into the original units, making it easier to interpret. 

print('Test MSE: ', mse)
print('Test RMSE: ', rmse)
print()



# Plot of train/test split model overlay with source signal
plt.figure(figsize = (10,6), dpi = 100)

plt.plot(t[split:], y_test, label = 'Source Signal', linewidth = 1)
plt.plot(t[split:], signal_pred, label = 'Predicted ML Signal', color = 'green', linestyle = '--', linewidth = 1)
plt.fill_between(t[split:], signal_pred - rmse, signal_pred + rmse, color = 'green', alpha = 0.2, label = rf'$\pm$ RMSE = {rmse:.2f}')      # Helps to visualise the error in the model's predictions. 

plt.xlim(np.min(t[split:]), np.max(t[split:]))

plt.xlabel('Time (s)', fontsize = 18)
plt.ylabel('Amplitude', fontsize = 18)

plt.title('Signal Reconstruction (Train/Test Split)', fontsize = 20)

plt.legend(loc = 'upper right')
plt.grid()
plt.show()



'''
--------------------------------------------------------------------------------------------------------------------------------------------------------
This section reconstructs the signal manually by using the indices we found earlier to determine the amplitude, frequency, and phase of the 
dominant sinusoidal components of the source signal. We use a for loop to sum these functions and replicate a Fourier series; the cosine function is 
considered through the phase. 
--------------------------------------------------------------------------------------------------------------------------------------------------------
'''



# Manually Reconstructing the Signal
phases = np.angle(signal_FFT[:N//2])                                            # np.angle(signal_FFT[:N//2]) returns the phase angles of the complex elements of the input argument. Taking this allows the reconstructed signal to also be accurately plotted along the time axis. 
rec_signal = np.zeros_like(t)                                                   # np.zeros_like(t) gives a numeric array of zeroes of the same size as the variable t. 

for idx in max_indices:                                                         # This for loop essentially loops through the specified indices from earlier and picks out the amplitudes, frequencies, and phases at these points to the feed into the rec_signal array to reconstruct the signal
    A = amplitude_pos[idx]
    f = freqs_pos[idx]
    phi = phases[idx]
    rec_signal += A * np.sin( 2 * np.pi * f * t + phi)
    


# Overlay Plot of Reconstructed Signal
plt.figure(figsize = (10,6), dpi = 100)

plt.plot(t, signal, linewidth = 1, label = 'Source Signal')
plt.plot(t, rec_signal, linewidth = 1, label = 'Manual Reconstruction')
plt.plot(t, signal_model, label = 'Full Model Fit Reconstruction', linestyle = '--')                # The split model isn't included in this plot as it becomes too cluttered; it is added to the next plot with more clarity. 

plt.xlim(np.min(t), np.max(t))

plt.xlabel('Time (s)', fontsize = 18)
plt.ylabel('Amplitude', fontsize = 18)

plt.title('Signal Reconstruction', fontsize = 20)

plt.legend(loc = 'upper right')
plt.grid()
plt.show()



# Zoomed-in Plot for Clarity
plt.figure(figsize = (10,6), dpi = 100)

plt.plot(t[900:1000], signal[900:1000], linewidth = 1, label = 'Source Signal')
plt.plot(t[900:1000], rec_signal[900:1000], linewidth = 1, label = 'Manual Reconstruction')
plt.plot(t[900:1000], signal_model[900:1000], label = 'Full Model Fit Reconstruction', linestyle = '--')
plt.plot(t[900:1000], signal_pred[200:300], label = 'Split Model Reconstruction', linestyle = ':')

plt.xlim(np.min(t[900:1000]), np.max(t[900:1000]))

plt.xlabel('Time (s)', fontsize = 18)
plt.ylabel('Amplitude', fontsize = 18)

plt.title('Signal Reconstruction (0.90s to 1.0s)', fontsize = 20)

plt.legend(loc = 'upper right')
plt.grid()
plt.show()

