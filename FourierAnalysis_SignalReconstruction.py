# -*- coding: utf-8 -*-
"""
This program generates a signal comprised of multiple sine waves and reconstructs the signal through a Fast Fourier Transform. 
The reconstructed signal is accurate and demonstrates the effectiveness of the method. Slight deviations in amplitude and phase may be explained by numeric approximations
carried out throughout the program.
The wave reconstruction assumes that we know the original signal was comprised of sine waves. 

@author: 2636038
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq


# Signal Generation

freq_s = 1000                       # Sampling frequency, Hz
T = 1                               # Period in seconds
t = np.linspace(0, T, freq_s*T)     # Time array
N = len(t)                          # Number of samples


f1 = 25                             # Input frequencies
f2 = 50
f3 = 75

signal1 = 1 * np.sin(2 * np.pi * f1 * t)            # Individual signals
signal2 = 2 * np.sin(2 * np.pi * f2 * t)
signal3 = 3 * np.sin(2 * np.pi * f3 * t)

 
signal = signal1 + signal2 + signal3                # Source signal

noise = 0.2 * np.random.randn(len(t))               # random.randn(n) produces a Gaussian distribution of n random numbers; Gaussian distribution is used for its realism and ease-of-use
signal_noisy = signal + noise

signal = signal_noisy



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
signal_FFT = fft(signal)                # fft(signal) converts the signal from the time domain to the frequency domain, producing a complex array of the same length as the signal variable

dt = t[1] - t[0]                        # Sampling interval for uniform sampling; can just take the interval between two arbitrary times

freqs = fftfreq(N, dt)                  # fftfreq(N, dt) returns the real frequencies corresponding to each value in signal_FFT
amplitude = np.abs(signal_FFT) / N      # Taking the absolute value of a complex element returns the magnitude of that element. Here, it's returning the strength of each frequency element in the signal; dividing by N scales the magnitude such that it reflects the amplitude of the original signal element

freqs_pos = freqs[:N//2]                # Taking only the positive frequencies for clarity ( [:N//2] ): the FFT of a sine wave is symmetric and so the negative frequencies can be discarded. Only the first half of the freqs variable is postitive so an integer division is done to provide only the positive elements. 
amplitude_pos = 2 * amplitude[:N//2]    # Multiply by 2 to compensate for the discarded negative frequencies, restoring the whole amplitude to the signal. [:N//2] again to only consider the amplitudes corresponding to the positive frequencies



# Identifying Dominant Frequencies 
max_indices = np.argsort(amplitude_pos)[-3:][::-1]                                                  # np.argsort(a) sorts an array in ascending order and returns the index of each element. From visual inspection of the FFT plot, 3 frequency peaks are dominant, and so np.argsort(amplitude_pos)[-3:][::-1] returns the indices of these three peaks in descending order. 
max_freqs = freqs_pos[max_indices]                                                                  # Returns the frequency at the indices found in the preceeding line
max_amp = amplitude_pos[max_indices]

print('The dominant frequencies are: ', max_freqs)
print('Their corresponding amplitudes are: ', max_amp)

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



# Reconstructing the Signal
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
plt.plot(t, rec_signal, linewidth = 1, label = 'Reconstructed Signal')

plt.xlim(np.min(t), np.max(t))

plt.xlabel('Time (s)', fontsize = 18)
plt.ylabel('Amplitude', fontsize = 18)

plt.title('Signal Reconstruction', fontsize = 20)

plt.legend(loc = 'upper right')
plt.grid()
plt.show()



# Zoomed-in Plot for Clarity
plt.figure(figsize = (10,6), dpi = 100)

plt.plot(t[:200], signal[:200], linewidth = 1, label = 'Source Signal')
plt.plot(t[:200], rec_signal[:200], linewidth = 1, label = 'Reconstructed Signal')

plt.xlim(np.min(t[:200]), np.max(t[:200]))

plt.xlabel('Time (s)', fontsize = 18)
plt.ylabel('Amplitude', fontsize = 18)

plt.title('Signal Reconstruction (0 to 0.2s)', fontsize = 20)

plt.legend(loc = 'upper right')
plt.grid()
plt.show()

