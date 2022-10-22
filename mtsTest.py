from multitaper_toolbox.python.multitaper_spectrogram_python import \
    multitaper_spectrogram  # import multitaper_spectrogram function from the multitaper_spectrogram_python.py file
import numpy as np  # import numpy
from scipy.signal import chirp  # import chirp generation function
import pandas as pd
import matplotlib.pyplot as plt

# reading the CSV file
csv_data = pd.read_csv('data.csv', header=None)
data = np.array(csv_data[0])

# Set spectrogram params
fs = 100  # Sampling Frequency
frequency_range = [0, 30]  # Limit frequencies from 0 to 25 Hz
time_bandwidth = 2  # Set time-half bandwidth
num_tapers = 3  # Set number of tapers (optimal is time_bandwidth*2 - 1)
window_params = [1, .05]  # Window size is 4s with step size of 1s
min_nfft = 2**10  # No minimum nfft
detrend_opt = 'constant'  # detrend each window by subtracting the average
multiprocess = True  # use multiprocessing
cpus = 4  # use 3 cores in multiprocessing
weighting = 'unity'  # weight each taper at 1
plot_on = False  # plot spectrogram
clim_scale = False  # do not auto-scale colormap
verbose = True  # print extra info
xyflip = False  # do not transpose spect output matrix

# Generate sample chirp data
t = np.arange(1 / fs, 600, 1 / fs)  # Create 10 min time array from 1/fs to 600 stepping by 1/fs
f_start = 1  # Set chirp freq range min (Hz)
f_end = 20  # Set chirp freq range max (Hz)
# data = chirp(t, f_start, t[-1], f_end, 'logarithmic')

# Compute the multitaper spectrogram
spect, stimes, sfreqs = multitaper_spectrogram(data, fs, frequency_range, time_bandwidth, num_tapers, window_params,
                                               min_nfft, detrend_opt, multiprocess, cpus,
                                               weighting, plot_on, clim_scale, verbose, xyflip)

baseline = np.percentile(spect, 2, axis=1, keepdims=True)


plt.figure(1)
plt.plot(baseline)

plt.figure(2)
extent = np.min(stimes), np.max(stimes), np.max(sfreqs), np.min(sfreqs)
plt.imshow(np.divide(spect, baseline), extent=extent)
plt.gca().invert_yaxis()

plt.show()




