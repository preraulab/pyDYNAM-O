

import scipy
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
import pandas as pd


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = signal.butter(order, [low, high], btype='band', output='sos')
    return sos


def SO_phase(data, fs, lowcut=0.3, highcut=1.5, order=50):
    sos = butter_bandpass(lowcut, highcut, fs, 10)
    data_filt = signal.sosfiltfilt(sos, data)

    analytic_signal = signal.hilbert(data_filt)
    phase = np.unwrap(np.angle(analytic_signal))
    return phase, analytic_signal, data_filt

def wrap_phase(phase):
    return np.angle(np.exp(1j * phase))


csv_data = pd.read_csv('../data_segment.csv', header=None)

# Get test data from the CSV file
fs = 100

# Get test data from the CSV file
data = np.array(csv_data[0])
data -= np.mean(data)
t = np.arange(len(data)) / fs

phase, analytic_signal, data_filt = SO_phase(data, fs)

y_interp = scipy.interpolate.interp1d(t, phase)
peak_times = 1000 + np.array(range(10))
interp_vals = y_interp(peak_times)


ax1 = plt.subplot(211)
plt.plot(t, data)
plt.plot(t, data_filt)
ax2 = plt.subplot(212,  sharex=ax1)
plt.plot(t, wrap_phase(phase))
plt.plot(peak_times, wrap_phase(interp_vals), 'rx')
plt.show()



