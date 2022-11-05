import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import cpu_count
from matplotlib import colors
from scipy import signal
import colorcet  # It looks like this is not used, but it puts the colorcet cmaps in matplotlib
from extractTFPeaks import pow2db
from multitaper_toolbox.python.multitaper_spectrogram_python import multitaper_spectrogram
from pyTODhelper import outside_colorbar


def butter_bandpass(lowcut, highcut, fs, order=50):
    """Performs a zero-phase butterworth bandpass filter on SOS

    :param lowcut: Low-end frequency cutoff
    :param highcut: High-end frequency cutoff
    :param fs: Sampling Frequency
    :param order: Filter order
    :return: Filtered data
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = signal.butter(order, [low, high], btype='band', output='sos')
    return sos


def get_SO_phase(data, fs, lowcut=0.3, highcut=1.5, order=50):
    """Computes unwrapped slow oscillation phase
    Note: Phase is unwrapped because wrapping does not returne to original angle given the unwrapping algorithm

    :param data: EEG time series data
    :param fs: Sampling frequency
    :param lowcut: Bandpass low-end cutoff
    :param highcut: Bandpass high-end cutoff
    :param order: Filter order
    :return: Unwrapped phase
    """
    sos = butter_bandpass(lowcut, highcut, fs, 10)
    data_filt = signal.sosfiltfilt(sos, data)

    analytic_signal = signal.hilbert(data_filt)
    phase = np.unwrap(np.angle(analytic_signal))
    return phase


def get_SO_power(data, fs, lowcut=0.3, highcut=1.5):
    """Computes slow oscillation power

    :param data: EEG time series data
    :param fs: Sampling frequency
    :param lowcut: Bandpass low-end cutoff
    :param highcut: Bandpass high-end cutoff
    :return: SO_power, SOpow_times
    """
    frequency_range = [lowcut, highcut]

    taper_params = [15, 29]  # Set taper params
    time_bandwidth = taper_params[0]  # Set time-half bandwidth
    num_tapers = taper_params[1]  # Set number of tapers (optimal is time_bandwidth*2 - 1)
    window_params = [30, 10]  # Window size is 4s with step size of 1s
    min_nfft = 0  # NFFT
    detrend_opt = 'linear'  # constant detrend
    multiprocess = True  # use multiprocessing
    cpus = max(cpu_count() - 1, 1)  # use max cores in multiprocessing
    weighting = 'unity'  # weight each taper at 1
    plot_on = False  # plot spectrogram
    clim_scale = False  # do not auto-scale colormap
    verbose = False  # print extra info
    xyflip = False  # do not transpose spect output matrix

    # Compute the multitaper spectrogram
    spect, SOpow_times, sfreqs = multitaper_spectrogram(data, fs, frequency_range, time_bandwidth, num_tapers,
                                                        window_params,
                                                        min_nfft, detrend_opt, multiprocess, cpus,
                                                        weighting, plot_on, clim_scale, verbose, xyflip)

    df = sfreqs[1] - sfreqs[0]
    SO_power = pow2db(np.sum(spect, axis=0) * df)
    return SO_power, SOpow_times


def wrap_phase(phase: np.ndarray) -> np.ndarray:
    """Wrap phase from -pi to pi

    :param phase: Unwrapped phase
    :return: Wrapped phase
    """
    return np.angle(np.exp(1j * phase))


def plot_figure():
    # Load in raw data
    print('Loading in raw data...', end=" ")
    # EEG data
    stats_table = pd.read_csv('example_night.csv')
    # Sampling Frequency
    fs = 100
    print('Done')

    # Load in data
    print('Loading in TF-peaks stats data...', end=" ")
    csv_data = pd.read_csv('data_night.csv', header=None)
    data = np.array(csv_data[0]).astype(np.float32)
    print('Done')

    # Number of jobs to use
    n_jobs = max(cpu_count() - 1, 1)

    # Limit frequencies from 4 to 25 Hz
    frequency_range = [4, 25]

    taper_params = [15, 29]  # Set taper params
    time_bandwidth = taper_params[0]  # Set time-half bandwidth
    num_tapers = taper_params[1]  # Set number of tapers (optimal is time_bandwidth*2 - 1)
    window_params = [30, 10]  # Window size is 4s with step size of 1s
    min_nfft = 0  # NFFT
    detrend_opt = 'linear'  # constant detrend
    multiprocess = True  # use multiprocessing
    cpus = n_jobs  # use max cores in multiprocessing
    weighting = 'unity'  # weight each taper at 1
    plot_on = False  # plot spectrogram
    clim_scale = False  # do not auto-scale colormap
    verbose = True  # print extra info
    xyflip = False  # do not transpose spect output matrix

    # Compute the multitaper spectrogram
    spect, stimes, sfreqs = multitaper_spectrogram(data, fs, frequency_range, time_bandwidth, num_tapers,
                                                   window_params,
                                                   min_nfft, detrend_opt, multiprocess, cpus,
                                                   weighting, plot_on, clim_scale, verbose, xyflip)

    # Plot the scatter plot
    peak_size = stats_table['volume'] / 100
    pmax = np.percentile(list(peak_size), 95)  # Don't let the size get too big
    peak_size[peak_size > pmax] = 0

    SO_power, SO_power_times = get_SO_power(data, fs, lowcut=0.3, highcut=1.5)

    # %% Plot figure
    fig = plt.figure(figsize=(8, 11))
    gs = gridspec.GridSpec(nrows=4, ncols=2, height_ratios=[.2, .01, .2, .3],
                           width_ratios=[.5, .5],
                           hspace=0.25, wspace=0.2,
                           left=0.1, right=0.90,
                           bottom=0.05, top=0.95)

    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :])
    ax3 = fig.add_subplot(gs[2, :])
    ax4 = fig.add_subplot(gs[3, 0])
    ax5 = fig.add_subplot(gs[3, 1])

    # Link axes
    ax2.sharex(ax1)
    ax3.sharex(ax1)
    ax3.sharey(ax1)

    # Plot spectrogram
    extent = stimes[0] / 3600, stimes[-1] / 3600, frequency_range[1], frequency_range[0]
    plt.axes(ax1)
    im = ax1.imshow(pow2db(spect), extent=extent, aspect='auto')
    clims = np.percentile(pow2db(spect), [5, 98])
    im.set_clim(clims[0], clims[1])
    ax1.set_ylabel('Frequency (Hz)')
    ax1.invert_yaxis()
    plt.xticks([])
    im.set_cmap(plt.cm.get_cmap('cet_rainbow4'))
    outside_colorbar(fig, ax1, im, gap=0.01, shrink=0.8, label="Power (db)")
    ax1.set_title('EEG Spectrogram')

    # Plot SO_power
    ax2.plot(np.divide(SO_power_times, 3600), SO_power)
    ax2.set_xlim([SO_power_times[0] / 3500, SO_power_times[-1] / 3600])
    pos1 = ax1.get_position().bounds
    pos2 = ax2.get_position().bounds
    ax2.set_position([pos2[0], pos2[1], pos1[2], pos1[1] - pos2[1]])
    ax2.set_xlabel('Time (hrs)')

    # Plot scatter plot
    x = np.divide(stats_table.peak_time, 3600)
    y = [stats_table.peak_frequency]
    c = [stats_table.phase]

    sp = ax3.scatter(x, y, peak_size, c, cmap='hsv')
    ax3.set_xlim([stimes[0] / 3500, stimes[-1] / 3600])
    ax3.set_ylim(frequency_range)

    # Shift the HSV colormap
    hsv = plt.colormaps['hsv'].resampled(2 ** 12)
    hsv_rot = colors.ListedColormap(hsv(np.roll(np.linspace(0, 1, 2 ** 12), -650)))
    sp.set_cmap(hsv_rot)
    cbar = outside_colorbar(fig, ax3, sp, gap=0.01, shrink=0.8, label="Phase (rad)")
    cbar.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    cbar.set_ticklabels(['-π', '-π/2', '0', 'π/2', 'π'])

    ax3.set_xlabel('Time (hrs)')
    ax3.set_ylabel('Frequency (Hz)')
    ax3.set_title('Extracted Time-Frequency Peaks')

    # SO-power Histogram
    ax4.set_title('SO-power Histogram')

    # SO-phase Histogram
    ax5.set_title('SO-phase Histogram')
    plt.show()


if __name__ == '__main__':
    # Load full night extracted TF-peaks and plot figure so far
    plot_figure()
