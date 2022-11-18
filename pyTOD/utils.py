from itertools import groupby

import numpy as np
from joblib import cpu_count
from matplotlib import gridspec
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from scipy.signal import convolve, butter, hilbert, sosfiltfilt
from scipy.stats import chi2
import matplotlib.pyplot as plt
import colorcet
from pyTOD.multitaper import multitaper_spectrogram


def nan_zscore(data):
    """Computes z-score ignoring nan values

    :param data: Input data
    :return: zscored data
    """
    # Compute modified z-score
    mid = np.nanmean(data)
    std = np.nanstd(data)

    return (data - mid) / std


def pow2db(y):
    """Converts power to dB, ignoring nans

    :param y: values to convert
    :return: val_dB value in dB
    """

    if isinstance(y, int) or isinstance(y, float):
        if y == 0:
            return np.nan
        else:
            ydB = (10 * np.log10(y) + 300) - 300
    else:
        if isinstance(y, list):  # if list, turn into array
            y = np.asarray(y)
        y = y.astype(float)  # make sure it's a float array so we can put nans in it
        y[y == 0] = np.nan
        ydB = (10 * np.log10(y) + 300) - 300

    return ydB


def min_prominence(num_tapers, alpha=0.95):
    """Set minimal peak height based on confidence interval lower bound of MTS

    :param num_tapers: Number of tapers
    :param alpha: Significance level
    :return: min prominence
    """
    chi2_df = 2 * num_tapers
    return -pow2db(chi2_df / chi2.ppf(alpha / 2 + 0.5, chi2_df)) * 2


def convertHMS(seconds: float) -> str:
    """Converts seconds to HH:MM:SS string

    :param seconds:
    :return:
    """
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    return "%02d:%02d:%02d" % (hour, minutes, seconds)


def arange_inc(start: float, stop: float, step: float) -> np.ndarray:
    """Inclusive numpy arange

    :param start: start value
    :param stop: stop value
    :param step: step value
    :return: range = [start:step:stop]
    """
    stop += (lambda x: step * max(0.1, x) if x < 0.5 else 0)((lambda n: n - int(n))((stop - start) / step + 1))
    return np.arange(start, stop, step)


def create_bins(range_start: float, range_end: float, bin_width: float, bin_step: float, bin_method: str = 'full'):
    """Create bins allowing for various overlap and windowing schemes

    :param bin_range: 1x2 array of start-stop values
    :param bin_width: Bin width
    :param bin_step: Bin step
    :param bin_method: 'full' starts the first bin at bin_range[0]. 'partial' starts the first bin
    with its bin center at bin_range(1) but ignores all values below bin_range[0]. 'full_extend' starts
    the first bin at bin_range[0] - bin_width/2.Note that it is possible to get values outside of bin_range
    with this setting.
    :return: bin_edges, bin_centers
    """

    bin_method = str.lower(bin_method)

    if bin_method == 'full':
        range_start_new = range_start + bin_width / 2
        range_end_new = range_end - bin_width / 2

        bin_centers = np.array(arange_inc(range_start_new, range_end_new, bin_step))
        bin_edges = np.vstack([bin_centers - bin_width / 2, bin_centers + bin_width / 2])
    elif bin_method == 'partial':
        bin_centers = np.array(arange_inc(range_start, range_end, bin_step))
        bin_edges = np.maximum(np.minimum([bin_centers - bin_width / 2, bin_centers + bin_width / 2],
                                          range_end), range_start)
    elif bin_method == 'extend' or bin_method == 'full extend' or bin_method == 'full_extend':
        range_start_new = range_start - np.floor((bin_width / 2) / bin_step) * bin_step
        range_end_new = range_end + np.floor((bin_width / 2) / bin_step) * bin_step

        bin_centers = np.array(arange_inc(range_start_new + (bin_width / 2), range_end_new - (bin_width / 2), bin_step))
        bin_edges = np.vstack([bin_centers - bin_width / 2, bin_centers + bin_width / 2])
    else:
        raise ValueError("bin_method should be full, partial, or extend")

    return bin_edges, bin_centers


def outside_colorbar(fig_obj, ax_obj, graphic_obj, gap=0.01, shrink=1, label=""):
    """Creates a colorbar that is outside the axis bounds and does not shrink the axis

    :param fig_obj: Figure object
    :param ax_obj: Axis object
    :param graphic_obj: Graphics object (image, scatterplot, etc.)
    :param gap: Gap between bar and axis
    :param shrink: Colorbar shrink factor
    :param label: Colorbar label
    :return: colorbar object
    """

    ax_pos = ax_obj.get_position().bounds  # Axis position

    # Create new colorbar and get position
    cbar = fig_obj.colorbar(graphic_obj, ax=ax_obj, shrink=shrink, label=label)
    ax_obj.set_position(ax_pos)
    cbar_pos = cbar.ax.get_position().bounds

    # Set new colorbar position
    cbar.ax.set_position([ax_pos[0] + ax_pos[2] + gap, cbar_pos[1], cbar_pos[2], cbar_pos[3]])

    return cbar


def consecutive(val):
    vals = [v[0] for v in groupby(val)]
    cons = [sum(1 for i in g) for v, g in groupby(val)]

    start_inds = np.cumsum(np.insert(cons, 0, 0))
    end_inds = np.add(start_inds[0:-1], cons)

    return list(zip(vals, start_inds, end_inds))


def find_flat(data, minsize=100):
    inds = np.full((len(data)), False)
    for c in consecutive(data):
        if c[2] - c[1] >= minsize:
            inds[c[1]:c[2]] = True

    return inds


def hypnoplot(time, stage, ax=None, plot_buffer=0.8):
    """Plots the hypnogram

    :param time: Stage times
    :param stage: Stage values 6:art, 5:W, 4:R, 3:N1, 2:N2, 1:N3, 0:Unknown
    :param ax: axis for plotting
    :param plot_buffer: how much space above/below
    :return:
    """
    if ax is None:
        ax = plt.axes()

    ax.step(time, stage, 'k-', where='post')
    ax.set_yticks([0, 1, 2, 3, 4, 5, 6], ['Undef', 'N3', 'N2', 'N1', 'R', 'W', 'Art'])
    ylim = (np.min(stage) - plot_buffer, np.max(stage) + plot_buffer)
    ax.set_ylim(ylim)

    ptime = np.append(time, time[-1])

    for c in consecutive(stage):
        if c[0] == 0:
            color = (.9, .9, .9)
        elif 1 <= c[0] <= 3:
            color = (.8, .8, 1)
        elif c[0] == 4:
            color = (.7, 1, .7)
        elif c[0] == 5:
            color = (1, .7, .7)
        else:
            color = (.6, .6, .6)

        ax.add_patch(Rectangle((ptime[c[1]], ylim[0]), ptime[c[2]] - ptime[c[1]], ylim[1] - ylim[0], facecolor=color))


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
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos


def butter_highpass(lowcut, fs, order=50):
    """Performs a zero-phase butterworth bandpass filter on SOS

    :param lowcut: Low-end frequency cutoff
    :param fs: Sampling Frequency
    :param order: Filter order
    :return: Filtered data
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    sos = butter(order, low, btype='hp', output='sos')
    return sos


def zscore_remove(data, crit, bad_inds, smooth_dur, detrend_dir):
    """Find artifacts by removing data until none left outside of criterion

    :param data: Data sequence
    :param crit: Critical value in terms of std from mean
    :param bad_inds: Data to remove prior to procedure
    :param smooth_dur: Duration (in samples) of smoothing window for mean filter
    :param detrend_dir: Duration (in samples) of detrending window for mean filter
    :return: Detected arifacts
    """
    # Get signal envelope
    signal_envelope = np.abs(hilbert(data))
    # Smooth envelope and take the log
    envelope_smooth = np.log(convolve(signal_envelope, np.ones(smooth_dur), 'same') / smooth_dur)
    # Detrend envelope using mean filter
    envelope_detrend = envelope_smooth - convolve(envelope_smooth, np.ones(detrend_dir), 'same') / detrend_dir

    envelope = nan_zscore(envelope_detrend)

    if bad_inds is None:
        detected_artifacts = np.full(len(envelope), False)
    else:
        detected_artifacts = bad_inds

    over_crit = np.logical_and(np.abs(envelope) > crit, ~detected_artifacts)

    # Keep removing data until there is nothing left outside the criterion
    while np.any(over_crit):
        detected_artifacts[over_crit] = True
        # Remove artifacts from the signal
        ysig = envelope[~detected_artifacts]
        ymid = np.nanmean(ysig)
        ystd = np.nanstd(ysig)
        envelope = (envelope - ymid) / ystd

        # Find new criterion
        over_crit = np.logical_and(np.abs(envelope) > crit, ~detected_artifacts)

    return detected_artifacts


def detect_artifacts(data, fs, hf_cut=35, bb_cut=0.1, crit_high=4.5, crit_broad=4.5,
                     smooth_duration=2, detrend_duration=5 * 60):
    """An iterative method to detect artifacts based on data distribution spread

    :param data: Signal data
    :param fs: Sampling frequency
    :param hf_cut: High-frequency filter cut (in Hz)
    :param bb_cut: Broadband filter cut (in Hz)
    :param crit_high: Criterion value for high-frequency data
    :param crit_broad: Criterion value for broadband data
    :param smooth_duration: Duration of smoothing window (in seconds)
    :param detrend_duration: Duration of detrending window (in seconds)_
    :return:
    """
    highfilt = butter_highpass(hf_cut, fs, order=50)
    broadfilt = butter_highpass(bb_cut, fs, order=50)

    data_high = sosfiltfilt(highfilt, data)
    data_broad = sosfiltfilt(broadfilt, data)

    bad_inds = np.abs(nan_zscore(data)) > 10 | np.isnan(data) | np.isinf(data) | find_flat(data)

    hf_artifacts = zscore_remove(data_high, crit_high, bad_inds, smooth_dur=smooth_duration * fs,
                                 detrend_dir=detrend_duration * fs)
    bb_artifacts = zscore_remove(data_broad, crit_broad, bad_inds, smooth_dur=smooth_duration * fs,
                                 detrend_dir=detrend_duration * fs)

    return np.logical_or(hf_artifacts, bb_artifacts)


def summary_plot(data, fs, stages, stats_table, SOpow_hist, SO_cbins, SO_power_norm, SO_power_times, SOphase_hist,
                 freq_cbins):
    """Creates a summary plot with hypnogram, spectrogram, SO-power, scatter plot, and SOPHs

    :param data: Time series data
    :param fs: Sampling frequency
    :param stages: Stages dataframe
    :param stats_table: Peak statistics table
    :param SOpow_hist: SO-power histogram
    :param SO_cbins: SO-power bin centers
    :param SO_power_norm: Normalization method
    :param SO_power_times: SO-power times
    :param SOphase_hist: SO-phase histogram
    :param freq_cbins: Frequency bin centers
    """
    # Limit frequencies from 4 to 25 Hz
    frequency_range = [4, 25]

    taper_params = [15, 29]  # Set taper params
    time_bandwidth = taper_params[0]  # Set time-half bandwidth
    num_tapers = taper_params[1]  # Set number of tapers (optimal is time_bandwidth*2 - 1)
    window_params = [30, 10]  # Window size is 4s with step size of 1s
    min_nfft = 0  # NFFT
    detrend_opt = 'linear'  # constant detrend
    multiprocess = True  # use multiprocessing
    cpus = cpu_count()  # use max cores in multiprocessing
    weighting = 'unity'  # weight each taper at 1

    # Compute the multitaper spectrogram
    spect, stimes, sfreqs = multitaper_spectrogram(data, fs, frequency_range, time_bandwidth, num_tapers,
                                                   window_params,
                                                   min_nfft, detrend_opt, multiprocess, cpus,
                                                   weighting, False, False, False, False)

    lab_size = 9
    clab_size = 8
    title_size = 12
    tick_size = 7

    fig = plt.figure(figsize=(8.5 * .7, 11 * .7))
    gs = gridspec.GridSpec(nrows=5, ncols=2, height_ratios=[0.01, .2, .01, .2, .3],
                           width_ratios=[.5, .5],
                           hspace=0.4, wspace=0.5,
                           left=0.08, right=.875,
                           bottom=0.05, top=0.95)

    ax0 = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, :])
    ax2 = fig.add_subplot(gs[2, :])
    ax3 = fig.add_subplot(gs[3, :])
    ax4 = fig.add_subplot(gs[4, 0])
    ax5 = fig.add_subplot(gs[4, 1])

    # Link axes
    ax0.sharex(ax1)
    ax2.sharex(ax1)
    ax3.sharex(ax1)
    ax3.sharey(ax1)
    ax4.sharey(ax1)
    ax5.sharey(ax1)

    pos0 = ax0.get_position().bounds
    pos1 = ax1.get_position().bounds
    pos2 = ax2.get_position().bounds
    ax0.set_position([pos1[0], pos1[1] + pos1[3], pos0[2], pos1[1] - pos2[1]])
    ax2.set_position([pos2[0], pos2[1], pos1[2], pos1[1] - pos2[1]])

    # Plot hypnogram
    plt.axes(ax0)
    hypnoplot(stages.Time.values / 3600, stages.Stage.values, ax0)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    ax0.set_title('EEG Spectrogram', fontsize=title_size, fontweight='bold')

    # Plot spectrogram
    extent = stimes[0] / 3600, stimes[-1] / 3600, frequency_range[1], frequency_range[0]
    plt.axes(ax1)
    im = ax1.imshow(pow2db(spect), extent=extent, aspect='auto')
    clims = np.percentile(pow2db(spect[~np.isnan(spect)]), [5, 98])
    im.set_clim(clims[0], clims[1])
    ax1.set_ylabel('Frequency (Hz)', fontsize=lab_size)
    ax1.invert_yaxis()
    im.set_cmap(plt.cm.get_cmap('cet_rainbow4'))
    cbar = outside_colorbar(fig, ax1, im, gap=0.01, shrink=0.8)
    cbar.set_label("Power (dB)", fontsize=clab_size)
    cbar.ax.tick_params(labelsize=7)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)

    # Plot SO_power
    plt.axes(ax2)
    ax2.plot(np.divide(SO_power_times, 3600), SO_power_norm, 'b', linewidth=1)
    ax2.set_xlim([SO_power_times[0] / 3500, SO_power_times[-1] / 3600])
    # ax2.set_xlabel('Time (hrs)', fontsize=lab_size)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)

    # Plot the scatter plot
    peak_size = stats_table['volume'] / 200
    pmax = np.percentile(list(peak_size), 95)  # Don't let the size get too big
    peak_size[peak_size > pmax] = 0
    peak_size = np.square(peak_size)

    x = np.divide(stats_table.peak_time, 3600)
    y = [stats_table.peak_frequency]
    c = [stats_table.phase]

    sp = ax3.scatter(x, y, peak_size, c, cmap='hsv')
    ax3.set_xlim([stimes[0] / 3500, stimes[-1] / 3600])
    ax3.set_ylim(frequency_range)

    # Shift the HSV colormap
    hsv = plt.colormaps['hsv'].resampled(2 ** 12)
    hsv_rot = ListedColormap(hsv(np.roll(np.linspace(0, 1, 2 ** 12), -650)))
    sp.set_cmap(hsv_rot)
    cbar = outside_colorbar(fig, ax3, sp, gap=0.01, shrink=0.8)
    cbar.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    cbar.set_ticklabels(['-π', '-π/2', '0', 'π/2', 'π'], fontsize=7)
    cbar.set_label("Phase (rad)", fontsize=clab_size)
    cbar.ax.tick_params(labelsize=lab_size)

    ax3.set_xlabel('Time (hrs)')
    ax3.set_ylabel('Frequency (Hz)', fontsize=lab_size)
    ax3.set_title('Extracted Time-Frequency Peaks', fontsize=title_size, fontweight='bold')
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)

    # SO-power Histogram
    ax4.set_title('SO-power Histogram', fontsize=title_size, fontweight='bold')
    extent = SO_cbins[0], SO_cbins[-1], freq_cbins[-1], freq_cbins[0]
    plt.axes(ax4)
    im = ax4.imshow(SOpow_hist, extent=extent, aspect='auto')
    clims = np.percentile(SOpow_hist, [5, 98])
    im.set_clim(clims[0], clims[1])
    ax4.set_ylabel('Frequency (Hz)', fontsize=lab_size)
    ax4.invert_yaxis()
    im.set_cmap(plt.cm.get_cmap('cet_gouldian'))
    cbar = outside_colorbar(fig, ax4, im, gap=0.01, shrink=0.6)
    cbar.set_label("Density", fontsize=clab_size)
    cbar.ax.tick_params(labelsize=7)
    ax4.set_xlabel('% SO-Power', fontsize=lab_size)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)

    # SO-phase Histogram
    ax5.set_title('SO-Phase Histogram', fontsize=title_size, fontweight='bold')
    extent = -np.pi, np.pi, freq_cbins[-1], freq_cbins[0]
    plt.axes(ax5)
    im = ax5.imshow(SOphase_hist, extent=extent, aspect='auto')
    clims = np.percentile(SOphase_hist, [5, 98])
    im.set_clim(clims[0], clims[1])
    ax5.set_ylabel('Frequency (Hz)', fontsize=lab_size)
    ax5.invert_yaxis()
    im.set_cmap(plt.cm.get_cmap('magma'))
    cbar = outside_colorbar(fig, ax5, im, gap=0.01, shrink=0.6)
    cbar.set_label("Proportion", fontsize=clab_size)
    cbar.ax.tick_params(labelsize=7)
    plt.xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi], ['-π', '-π/2', '0', 'π/2', 'π'])
    plt.xlabel('SO-Phase (rad)', fontsize=lab_size)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)

    plt.show()
