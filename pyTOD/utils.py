from itertools import groupby

import matplotlib
import numpy
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
    """
    Compute modified z-score for a numpy array.

    Parameters
    ----------
    data : numpy array
        The data to be normalized.

    Returns
    -------
    numpy array
        The normalized data.
    """
    # Compute modified z-score
    mid = np.nanmean(data)
    std = np.nanstd(data)

    return (data - mid) / std


def pow2db(y):
    """Converts power to dB ignoring nans.

    Parameters
    ----------
    y : float or array
        Power to be converted to dB.

    Returns
    -------
    ydB : float or array
        Power in dB.

    Notes
    -----
    This function is a wrapper for the following:

    .. math::
        ydB = (10 * log10(y) + 300) - 300

    Examples
    --------
    >>> pow2db(1)
    0.0
    >>> pow2db(0)
    nan
    >>> pow2db([1, numpy.nan, 1])
    array([  0.,  nan,   0.])
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


def min_prominence(num_tapers: int, alpha: float = 0.95) -> float:
    """Set minimal peak height based on confidence interval lower bound of MTS

    Parameters
    ----------
    num_tapers : int
        Number of tapers used in MTS.
    alpha : float, optional
        Confidence level.

    Returns
    -------
    min_prominence : float
        Minimal peak height.

    Notes
    -----
    The minimal peak height is calculated as the lower bound of the confidence
    interval of the MTS spectrum. The confidence interval is calculated using
    the chi-squared distribution.
    """
    chi2_df = 2 * num_tapers
    return -pow2db(chi2_df / chi2.ppf(alpha / 2 + 0.5, chi2_df)) * 2


def convertHMS(seconds: float) -> str:
    """Convert seconds to hours, minutes, and seconds.

    Parameters
    ----------
    seconds : float
        The number of seconds to convert.

    Returns
    -------
    str
        A string of the form "HH:MM:SS"

    Examples
    --------
    >>> convertHMS(3661)
    '01:01:01'
    """
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    return "%02d:%02d:%02d" % (hour, minutes, seconds)


def arange_inc(start: float, stop: float, step: float) -> np.ndarray:
    """Inclusive numpy arange that includes endpoints

    Parameters
    ----------
    start : float
        The start of the range.
    stop : float
        The end of the range.
    step : float
        The step size.

    Returns
    -------
    np.ndarray
        The range.
    """
    stop += (lambda x: step * max(0.1, x) if x < 0.5 else 0)((lambda n: n - int(n))((stop - start) / step + 1))
    return np.arange(start, stop, step)


def create_bins(range_start: float, range_end: float, bin_width: float, bin_step: float, bin_method: str = 'full'):
    """Create bins allowing for various overlap and windowing schemes

    Parameters
    ----------
    range_start : float
        The start of the range of the bins.
    range_end : float
        The end of the range of the bins.
    bin_width : float
        The width of the bins.
    bin_step : float
        The step size between the bins.
    bin_method : str, optional
        The method for creating the bins.
        The default is 'full'.
        'full' : Only full-width bins, with bin edges starting at range_start and ending range_end
        'partial' : Includes partial width bins, with bin centers starting at range_start and ending range_end
        'extend' : Only full-width bins extending +- bin_width/2 outside the range, with bin centers starting at
                   range_start and ending range_end

    Returns
    -------
    bin_edges : ndarray
        The edges of the bins.
    bin_centers : ndarray
        The centers of the bins.
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

    Parameters
    ----------
    fig_obj : matplotlib.figure.Figure
        The figure object that contains the axis.
    ax_obj : matplotlib.axes.Axes
        The axis object that contains the graphic object.
    graphic_obj : graphics object (image, scatterplot, etc.)
        The graphic object that is to be plotted.
    gap : float, optional
        The gap between the axis and the colorbar.
    shrink : float, optional
        The shrink factor of the colorbar.
    label : str, optional
        The label of the colorbar.

    Returns
    -------
    cbar : matplotlib.colorbar.Colorbar
        The colorbar object.
    """

    ax_pos = ax_obj.get_position().bounds  # Axis position

    # Create new colorbar and get position
    cbar = fig_obj.colorbar(graphic_obj, ax=ax_obj, shrink=shrink, label=label)
    ax_obj.set_position(ax_pos)
    cbar_pos = cbar.ax.get_position().bounds

    # Set new colorbar position
    cbar.ax.set_position([ax_pos[0] + ax_pos[2] + gap, cbar_pos[1], cbar_pos[2], cbar_pos[3]])

    return cbar


def consecutive(data):
    """This function takes a list of values and returns a list of tuples.
    Each tuple contains the value, the start index, and the end index
    of a consecutive sequence of that value.

    Parameters
    ----------
    data : list
        A list of values.

    Returns
    -------
    list
        A list of tuples. Each tuple contains the value, the start index,
        and the end index of a consecutive sequence of that value.

    Examples
    --------
    >>> consecutive([1, 1, 1, 2, 2, 3, 3, 3, 3])
    [(1, 0, 2), (2, 3, 4), (3, 5, 8)]
    """
    vals = [v[0] for v in groupby(data)]
    cons = [sum(1 for i in g) for v, g in groupby(data)]

    start_inds = np.cumsum(np.insert(cons, 0, 0))
    end_inds = np.add(start_inds[0:-1], cons) - 1

    return list(zip(vals, start_inds, end_inds))


def find_flat(data: list, minsize: int = 100) -> list:
    """Finds the indices of flat regions in a 1D array.

    Parameters
    ----------
    data : list
        The data to search for flat regions.
    minsize : int, optional
        The minimum size of a flat region to be considered.

    Returns
    -------
    numpy.ndarray
    A boolean array of the same length as `data` with True values at the indices of flat regions.
    """
    inds = np.full((len(data)), False)

    # Find trains of consecutive equal data
    for c in consecutive(data):
        if c[2] - c[1] + 1 >= minsize:
            inds[c[1]:c[2] + 1] = True

    return inds


def hypnoplot(time: list, stage: list, ax: matplotlib.axes.Axes = None, plot_buffer: float = 0.8):
    """
    Plots the hypnogram

    Parameters
    ----------
    time : list
        Time vector in seconds.
    stage : list
        Sleep stage vector.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If not specified, a new figure will be created.
    plot_buffer : float, optional
        The amount of space to leave above and below the hypnogram.

    Returns
    -------
    None
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
    """Design a Butterworth bandpass filter.

    Parameters
    ----------
    lowcut : float
        Low cutoff frequency in Hz.
    highcut : float
        High cutoff frequency in Hz.
    fs : float
        Sample rate in Hz.
    order : int, optional
        Filter order.

    Returns
    -------
    sos : ndarray
        Second-order sections representation of the filter.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos


def butter_highpass(lowcut, fs, order=50):
    """Performs a zero-phase butterworth bandpass filter on SOS

    Parameters
    ----------
    lowcut : float
        Low cutoff frequency in Hz.
    highcut : float
        High cutoff frequency in Hz.
    fs : float
        Sample rate in Hz.
    order : int, optional
        Filter order.

    Returns
    -------
    filt_data : ndarray
        Filtered data
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    sos = butter(order, low, btype='hp', output='sos')
    return sos


def zscore_remove(data, crit, bad_inds, smooth_dur, detrend_dir):
    """Find artifacts by removing data until none left outside of criterion

    Parameters
    ----------
    data : array_like
        Data sequence
    crit : float
        Critical value in terms of std from mean
    bad_inds : array_like, optional
        Data to remove prior to procedure
    smooth_dur : int
        Duration (in samples) of smoothing window for mean filter
    detrend_dir : int
        Duration (in samples) of detrending window for mean filter

    Returns
    -------
    detected_artifacts : array_like
        Detected artifacts
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

        Parameters
    ----------
    data : array_like
        The data to be filtered.
    fs : float
        The sampling frequency of the data.
    hf_cut : float, optional
        The high-pass cutoff frequency for the high-frequency artifact detection.
        Default is 35 Hz.
    bb_cut : float, optional
        The high-pass cutoff frequency for the broadband artifact detection.
        Default is 0.1 Hz.
    crit_high : float, optional
        The z-score threshold for the high-frequency artifact detection.
        Default is 4.5.
    crit_broad : float, optional
        The z-score threshold for the broadband artifact detection.
        Default is 4.5.
    smooth_duration : float, optional
        The duration of the smoothing window in seconds. Default is 2 seconds.
    detrend_duration : float, optional
        The duration of the detrending window in seconds. Default is 5 minutes.

    Returns
    -------
    artifacts : array_like
        A boolean array of the same length as the input data, where True indicates an artifact.
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

        Parameters
    ----------
    data : ndarray
        The raw EEG data.
    fs : int
        The sampling frequency of the data.
    stages : pandas.DataFrame
        A pandas DataFrame containing the sleep stages.
    stats_table : pandas.DataFrame
        A pandas DataFrame containing the statistics of the extracted peaks.
    SOpow_hist : ndarray
        A 2D histogram of the SO-power.
    SO_cbins : ndarray
        The bins used to create the SO-power histogram.
    SO_power_norm : ndarray
        The normalized SO-power.
    SO_power_times : ndarray
        The times corresponding to the SO-power.
    SOphase_hist : ndarray
        A 2D histogram of the SO-phase.
    freq_cbins : ndarray
        The bins used to create the SO-phase histogram.

    Returns
    -------
    None
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
