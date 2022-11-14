import copy
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d
from scipy.signal import convolve, butter, hilbert, sosfiltfilt
from extractTFPeaks import *
from matplotlib.colors import ListedColormap
import colorcet  # It looks like this is not used, but it puts the colorcet cmaps in matplotlib


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


def detect_artifacts(data, fs, hf_cut=25, bb_cut=0.1, crit_high=4.5, crit_broad=4.5,
                     smooth_duration=2, detrend_duration=5*60):
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

    hf_artifacts = zscore_remove(data_high, crit_high, bad_inds, smooth_dur=smooth_duration * fs, detrend_dir=detrend_duration * fs)
    bb_artifacts = zscore_remove(data_broad, crit_broad, bad_inds, smooth_dur=smooth_duration * fs, detrend_dir=detrend_duration * fs)

    return np.logical_or(hf_artifacts, bb_artifacts)


def wrap_phase(phase: np.ndarray) -> np.ndarray:
    """Wrap phase from -pi to pi

    :param phase: Unwrapped phase
    :return: Wrapped phase
    """
    return np.angle(np.exp(1j * phase))


def get_SO_phase(data, fs, lowcut=0.3, highcut=1.5, order=50):
    """Computes unwrapped slow oscillation phase
    Note: Phase is unwrapped because wrapping does not return to original angle given the unwrapping algorithm

    :param data: EEG time series data
    :param fs: Sampling frequency
    :param lowcut: Bandpass low-end cutoff
    :param highcut: Bandpass high-end cutoff
    :param order: Filter order
    :return: Unwrapped phase
    """
    sos = butter_bandpass(lowcut, highcut, fs, 10)
    data_filt = sosfiltfilt(sos, data)

    analytic_signal = hilbert(data_filt)
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


def SO_power_histogram(peak_times, peak_freqs, data, fs, artifacts, freq_range=None, freq_window=None,
                       SO_range=None, SO_window=None, norm_method='percent', min_time_in_bin=1, verbose=True):
    """Compute a slow oscillation power histogram

    :param peak_times: Peak times
    :param peak_freqs: Peak frequencies
    :param data: Time series data
    :param fs: Sampling frequency
    :param artifacts: Artifacts list
    :param freq_range: Frequency range
    :param freq_window: Frequency window width and step size
    :param SO_range: SO-power range
    :param SO_window: SO-power window width and step size
    :param norm_method: Normalization method ('shift', 'percent', or 'none')
    :param min_time_in_bin: Minimum time in bin to count towards histogram
    :param verbose: Verbose flag
    :return: SOpow_hist, freq_cbins, SO_cbins, SO_power_norm, SOpow_times
    """
    good_data = copy.deepcopy(data)
    good_data[artifacts] = np.nan
    SO_power, SOpow_times = get_SO_power(good_data, fs, lowcut=0.3, highcut=1.5)
    inds = ~np.isnan(SO_power)

    # Normalize  power
    if type(norm_method) == float:
        shift_ptile = norm_method
        norm_method = 'shift'
    else:
        shift_ptile = 5

    if norm_method == 'percentile' or norm_method == 'percent':
        low_val = 1
        high_val = 99
        ptile = np.percentile(SO_power[inds], [low_val, high_val])
        SO_power_norm = SO_power - ptile[0]
        SO_power_norm = np.divide(SO_power_norm, (ptile[1] - ptile[0])) * 100
    elif norm_method == 'shift' or norm_method == 'p5shift':
        ptile = np.percentile(SO_power[inds], [shift_ptile])
        SO_power_norm = np.subtract(SO_power, ptile)
    elif norm_method == 'absolute' or norm_method == 'none':
        SO_power_norm = SO_power
    else:
        raise ValueError("Not a valid normalization option, choose 'p5shift', 'percent', or 'none'")

    if freq_range is None:
        freq_range = [np.min(peak_freqs), np.max(peak_freqs)]

    if freq_window is None:
        freq_window = [(freq_range[1] - freq_range[0]) / 5, (freq_range[1] - freq_range[0]) / 100]

    freq_bin_edges, freq_cbins = create_bins(freq_range[0], freq_range[1], freq_window[0], freq_window[1], 'partial')
    num_freqbins = len(freq_cbins)

    if SO_range is None:
        SO_range = [np.min(SO_power_norm[inds]), np.max(SO_power_norm[inds])]

    if SO_window is None:
        SO_window = [(SO_range[1] - SO_range[0]) / 5, (SO_range[1] - SO_range[0]) / 100]

    SO_bin_edges, SO_cbins = create_bins(SO_range[0], SO_range[1], SO_window[0], SO_window[1], 'partial')
    num_SObins = len(SO_cbins)

    if verbose:
        print('  SO-Power Histogram Settings:')
        print('    Normalization Method: ' + str(norm_method))
        print('    Frequency Window Size: ' + str(freq_window[0]) + ' Hz, Window Step: ' + str(freq_window[1]) + ' Hz')
        print('    Frequency Range: ', str(freq_range[0]) + '-' + str(freq_range[1]) + ' Hz')
        print('    SO-Power Window Size: ' + str(SO_window[0]) + ', Window Step: ' + str(SO_window[1]))
        print('    SO-Power Range: ' + str(SO_range[0]) + '-', str(SO_range[1]))
        print('    Minimum time required in each SO-Power bin: ' + str(min_time_in_bin) + ' min')

    # Initialize SO_pow x freq matrix
    SOpow_hist = np.empty(shape=(num_freqbins, num_SObins)) * np.nan

    # Initialize time in bin
    time_in_bin = np.zeros((num_SObins, 1))

    # Compute peak phase
    pow_interp = interp1d(SOpow_times[inds], SO_power_norm[inds], fill_value="extrapolate")
    peak_SOpow = pow_interp(peak_times)

    # SO-power time step size
    d_stimes = SOpow_times[1] - SOpow_times[0]

    for s_bin in range(num_SObins):
        TIB_inds = np.logical_and(SO_power_norm >= SO_bin_edges[0, s_bin], SO_power_norm < SO_bin_edges[1, s_bin])
        SO_inds = np.logical_and(peak_SOpow >= SO_bin_edges[0, s_bin], peak_SOpow < SO_bin_edges[1, s_bin])

        # Time in bin in minutes
        time_in_bin[s_bin] = (np.sum(TIB_inds) * d_stimes / 60)

        # Check for min time in bin
        if time_in_bin[s_bin] < min_time_in_bin:
            continue

        if np.sum(SO_inds):
            for f_bin in range(num_freqbins):
                # Get indices in freq bin
                freq_inds = np.logical_and(peak_freqs >= freq_bin_edges[0, f_bin],
                                           peak_freqs < freq_bin_edges[1, f_bin])

                # Fill histogram with peak rate in this freq / SOpow bin
                SOpow_hist[f_bin, s_bin] = np.sum(SO_inds & freq_inds) / time_in_bin[s_bin]
        else:
            SOpow_hist[:, s_bin] = np.nan

    return SOpow_hist, freq_cbins, SO_cbins, SO_power_norm, SOpow_times


def SO_phase_histogram(peak_times, peak_freqs, data, fs, freq_range=None, freq_window=None,
                       phase_range=None, phase_window=None, min_time_in_bin=1, verbose=True):
    """Compute a slow oscillation phase histogram

    :param peak_times: Peak times
    :param peak_freqs: Peak frequencies
    :param data: Time series data
    :param fs: Sampling frequency
    :param freq_range: Frequency range
    :param freq_window: Frequency window width and step size
    :param phase_range: SO-power range
    :param phase_window: SO-power window width and step size
    :param min_time_in_bin: Minimum time in bin to count towards histogram
    :param verbose: Verbose flag
    :return: SOphase_hist, freq_cbins, phase_cbins
    """
    SO_phase = wrap_phase(get_SO_phase(data, fs, lowcut=0.3, highcut=1.5))

    # Set defaults
    if freq_range is None:
        freq_range = [np.min(peak_freqs), np.max(peak_freqs)]

    if not freq_window:
        freq_window = [(freq_range[1] - freq_range[0]) / 5, (freq_range[1] - freq_range[0]) / 100]

    freq_bin_edges, freq_cbins = create_bins(freq_range[0], freq_range[1],
                                             freq_window[0], freq_window[1], 'partial')
    num_freqbins = len(freq_cbins)

    if phase_range is None:
        phase_range = [-np.pi, np.pi]

    if phase_window is None:
        phase_window = [(2 * np.pi) / 5, (2 * np.pi) / 100]

    # Extend bins to wrap around
    phase_bin_edges, phase_cbins = create_bins(phase_range[0], phase_range[1],
                                               phase_window[0], phase_window[1], 'extend')
    num_phasebins = len(phase_cbins)

    if verbose:
        print('  SO-Power Histogram Settings:')
        print('    Frequency Window Size: ' + str(freq_window[0]) + ' Hz, Window Step: ' + str(freq_window[1]) + ' Hz')
        print('    Frequency Range: ', str(freq_range[0]) + '-' + str(freq_range[1]) + ' Hz')
        print('    SO-Phase Window Size: ' + str(phase_window[0] / np.pi) + 'π, Window Step: ' +
              str(phase_window[1] / np.pi) + 'π')
        print('    SO-Phase Range: ' + str(phase_range[0] / np.pi) + 'π - ', str(phase_range[1] / np.pi) + 'π')
        print('    Minimum time required in each phase bin: ' + str(min_time_in_bin) + ' min')

    # Initialize SO_pow x freq matrix
    SOphase_hist = np.empty(shape=(num_freqbins, num_phasebins)) * np.nan

    # Initialize time in bin
    time_in_bin = np.zeros((num_phasebins, 1))

    # Compute peak phase
    inds = ~np.isnan(SO_phase)
    pow_interp = interp1d(np.arange(len(data[inds])) / fs, SO_phase[inds], fill_value="extrapolate")
    peak_SOphase = pow_interp(peak_times)

    for p_bin in range(num_phasebins):
        if phase_bin_edges[0, p_bin] <= -np.pi:
            wrapped_edge_lowlim = phase_bin_edges[0, p_bin] + (2 * np.pi)
            TIB_inds = np.logical_or(SO_phase >= wrapped_edge_lowlim,
                                     SO_phase < phase_bin_edges[1, p_bin])
            phase_inds = np.logical_or(peak_SOphase >= wrapped_edge_lowlim,
                                       peak_SOphase < phase_bin_edges[1, p_bin])
        elif phase_bin_edges[1, p_bin] >= np.pi:
            wrapped_edge_highlim = phase_bin_edges[1, p_bin] - (2 * np.pi)
            TIB_inds = np.logical_or(SO_phase < wrapped_edge_highlim,
                                     SO_phase >= phase_bin_edges[0, p_bin])
            phase_inds = np.logical_or(peak_SOphase < wrapped_edge_highlim,
                                       peak_SOphase >= phase_bin_edges[0, p_bin])
        else:
            TIB_inds = np.logical_and(SO_phase >= phase_bin_edges[0, p_bin],
                                      SO_phase < phase_bin_edges[1, p_bin])
            phase_inds = np.logical_and(peak_SOphase >= phase_bin_edges[0, p_bin],
                                        peak_SOphase < phase_bin_edges[1, p_bin])

        time_in_bin[p_bin] = (np.sum(TIB_inds) * (1 / fs) / 60)

        if time_in_bin[p_bin] < min_time_in_bin:
            continue

        if np.sum(phase_inds):
            for f_bin in range(num_freqbins):
                # Get indices in freq bin
                freq_inds = np.logical_and(peak_freqs >= freq_bin_edges[0, f_bin],
                                           peak_freqs < freq_bin_edges[1, f_bin])

                # Fill histogram with count of peaks in this freq / SOpow bin
                SOphase_hist[f_bin, p_bin] = np.sum(phase_inds & freq_inds) / time_in_bin[p_bin]
        else:
            SOphase_hist[:, p_bin] = 0

    # Normalize
    SOphase_hist = SOphase_hist / np.nansum(SOphase_hist, axis=1)[:, np.newaxis]

    return SOphase_hist, freq_cbins, phase_cbins


def plot_figure():
    # Load in data
    print('Loading in raw data...', end=" ")
    # EEG data
    csv_data = pd.read_csv('data/night_data.csv', header=None)
    data = np.array(csv_data[0]).astype(np.float32)
    stages = pd.read_csv('data/night_stages.csv')
    stages.Time -= 3180

    # Sampling Frequency
    fs = 100
    print('Done')

    # Load in stats table of peak data
    print('Loading in TF-peaks stats data...', end=" ")
    stats_table = pd.read_csv('data/night_peaks.csv')
    print('Done')

    # Compute peak phase
    t = np.arange(len(data)) / fs
    phase = get_SO_phase(data, fs)

    # Compute peak phase
    peak_interp = interp1d(t, phase)
    peak_phase = wrap_phase(peak_interp(stats_table['peak_time'].values))
    stats_table['phase'] = peak_phase

    print('Detecting artifacts...', end=" ")
    artifacts = detect_artifacts(data, fs)
    print('Done')

    # Compute peak phase
    stage_interp = interp1d(stages.Time.values, stages.Stage.values, kind='previous',
                                        fill_value="extrapolate")
    # Compute peak stage
    peak_stages = stage_interp(stats_table.peak_time)
    stats_table['stage'] = peak_stages

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
    verbose = False  # print extra info
    xyflip = False  # do not transpose spect output matrix

    # Compute the multitaper spectrogram
    print('Computing visualization spectrogram...', end=" ")
    spect, stimes, sfreqs = multitaper_spectrogram(data, fs, frequency_range, time_bandwidth, num_tapers,
                                                   window_params,
                                                   min_nfft, detrend_opt, multiprocess, cpus,
                                                   weighting, plot_on, clim_scale, verbose, xyflip)
    print('Done')

    print('Computing SO-Power Histogram...', end=" ")
    SOpow_hist, freq_cbins, SO_cbins, SO_power_norm, SO_power_times = \
        SO_power_histogram(stats_table.peak_time, stats_table.peak_frequency,
                           data, fs, artifacts, freq_range=[4, 25], freq_window=[1, 0.2],
                           norm_method='shift', verbose=False)
    print('Done')

    print('Computing SO-Phase Histogram...', end=" ")
    SOphase_hist, freq_cbins, phase_cbins = \
        SO_phase_histogram(stats_table.peak_time, stats_table.peak_frequency,
                           data, fs, freq_range=[4, 25], freq_window=[1, 0.2], verbose=False)
    print('Done')

    # %%  Plot figure
    print('Plotting figure...', end=" ")
    fig = plt.figure(figsize=(8.5 * .8, 11 * .8))
    gs = gridspec.GridSpec(nrows=5, ncols=2, height_ratios=[0.01, .2, .01, .2, .3],
                           width_ratios=[.5, .5],
                           hspace=0.4, wspace=0.5,
                           left=0.1, right=0.90,
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
    hypnoplot(stages.Time.values / 3600, stages.Stage.values, ax0)
    ax0.set_xticks([])
    ax0.set_yticklabels(ax0.get_yticklabels(), fontsize=6)
    ax0.set_title('EEG Spectrogram')

    # Plot spectrogram
    extent = stimes[0] / 3600, stimes[-1] / 3600, frequency_range[1], frequency_range[0]
    plt.axes(ax1)
    im = ax1.imshow(pow2db(spect), extent=extent, aspect='auto')
    clims = np.percentile(pow2db(spect[~np.isnan(spect)]), [5, 98])
    im.set_clim(clims[0], clims[1])
    ax1.set_ylabel('Frequency (Hz)')
    ax1.invert_yaxis()
    ax1.set_xticks([])
    im.set_cmap(plt.cm.get_cmap('cet_rainbow4'))
    cbar = outside_colorbar(fig, ax1, im, gap=0.01, shrink=0.8)
    cbar.set_label("Power (dB)", fontsize=10)
    cbar.ax.tick_params(labelsize=7)

    # Plot SO_power
    ax2.plot(np.divide(SO_power_times, 3600), SO_power_norm, 'b', linewidth=1)
    ax2.set_xlim([SO_power_times[0] / 3500, SO_power_times[-1] / 3600])
    ax2.set_xlabel('Time (hrs)')

    # Plot the scatter plot
    peak_size = stats_table['volume'] / 300
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
    cbar.set_label("Phase (rad)", fontsize=10)
    cbar.ax.tick_params(labelsize=7)

    ax3.set_xlabel('Time (hrs)')
    ax3.set_ylabel('Frequency (Hz)')
    ax3.set_title('Extracted Time-Frequency Peaks')

    # SO-power Histogram
    ax4.set_title('SO-power Histogram')
    extent = SO_cbins[0], SO_cbins[-1], freq_cbins[-1], freq_cbins[0]
    plt.axes(ax4)
    im = ax4.imshow(SOpow_hist, extent=extent, aspect='auto')
    clims = np.percentile(SOpow_hist, [5, 98])
    im.set_clim(0, 8.5)  #clims[0], clims[1])
    ax4.set_ylabel('Frequency (Hz)')
    ax4.invert_yaxis()
    im.set_cmap(plt.cm.get_cmap('cet_gouldian'))
    cbar = outside_colorbar(fig, ax4, im, gap=0.01, shrink=0.6)
    cbar.set_label("Density", fontsize=10)
    cbar.ax.tick_params(labelsize=7)
    ax4.set_xlabel('% SO-Power')

    # SO-phase Histogram
    ax5.set_title('SO-power Histogram')
    extent = -np.pi, np.pi, freq_cbins[-1], freq_cbins[0]
    plt.axes(ax5)
    im = ax5.imshow(SOphase_hist, extent=extent, aspect='auto')
    clims = np.percentile(SOphase_hist, [5, 98])
    im.set_clim(0.0085, 0.0118)  # ]clims[0], clims[1])
    ax5.set_ylabel('Frequency (Hz)')
    ax5.invert_yaxis()
    im.set_cmap(plt.cm.get_cmap('magma'))
    cbar = outside_colorbar(fig, ax5, im, gap=0.01, shrink=0.6)
    cbar.set_label("Proportion", fontsize=10)
    cbar.ax.tick_params(labelsize=7)
    plt.xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi], ['-π', '-π/2', '0', 'π/2', 'π'])
    plt.xlabel('SO-Phase (rad)')

    print('Done')
    plt.show()


if __name__ == '__main__':
    # Load full night extracted TF-peaks and plot figure so far
    plot_figure()
