import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import colorcet  # It looks like this is not used, but it puts the colorcet cmaps in matplotlib
from extractTFPeaks import *
from SOhistograms import *


def run_example_data(data_range='segment', quality='fast', save_peaks=False, load_peaks=True):
    # Load in data
    print('Loading in raw data...', end=" ")
    # EEG data and stages
    csv_data = pd.read_csv('data/' + data_range + '_data.csv', header=None)
    data = np.array(csv_data[0]).astype(np.float32)
    stages = pd.read_csv('data/' + data_range + '_stages.csv')

    # Sampling Frequency
    fs = 100
    print('Done')

    # Sampling Frequency
    fs = 100

    if not load_peaks:
        # %% DETECT TF-PEAKS
        if quality == 'paper':
            downsample = []
            segment_dur = 60
            merge_thresh = 8
        elif quality == 'precision':
            downsample = []
            segment_dur = 30
            merge_thresh = 8
        elif quality == 'fast':
            downsample = [2, 2]
            segment_dur = 30
            merge_thresh = 11
        elif quality == 'draft':
            downsample = [5, 1]
            segment_dur = 30
            merge_thresh = 13
        else:
            raise ValueError("Specify settings 'precision', 'fast', or 'draft'")

        trim_volume = 0.8  # Trim TF-peaks to retain 80 of original volume
        max_merges = np.inf  # Set limit on number merges if needs be

        stats_table = run_TFpeak_detection(data, fs, downsample, segment_dur, merge_thresh, max_merges, trim_volume)

        if save_peaks:
            print('Writing stats_table to file...')
            stats_table.to_csv('data/' + data_range + '_peaks.csv')
            print('Done')
    else:
        stats_table = pd.read_csv('data/' + data_range + '_peaks.csv')

    SOpow_hist, freq_cbins, SO_cbins, SO_power_norm, SO_power_times, \
    SOphase_hist, freq_cbins, phase_cbins = compute_SOPHs(data, fs, stages, stats_table)

    plot_figure(data, fs, stages, stats_table, SOpow_hist, SO_cbins, SO_power_norm, SO_power_times, SOphase_hist,
                freq_cbins)


def run_TFpeak_detection(data=None, fs=None, downsample=None, segment_dur=60, merge_thresh=8,
                         max_merges=np.inf, trim_volume=0.8):
    # Handle no downsampling
    if downsample is None:
        downsample = []

    # %% COMPUTE MULTITAPER SPECTROGRAM
    # Number of jobs to use
    n_jobs = max(cpu_count() - 1, 1)

    # Limit frequencies from 0 to 25 Hz
    frequency_range = [0, 30]

    taper_params = [2, 3]  # Set taper params
    time_bandwidth = taper_params[0]  # Set time-half bandwidth
    num_tapers = taper_params[1]  # Set number of tapers (optimal is time_bandwidth*2 - 1)
    window_params = [1, .05]  # Window size is 4s with step size of 1s
    min_nfft = 2 ** 10  # NFFT
    detrend_opt = 'constant'  # constant detrend
    multiprocess = True  # use multiprocessing
    cpus = n_jobs  # use max cores in multiprocessing
    weighting = 'unity'  # weight each taper at 1
    plot_on = False  # plot spectrogram
    clim_scale = False  # do not auto-scale colormap
    verbose = False  # print extra info
    xyflip = False  # do not transpose spect output matrix

    # MTS frequency resolution
    df = taper_params[0] / window_params[0] * 2

    # Set min duration and bandwidth based on spectral parameters
    dur_min = window_params[0] / 2
    bw_min = df / 2

    # Max duration and bandwidth are set to be large values
    dur_max = 5
    bw_max = 15

    # Set minimal peak height based on confidence interval lower bound of MTS
    chi2_df = 2 * taper_params[1]
    alpha = 0.95
    prom_min = -pow2db(chi2_df / chi2.ppf(alpha / 2 + 0.5, chi2_df)) * 2

    # Compute the multitaper spectrogram
    spect, stimes, sfreqs = multitaper_spectrogram(data, fs, frequency_range, time_bandwidth, num_tapers,
                                                   window_params,
                                                   min_nfft, detrend_opt, multiprocess, cpus,
                                                   weighting, plot_on, clim_scale, verbose, xyflip)

    # Define spectral coords dx dy
    d_time = stimes[1] - stimes[0]
    d_freq = sfreqs[1] - sfreqs[0]

    # Remove baseline
    baseline = np.percentile(spect, 2, axis=1, keepdims=True)
    spect_baseline = np.divide(spect, baseline)

    # Set the size of the spectrogram samples
    window_idxs, start_times = process_segments_params(segment_dur, stimes)
    num_windows = len(start_times)

    # Set up the parameters to pass to each window
    dp_params = (d_time, d_freq, merge_thresh, max_merges, trim_volume, downsample, dur_min, dur_max,
                 bw_min, bw_max, prom_min, plot_on, verbose)

    #  Run jobs in parallel
    print('Running peak detection in parallel with ' + str(n_jobs) + ' jobs...')
    tic_outer = timeit.default_timer()

    # # First segment test
    # stats_table = detect_tfpeaks(spect_baseline[:, window_idxs[0]], start_times[0], *dp_params)

    stats_tables = Parallel(n_jobs=8)(delayed(detect_tfpeaks)(
        spect_baseline[:, window_idxs[num_window]], start_times[num_window], *dp_params)
                                           for num_window in tqdm(range(num_windows)))

    stats_table = pd.concat(stats_tables, ignore_index=True)

    toc_outer = timeit.default_timer()
    print('Peak detection took ' + convertHMS(toc_outer - tic_outer))

    # Fix the stats_table to sort by time and reset labels
    del stats_table['label']
    stats_table.sort_values('peak_time')
    stats_table.reset_index()

    return stats_table


def compute_SOPHs(data, fs, stages, stats_table):
    """Compute SO-power and SO-phase histograms for detected peaks

    :param data: Time series data
    :param fs: Sampling frequency
    :param stages: Time/Stage dataframe
    :param stats_table: Peak statistics table
    :return: SOpow_hist, freq_cbins, SO_cbins, SO_power_norm, SO_power_times, SOphase_hist, freq_cbins, phase_cbins
    """
    print('Detecting artifacts...', end=" ")
    artifacts = detect_artifacts(data, fs)
    print('Done')

    # Compute peak phase
    t = np.arange(len(data)) / fs
    phase = get_SO_phase(data, fs)

    # Compute peak phase
    peak_interp = interp1d(t, phase)
    peak_phase = wrap_phase(peak_interp(stats_table['peak_time'].values))
    stats_table['phase'] = peak_phase

    # Compute peak stage
    stage_interp = interp1d(stages.Time.values, stages.Stage.values, kind='previous',
                            fill_value="extrapolate")

    peak_stages = stage_interp(stats_table.peak_time)
    stats_table['stage'] = peak_stages

    # Compute artifact peaks
    art_interp = interp1d(t, artifacts, kind='nearest',
                          fill_value="extrapolate")
    artifact_time = art_interp(stats_table.peak_time)
    stats_table['artifact_time'] = artifact_time.astype(bool)

    # Only consider sleep peaks at non-artifact times
    stats_table = stats_table.query('stage<5 and not artifact_time')

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

    return SOpow_hist, freq_cbins, SO_cbins, SO_power_norm, SO_power_times, SOphase_hist, freq_cbins, phase_cbins


def plot_figure(data, fs, stages, stats_table, SOpow_hist, SO_cbins, SO_power_norm, SO_power_times, SOphase_hist, freq_cbins):
    # Number of jobs to use
    n_jobs = max(cpu_count(), 1)

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
    peak_size = stats_table['volume'] / 250
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
    im.set_clim(clims[0], clims[1])
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
    im.set_clim(clims[0], clims[1])
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
    quality = 'fast'
    data_range = 'night'
    save_peaks = True
    load_peaks = False
    run_example_data(data_range, quality, save_peaks, load_peaks)

