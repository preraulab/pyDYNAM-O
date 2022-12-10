import timeit

import numpy as np
import pandas as pd
from joblib import cpu_count, Parallel, delayed
from scipy.interpolate import interp1d
from tqdm import tqdm

from pyTOD.SOPH import wrap_phase, get_SO_phase, SO_power_histogram, SO_phase_histogram
from pyTOD.TFpeaks import process_segments_params, detect_TFpeaks
from pyTOD.multitaper import multitaper_spectrogram
from pyTOD.utils import convertHMS, detect_artifacts, min_prominence, summary_plot


def run_example_data(data_range='segment', quality='fast', save_peaks=False, load_peaks=True):
    """Example data script

    Parameters
    ----------
    data_range : str, optional
        The range of data to use. Can be 'segment' or 'full'. Default: 'segment'
    quality : str, optional
        The quality of the TF-peak detection. Can be 'paper', 'precision', 'fast', or 'draft'. Default: 'fast'
    save_peaks : bool, optional
        Whether to save the TF-peak stats table to file. Default: False
    load_peaks : bool, optional
        Whether to load the TF-peak stats table from file. Default: True

    Returns
    -------
    None
    """
    # Load in data
    print('Loading in raw data...', end=" ")
    # EEG data and stages
    csv_data = pd.read_csv('data/' + data_range + '_data.csv', header=None)
    data = np.array(csv_data[0]).astype(np.float32)
    stages = pd.read_csv('data/' + data_range + '_stages.csv')
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

        stats_table, \
        SOpow_hist, freq_cbins, SO_cbins, SO_power_norm, SO_power_times, \
        SOphase_hist, freq_cbins, phase_cbins = run_TFpeaks_SOPH(data, fs, stages, downsample, segment_dur,
                                                                 merge_thresh, max_merges, trim_volume, True)

        if save_peaks:
            print('Writing stats_table to file...', end=" ")
            stats_table.to_csv('data/' + data_range + '_peaks.csv')
            print('Done')
    else:
        stats_table = pd.read_csv('data/' + data_range + '_peaks.csv')

        SOpow_hist, freq_cbins, SO_cbins, SO_power_norm, SO_power_times, \
        SOphase_hist, freq_cbins, phase_cbins = compute_SOPHs(data, fs, stages, stats_table)

        summary_plot(data, fs, stages, stats_table, SOpow_hist, SO_cbins, SO_power_norm, SO_power_times, SOphase_hist,
                     freq_cbins)


def compute_TFpeaks(data=None, fs=None, downsample=None, segment_dur=30, merge_thresh=8,
                    max_merges=np.inf, trim_volume=0.8, verbose=True):
    """Extract TF-peaks from the data using the pyTOD packages

    Parameters
    ----------
    data : array_like
        The data to be analyzed.
    fs : float
        The sampling frequency of the data.
    downsample : array_like, optional
        The downsampling factor for each dimension.
    segment_dur : float, optional
        The duration of each segment to be analyzed.
    merge_thresh : float, optional
        The threshold for merging peaks.
    max_merges : int, optional
        The maximum number of merges to perform.
    trim_volume : float, optional
        The fraction of the volume to trim from the peaks.
    verbose : bool, optional
        Whether to print progress messages.

    Returns
    -------
    stats_table : pandas.DataFrame
        A table containing the TF-peak statistics.
    """
    # Handle no downsampling
    if downsample is None:
        downsample = []

    # %% Compute the multitaper spectrogram (MTS)
    frequency_range = [0, 30]  # MTS frequency range
    taper_params = [2, 3]  # Set taper params
    time_bandwidth = taper_params[0]  # Set time-half bandwidth
    num_tapers = taper_params[1]  # Set number of tapers (optimal is time_bandwidth*2 - 1)
    window_params = [1, .05]  # Window size is 4s with step size of 1s
    min_nfft = 2 ** 10  # NFFT
    detrend_opt = 'constant'  # constant detrend = mean zero
    multiprocess = True  # use multiprocessing
    cpus = cpu_count()  # use max cores in multiprocessing
    weighting = 'unity'  # weight each taper at 1

    # MTS frequency resolution
    df = taper_params[0] / window_params[0] * 2

    # Set min duration and bandwidth based on spectral parameters
    dur_min = window_params[0] / 2
    bw_min = df / 2

    # Max duration and bandwidth are set to be large values
    dur_max = 5
    bw_max = 15

    # Set minimal peak height based on confidence interval lower bound of MTS
    prom_min = min_prominence(taper_params[1], 0.95)

    # Compute the multitaper spectrogram
    spect, stimes, sfreqs = multitaper_spectrogram(data, fs, frequency_range, time_bandwidth, num_tapers,
                                                   window_params,
                                                   min_nfft, detrend_opt, multiprocess, cpus,
                                                   weighting, False, False, False, False)

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
                 bw_min, bw_max, prom_min, False, False)

    #  Run jobs in parallel
    if verbose:
        print('Running peak detection in parallel with ' + str(cpus) + ' jobs...')
        tic_outer = timeit.default_timer()

    # # Use this if you would like to run a test segment
    # stats_table = detect_TFpeaks(spect_baseline[:, window_idxs[0]], start_times[0], *dp_params)

    stats_tables = Parallel(n_jobs=cpus)(delayed(detect_TFpeaks)(
        spect_baseline[:, window_idxs[num_window]], start_times[num_window], *dp_params)
                                         for num_window in tqdm(range(num_windows)))

    stats_table = pd.concat(stats_tables, ignore_index=True)

    if verbose:
        toc_outer = timeit.default_timer()
        print('Took ' + convertHMS(toc_outer - tic_outer))

    # Fix the stats_table to sort by time and reset labels
    del stats_table['label']
    stats_table.sort_values('peak_time')
    stats_table.reset_index()

    return stats_table


def compute_SOPHs(data, fs, stages, stats_table, verbose=True):
    """Compute SO-power and SO-phase histograms for detected peaks

    Parameters
    ----------
    data : numpy.ndarray
        Time series data
    fs : float
        Sampling frequency
    stages : pandas.DataFrame
        Time/Stage dataframe
    stats_table : pandas.DataFrame
        Peak statistics table
    verbose : bool, optional
        Verbose flag, by default True

    Returns
    -------
    SOpow_hist : numpy.ndarray
        SO-power histogram
    freq_cbins : numpy.ndarray
        Frequency bins for SO-power histogram
    SO_cbins : numpy.ndarray
        SO-power bins for SO-power histogram
    SO_power_norm : numpy.ndarray
        Normalized SO-power for each peak
    SO_power_times : numpy.ndarray
        Time of each peak in SO-power histogram
    SOphase_hist : numpy.ndarray
        SO-phase histogram
    freq_cbins : numpy.ndarray
        Frequency bins for SO-phase histogram
    phase_cbins : numpy.ndarray
        Phase bins for SO-phase histogram
    """
    if verbose:
        print('Detecting artifacts...', end=" ")
        tic_art = timeit.default_timer()
    artifacts = detect_artifacts(data, fs)
    if verbose:
        print('Took ' + convertHMS(timeit.default_timer() - tic_art))

    # Compute peak phase
    t = np.arange(len(data)) / fs
    phase = get_SO_phase(data, fs)

    # Compute peak phase for plotting
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

    # Only consider non-Wake peaks at non-artifact times
    stats_table = stats_table.query('stage<5 and not artifact_time')

    if verbose:
        print('Computing SO-Power Histogram...', end=" ")
        tic_SOpow = timeit.default_timer()

    SOpow_hist, freq_cbins, SO_cbins, SO_power_norm, SO_power_times = \
        SO_power_histogram(stats_table.peak_time, stats_table.peak_frequency,
                           data, fs, artifacts, freq_range=[4, 25], freq_window=[1, 0.2],
                           norm_method='shift', verbose=False)
    if verbose:
        print('Took ' + convertHMS(timeit.default_timer() - tic_SOpow))

        print('Computing SO-Phase Histogram...', end=" ")
        tic_SOhase = timeit.default_timer()
    SOphase_hist, freq_cbins, phase_cbins = \
        SO_phase_histogram(stats_table.peak_time, stats_table.peak_frequency,
                           data, fs, freq_range=[4, 25], freq_window=[1, 0.2], verbose=False)
    if verbose:
        print('Took ' + convertHMS(timeit.default_timer() - tic_SOhase))

    return SOpow_hist, freq_cbins, SO_cbins, SO_power_norm, SO_power_times, SOphase_hist, freq_cbins, phase_cbins


def run_TFpeaks_SOPH(data, fs, stages, downsample=None, segment_dur=30, merge_thresh=8,
                     max_merges=np.inf, trim_volume=0.8, plot_on=True):
    """Extracts TF-peaks then cmputes SO-power and Phase histograms using the pyTOD package

    Parameters
    ----------
    data : ndarray
        The data to be analyzed.
    fs : int
        The sampling frequency of the data.
    stages : ndarray
        The sleep stages of the data.
    downsample : int, optional
        The downsampling factor to be applied to the data. The default is None.
    segment_dur : int, optional
        The duration of each segment in seconds. The default is 30.
    merge_thresh : int, optional
        The minimum number of peaks that must be present in a segment for it to be considered a peak. The default is 8.
    max_merges : int, optional
        The maximum number of merges that can be performed on a segment. The default is np.inf.
    trim_volume : float, optional
        The fraction of the data to be trimmed from the beginning and end of each segment. The default is 0.8.
    plot_on : bool, optional
        Whether or not to plot the summary figure. The default is True.

    Returns
    -------
    stats_table : ndarray
        A table containing the statistics of each TF-peak segment.
    SOpow_hist : ndarray
        A histogram of the SO-power in each TF-peak segment.
    freq_cbins : ndarray
        The frequency bins used to compute the SO-power histogram.
    SO_cbins : ndarray
        The SO-power bins used to compute the SO-power histogram.
    SO_power_norm : ndarray
        The normalized SO-power in each TF-peak segment.
    SO_power_times : ndarray
        The time points corresponding to each SO-power value in each TF-peak segment.
    SOphase_hist : ndarray
        A histogram of the SO-phase in each TF-peak segment.
    freq_cbins : ndarray
        The frequency bins used to compute the SO-phase histogram.
    phase_cbins : ndarray
        The SO-phase bins used to compute the SO-phase histogram.
    """

    # Extract TF-peaks and compute peak statistics table
    stats_table = compute_TFpeaks(data, fs, downsample, segment_dur, merge_thresh, max_merges, trim_volume)

    # Compute SO-power and SO-phase Histograms
    SOpow_hist, freq_cbins, SO_cbins, SO_power_norm, SO_power_times, \
    SOphase_hist, freq_cbins, phase_cbins = compute_SOPHs(data, fs, stages, stats_table)

    # Create summary plot if selected
    if plot_on:
        summary_plot(data, fs, stages, stats_table, SOpow_hist, SO_cbins, SO_power_norm, SO_power_times, SOphase_hist,
                     freq_cbins)

    return stats_table, SOpow_hist, freq_cbins, SO_cbins, SO_power_norm, SO_power_times, SOphase_hist, freq_cbins, \
           phase_cbins


if __name__ == '__main__':
    quality = 'fast'  # Quality setting 'precision','fast', or 'draft'
    data_range = 'night'  # Segment vs. night
    save_peaks = False  # Save csv of peaks if computing
    load_peaks = True  # Load from csv vs computing

    # Run example data
    run_example_data(data_range, quality, save_peaks, load_peaks)
