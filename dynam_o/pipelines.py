""" Advanced users are recommended to edit these pipelines based on custom needs """
import timeit

import numpy as np
import pandas as pd
from joblib import cpu_count, Parallel, delayed
from scipy.interpolate import interp1d
from tqdm import tqdm

from dynam_o.SOPH import so_power_histogram, so_phase_histogram
from dynam_o.TFpeaks import process_segments_params, detect_tfpeaks
from dynam_o.multitaper import multitaper_spectrogram
from dynam_o.utils import convert_hms, detect_artifacts, min_prominence, summary_plot


def compute_tfpeaks(data=None, fs=None, downsample=None, segment_dur=30, merge_thresh=8,
                    max_merges=np.inf, trim_volume=0.8, verbose=True):
    """Extract TF-peaks from the data using the dynam_o packages

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

    # Compute the multitaper spectrogram (MTS)
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
    spect_baseline = spect / baseline

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
    # stats_table = detect_tfpeaks(spect_baseline[:, window_idxs[0]], start_times[0], *dp_params)

    stats_tables = Parallel(n_jobs=cpus)(delayed(detect_tfpeaks)(
        spect_baseline[:, window_idxs[num_window]], start_times[num_window], *dp_params)
                                         for num_window in tqdm(range(num_windows)))

    stats_table = pd.concat(stats_tables, ignore_index=True)

    if verbose:
        toc_outer = timeit.default_timer()
        # noinspection PyUnboundLocalVariable
        print('Took ' + convert_hms(toc_outer - tic_outer))

    # Fix the stats_table to sort by time and reset labels
    del stats_table['label']
    stats_table.sort_values('peak_time')
    stats_table.reset_index()

    return stats_table


def compute_sophs(data, fs, stages, stats_table, stages_include=None, norm_method='percent', verbose=True):
    """Compute SO-power and SO-phase histograms for detected peaks

    Parameters
    ----------
    data : numpy.ndarray
        Time series data
    fs : int
        Sampling frequency
    stages : pandas.DataFrame
        Time/Stage dataframe
    stats_table : pandas.DataFrame
        Peak statistics table
    stages_include : array_like
        A list of stages within which peaks are included in the histograms
    norm_method : str, float, optional
        Normalization method for SO power ('percent','shift', and 'none'). The default is 'percent'.
    verbose : bool, optional
        Verbose flag, by default True

    Returns
    -------
    SOpower_hist : numpy.ndarray
        SO-power histogram
    SOphase_hist : numpy.ndarray
        SO-phase histogram
    SOpower_cbins : numpy.ndarray
        SO-power bins for SO-power histogram
    SOphase_cbins : numpy.ndarray
        Phase bins for SO-phase histogram
    freq_cbins : numpy.ndarray
        Frequency bins for SO-power/phase histograms
    peak_selection_inds : boolean array
        Indices of peaks included in the histograms
    SO_power_norm : numpy.ndarray
        Normalized SO-power for each peak
    SO_power_times : numpy.ndarray
        Time of each peak in SO-power histogram
    SO_power_label : string
        Label of the SO-power axis
    SO_phase : numpy.ndarray
        SO-phase for each peak
    SO_phase_times : numpy.ndarray
        Time of each peak in SO-phase histogram
    """
    if verbose:
        print('Detecting artifacts...', end=" ")
        tic_art = timeit.default_timer()

    artifacts = detect_artifacts(data, fs)

    if verbose:
        # noinspection PyUnboundLocalVariable
        print('Took ' + convert_hms(timeit.default_timer() - tic_art))

    # Compute artifact peaks
    art_interp = interp1d(np.arange(len(data)) / fs, artifacts, kind='nearest')
    stats_table['artifact_time'] = art_interp(stats_table.peak_time).astype(bool)

    # Compute peak stage
    stage_interp = interp1d(stages.Time.values, stages.Stage.values, kind='previous', bounds_error=False, fill_value=0)
    stats_table['stage'] = stage_interp(stats_table.peak_time)

    """ SOPH computation """
    if stages_include is None:
        stages_include = [1, 2, 3, 4]

    if verbose:
        print('Computing SO-Power Histogram...', end=" ")
        tic_SOpower = timeit.default_timer()

    SOpower_hist, freq_cbins, SOpower_cbins, peak_SOpower, peak_inds_SOpower, \
        SO_power_norm, SO_power_times, SO_power_label = \
        so_power_histogram(stats_table.peak_time, stats_table.peak_frequency,
                           data, fs, artifacts, stages=stages, freq_range=[4, 25], freq_window=[1, 0.2],
                           soph_stages=stages_include, norm_method=norm_method, verbose=verbose)

    if verbose:
        # noinspection PyUnboundLocalVariable
        print('Took ' + convert_hms(timeit.default_timer() - tic_SOpower))

        print('Computing SO-Phase Histogram...', end=" ")
        tic_SOphase = timeit.default_timer()

    SOphase_hist, _, SOphase_cbins, peak_SOphase, peak_inds_SOphase, \
        SO_phase, SO_phase_times = \
        so_phase_histogram(stats_table.peak_time, stats_table.peak_frequency,
                           data, fs, artifacts, stages=stages, freq_range=[4, 25], freq_window=[1, 0.2],
                           soph_stages=stages_include, verbose=verbose)
    stats_table['phase'] = peak_SOphase

    if verbose:
        # noinspection PyUnboundLocalVariable
        print('Took ' + convert_hms(timeit.default_timer() - tic_SOphase))

    # Ensure the same number of peaks are included in the SOpower and SOphase histograms
    assert np.all(peak_inds_SOpower == peak_inds_SOphase), 'SOpower and SOphase histograms included different TF peaks.'
    peak_selection_inds = peak_inds_SOpower

    return SOpower_hist, SOphase_hist, SOpower_cbins, SOphase_cbins, freq_cbins, peak_selection_inds,\
        SO_power_norm, SO_power_times, SO_power_label, SO_phase, SO_phase_times


def run_tfpeaks_soph(data, fs, stages, downsample=None, segment_dur=30, merge_thresh=8,
                     max_merges=np.inf, trim_volume=0.8, norm_method='percent', plot_on=True):
    """Extracts TF-peaks then computes SO-power and Phase histograms using the dynam_o package

    Parameters
    ----------
    data : ndarray
        The data to be analyzed.
    fs : int
        The sampling frequency of the data.
    stages : pandas.DataFrame
        The sleep stages of the data.
    downsample : list, optional
        The downsampling factor to be applied to the data. The default is None.
    segment_dur : int, optional
        The duration of each segment in seconds. The default is 30.
    merge_thresh : int, optional
        The minimum number of peaks that must be present in a segment for it to be considered a peak. The default is 8.
    max_merges : int, optional
        The maximum number of merges that can be performed on a segment. The default is np.inf.
    trim_volume : float, optional
        The fraction of the data to be trimmed from the beginning and end of each segment. The default is 0.8.
    norm_method : str, float, optional
        Normalization method for SO power ('percent','shift', and 'none'). The default is 'percent'.
    plot_on : bool, optional
        Whether to plot the summary figure. The default is True.

    Returns
    -------
    stats_table : pandas.DataFrame
        A table containing the statistics.
    SOpower_hist : ndarray
        A histogram of the SO-power
    SOphase_hist : ndarray
        A histogram of the SO-phase
    SOpower_cbins : ndarray
        The SO-power bins used to compute the SO-power histogram.
    SOphase_cbins : ndarray
        The SO-phase bins used to compute the SO-phase histogram.
    freq_cbins : ndarray
        The frequency bins used to compute the SO-power/phase histograms.
    SO_power_norm : ndarray
        The normalized SO-power value.
    SO_power_times : ndarray
        The time points corresponding to each SO-power value.
    SO_phase : ndarray
        The SO-phase value.
    SO_phase_times : ndarray
        The time points corresponding to each SO-phase value.
    """

    # Extract TF-peaks and compute peak statistics table
    stats_table = compute_tfpeaks(data, fs, downsample, segment_dur, merge_thresh, max_merges, trim_volume)

    # Compute SO-power and SO-phase Histograms
    SOpower_hist, SOphase_hist, SOpower_cbins, SOphase_cbins, freq_cbins, peak_selection_inds,\
        SO_power_norm, SO_power_times, SO_power_label, SO_phase, SO_phase_times = \
        compute_sophs(data, fs, stages, stats_table, norm_method=norm_method)

    # Create summary plot if selected
    if plot_on:
        summary_plot(data, fs, stages, stats_table[peak_selection_inds], SOpower_hist, SOpower_cbins,
                     SO_power_norm, SO_power_times, SO_power_label, SOphase_hist, freq_cbins)

    return stats_table, SOpower_hist, SOphase_hist, SOpower_cbins, SOphase_cbins, freq_cbins, \
        SO_power_norm, SO_power_times, SO_phase, SO_phase_times
