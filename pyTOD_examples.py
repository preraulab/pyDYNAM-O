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

    :param data_range: 'night' or 'segment'
    :param quality: 'precision','fast',or 'draft'
    :param save_peaks: saves peak data
    :param load_peaks: runs from saved peak data
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

    :param data: Time series data
    :param fs: Sampling frequency
    :param downsample: Downsampling amount
    :param segment_dur: Segment duration for breaking up peak detection
    :param merge_thresh: Peak merge threshold
    :param max_merges: Number of maximum merges
    :param trim_volume: Percent of peak to retain
    :param verbose: Verbose flag
    :return: Peak statistics table
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

    :param verbose: Verbose flag
    :param data: Time series data
    :param fs: Sampling frequency
    :param stages: Time/Stage dataframe
    :param stats_table: Peak statistics table
    :return: SOpow_hist, freq_cbins, SO_cbins, SO_power_norm, SO_power_times, SOphase_hist, freq_cbins, phase_cbins
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

    :param data: Time series data
    :param fs: Sample frequency
    :param stages: Stages data frame
    :param downsample: Downsample amount for the watershed output
    :param segment_dur: Segment duration
    :param merge_thresh: Merge threshold
    :param max_merges: Number of maximum merges
    :param trim_volume: Percent of volume to retain
    :param plot_on: Plotting flag
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
    data_range = 'segment'  # Segment vs. night
    save_peaks = True  # Save csv of peaks if computing
    load_peaks = False  # Load from csv vs computing

    # Run example data
    run_example_data(data_range, quality, save_peaks, load_peaks)
