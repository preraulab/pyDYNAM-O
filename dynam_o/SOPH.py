import copy
import re

import numpy as np
from joblib import cpu_count
from scipy.interpolate import interp1d
from scipy.signal import hilbert, sosfiltfilt

from dynam_o.multitaper import multitaper_spectrogram
from dynam_o.utils import butter_bandpass, pow2db, create_bins


def wrap_phase(phase: np.ndarray) -> np.ndarray:
    """Wrap phase from -pi to pi

    Parameters
    ----------
    phase : np.ndarray
        Unwrapped phase

    Returns
    -------
    np.ndarray
        Wrapped phase
    """
    return np.angle(np.exp(1j * phase))


def compute_so_phase(data, fs, lowcut=0.3, highcut=1.5, stages=None, isexcluded=None, order=10):
    """Computes unwrapped slow oscillation phase
    Note: Phase is unwrapped because wrapping does not return to original angle given the unwrapping algorithm

    Parameters
    ----------
    data : EEG time series data
    fs : Sampling frequency
    lowcut : Bandpass low-end cutoff
    highcut : Bandpass high-end cutoff
    stages : pandas.DataFrame, optional
        Time/Stage dataframe.
    isexcluded : Time points that should be excluded
    order : Filter order

    Returns
    -------
    phase : Unwrapped phase
    t : Time vector for phase
    phase_stages : Stages of phase values
    """
    sos = butter_bandpass(lowcut, highcut, fs, order)
    data_filt = sosfiltfilt(sos, data)

    analytic_signal = hilbert(data_filt)
    phase = np.unwrap(np.angle(analytic_signal))
    t = np.arange(len(data)) / fs

    if isexcluded is None:
        isexcluded = np.zeros(data.shape, dtype=bool)
    phase[isexcluded] = np.nan

    # Get stages of phase values
    if stages is not None:
        stage_interp = interp1d(stages.Time.values, stages.Stage.values, kind='previous', bounds_error=False,
                                fill_value=0)
        phase_stages = stage_interp(t)
    else:
        phase_stages = True

    return phase, t, phase_stages


def compute_so_power(data, fs, lowcut=0.3, highcut=1.5, stages=None, isexcluded=None,
                     norm_method='percent', retain_fs=True):
    """Computes slow oscillation power and normalize

    Parameters
    ----------
    data : EEG time series data
    fs : Sampling frequency
    lowcut : Bandpass low-end cutoff
    highcut : Bandpass high-end cutoff
    stages : pandas.DataFrame, optional
        Time/Stage dataframe.
    isexcluded : Time points that should be excluded
    norm_method : str, float, optional
        Normalization method for SO power ('percent','shift', and 'none'). The default is 'percent'.
    retain_fs : Logical flag for up-sampling SO_power to fs

    Returns
    -------
    SO_power_norm : Normalized slow oscillation (SO) power
    SO_power_times : Time vector for SO_power_norm
    SO_power_stages : Stages of SO_power_norm values
    SO_power_label : Label of the SO-power axis
    """
    if isexcluded is None:
        isexcluded = np.zeros(data.shape, dtype=bool)
    good_data = copy.deepcopy(data)
    good_data[isexcluded] = np.nan

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
    spect, SO_power_times, sfreqs = multitaper_spectrogram(good_data, fs, frequency_range, time_bandwidth, num_tapers,
                                                           window_params,
                                                           min_nfft, detrend_opt, multiprocess, cpus,
                                                           weighting, plot_on, clim_scale, verbose, xyflip)

    df = sfreqs[1] - sfreqs[0]
    SO_power = pow2db(np.sum(spect, axis=0) * df)
    inds = ~np.isnan(SO_power)

    # Get stages of SOpower values
    if stages is not None:
        stage_interp = interp1d(stages.Time.values, stages.Stage.values, kind='previous', bounds_error=False,
                                fill_value=0)
        SOpower_stages = stage_interp(SO_power_times)
    else:
        SOpower_stages = True

    # Normalize SO power
    if type(norm_method) == float:
        shift_ptile = norm_method
        shift_stages = [1, 2, 3, 4]
        norm_method = 'shift'
    elif re.search("^p\\d+shift\\d*$", norm_method):
        shift_ptile = float(norm_method[1:norm_method.find('shift')])
        shift_stages = [int(x) for x in norm_method[norm_method.find('shift')+5:]]
        if not shift_stages:
            shift_stages = [1, 2, 3, 4]
        norm_method = 'shift'
    else:
        shift_ptile = 2  # default shift is the 2nd percentile
        shift_stages = [1, 2, 3, 4]

    if norm_method == 'percentile' or norm_method == 'percent':
        low_val = 1
        high_val = 99
        ptile = np.percentile(SO_power[inds], [low_val, high_val])
        SO_power_norm = SO_power - ptile[0]
        SO_power_norm = SO_power_norm / (ptile[1] - ptile[0]) * 100
        SO_power_label = '% SO-Power'
    elif norm_method == 'shift':
        if type(SOpower_stages) == bool and SOpower_stages:
            SOpower_stages_valid = np.ones(SO_power.shape, dtype=bool)
        else:
            SOpower_stages_valid = np.isin(SOpower_stages, shift_stages)
        ptile = np.percentile(SO_power[np.logical_and(SOpower_stages_valid, inds)], shift_ptile)
        SO_power_norm = SO_power - ptile
        SO_power_label = 'SO-Power (dB)'
    elif norm_method == 'absolute' or norm_method == 'none':
        SO_power_norm = SO_power
        SO_power_label = 'SO-Power (dB)'
    else:
        raise ValueError("Not a valid normalization option, choose 'percent', 'shift', or 'none'")

    # Up-sample to data sampling rate
    if retain_fs:
        SO_power_norm_notnan = SO_power_norm[inds]
        t = np.arange(len(data)) / fs
        SO_power_interp = interp1d(np.concatenate(([t[0]], SO_power_times[inds], [t[-1]])),
                                   np.concatenate(([SO_power_norm_notnan[0]],
                                                   SO_power_norm_notnan, [SO_power_norm_notnan[-1]])))
        SO_power_norm = SO_power_interp(t)
        SO_power_norm[isexcluded] = np.nan
        SO_power_times = t
        # noinspection PyUnboundLocalVariable
        SOpower_stages = stage_interp(SO_power_times) if stages is not None else SOpower_stages

    return SO_power_norm, SO_power_times, SOpower_stages, SO_power_label


def so_power_histogram(peak_times, peak_freqs, data, fs, artifacts, stages=None, freq_range=None, freq_window=None,
                       SO_range=None, SO_window=None, norm_method='percent', soph_stages=None,
                       min_time_in_bin=1, verbose=True):
    """Compute a slow oscillation power histogram

    Parameters
    ----------
    peak_times : array_like
        Times of peaks in seconds.
    peak_freqs : array_like
        Frequencies of peaks in Hz.
    data : array_like
        Data to compute SO power from.
    fs : int
        Sampling frequency of data.
    artifacts : array_like
        Indices of artifacts in data.
    stages : pandas.DataFrame, optional
        Time/Stage dataframe.
    freq_range : array_like, optional
        Frequency range to compute histogram over. The default is None.
    freq_window : array_like, optional
        Frequency window size and step size. The default is None.
    SO_range : array_like, optional
        SO power range to compute histogram over. The default is None.
    SO_window : array_like, optional
        SO power window size and step size. The default is None.
    norm_method : str, float, optional
        Normalization method for SO power ('percent','shift', and 'none'). The default is 'percent'.
    soph_stages : array_like, optional
        Sleep stages to be included in the histogram. Default: [1,2,3].
    min_time_in_bin : int, optional
        Minimum time required in each SO power bin. The default is 1.
    verbose : bool, optional
        Print settings to console. The default is True.

    Returns
    -------
    SOpower_hist : array_like
        SO power histogram.
    freq_cbins : array_like
        Center frequencies of frequency bins.
    SOpower_cbins : array_like
        Center SO powers of SO power bins.
    peak_SOpower : array_like
        SO power at peak_times.
    peak_selection_inds : boolean array
        Indices of peaks included in the histogram
    SO_power_norm : array_like
        Normalized SO power.
    SO_power_times : array_like
        Times of SO power samples.
    SO_power_label : string
        Label of the SO-power axis
    """
    # Compute SO power
    SO_power_norm, SO_power_times, SO_power_stages, SO_power_label = compute_so_power(
        data, fs, lowcut=0.3, highcut=1.5, stages=stages, isexcluded=artifacts, norm_method=norm_method)

    # Set defaults
    if freq_range is None:
        freq_range = [np.min(peak_freqs), np.max(peak_freqs)]

    if freq_window is None:
        freq_window = [(freq_range[1] - freq_range[0]) / 5, (freq_range[1] - freq_range[0]) / 100]

    freq_bin_edges, freq_cbins = create_bins(freq_range[0], freq_range[1], freq_window[0], freq_window[1], 'partial')
    num_freqbins = len(freq_cbins)

    if SO_range is None:
        SO_range = [np.nanmin(SO_power_norm), np.nanmax(SO_power_norm)]

    if SO_window is None:
        SO_window = [(SO_range[1] - SO_range[0]) / 5, (SO_range[1] - SO_range[0]) / 100]

    SO_bin_edges, SOpower_cbins = create_bins(SO_range[0], SO_range[1], SO_window[0], SO_window[1], 'partial')
    num_SObins = len(SOpower_cbins)

    # Print settings
    if verbose:
        print('\n  SO-Power Histogram Settings:')
        print('    Normalization Method: ' + str(norm_method))
        print('    Frequency Window Size: ' + str(freq_window[0]) + ' Hz, Window Step: ' + str(freq_window[1]) + ' Hz')
        print('    Frequency Range: ', str(freq_range[0]) + '-' + str(freq_range[1]) + ' Hz')
        print('    SO-Power Window Size: ' + f'{SO_window[0]:.3f}' + ', Window Step: ' + f'{SO_window[1]:.3f}')
        print('    SO-Power Range: ' + f'{SO_range[0]:.3f}' + ' - ' + f'{SO_range[1]:.3f}')
        print('    Minimum time required in each SO-Power bin: ' + str(min_time_in_bin) + ' min')

    # Initialize SO_power x freq matrix
    SOpower_hist = np.full(shape=(num_freqbins, num_SObins), fill_value=np.nan)

    # Initialize time in bin
    time_in_bin = np.zeros(num_SObins)

    # SO-power time step size
    d_times = SO_power_times[1] - SO_power_times[0]

    # Compute peak SO_power
    SO_interp = interp1d(np.concatenate(([SO_power_times[0]-d_times], SO_power_times, [SO_power_times[-1]+d_times])),
                         np.concatenate(([SO_power_norm[0]], SO_power_norm, [SO_power_norm[-1]])))
    peak_SOpower = SO_interp(peak_times)

    # Only include values from desired stages in the histogram
    if soph_stages is None:
        soph_stages = [1, 2, 3]

    if stages is not None:
        stage_interp = interp1d(stages.Time.values, stages.Stage.values, kind='previous', bounds_error=False,
                                fill_value=0)
        # Select TF-peaks
        peak_stages = stage_interp(peak_times)
        peak_selection_inds = np.logical_and(np.isin(peak_stages, soph_stages), ~np.isnan(peak_SOpower))

        # Select SO_power
        SO_selection_inds = np.logical_and(np.isin(SO_power_stages, soph_stages), ~np.isnan(SO_power_norm))

        peak_Cmetric = peak_SOpower[peak_selection_inds]
        peak_freqs = peak_freqs[peak_selection_inds]
        Cmetric = SO_power_norm[SO_selection_inds]
    else:
        peak_selection_inds = np.ones(peak_SOpower.shape, dtype=bool)
        peak_Cmetric = peak_SOpower
        Cmetric = SO_power_norm

    # Compute the histogram
    for s_bin in range(num_SObins):
        TIB_inds = np.logical_and(Cmetric >= SO_bin_edges[0, s_bin], Cmetric < SO_bin_edges[1, s_bin])
        SO_inds = np.logical_and(peak_Cmetric >= SO_bin_edges[0, s_bin], peak_Cmetric < SO_bin_edges[1, s_bin])

        # Time in bin in minutes
        time_in_bin[s_bin] = (np.sum(TIB_inds) * d_times / 60)

        # Check for min time in bin
        if time_in_bin[s_bin] < min_time_in_bin:
            continue

        if np.sum(SO_inds):
            for f_bin in range(num_freqbins):
                # Get indices in freq bin
                freq_inds = np.logical_and(peak_freqs >= freq_bin_edges[0, f_bin],
                                           peak_freqs < freq_bin_edges[1, f_bin])

                # Fill histogram with peak rate in this freq / SOpower bin
                SOpower_hist[f_bin, s_bin] = np.sum(SO_inds & freq_inds) / time_in_bin[s_bin]
        else:
            SOpower_hist[:, s_bin] = 0

    return SOpower_hist, freq_cbins, SOpower_cbins, peak_SOpower, peak_selection_inds, \
        SO_power_norm, SO_power_times, SO_power_label


def so_phase_histogram(peak_times, peak_freqs, data, fs, artifacts, stages=None, freq_range=None, freq_window=None,
                       phase_range=None, phase_window=None, soph_stages=None, min_time_in_bin=0, verbose=True):
    """Compute a slow oscillation phase histogram

    Parameters
    ----------
    peak_times : numpy.ndarray
        Times of detected peaks in the recording.
    peak_freqs : numpy.ndarray
        Frequencies of detected peaks in the recording.
    data : numpy.ndarray
        The recording.
    fs : int, float
        Sampling rate of the recording.
    artifacts : array_like
        Indices of artifacts in data.
    stages : pandas.DataFrame, optional
        Time/Stage dataframe.
    freq_range : list, optional
        The range of frequencies to include in the histogram. Defaults to [np.min(peak_freqs), np.max(peak_freqs)].
    freq_window : list, optional
        The size and step of the frequency bins.
        Defaults to [(freq_range[1] - freq_range[0]) / 5, (freq_range[1] - freq_range[0]) / 100].
    phase_range : list, optional
        The range of SO-phases to include in the histogram. Defaults to [-np.pi, np.pi].
    phase_window : list, optional
        The size and step of the SO-phase bins. Defaults to [(2 * np.pi) / 5, (2 * np.pi) / 100].
    soph_stages : array_like, optional
        Sleep stages to be included in the histogram. Default: [1,2,3].
    min_time_in_bin : int, optional
        The minimum amount of time required in each SO-phase bin for it to be included in the histogram.
        Defaults to 0 minute.
    verbose : bool, optional
        Verbose setting. Defaults to True.

    Returns
    -------
    SOphase_hist : numpy.ndarray
        The SO-phase histogram.
    freq_cbins : numpy.ndarray
        The center frequencies of the frequency bins.
    SOphase_cbins : numpy.ndarray
        The center SO-phases of the SO-phase bins.
    peak_SOphase : array_like
        SO phase at peak_times.
    peak_selection_inds : boolean array
        Indices of peaks included in the histogram
    SO_phase : array_like
        SO phase.
    SO_phase_times : array_like
        Times of SO phase samples.
    """
    # Compute SO phase
    SO_phase, SO_phase_times, SO_phase_stages = compute_so_phase(data, fs, lowcut=0.3, highcut=1.5,
                                                                 stages=stages, isexcluded=artifacts)
    SO_phase = wrap_phase(SO_phase)

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
    phase_bin_edges, SOphase_cbins = create_bins(phase_range[0], phase_range[1],
                                                 phase_window[0], phase_window[1], 'extend')
    num_phasebins = len(SOphase_cbins)

    # Print settings
    if verbose:
        print('\n  SO-Phase Histogram Settings:')
        print('    Frequency Window Size: ' + str(freq_window[0]) + ' Hz, Window Step: ' + str(freq_window[1]) + ' Hz')
        print('    Frequency Range: ', str(freq_range[0]) + '-' + str(freq_range[1]) + ' Hz')
        print('    SO-Phase Window Size: ' + str(phase_window[0] / np.pi) + 'π, Window Step: ' +
              str(phase_window[1] / np.pi) + 'π')
        print('    SO-Phase Range: ' + str(phase_range[0] / np.pi) + 'π - ' + str(phase_range[1] / np.pi) + 'π')
        print('    Minimum time required in each SO-phase bin: ' + str(min_time_in_bin) + ' min')

    # Initialize SO_phase x freq matrix
    SOphase_hist = np.full(shape=(num_freqbins, num_phasebins), fill_value=np.nan)

    # Initialize time in bin
    time_in_bin = np.zeros(num_phasebins)

    # SO-phase time step size
    d_times = SO_phase_times[1] - SO_phase_times[0]

    # Compute peak phase
    phase_interp = interp1d(np.concatenate(([SO_phase_times[0]-d_times], SO_phase_times, [SO_phase_times[-1]+d_times])),
                            np.concatenate(([SO_phase[0]], SO_phase, [SO_phase[-1]])))
    peak_SOphase = phase_interp(peak_times)

    # Only include values from desired stages in the histogram
    if soph_stages is None:
        soph_stages = [1, 2, 3]

    if stages is not None:
        stage_interp = interp1d(stages.Time.values, stages.Stage.values, kind='previous', bounds_error=False,
                                fill_value=0)
        # Select TF-peaks
        peak_stages = stage_interp(peak_times)
        peak_selection_inds = np.logical_and(np.isin(peak_stages, soph_stages), ~np.isnan(peak_SOphase))

        # Select SO_phase
        SO_selection_inds = np.logical_and(np.isin(SO_phase_stages, soph_stages), ~np.isnan(SO_phase))

        peak_Cmetric = peak_SOphase[peak_selection_inds]
        peak_freqs = peak_freqs[peak_selection_inds]
        Cmetric = SO_phase[SO_selection_inds]
    else:
        peak_selection_inds = np.ones(peak_SOphase.shape, dtype=bool)
        peak_Cmetric = peak_SOphase
        Cmetric = SO_phase

    # Compute the histogram
    for p_bin in range(num_phasebins):
        if phase_bin_edges[0, p_bin] <= -np.pi:
            wrapped_edge_lowlim = phase_bin_edges[0, p_bin] + (2 * np.pi)
            TIB_inds = np.logical_or(Cmetric >= wrapped_edge_lowlim,
                                     Cmetric < phase_bin_edges[1, p_bin])
            phase_inds = np.logical_or(peak_Cmetric >= wrapped_edge_lowlim,
                                       peak_Cmetric < phase_bin_edges[1, p_bin])
        elif phase_bin_edges[1, p_bin] >= np.pi:
            wrapped_edge_highlim = phase_bin_edges[1, p_bin] - (2 * np.pi)
            TIB_inds = np.logical_or(Cmetric < wrapped_edge_highlim,
                                     Cmetric >= phase_bin_edges[0, p_bin])
            phase_inds = np.logical_or(peak_Cmetric < wrapped_edge_highlim,
                                       peak_Cmetric >= phase_bin_edges[0, p_bin])
        else:
            TIB_inds = np.logical_and(Cmetric >= phase_bin_edges[0, p_bin],
                                      Cmetric < phase_bin_edges[1, p_bin])
            phase_inds = np.logical_and(peak_Cmetric >= phase_bin_edges[0, p_bin],
                                        peak_Cmetric < phase_bin_edges[1, p_bin])

        time_in_bin[p_bin] = (np.sum(TIB_inds) * d_times / 60)

        if time_in_bin[p_bin] < min_time_in_bin:
            continue

        if np.sum(phase_inds):
            for f_bin in range(num_freqbins):
                # Get indices in freq bin
                freq_inds = np.logical_and(peak_freqs >= freq_bin_edges[0, f_bin],
                                           peak_freqs < freq_bin_edges[1, f_bin])

                # Fill histogram with peak rate in this freq / SOphase bin
                SOphase_hist[f_bin, p_bin] = np.sum(phase_inds & freq_inds) / time_in_bin[p_bin]
        else:
            SOphase_hist[:, p_bin] = 0

    # Normalize across each frequency row
    SOphase_hist = SOphase_hist / np.nansum(SOphase_hist, axis=1)[:, np.newaxis]

    return SOphase_hist, freq_cbins, SOphase_cbins, peak_SOphase, peak_selection_inds, \
        SO_phase, SO_phase_times
