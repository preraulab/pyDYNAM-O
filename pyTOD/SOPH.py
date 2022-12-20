import copy

import numpy as np
from joblib import cpu_count
from scipy.interpolate import interp1d
from scipy.signal import hilbert, sosfiltfilt

from pyTOD.multitaper import multitaper_spectrogram
from pyTOD.utils import butter_bandpass, pow2db, create_bins


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


def get_so_phase(data, fs, lowcut=0.3, highcut=1.5, order=10):
    """Computes unwrapped slow oscillation phase
    Note: Phase is unwrapped because wrapping does not return to original angle given the unwrapping algorithm

    Parameters
    ----------
    data : EEG time series data
    fs : Sampling frequency
    lowcut : Bandpass low-end cutoff
    highcut : Bandpass high-end cutoff
    order : Filter order

    Returns
    -------
    phase : Unwrapped phase
    """
    sos = butter_bandpass(lowcut, highcut, fs, order)
    data_filt = sosfiltfilt(sos, data)

    analytic_signal = hilbert(data_filt)
    phase = np.unwrap(np.angle(analytic_signal))
    return phase


def get_so_power(data, fs, lowcut=0.3, highcut=1.5):
    """Computes slow oscillation power

    Parameters
    ----------
    data : EEG time series data
    fs : Sampling frequency
    lowcut : Bandpass low-end cutoff
    highcut : Bandpass high-end cutoff

    Returns
    -------
    SO_power : Slow oscillation power
    SO_power_times : Time vector for SO_power
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
    spect, SO_power_times, sfreqs = multitaper_spectrogram(data, fs, frequency_range, time_bandwidth, num_tapers,
                                                           window_params,
                                                           min_nfft, detrend_opt, multiprocess, cpus,
                                                           weighting, plot_on, clim_scale, verbose, xyflip)

    df = sfreqs[1] - sfreqs[0]
    SO_power = pow2db(np.sum(spect, axis=0) * df)
    return SO_power, SO_power_times


def so_power_histogram(peak_times, peak_freqs, data, fs, artifacts, freq_range=None, freq_window=None,
                       SO_range=None, SO_window=None, norm_method='percent', min_time_in_bin=1, verbose=True):
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
    min_time_in_bin : int, optional
        Minimum time required in each SO power bin. The default is 1.
    verbose : bool, optional
        Print settings to console. The default is True.

    Returns
    -------
    SOpow_hist : array_like
        SO power histogram.
    freq_cbins : array_like
        Center frequencies of frequency bins.
    SO_cbins : array_like
        Center SO powers of SO power bins.
    SO_power_norm : array_like
        Normalized SO power.
    SO_power_times : array_like
        Times of SO power samples.
    SO_power_label : string
        Label of the SO-power axis
    """
    good_data = copy.deepcopy(data)
    good_data[artifacts] = np.nan
    SO_power, SO_power_times = get_so_power(good_data, fs, lowcut=0.3, highcut=1.5)
    inds = ~np.isnan(SO_power)

    # Normalize  power
    if type(norm_method) == float:
        shift_ptile = norm_method
        norm_method = 'shift'
    else:
        shift_ptile = 5  # default shift is the 5th percentile

    if norm_method == 'percentile' or norm_method == 'percent':
        low_val = 1
        high_val = 99
        ptile = np.percentile(SO_power[inds], [low_val, high_val])
        SO_power_norm = SO_power - ptile[0]
        # SO_power_norm = np.divide(SO_power_norm, (ptile[1] - ptile[0])) * 100
        SO_power_norm = SO_power_norm / (ptile[1] - ptile[0]) * 100
        SO_power_label = '% SO-Power'
    elif norm_method == 'p5shift' or norm_method == 'shift':
        ptile = np.percentile(SO_power[inds], [shift_ptile])
        # SO_power_norm = np.subtract(SO_power, ptile)
        SO_power_norm = SO_power - ptile
        SO_power_label = 'SO-Power (dB)'
    elif norm_method == 'absolute' or norm_method == 'none':
        SO_power_norm = SO_power
        SO_power_label = 'SO-Power (dB)'
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
    pow_interp = interp1d(SO_power_times[inds], SO_power_norm[inds], fill_value="extrapolate")
    peak_SOpow = pow_interp(peak_times)

    # SO-power time step size
    d_stimes = SO_power_times[1] - SO_power_times[0]

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

    return SOpow_hist, freq_cbins, SO_cbins, SO_power_norm, SO_power_times, SO_power_label


def so_phase_histogram(peak_times, peak_freqs, data, fs, freq_range=None, freq_window=None,
                       phase_range=None, phase_window=None, min_time_in_bin=1, verbose=True):
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
    freq_range : list, optional
        The range of frequencies to include in the histogram. Defaults to [np.min(peak_freqs), np.max(peak_freqs)].
    freq_window : list, optional
        The size and step of the frequency bins.
        Defaults to [(freq_range[1] - freq_range[0]) / 5, (freq_range[1] - freq_range[0]) / 100].
    phase_range : list, optional
        The range of SO-phases to include in the histogram. Defaults to [-np.pi, np.pi].
    phase_window : list, optional
        The size and step of the SO-phase bins. Defaults to [(2 * np.pi) / 5, (2 * np.pi) / 100].
    min_time_in_bin : int, optional
        The minimum amount of time required in each SO-phase bin for it to be included in the histogram.
        Defaults to 1 minute.
    verbose : bool, optional
        Verbose setting. Defaults to True.

    Returns
    -------
    SOphase_hist : numpy.ndarray
        The SO-phase histogram.
    freq_cbins : numpy.ndarray
        The center frequencies of the frequency bins.
    phase_cbins : numpy.ndarray
        The center SO-phases of the SO-phase bins.
    """
    SO_phase = wrap_phase(get_so_phase(data, fs, lowcut=0.3, highcut=1.5))

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
