import os
import numpy as np
import pandas as pd
from pyTOD.utils import summary_plot
from pyTOD.pipelines import compute_sophs, run_tfpeaks_soph


def run_example_data(data_range='segment', quality='fast', save_peaks=False, load_peaks=True):
    """Example data script

    Parameters
    ----------
    data_range : str, optional
        The range of data to use. Can be 'segment' or 'night'. Default: 'segment'
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

    # Configure example_data path
    try:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        if dir_path.find('examples') > 0:
            example_data_dir = os.path.join(dir_path, '_example_data')
        else:
            example_data_dir = os.path.join(dir_path, 'examples/_example_data')
    except NameError:  # __file__ isn't available for iPython console sessions
        dir_path = os.getcwd()
        example_data_dir = os.path.join(dir_path[:dir_path.find('pyTOD') + 5],
                                        'examples/_example_data')  # assumes the top level repo name is pyTOD

    # EEG data and stages
    csv_data = pd.read_csv(example_data_dir + '/' + data_range + '_data.csv', header=None)
    data = np.array(csv_data[0]).astype(np.float32)
    stages = pd.read_csv(example_data_dir + '/' + data_range + '_stages.csv')
    print('Done')

    # Sampling Frequency
    fs = 100

    if not load_peaks:
        # DETECT TF-PEAKS
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
            SOphase_hist, freq_cbins, phase_cbins = run_tfpeaks_soph(data, fs, stages, downsample, segment_dur,
                                                                     merge_thresh, max_merges, trim_volume,
                                                                     norm_method='p5shift', plot_on=True)

        if save_peaks:
            print('Writing stats_table to file...', end=" ")
            stats_table.to_csv(example_data_dir + '/' + data_range + '_peaks.csv')
            print('Done')

    else:  # load saved TF-peaks from file
        stats_table = pd.read_csv(example_data_dir + '/' + data_range + '_peaks.csv')

        SOpow_hist, freq_cbins, SO_cbins, SO_power_norm, SO_power_times, SO_power_label, \
            SOphase_hist, freq_cbins, phase_cbins = compute_sophs(data, fs, stages, stats_table, norm_method='p5shift')

        summary_plot(data, fs, stages, stats_table, SOpow_hist, SO_cbins, SO_power_norm, SO_power_times, SO_power_label,
                     SOphase_hist, freq_cbins)


if __name__ == '__main__':
    data_range = 'night'  # 'segment' vs. 'night'
    quality = 'fast'  # Quality setting 'precision','fast', or 'draft'
    save_peaks = False  # Save csv of peaks if computing
    load_peaks = True  # Load from csv vs computing

    # Run example data
    run_example_data(data_range, quality, save_peaks, load_peaks)
