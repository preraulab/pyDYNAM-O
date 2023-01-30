import os
import numpy as np
import pandas as pd
from dynam_o.utils import summary_plot
from dynam_o.pipelines import compute_sophs, run_tfpeaks_soph


def run_example_data(data_range='segment', quality='fast', norm_method='percent', load_peaks=True, save_peaks=False):
    """Example data script

    Parameters
    ----------
    data_range : str, optional
        The range of data to use. Can be 'segment' or 'night'. Default: 'segment'
    quality : str, optional
        The quality of the TF-peak detection. Can be 'paper', 'precision', 'fast', or 'draft'. Default: 'fast'
    norm_method : str, float, optional
        Normalization method for SO power ('percent','shift', and 'none'). Default: 'percent'
    load_peaks : bool, optional
        Whether to load the TF-peak stats table from file. Default: True
    save_peaks : bool, optional
        Whether to save the TF-peak stats table to file. Default: False

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
        example_data_dir = os.path.join(dir_path[:dir_path.find('pyDYNAM-O') + 5],
                                        'examples/_example_data')  # assumes the top level repo name is pyDYNAM-O

    # EEG data and stages
    csv_data = pd.read_csv(example_data_dir + '/' + data_range + '_data.csv', header=None)
    data = np.array(csv_data[0]).astype(np.float32)
    stages = pd.read_csv(example_data_dir + '/' + data_range + '_stages.csv')
    print('Done')

    # Sampling frequency of the example data
    fs = 100

    if save_peaks:
        assert ~load_peaks, 'Set load_peaks=False in order to compute TF-peaks and save stats_table.'

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

        stats_table, *_ = run_tfpeaks_soph(data, fs, stages, downsample, segment_dur,
                                           merge_thresh, max_merges, trim_volume, norm_method=norm_method)

        if save_peaks:
            print('Writing stats_table to file...', end=" ")
            stats_table.to_csv(example_data_dir + '/' + data_range + '_peaks.csv')
            print('Done')

    else:  # load saved TF-peaks from file
        stats_table = pd.read_csv(example_data_dir + '/' + data_range + '_peaks.csv')

        SOpower_hist, SOphase_hist, SOpower_cbins, SOphase_cbins, freq_cbins, peak_selection_inds,\
            SO_power_norm, SO_power_times, SO_power_label, *_ \
            = compute_sophs(data, fs, stages, stats_table, norm_method=norm_method)

        summary_plot(data, fs, stages, stats_table[peak_selection_inds], SOpower_hist, SOpower_cbins,
                     SO_power_norm, SO_power_times, SO_power_label, SOphase_hist, freq_cbins)


def main():
    """
    Evoked when executing the .py file directly.
    Adjust parameters here to analyze the example data with different settings.
    """
    data_range = 'segment'  # 'segment' vs. 'night'
    quality = 'fast'  # Quality setting 'precision','fast', or 'draft'
    norm_method = 'p5shift'  # Normalization of SO power 'percent','shift', or 'none'
    load_peaks = True  # Load from csv vs computing
    save_peaks = False  # Save csv of peaks if computing

    # Run example data
    run_example_data(data_range, quality, norm_method, load_peaks, save_peaks)


if __name__ == '__main__':
    main()
