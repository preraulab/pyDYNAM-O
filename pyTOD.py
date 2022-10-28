import numpy
import pandas
import skimage.future.graph
from skimage import measure, segmentation, future, color, morphology
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import timeit
from scipy.stats.distributions import chi2
from multitaper_toolbox.python.multitaper_spectrogram_python import multitaper_spectrogram
from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm


def edge_weight(graph_rag: skimage.future.graph.RAG, graph_edge: tuple, graph_data: numpy.ndarray) -> float:
    """
    Computes the edge weight between two regions

    :param graph_rag: RAG graph
    :param graph_edge: edge for which to compute weight
    :param graph_data: spectrogram data
    :type graph_rag: skimage.future.graph.rag.RAG
    :type graph_edge: tuple
    :type graph_rag: skimage.future.graph.rag.RAG
    """
    # Get border and region tuples
    i_border = graph_rag.nodes[graph_edge[0]]["border"]
    i_region = graph_rag.nodes[graph_edge[0]]["region"]
    j_border = graph_rag.nodes[graph_edge[1]]["border"]
    j_region = graph_rag.nodes[graph_edge[1]]["region"]

    A_ij = i_border.intersection(j_border)

    # Diagonal neighbor is not really connected
    if not len(A_ij):
        return np.nan  # Set to nan to not be a neighbor

        # #Computationally costly procedure to join to other region
        # img = np.zeros(graph_data.shape)
        # for i in i_region:
        #     img[i[0], i[1]] = 1
        # bx, by = np.where(morphology.dilation(img, np.ones([3, 3])))
        # joint_pxls = set(zip(bx, by)).intersection(j_border)
        # i_region.update(joint_pxls)
        # i_border.update(joint_pxls)
        # A_ij = i_border.intersection(j_border)

    A_ij_max = np.max([graph_data[i] for i in A_ij])

    # Store for reuse efficiency
    i_border_vals = [graph_data[i] for i in i_border]
    j_border_vals = [graph_data[i] for i in j_border]

    # Minimum border values
    B_i_min = np.min(i_border_vals)
    B_j_min = np.min(j_border_vals)

    # Max region values
    i_max = np.max([graph_data[i] for i in i_region] + i_border_vals)
    j_max = np.max([graph_data[i] for i in j_region] + j_border_vals)

    # Compute weight for i into j
    # C_ij = A_ij_max - B_i_min
    # D_ij = j_max - A_ij_max
    # w_ij = C_ij - D_ij = 2 * A_ij_max - B_i_min - j_max
    w_ij = - B_i_min - j_max

    # Compute weight for j into i
    w_ji = - B_j_min - i_max

    # Moved constant 2 * A_ij_max to final weight computation for efficiency
    # Take the max weight of w_ij and w_ji
    w_max = 2 * A_ij_max + np.max([w_ij, w_ji])

    return w_max


def merge_regions(graph_rag: skimage.future.graph.RAG, src: int, dst: int):
    """
    Merges the regions and borders for use in hierarchical merge

    :type graph_rag: skimage.future.graph.rag.RAG
    :param src: Source node
    :param dst: Destination node
    """
    # Region is union of regions
    graph_rag[dst]["region"] = graph_rag[dst]["region"].union(graph_rag[src]["region"])
    # Border is symmetric difference of borders
    graph_rag[dst]["border"] = graph_rag[dst]["border"].symmetric_difference(graph_rag[src]["border"])
    # print(str(src) + ' > ' + str(dst) + ' weight: ' + str(graph_rag.edges[src, dst]['weight']))


def merge_weight(graph_rag: skimage.future.graph.RAG, src: int, dst: int, neighbor: int,
                 data: numpy.ndarray) -> dict:
    """
    Computes weight for use in hierarchical merge

    :rtype dict
    :return Weight between src and neighbors
    :param graph_rag: skimage.future.graph.rag.RAG
    :param src: Source node (unused but required)
    :param dst: Destination node (merged already)
    :param neighbor: Neighbor node
    :param data: Segment data
    """

    # Convert weight output to dictionary form
    n_weight = edge_weight(graph_rag, tuple([dst, neighbor]), data)
    # print('   ' + str(list([dst, neighbor])) + ' new weight: ' + str(n_weight))
    return {'weight': n_weight}


def trim_region(graph_rag: skimage.future.graph.RAG, labels_merged, graph_data: numpy.ndarray, region_num: int,
                trim_volume: float):
    """
    Computes the edge weight between two regions
    NOTE: Weights must be flipped to be negative to match ascending hierarchical merging

    :param trim_volume: Percentage to trim
    :type: float
    :param graph_rag: RAG
    :type graph_rag: skimage.future.graph.rag.RAG
    :param graph_data: Spectrogram data
    :type graph_data: numpy.ndarray
    :param region_num: Region number
    :type region_num: int
    """

    # Get the region pixel values
    reg_idx = list(zip(*graph_rag.nodes[region_num]["region"]))
    reg_vals = np.sort(graph_data[reg_idx[0], reg_idx[1]])

    # Find the cutoff for the volume
    percent_vol = np.cumsum(reg_vals) / np.sum(reg_vals)
    trim_idx = next(x for x, val in enumerate(percent_vol)
                    if val > 1 - trim_volume)
    trim_level = reg_vals[trim_idx]
    rplot = labels_merged == region_num

    # Slice at the trim level
    img = np.logical_and(rplot, graph_data > trim_level)

    # Fill all holes
    img = morphology.remove_small_holes(img)

    # Turn into a label image
    label_img = measure.label(img)

    # Select the largest region if there are subregions after trim
    if np.max(label_img) > 1:
        rp = measure.regionprops(label_img)
        max_area_label = rp[np.argmax([prop.area for prop in rp])].label
        label_img = label_img == max_area_label
    else:
        label_img = label_img > 0

    return label_img


def plot_node(graph_rag, node_num):
    """
    Plots a node for diagnostics

    :param graph_rag: RAG graph
    :param node_num: Node number
    :type graph_rag: skimage.future.graph.rag.RAG
    :type node_num: int
    """
    for i in graph_rag.nodes[node_num]["border"]:
        plt.plot(i[1], i[0], 'kx')
    for i in graph_rag.nodes[node_num]["region"]:
        plt.plot(i[1], i[0], 'b.')


def nn_resample(data: numpy.ndarray, shape: tuple) -> numpy.ndarray:
    """
    Nearest neighbors resampling of the matrix

    :param data: data matrix to upsample
    :type data: numpy.ndarray
    :param shape: new shape
    :type shape: tuple
    :return: new matrix
    """

    def per_axis(in_sz, out_sz):
        ratio = 0.5 * in_sz / out_sz
        return np.round(np.linspace(ratio - 0.5, in_sz - ratio - 0.5, num=out_sz)).astype(int)

    return data[per_axis(data.shape[0], shape[0])[:, None],
                per_axis(data.shape[1], shape[1])]


def pow2db(val):
    """
    Converts power to dB
    :param val: values to convert
    :return: val_dB value in dB
    """
    return (10 * np.log10(val) + 300) - 300


def convertHMS(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    return "%d:%02d:%02d" % (hour, minutes, seconds)

def process_segments_params(segment_dur: float, stimes: numpy.ndarray):
    """Gets parameters for segmenting the spectrogram

    :param segment_dur: The duration of the segment in seconds
    :param stimes: spectrogram times
    :return: window_idx, start_times
    """
    # create frequency vector
    d_stimes = stimes[1] - stimes[0]
    win_samples = np.round(segment_dur / d_stimes)

    window_start = np.arange(0, len(stimes) - 1, win_samples)

    # Get indexes for each window
    window_idx = [np.arange(window_start[i], np.min([window_start[i] + win_samples, len(stimes)]), 1).astype(int)
                  for i in np.arange(0, len(window_start), 1)]

    start_times = stimes[window_start.astype(int)]

    return window_idx, start_times


def detect_tfpeaks(segment_data: numpy.ndarray, start_time=0, merge_threshold=8, trim_volume=0.8, downsample=None, dur_min=-np.inf,
                   dur_max=np.inf, bw_min=-np.inf, bw_max=np.inf, prom_min=-np.inf, plot_on=True,
                   verbose=True) -> pandas.DataFrame:
    # Downsample data
    if downsample is None:
        downsample = []
    if downsample:
        segment_data_LR = segment_data[::downsample[0], ::downsample[1]]
    else:
        segment_data_LR = segment_data

    # Run watershed segmentation with empty border regions
    labels = segmentation.watershed(-segment_data_LR, connectivity=2, watershed_line=True)

    # Expand labels by 1 to join them. This will be used to compute the RAG
    join_labels = segmentation.expand_labels(labels, distance=10)

    # Create a region adjacency graph (RAG))
    RAG = future.graph.RAG(join_labels, connectivity=2)

    # Add labels, borders, and regions to each RAG node
    for n in RAG:
        RAG.nodes[n]['labels'] = [n]

        curr_region = (labels == n)
        # Compute the borders by intersecting 1 pixel dilation with zero-valued watershed border regions
        bx, by = np.where(morphology.dilation(curr_region, np.ones([3, 3])) & (labels == 0))
        # Get regions by bing full region
        rx, ry = np.where(curr_region)

        # Zip into sets of tuples for easy set operations (e.g. intersection)
        border = set(zip(bx, by))
        region = set(zip(rx, ry)).union(border)  # Add border to region

        # Set node properties
        RAG.nodes[n]["border"] = border
        RAG.nodes[n]["region"] = region

    if verbose:
        print('Computing weights...')
        tic = timeit.default_timer()

    # Compute the initial RAG weights
    for edge in RAG.edges:
        weight = edge_weight(RAG, edge, segment_data_LR)

        if np.isnan(weight):
            RAG.remove_edge(edge[0], edge[1])
            # print("Removing: " + str(edge))
        else:
            RAG.edges[edge]["weight"] = weight
            # print('Edge ' + str(edge) + ' weight: ' + str(weight))

    if verbose:
        toc = timeit.default_timer()
        print(f'      Weights took {toc - tic:.3f}s')

    if verbose:
        print('Starting merge...')

    tic_merge = timeit.default_timer()
    get_max_time = 0
    merge_borders_time = 0
    merge_node_time = 0

    max_val = np.inf
    while max_val > merge_threshold:
        # tic = timeit.default_timer()
        edge, mv = max(dict(RAG.edges).items(), key=lambda x: x[1]['weight'])
        src, dst = edge
        max_val = mv['weight']
        # toc = timeit.default_timer()
        # get_max_time += toc-tic

        # tic = timeit.default_timer()
        # Region is union of regions
        RAG.nodes[dst]["region"] = RAG.nodes[dst]["region"].union(RAG.nodes[src]["region"])
        # Border is symmetric difference of borders
        RAG.nodes[dst]["border"] = RAG.nodes[dst]["border"].symmetric_difference(RAG.nodes[src]["border"])
        # toc = timeit.default_timer()
        # merge_borders_time += toc - tic

        # tic = timeit.default_timer()
        RAG.merge_nodes(src, dst, merge_weight, extra_arguments=[segment_data_LR])
        #  toc = timeit.default_timer()
        #  merge_node_time += toc - tic

    if verbose:
        toc_merge = timeit.default_timer()
        print(f'      Merging took {toc_merge - tic_merge:.3f}s')
        print(f'            Max edge took {get_max_time:.3f}s')
        print(f'            Border merge took {merge_borders_time:.3f}s')
        print(f'            Node merge took {merge_node_time:.3f}s')

    # Create a new label image
    labels_merged = np.zeros(labels.shape, dtype=int)
    for n in RAG:
        for p in RAG.nodes[n]["region"]:
            labels_merged[p] = n
        for b in RAG.nodes[n]["border"]:
            labels_merged[b] = 0

    if downsample:
        labels_merged = nn_resample(labels_merged, segment_data.shape)

    if verbose:
        print('Trimming...')
        tic = timeit.default_timer()
    # Set up the trim images
    trim_labels = np.zeros(segment_data.shape)

    # Trim only regions that are bigger than the min bw/dur
    for r in RAG.nodes:
        # Get the values of the merged nodes
        r_border = list(zip(*RAG.nodes[r]["border"]))
        r_region = list(zip(*RAG.nodes[r]["region"]))

        # Compute bandwidth and duration
        bw = (np.max(r_border[0]) - np.min(r_border[0])) * d_freq
        dur = (np.max(r_border[1]) - np.min(r_border[1])) * d_time

        data_vals = segment_data[r_region[0], r_region[1]]
        height = pow2db(max(data_vals) - min(data_vals))

        # Only trim if within parameters
        if (bw >= bw_min) & (dur >= dur_min) & (height >= prom_min):
            trim_labels += trim_region(RAG, labels_merged, segment_data, r, trim_volume)

    # Get the label image of the trimmed regions
    trim_labels = measure.label(trim_labels)

    if verbose:
        toc = timeit.default_timer()
        print(f'      Trimming took {toc - tic:.3f}s')

    # Generate stats table
    if verbose:
        print('Building stats table')
        tic = timeit.default_timer()

    stats_table = pd.DataFrame(measure.regionprops_table(trim_labels, segment_data, properties=('label',
                                                                                                'centroid_weighted',
                                                                                                'bbox',
                                                                                                'intensity_min',
                                                                                                'intensity_max')))
    # Compute stats
    stats_table['prominence'] = stats_table['intensity_max'] - stats_table['intensity_min']

    # Use bounding box to get bw and dur
    minr = stats_table['bbox-0']
    minc = stats_table['bbox-1']
    maxr = stats_table['bbox-2']
    maxc = stats_table['bbox-3']

    stats_table['duration'] = (maxc - minc) * d_time
    stats_table['bandwidth'] = (maxr - minr) * d_freq

    # Compute time and freq from centroid
    stats_table['peak_time'] = stats_table['centroid_weighted-1'] * d_time + start_time
    stats_table['peak_frequency'] = stats_table['centroid_weighted-0'] * d_freq

    # Compute volume from label data and spectrogram
    stats_table['volume'] = [np.sum(segment_data[np.where(trim_labels == i)]) * d_time * d_freq for i in
                             stats_table['label']]

    # Drop unneeded columns
    del stats_table['centroid_weighted-0']
    del stats_table['centroid_weighted-1']
    del stats_table['bbox-0']
    del stats_table['bbox-1']
    del stats_table['bbox-2']
    del stats_table['bbox-3']
    del stats_table['intensity_max']
    del stats_table['intensity_min']

    # Query stats table for final results
    stats_table = stats_table.query(
        'duration>@dur_min & duration<@dur_max & bandwidth>@bw_min & bandwidth<@bw_max & '
        'prominence>@prom_min')

    filtered_labels = np.zeros(segment_data.shape)
    for n in stats_table['label']:
        filtered_labels += (trim_labels == n)

    # Get updated label array
    filtered_labels = measure.label(filtered_labels)

    if verbose:
        toc = timeit.default_timer()
        print(f'      Stats table took {toc - tic:.3f}s')

        # # Display region properties
        # print('Stats table:')
        # print(stats_table.to_string())
        # print(' ')

    if plot_on:
        # Plot post-merged network
        img_extent = 0, list(segment_data.shape)[1] * d_time, list(segment_data.shape)[0] * d_freq, 0

        plt.subplot(141)
        plt.imshow(np.log(segment_data), extent=img_extent, cmap='jet')
        plt.gca().invert_yaxis()
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.title('Spectrogram')

        plt.subplot(142)
        image_label_overlay = color.label2rgb(labels, bg_label=0)
        plt.imshow(image_label_overlay, extent=img_extent)
        plt.gca().invert_yaxis()
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.title('Original Regions')

        plt.subplot(143)
        image_label_overlay = color.label2rgb(labels_merged, bg_label=0)
        plt.imshow(image_label_overlay, extent=img_extent)
        plt.gca().invert_yaxis()
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.title('Merged Regions')

        plt.subplot(144)
        image_label_overlay = color.label2rgb(filtered_labels, bg_label=0)
        plt.imshow(image_label_overlay, extent=img_extent)
        plt.gca().invert_yaxis()
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.title('Trimmed/Filtered Regions')

        # Show Figures
        plt.show()

    return stats_table


csv_data = pd.read_csv('data_segment.csv', header=None)

# Get test data from the CSV file
data_csv = np.array(csv_data[0]).astype(np.float32)
# Sampling Frequency
fs = 100

# Number of jobs to use
n_jobs = max(cpu_count() - 1, 1)

#%% COMPUTE MULTITAPER SPECTROGRAM
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
verbose = True  # print extra info
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
spect, stimes, sfreqs = multitaper_spectrogram(data_csv, fs, frequency_range, time_bandwidth, num_tapers,
                                               window_params,
                                               min_nfft, detrend_opt, multiprocess, cpus,
                                               weighting, plot_on, clim_scale, verbose, xyflip)

# Define spectral coords dx dy
d_time = stimes[1] - stimes[0]
d_freq = sfreqs[1] - sfreqs[0]

# Remove baseline
baseline = np.percentile(spect, 3, axis=1, keepdims=True)
spect_baseline = np.divide(spect, baseline)

#%% DETECT TF-PEAKS

# Set TF-peak detection settings
merge_thresh = 11
trim_vol = 0.8
segment_dur = 30  # Segment time in seconds
downsample = [2, 2]

# Set the size of the spectrogram samples
window_idxs, start_times = process_segments_params(segment_dur, stimes)
num_windows = len(start_times)

# Set up the parameters to pass to each window
dp_params = (merge_thresh, trim_vol, downsample, dur_min, dur_max, bw_min, bw_max, prom_min, False, False)

detect_tfpeaks(spect_baseline[:, window_idxs[0]], start_times[0], *dp_params)

# # Run jobs in parallel
# print('Running peak detection in parallel with ' + str(n_jobs) + ' jobs...')
# tic = timeit.default_timer()
#
# stats_table = pd.concat(Parallel(n_jobs=n_jobs)(delayed(detect_tfpeaks)(
#     spect_baseline[:, window_idxs[num_window]], start_times[num_window], *dp_params) for num_window in tqdm(range(num_windows))), ignore_index=True)
# toc = timeit.default_timer()
# print('      Peak detection took ' + convertHMS(toc-tic))
#
# # Fix the stats_table to sort by time and reset labels
# del stats_table['label']
# stats_table.sort_values('peak_time')
# stats_table.reset_index()
#
# # Plot post-merged network
# img_extent = stimes[0], stimes[len(stimes)-1], sfreqs[len(sfreqs)-1], sfreqs[0]
#
# peak_size = stats_table['volume']/15
# pmax = np.percentile(list(peak_size), 95)
# peak_size[peak_size>pmax] = 0
#
# x = [stats_table.peak_time - 2*(stimes[1]-stimes[0])]
# y = [stats_table.peak_frequency]
# plt.scatter(x, y, peak_size, facecolors='none', edgecolors='k')
# plt.xlabel('Time (s)')
# plt.ylabel('Frequency (Hz)')
#
# # Show Figures
# plt.show()