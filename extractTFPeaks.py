import timeit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.future.graph
from joblib import Parallel, delayed, cpu_count
from scipy.stats.distributions import chi2
from skimage import measure, segmentation, future, color, morphology
from skimage.transform import resize
from tqdm import tqdm
from multitaper_toolbox.python.multitaper_spectrogram_python import multitaper_spectrogram
from pyTODhelper import *
import heapq


def edge_weight(graph_rag: skimage.future.graph.RAG, graph_edge: tuple, graph_data: np.ndarray) -> float:
    """
    Computes the edge weight between two regions

    :return: Edge weight
    :param graph_rag: RAG graph
    :param graph_edge: edge for which to compute weight
    :param graph_data: spectrogram data
    """
    # Get border and region tuples
    i_border, i_region = list(graph_rag.nodes[graph_edge[0]].values())[1:]
    j_border, j_region = list(graph_rag.nodes[graph_edge[1]].values())[1:]

    A_ij = np.intersect1d(i_border, j_border)

    # Diagonal neighbor is not really connected
    if not len(A_ij):
        return np.nan  # Set to nan to not be a neighbor

    A_ij_max = np.max(graph_data[np.unravel_index(A_ij, graph_data.shape)])

    # Store for reuse efficiency
    i_border_vals = graph_data[np.unravel_index(i_border, graph_data.shape)]
    j_border_vals = graph_data[np.unravel_index(i_border, graph_data.shape)]

    # Minimum border values
    B_i_min = np.min(i_border_vals)
    B_j_min = np.min(j_border_vals)

    i_region_vals = graph_data[np.unravel_index(i_region, graph_data.shape)]
    j_region_vals = graph_data[np.unravel_index(j_region, graph_data.shape)]

    # Max region values
    i_max = np.max(np.concatenate([i_border_vals, i_region_vals]))
    j_max = np.max(np.concatenate([j_border_vals, j_region_vals]))

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


def merge_weight(graph_rag: skimage.future.graph.RAG, src: int, dst: int, neighbor: int,
                 graph_data: np.ndarray) -> dict:
    """
    Computes the edge weight between two regions in merge. (mirrors edge_weight)
    NOTE: Keeping as a distinct function saves some time rather than calling a wrapper to edge_weight

    :return: edge weight
    :param neighbor: neighbor node
    :param graph_rag: RAG graph
    :param src: source (unused)
    :param dst: destination node (merged already)
    :param graph_data: spectrogram data
    """

    # Get border and region tuples
    i_border, i_region = list(graph_rag.nodes[dst].values())[1:]
    j_border, j_region = list(graph_rag.nodes[neighbor].values())[1:]

    A_ij = np.intersect1d(i_border, j_border)

    # Diagonal neighbor is not really connected
    if not len(A_ij):
        return np.nan  # Set to nan to not be a neighbor

    A_ij_max = np.max(graph_data[np.unravel_index(A_ij, graph_data.shape)])

    # Store for reuse efficiency
    i_border_vals = graph_data[np.unravel_index(i_border, graph_data.shape)]
    j_border_vals = graph_data[np.unravel_index(i_border, graph_data.shape)]

    # Minimum border values
    B_i_min = np.min(i_border_vals)
    B_j_min = np.min(j_border_vals)

    i_region_vals = graph_data[np.unravel_index(i_region, graph_data.shape)]
    j_region_vals = graph_data[np.unravel_index(j_region, graph_data.shape)]

    # Max region values
    i_max = np.max(np.concatenate([i_border_vals, i_region_vals]))
    j_max = np.max(np.concatenate([j_border_vals, j_region_vals]))

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

    # Return as dict
    return {'weight': w_max}


def merge_regions(graph_rag: skimage.future.graph.RAG, src: int, dst: int):
    """
    Merges the regions and borders for use in hierarchical merge

    :param graph_rag: RAG graph
    :param src: Source node
    :param dst: Destination node
    """
    # Region is union of regions
    graph_rag.nodes[dst]["region"] = np.union1d(graph_rag.nodes[dst]["region"], graph_rag.nodes[src]["region"])
    # Border is symmetric difference of borders
    graph_rag.nodes[dst]["border"] = np.setxor1d(graph_rag.nodes[dst]["border"], graph_rag.nodes[src]["border"])


def trim_region(graph_rag: skimage.future.graph.RAG, labels_merged: np.ndarray, graph_data: np.ndarray, region_num: int,
                trim_volume: float):
    """
    Computes the edge weight between two regions
    NOTE: Weights must be flipped to be negative to match ascending hierarchical merging

    :param labels_merged: labels of merged regions
    :param trim_volume: Percentage to trim
    :type: float
    :param graph_rag: RAG
    :type graph_rag: skimage.future.graph.rag.RAG
    :param graph_data: Spectrogram data
    :type graph_data: np.ndarray
    :param region_num: Region number
    :type region_num: int
    """

    # Get the region pixel values
    reg_idx = np.where(labels_merged == region_num)  # list(zip(*graph_rag.nodes[region_num]["region"]))
    reg_vals = np.sort(graph_data[reg_idx])  # graph_data[reg_idx[0], reg_idx[1]])

    # Find the cutoff for the volume
    percent_vol = np.cumsum(reg_vals) / np.sum(reg_vals)
    trim_idx = next(x for x, val in enumerate(percent_vol)
                    if val > 1 - trim_volume)
    trim_level = reg_vals[trim_idx]
    rplot = np.zeros(graph_data.shape)
    rplot[reg_idx] = 1

    # Slice at the trim level
    img = np.logical_and(rplot, graph_data > trim_level)

    # Fill all holes
    label_img = measure.label(morphology.remove_small_holes(img), connectivity=2)

    # Select the largest region if there are subregions after trim
    if np.max(label_img) > 1:
        rp = measure.regionprops(label_img)
        max_area_label = rp[np.argmax([prop.area for prop in rp])].label
        label_img = (label_img == max_area_label)

    return label_img * region_num


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


def process_segments_params(segment_dur: float, stimes: np.ndarray):
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


def _revalidate_node_edges(rag, node, heap_list):
    """Handles validation and invalidation of edges incident to a node.
    This function invalidates all existing edges incident on `node` and inserts
    new items in `heap_list` updated with the valid weights.
    rag : RAG
        The Region Adjacency Graph.
    node : int
        The id of the node whose incident edges are to be validated/invalidated
        .
    heap_list : list
        The list containing the existing heap of edges.
    """
    # networkx updates data dictionary if edge exists
    # this would mean we have to reposition these edges in
    # heap if their weight is updated.
    # instead we invalidate them

    for nbr in rag.neighbors(node):
        data = rag[node][nbr]
        try:
            # invalidate edges incident on `dst`, they have new weights
            data['heap item'][3] = False
            _invalidate_edge(rag, node, nbr)
        except KeyError:
            # will handle the case where the edge did not exist in the existing
            # graph
            pass

        wt = data['weight']
        heap_item = [wt, node, nbr, True]
        data['heap item'] = heap_item
        heapq.heappush(heap_list, heap_item)


def _rename_node(graph, node_id, copy_id):
    """ Rename `node_id` in `graph` to `copy_id`. """

    graph._add_node_silent(copy_id)
    graph.nodes[copy_id].update(graph.nodes[node_id])

    for nbr in graph.neighbors(node_id):
        wt = graph[node_id][nbr]['weight']
        graph.add_edge(nbr, copy_id, {'weight': wt})

    graph.remove_node(node_id)


def _invalidate_edge(graph, n1, n2):
    """ Invalidates the edge (n1, n2) in the heap. """
    graph[n1][n2]['heap item'][3] = False


def merge_hierarchical(labels, rag, thresh, rag_copy, in_place_merge,
                       merge_func, weight_func, segment_data_LR):
    """Perform hierarchical merging of a RAG.
    Greedily merges the most similar pair of nodes until no edges lower than
    `thresh` remain.
    Parameters
    ----------
    labels : ndarray
        The array of labels.
    rag : RAG
        The Region Adjacency Graph.
    thresh : float
        Regions connected by an edge with weight smaller than `thresh` are
        merged.
    rag_copy : bool
        If set, the RAG copied before modifying.
    in_place_merge : bool
        If set, the nodes are merged in place. Otherwise, a new node is
        created for each merge..
    merge_func : callable
        This function is called before merging two nodes. For the RAG `graph`
        while merging `src` and `dst`, it is called as follows
        ``merge_func(graph, src, dst)``.
    weight_func : callable
        The function to compute the new weights of the nodes adjacent to the
        merged node. This is directly supplied as the argument `weight_func`
        to `merge_nodes`.
    Returns
    -------
    out : ndarray
        The new labeled array.
    """
    # if rag_copy:
    #     rag = rag.copy()

    edge_heap = []
    for n1, n2, data in rag.edges(data=True):
        # Push a valid edge in the heap
        wt = data['weight']
        heap_item = [wt, n1, n2, True]
        heapq.heappush(edge_heap, heap_item)

        # Reference to the heap item in the graph
        data['heap item'] = heap_item

    heapq._heapify_max(edge_heap)

    while len(edge_heap) > 0 and edge_heap[0][0] > thresh:
        _, n1, n2, valid = heapq._heappop_max(edge_heap)

        # Ensure popped edge is valid, if not, the edge is discarded
        if valid:
            # Invalidate all neigbors of `src` before its deleted

            for nbr in rag.neighbors(n1):
                _invalidate_edge(rag, n1, nbr)

            for nbr in rag.neighbors(n2):
                _invalidate_edge(rag, n2, nbr)

            if not in_place_merge:
                next_id = rag.next_id()
                _rename_node(rag, n2, next_id)
                src, dst = n1, next_id
            else:
                src, dst = n1, n2

            # # Region is union of regions
            # rag.nodes[dst]["region"] = np.union1d(rag.nodes[dst]["region"], RAG.nodes[src]["region"])
            # #     # Border is symmetric difference of borders
            # rag.nodes[dst]["border"] = np.setxor1d(rag.nodes[dst]["border"], RAG.nodes[src]["border"])

        merge_regions(rag, src, dst)

        # merge_func(rag, src, dst)
        new_id = rag.merge_nodes(src, dst, merge_weight, extra_arguments=[segment_data_LR])
        _revalidate_node_edges(rag, new_id, edge_heap)

    label_map = np.arange(labels.max() + 1)
    for ix, (n, d) in enumerate(rag.nodes(data=True)):
        for label in d['labels']:
            label_map[label] = ix

    return label_map[labels]


def detect_tfpeaks(segment_data: np.ndarray, start_time=0, d_time=1, d_freq=1, merge_threshold=8, max_merges=np.inf,
                   trim_volume=0.8,
                   downsample=None, dur_min=-np.inf, dur_max=np.inf, bw_min=-np.inf, bw_max=np.inf, prom_min=-np.inf,
                   plot_on=True, verbose=True) -> pd.DataFrame:
    """Detects TF-peaks within a spectrogram

    :return: Table of peak statistics
    :param segment_data: Spectrogram segment for peak detection
    :param start_time: Start time of the segment
    :param d_time: Time bin size
    :param d_freq: Frequency bin size
    :param merge_threshold: Threshold to stop merging
    :param max_merges: Maximum number of merges to perform
    :param trim_volume: % of total peak volume to keep
    :param downsample: time x freq step to downsample
    :param dur_min: min peak duration
    :param dur_max: max peak duration
    :param bw_min: min bandwidth
    :param bw_max: max bandwidth
    :param prom_min: min prominence
    :param plot_on: Flag for plotting
    :param verbose: Verbose setting
    """

    if verbose:
        tic_all = timeit.default_timer()

    # Downsample data
    if downsample is None:
        downsample = []
    if downsample:
        segment_data_LR = segment_data[::downsample[0], ::downsample[1]]
    else:
        segment_data_LR = segment_data

    # Run watershed segmentation with empty border regions
    labels = segmentation.watershed(-segment_data_LR, connectivity=2, watershed_line=True)

    # Expand labels by 5 (to be safe) to join them. This will be used to compute the RAG
    join_labels = segmentation.expand_labels(labels, distance=5)

    # Create a region adjacency graph (RAG))
    RAG = future.graph.RAG(join_labels, connectivity=2)

    # Add labels, borders, and regions to each RAG node
    for n in RAG:
        RAG.nodes[n]['labels'] = [n]

        curr_region = (labels == n)
        # Compute the borders by intersecting 1 pixel dilation with zero-valued watershed border regions
        bx, by = np.where(morphology.dilation(curr_region, np.ones([3, 3])) & (labels == 0))
        border = np.array([np.ravel_multi_index((a, b), labels.shape) for a, b in zip(bx, by)])

        # Get regions by bing full region
        rx, ry = np.where(curr_region)
        region = np.array([np.ravel_multi_index((a, b), labels.shape) for a, b in zip(rx, ry)])

        # Set node properties
        RAG.nodes[n]["border"] = border
        RAG.nodes[n]["region"] = region

    if verbose:
        print('Computing weights...')
        tic_weights = timeit.default_timer()

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
        toc_weights = timeit.default_timer()
        print(f'      Weights took {toc_weights - tic_weights:.3f}s')

    if verbose:
        print('Starting merge...')
        tic_merge = timeit.default_timer()
        get_max_time = 0
        merge_borders_time = 0
        merge_node_time = 0

    # Set initial max value
    max_val = np.inf

    # Unclear if any advantage to for vs. while loop construction
    for num_merges in range(max_merges):
        if max_val <= merge_threshold:
            break

        if verbose:
            tic_max = timeit.default_timer()

        # Get max val and edge
        (max_val, src, dst) = max([(d['weight'], u, v) for (u, v, d) in RAG.edges(data=True)])

        if verbose:
            toc_max = timeit.default_timer()
            get_max_time += toc_max - tic_max

        if verbose:
            tic_borders = timeit.default_timer()

        merge_regions(RAG, src, dst)

        if verbose:
            toc_borders = timeit.default_timer()
            merge_borders_time += toc_borders - tic_borders
            tic_node = timeit.default_timer()

        RAG.merge_nodes(src, dst, merge_weight, extra_arguments=[segment_data_LR])

        if verbose:
            toc_node = timeit.default_timer()
            merge_node_time += toc_node - tic_node

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
            labels_merged[np.unravel_index(p, labels_merged.shape)] = n
    for n in RAG:
        for b in RAG.nodes[n]["border"]:
            labels_merged[np.unravel_index(b, labels_merged.shape)] = 0

    join_labels_merged = segmentation.expand_labels(labels_merged, distance=5)

    if downsample:
        labels_merged = resize(join_labels_merged, segment_data.shape, 0)

    if verbose:
        print('Trimming...')
        tic_trim = timeit.default_timer()

    # Set up the trim images
    trim_labels = np.zeros(segment_data.shape)

    c = 1
    for n in np.unique(labels_merged):
        curr_region = (labels_merged == n)
        rx, ry = np.where(curr_region)
        bw = (np.max(rx) - np.min(rx)) * d_freq
        dur = (np.max(ry) - np.min(ry)) * d_time
        data_vals = segment_data[curr_region]
        height = pow2db(max(data_vals) - min(data_vals))

        # Only trim if within parameters
        if (bw >= bw_min) & (dur >= dur_min) & (height >= prom_min):
            # Add to new labels image
            trim_labels += (trim_region(RAG, labels_merged, segment_data, n, trim_volume) > 0) * c
            c = c + 1

    # Convert to int to work as a labeled image
    trim_labels = trim_labels.astype(int)

    if verbose:
        toc_trim = timeit.default_timer()
        print(f'      Trimming took {toc_trim - tic_trim:.3f}s')

    # Generate stats table
    if verbose:
        print('Building stats table')
        tic_stats = timeit.default_timer()

    # Compute the stats table for just the needed stats
    stats_table = pd.DataFrame(measure.regionprops_table(trim_labels, segment_data, properties=('label',
                                                                                                'centroid_weighted',
                                                                                                'bbox',
                                                                                                'intensity_min',
                                                                                                'intensity_max')))
    # Calculate all custom stats
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
                             stats_table.label.values]

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
        'duration>@dur_min & duration<@dur_max & bandwidth>@bw_min & bandwidth<@bw_max & prominence>@prom_min')

    if verbose:
        toc_stats = timeit.default_timer()
        print(f'      Stats table took {toc_stats - tic_stats:.3f}s')

    if plot_on:
        # Make labels post filtering for display
        filtered_labels = np.zeros(segment_data.shape)
        for n in stats_table.label:
            filtered_labels += (trim_labels == n) * n

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

    if verbose:
        toc_all = timeit.default_timer()
        print(f'TOTAL SEGMENT TIME:  {toc_all - tic_all:.3f}s')

    return stats_table


def run_TFpeak_extraction(quality='fast'):
    # if not data:
    # Load in data
    csv_data = pd.read_csv('data/night_data.csv', header=None)
    data = np.array(csv_data[0]).astype(np.float32)

    # Sampling Frequency
    fs = 100

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

    # Set TF-peak detection settings
    trim_vol = 0.8
    max_merges = 500
    plot_on = False
    verbose = False

    # Set the size of the spectrogram samples
    window_idxs, start_times = process_segments_params(segment_dur, stimes)
    num_windows = len(start_times)

    # Set up the parameters to pass to each window
    dp_params = (d_time, d_freq, merge_thresh, max_merges, trim_vol, downsample, dur_min, dur_max,
                 bw_min, bw_max, prom_min, plot_on, verbose)

    #  Run jobs in parallel
    print('Running peak detection in parallel with ' + str(n_jobs) + ' jobs...')
    tic_outer = timeit.default_timer()

    # Single chunk test
    stats_table = detect_tfpeaks(spect_baseline[:, window_idxs[0]], start_times[0], *dp_params)

    stats_tables = Parallel(n_jobs=n_jobs)(delayed(detect_tfpeaks)(
        spect_baseline[:, window_idxs[num_window]], start_times[num_window], *dp_params)
                                           for num_window in tqdm(range(num_windows)))

    stats_table = pd.concat(stats_tables, ignore_index=True)

    toc_outer = timeit.default_timer()
    print('Peak detection took ' + convertHMS(toc_outer - tic_outer))

    # Fix the stats_table to sort by time and reset labels
    del stats_table['label']
    stats_table.sort_values('peak_time')
    stats_table.reset_index()

    print('Writing stats_table to file...')
    stats_table.to_csv('data/data_night_peaks.csv')
    print('Done')


if __name__ == '__main__':
    # Extract TF-peaks from scratch

    print('here')
    run_TFpeak_extraction()
    print('done')
