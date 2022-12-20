import timeit

import pandas as pd
import skimage.future.graph
from skimage import measure, segmentation, future, color, morphology
from skimage.transform import resize

from pyTOD.utils import *


def edge_weight(graph_rag: skimage.future.graph.RAG, graph_edge: tuple, graph_data: np.ndarray) -> float:
    """
    Computes the edge weight between two regions

    Parameters
    ----------
    graph_rag : skimage.future.graph.RAG
        Region adjacency graph
    graph_edge : tuple
        Edge tuple
    graph_data : np.ndarray
        Graph data

    Returns
    -------
    float
        Edge weight
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

    Parameters
    ----------
    graph_rag : skimage.future.graph.RAG
        The region adjacency graph
    src : int
        The source region
    dst : int
        The destination region
    neighbor : int
        The neighbor region
    graph_data : np.ndarray
        The data array for the graph

    Returns
    -------
    dict
        A dictionary containing the weight of the edge between the two regions
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

    Parameters
    ----------
    graph_rag : skimage.future.graph.RAG
        The region adjacency graph
    src : int
        The source region
    dst : int
        The destination region

    Returns
    -------
    None
    """
    # Region is union of regions
    graph_rag.nodes[dst]["region"] = np.union1d(graph_rag.nodes[dst]["region"], graph_rag.nodes[src]["region"])
    # Border is symmetric difference of borders
    graph_rag.nodes[dst]["border"] = np.setxor1d(graph_rag.nodes[dst]["border"], graph_rag.nodes[src]["border"])


def trim_region(graph_rag: skimage.future.graph.RAG, labels_merged: np.ndarray, graph_data: np.ndarray, region_num: int,
                trim_volume: float):
    """Trims a region to a given volume percentage

    Parameters
    ----------
    graph_rag : skimage.future.graph.RAG
        The region adjacency graph
    labels_merged : np.ndarray
        The merged labels array
    graph_data : np.ndarray
        The data array used to compute the edge weights
    region_num : int
        The region number to trim
    trim_volume : float
        The volume to trim to, between 0 and 1

    Returns
    -------
    np.ndarray
        The trimmed region

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

    Parameters
    ----------
    graph_rag : RAG
        The region adjacency graph
    node_num : int
        The node number to plot

    Returns
    -------
    None
    """
    for i in graph_rag.nodes[node_num]["border"]:
        plt.plot(i[1], i[0], 'kx')
    for i in graph_rag.nodes[node_num]["region"]:
        plt.plot(i[1], i[0], 'b.')


def process_segments_params(segment_dur: float, stimes: np.ndarray):
    """Gets parameters for segmenting the spectrogram

    Parameters
    ----------
    segment_dur : float
        The duration of the segment in seconds
    stimes : np.ndarray
        spectrogram times

    Returns
    -------
    window_idx : list
        list of indexes for each window
    start_times : np.ndarray
        start times for each window
    """
    # create frequency vector
    d_stimes = stimes[1] - stimes[0]
    win_samples = np.round(segment_dur / d_stimes)

    window_start = np.arange(0, len(stimes) - 1, win_samples)

    # Get indexes for each window
    window_idxs = [np.arange(window_start[i], np.min([window_start[i] + win_samples, len(stimes)]), 1).astype(int)
                   for i in np.arange(0, len(window_start), 1)]

    start_times = stimes[window_start.astype(int)]

    return window_idxs, start_times


def detect_TFpeaks(segment_data: np.ndarray, start_time=0, d_time=1, d_freq=1, merge_threshold=8, max_merges=np.inf,
                   trim_volume=0.8, downsample=None, dur_min=-np.inf, dur_max=np.inf, bw_min=-np.inf, bw_max=np.inf,
                   prom_min=-np.inf, plot_on=True, verbose=True) -> pd.DataFrame:
    """Detects TF-peaks within a spectrogram

    Parameters
    ----------
    segment_data : np.ndarray
        The spectrogram data to segment.
    start_time : int, optional
        The start time of the spectrogram in seconds. The default is 0.
    d_time : int, optional
        The time resolution of the spectrogram in seconds. The default is 1.
    d_freq : int, optional
        The frequency resolution of the spectrogram in Hz. The default is 1.
    merge_threshold : int, optional
        The threshold for merging regions. The default is 8.
    max_merges : int, optional
        The maximum number of merges to perform. The default is np.inf.
    trim_volume : float, optional
        The volume threshold for trimming regions. The default is 0.8.
    downsample : list, optional
        A list of downsample factors for the spectrogram data. The default is None.
    dur_min : int, optional
        The minimum duration for a peak in seconds. The default is -np.inf.
    dur_max : int, optional
        The maximum duration for a peak in seconds. The default is np.inf.
    bw_min : int, optional
        The minimum bandwidth for a peak in Hz. The default is -np.inf.
    bw_max : int, optional
        The maximum bandwidth for a peak in Hz. The default is np.inf.
    prom_min : int, optional
        The minimum prominence for a peak in dB. The default is -np.inf.
    plot_on : bool, optional
        Whether to plot the segmentation results or not. The default is True.
    verbose : bool, optional
        Whether to print verbose output or not. The default is True.

    Returns
    -------
    stats_table: pd.DataFrame
        A table of stats for each peak detected in the spectrogram.

        Columns:
            label: the label of the peak in the spectrogram image
            peak_time: the time of the peak in seconds
            peak_frequency: the frequency of the peak in Hz
            duration: the duration of the peak in seconds
            bandwidth: the bandwidth of the peak in Hz
            prominence: the prominence of the peak in dB
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

    # Merge loop
    num_merges = 1
    while max_val >= merge_threshold:
        if num_merges > max_merges:
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

        # merge_regions(RAG, src, dst)
        # Region is union of regions
        RAG.nodes[dst]["region"] = np.union1d(RAG.nodes[dst]["region"], RAG.nodes[src]["region"])
        # Border is symmetric difference of borders
        RAG.nodes[dst]["border"] = np.setxor1d(RAG.nodes[dst]["border"], RAG.nodes[src]["border"])

        if verbose:
            toc_borders = timeit.default_timer()
            merge_borders_time += toc_borders - tic_borders
            tic_node = timeit.default_timer()

        RAG.merge_nodes(src, dst, merge_weight, extra_arguments=[segment_data_LR])

        if verbose:
            toc_node = timeit.default_timer()
            merge_node_time += toc_node - tic_node

        # Update the merge counter
        num_merges += 1

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
