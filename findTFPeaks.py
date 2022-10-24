import numpy
import skimage.future.graph
from skimage.io import imread, imshow
from skimage import measure, segmentation, future, color, data, morphology
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import timeit
from scipy.stats.distributions import chi2
from multitaper_toolbox.python.multitaper_spectrogram_python import \
    multitaper_spectrogram  # import multitaper_spectrogram function from the multitaper_spectrogram_python.py file


def pow2db(val):
    """
    Converts power to dB
    :param val: values to convert
    :return: val_dB value in dB
    """
    val_dB = (10 * np.log10(val) + 300) - 300
    return val_dB


def plot_node(graph_rag, node_num):
    """
    Plots a node
    :param graph_rag: RAG graph
    :param node_num: Node number
    """
    for i in graph_rag.nodes[node_num]["border"]:
        plt.plot(i[1], i[0], 'kx')
    for i in graph_rag.nodes[node_num]["region"]:
        plt.plot(i[1], i[0], 'b.')


def edge_weight(graph_rag: skimage.future.graph.RAG, graph_edge: tuple, graph_data: numpy.ndarray) -> float:
    """
    Computes the edge weight between two regions

    :param graph_data: numpy.ndarray
    :param graph_edge: tuple
    :param graph_rag: skimage.future.graph.rag.RAG
    """
    # Get border and region tuples
    i_border = graph_rag.nodes[graph_edge[0]]["border"]
    i_region = graph_rag.nodes[graph_edge[0]]["region"]
    j_border = graph_rag.nodes[graph_edge[1]]["border"]
    j_region = graph_rag.nodes[graph_edge[1]]["region"]

    A_ij = i_border.intersection(j_border)

    # Expand border and region if no overlap
    if len(A_ij) == 0:
        return np.nan
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
    w_ij = 2 * A_ij_max - B_i_min - j_max

    # Compute weight for j into i
    # C_ji = A_ij_max - B_j_min
    # D_ji = i_max - A_ij_max
    # w_ji = C_ji - D_ji = 2 * A_ij_max - B_j_min - i_max
    w_ji = 2 * A_ij_max - B_j_min - i_max

    # NOTES:
    # 1) Weights flipped sign for the hierarchical merge function which goes from low to high
    # 2) Moved constant 2 * A_ij_max to final weight computation for efficiency

    # Take the max weight of w_ij and w_ji
    w_max = np.max([w_ij, w_ji])

    return w_max


def merge_regions(graph_rag: skimage.future.graph.RAG, src: int, dst: int):
    """
    Merges the regions and borders for use in hierarchical merge

    :param graph_rag: skimage.future.graph.rag.RAG
    :param src: Source node
    :param dst: Destination node
    """
    # Region is union of regions
    graph_rag.nodes[dst]["region"] = graph_rag.nodes[dst]["region"].union(graph_rag.nodes[src]["region"])
    # Border is symmetric difference of borders
    graph_rag.nodes[dst]["border"] = graph_rag.nodes[dst]["border"].symmetric_difference(graph_rag.nodes[src]["border"])
    # print(str(src) + ' > ' + str(dst) + ' weight: ' + str(graph_rag.edges[src, dst]['weight']))


# NOTE: Is there a better way to get segment_data into this other than making it global?
def merge_weight(graph_rag, src, dst, neighbor) -> dict:
    """
    Computes weight for use in hierarchical merge

    :rtype dict
    :return Weight between src and neighbors
    :param graph_rag: skimage.future.graph.rag.RAG
    :param src: Source node (unused but required)
    :param dst: Destination node (merged already)
    :param neighbor: Neighbor node
    """

    # Convert weight output to dictionary form
    n_weight = edge_weight(graph_rag, tuple([dst, neighbor]), segment_data)
    # print('   ' + str(list([dst, neighbor])) + ' new weight: ' + str(n_weight))
    return {'weight': n_weight}


def trim_region(graph_rag: skimage.future.graph.RAG, graph_data: numpy.ndarray, region_num: int, trim_volume: float):
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
        max_area_label = rp[np.argmax([p.area for p in rp])].label
        label_img = label_img == max_area_label
    else:
        label_img = label_img > 0

    return label_img

# reading the CSV file
csv_data = pd.read_csv('data.csv', header=None)
data_csv = np.array(csv_data[0])

# Set spectrogram params
fs = 100  # Sampling Frequency
frequency_range = [0, 30]  # Limit frequencies from 0 to 25 Hz
taper_params = [2, 3]  # Set taper params
time_bandwidth = taper_params[0]  # Set time-half bandwidth
num_tapers = taper_params[1]  # Set number of tapers (optimal is time_bandwidth*2 - 1)
window_params = [1, .05]  # Window size is 4s with step size of 1s
min_nfft = 2 ** 10  # NFFT
detrend_opt = 'constant'  # constant detrend
multiprocess = True  # use multiprocessing
cpus = 4  # use 4 cores in multiprocessing
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
spect, stimes, sfreqs = multitaper_spectrogram(data_csv, fs, frequency_range, time_bandwidth, num_tapers, window_params,
                                               min_nfft, detrend_opt, multiprocess, cpus,
                                               weighting, plot_on, clim_scale, verbose, xyflip)

# Define spectral coords dx dy
d_time = stimes[1] - stimes[0]
d_freq = sfreqs[1] - sfreqs[0]

baseline = np.percentile(spect, 3, axis=1, keepdims=True)

segment_data = np.divide(spect, baseline)
# Run watershed segmentation with empty border regions
labels = segmentation.watershed(-segment_data, connectivity=2, watershed_line=True)

# Expand labels by 1 to join them. This will be used to compute the RAG
join_labels = segmentation.expand_labels(labels, distance=10)

# Create a region adjacency graph based
labelRAG = future.graph.RAG(join_labels, connectivity=2)
for n in labelRAG.nodes():
    labelRAG.nodes[n]['labels'] = [n]

# Get all border and region pixels
for n in labelRAG:
    curr_region = (labels == n)
    bx, by = np.where(morphology.dilation(curr_region, np.ones([3, 3])) & (labels == 0))
    rx, ry = np.where(curr_region)

    # Zip into sets of tuples for easy set operations (e.g. intersection)
    border = set(zip(bx, by))
    region = set(zip(rx, ry))

    # Add border pixels to region
    for b in border:
        region.add(b)

    # Set node properties
    labelRAG.nodes[n]["border"] = border
    labelRAG.nodes[n]["region"] = region

print('Computing weights...')
tic = timeit.default_timer()
# Compute the initial RAG weights
for edge in labelRAG.edges:
    weight = edge_weight(labelRAG, edge, segment_data)

    if np.isnan(weight):
        labelRAG.remove_edge(edge[0], edge[1])
        # print("Removing: " + str(edge))
    else:
        labelRAG.edges[edge]["weight"] = weight
        # print('Edge ' + str(edge) + ' weight: ' + str(weight))
toc = timeit.default_timer()
print(f'      Weights took {toc - tic:.3f}s')
# # Show region boundaries with holes
# marked_bounds = segmentation.mark_boundaries(segment_data, labels, color=(1, 0, 1), outline_color=None, mode='outer',
#                                              background_label=0)

# Compute Region Properties
props_original = measure.regionprops(labels, segment_data)

# image_label_overlay = color.label2rgb(labels, bg_label=0)
# plt.imshow(image_label_overlay)
# for region in props_original:
#     for n in list(labelRAG.neighbors(region.label)):
#         for p in props_original:
#             if p["label"] == n:
#                 yn, xn = p.centroid_weighted
#                 plt.plot([x0, xn], [y0, yn], 'r-')
#     y0, x0 = region.centroid_weighted
#     plt.plot(x0, y0, '.k')


print('Starting merge...')

max_val = np.inf
tic = timeit.default_timer()
while max_val > 8:
    all_weights = [labelRAG.edges[i]["weight"] for i in labelRAG.edges]
    max_idx = np.argmax(all_weights)
    max_val = all_weights[max_idx]
    max_edge = list(labelRAG.edges)[max_idx]
    # print(max_val)
    src = max_edge[0]
    dst = max_edge[1]
    # Region is union of regions
    labelRAG.nodes[dst]["region"] = labelRAG.nodes[dst]["region"].union(labelRAG.nodes[src]["region"])
    # Border is symmetric difference of borders
    labelRAG.nodes[dst]["border"] = labelRAG.nodes[dst]["border"].symmetric_difference(labelRAG.nodes[src]["border"])
    labelRAG.merge_nodes(src, dst, merge_weight)
toc = timeit.default_timer()
print(f'      Merging took {toc - tic:.3f}s')
# # Perform hierarchical merging
# future.graph.merge_hierarchical(labels, labelRAG, thresh=-29, rag_copy=False,
#                                 in_place_merge=True,
#                                 merge_func=merge_regions,
#                                 weight_func=merge_weight)

# Create a new label image
labels_merged = np.zeros(labels.shape, dtype=int)
for n in labelRAG:
    for p in labelRAG.nodes[n]["region"]:
        labels_merged[p] = n
    for b in labelRAG.nodes[n]["border"]:
        labels_merged[b] = 0
# plt.title('Original')

# Compute region properties for plotting
props_all_merged = measure.regionprops(labels_merged, segment_data)

# plt.subplot(121)
# image_label_overlay = color.label2rgb(labels_merged, bg_label=0)
# plt.imshow(image_label_overlay)
# for region in props_all_merged:
#     y0, x0 = region.centroid_weighted
#     # for n in list(labelRAG.neighbors(region.label)):
#     #     for p in props_all_merged:
#     #         if p["label"] == n:
#     #             yn, xn = p.centroid_weighted
#     #             plt.plot([x0, xn], [y0, yn], 'r-')
#     plt.plot(x0, y0, '.k')

# Trim volume
trim_vol = 0.8

print('Trimming...')
tic = timeit.default_timer()
# Set up the trim images
trim_labels = np.zeros(segment_data.shape)
for r in labelRAG.nodes:
    # Get the values of the merged nodes
    r_border = list(zip(*labelRAG.nodes[r]["border"]))
    r_region = list(zip(*labelRAG.nodes[r]["region"]))

    # Compute bandwidth and duration
    bw = (np.max(r_border[0]) - np.min(r_border[0])) * d_freq
    dur = (np.max(r_border[1]) - np.min(r_border[1])) * d_time

    data_vals = segment_data[r_region[0], r_region[1]]
    height = pow2db(max(data_vals) - min(data_vals))

    # Only trim if within parameters
    if (bw >= bw_min) & (dur >= dur_min) & (height >= prom_min):
        trim_labels += trim_region(labelRAG, segment_data, r, trim_vol)

trim_labels = measure.label(trim_labels)

# Compute region properties for plotting
props_all_trimmed = measure.regionprops(trim_labels, segment_data)
toc = timeit.default_timer()
print(f'      Trimming took {toc - tic:.3f}s')

# Table for data
print('Building stats table')
tic = timeit.default_timer()
stats_table = pd.DataFrame(measure.regionprops_table(trim_labels, segment_data, properties=('centroid_weighted',
                                                                                            'bbox',
                                                                                            'intensity_min',
                                                                                            'intensity_max', 'label')))
stats_table['prominence'] = stats_table['intensity_max'] - stats_table['intensity_min']
minr = stats_table['bbox-0']
minc = stats_table['bbox-1']
maxr = stats_table['bbox-2']
maxc = stats_table['bbox-3']

stats_table['duration'] = (maxc - minc) * d_time
stats_table['bandwidth'] = (maxr - minr) * d_freq

stats_table['peak_time'] = stats_table['centroid_weighted-1'] * d_time
stats_table['peak_frequency'] = stats_table['centroid_weighted-0'] * d_freq

# Query stats table for final results
stats_table = stats_table.query('duration>@dur_min & duration<@dur_max & bandwidth>@bw_min & bandwidth<@bw_max & '
                                'prominence>@prom_min')

filtered_labels = np.zeros(segment_data.shape)
for i in stats_table['label']:
    filtered_labels += (trim_labels == i)

filtered_labels = measure.label(filtered_labels)

toc = timeit.default_timer()
print(f'      Stats table took {toc - tic:.3f}s')

# Display region properties
print('Stats table:')
print(stats_table.to_string())
print(' ')

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
image_label_overlay = color.label2rgb(filtered_labels,  bg_label=0)
plt.imshow(image_label_overlay, extent=img_extent)
plt.gca().invert_yaxis()
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Trimmed/Filtered Regions')

for index, row in stats_table.iterrows():
    plt.plot(row['peak_time'], row['peak_frequency'], 'm.')

# Show Figures
plt.show()
