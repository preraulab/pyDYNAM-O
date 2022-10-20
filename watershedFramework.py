from skimage.io import imread, imshow
from skimage import measure, segmentation, future, color, data, morphology
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def edge_weight(graph_rag, graph_edge, graph_data):
    """
    Computes the edge weight between two regions
    NOTE: Weights must be flipped to be negative to match ascending hierarchical merging

    :param graph_data: numpy.ndarray
    :param graph_edge: tuple
    :param graph_rag: skimage.future.graph.rag.RAG
    """
    i_border = graph_rag.nodes[graph_edge[0]]["border"]
    i_region = graph_rag.nodes[graph_edge[0]]["region"]
    j_border = graph_rag.nodes[graph_edge[1]]["border"]
    j_region = graph_rag.nodes[graph_edge[1]]["region"]
    
    A_ij = i_border.intersection(j_border)
    A_ij_max = np.max([graph_data[i] for i in A_ij])

    B_i_min = np.min([graph_data[i] for i in i_border])
    B_j_min = np.min([graph_data[i] for i in j_border])
    
    i_max = np.max([graph_data[i] for i in i_region] + [graph_data[i] for i in i_border])
    j_max = np.max([graph_data[i] for i in j_region] + [graph_data[i] for i in j_border])
    # C_ij = A_ij_max - B_i_min
    # D_ij = j_max - A_ij_max
    # w_ij = C_ij - D_ij = 2 * A_ij_max - B_i_min - j_max
    w_ij = 2 * A_ij_max - B_i_min - j_max

    # C_ji = A_ij_max - B_j_min
    # D_ji = i_max - A_ij_max
    # w_ij = C_ji - D_ji = 2 * A_ij_max - B_j_min - i_max
    w_ji = 2 * A_ij_max - B_j_min - i_max

    # NOTE: WEIGHTS MUST BE NEGATIVE DUE TO HIERARCHICAL MERGE
    # WHICH MERGES FROM LOW TO HIGH
    w = -np.max([w_ij, w_ji])
    
    # print('Computing weight for edges ' + str(edge))
    # print("B_i_min: " + str(B_i_min))
    # print("A_ij_max: " + str(A_ij_max))
    # print("B_j_min: " + str(B_j_min))
    # print("i_max: " + str(i_max))
    # print("j_max: " + str(j_max))
    # print("    w_ij: " + str(w_ij))
    # print("    w_ji: " + str(w_ji))
    # print('Weight = ' + str(w))
    return w


def merge_regions(graph_rag, src, dst):
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


def merge_weight(graph_rag, src, dst, neighbor):
    """
    Computes weight for use in hierarchical merge

    :param graph_rag: skimage.future.graph.rag.RAG
    :param src: Source node (unused but required)
    :param dst: Destination node (merged already)
    :param neighbor: Neighbor node
    """

    # Convert weight to dictionary form
    n_weight = edge_weight(graph_rag, list([dst, neighbor]), segment_data)
    return {'weight': n_weight}


# Load in an image
segment_data = imread('https://i.stack.imgur.com/nkQpj.png')

# Run watershed segmentation with empty border regions
labels = segmentation.watershed(-segment_data, connectivity=2, watershed_line=True)

# Expand labels by 1 to join them. This will be used to compute the RAG
join_labels = segmentation.expand_labels(labels, distance=1)

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

# Compute the initial RAG weights
for edge in labelRAG.edges:
    weight = edge_weight(labelRAG, edge, segment_data)
    labelRAG.edges[edge]["weight"] = weight

# Show region boundaries with holes
marked_bounds = segmentation.mark_boundaries(segment_data, labels, color=(1, 0, 1), outline_color=None, mode='outer',
                                             background_label=0)

# # TEST: Display the region properties table
# props_table_original = measure.regionprops_table(labels, segment_data, properties=('centroid_weighted',
#                                                  'bbox',
#                                                  'intensity_min',
#                                                  'intensity_max'))
# # Display region properties
# print('Region Props:')
# print(pd.DataFrame(props_table_original).to_string())
# print(' ')

# Compute Region Properties
props_original = measure.regionprops(labels, segment_data)

# Display RAG adjacency graph
print('Initial Adjacency Graph')
for n in labelRAG:
    print("    Label " + str(n) + " connects to: " + str(list(labelRAG.neighbors(n))))
print(' ')

# Display unique edges
print('Initial Edge Weights')
for u, v, weight in labelRAG.edges.data("weight"):
    print(str(tuple([u, v])) + " weight = " + str(weight))

#  Plot Initial Segmentation Results
plots = {'Original': segment_data, 'Watershed Labels': labels, 'Joined Labels': join_labels, 'Overlay': marked_bounds}
fig, ax = plt.subplots(1, len(plots))
for n, (title, img_plt) in enumerate(plots.items()):
    cmap = plt.cm.gnuplot if n == len(plots) - 1 else plt.cm.gray
    if n == 1 or n == 2:
        cmap = 'jet'

    ax[n].imshow(img_plt, cmap=cmap)
    ax[n].axis('off')
    ax[n].set_title(title)

plt.tight_layout()
plt.suptitle('Initial Watershed Segmentation')

# Plot pre-merged network (plot post-merge later)
plt.figure(2)
plt.subplot(121)
image_label_overlay = color.label2rgb(labels, image=segment_data, bg_label=0)
plt.imshow(image_label_overlay)
for region in props_original:
    y0, x0 = region.centroid_weighted
    plt.plot(x0, y0, 'ok', markersize=10, markerfacecolor="k", markeredgecolor="w")
    plt.text(x0-1, y0-1, region.label, size=20, color="k", bbox=dict(facecolor='white', edgecolor='none', alpha=0.5))
    minr, minc, maxr, maxc = region.bbox
    bx = (minc, maxc, maxc, minc, minc)
    by = (minr, minr, maxr, maxr, minr)
    plt.plot(bx, by, '-b', linewidth=2.5)

    for n in list(labelRAG.neighbors(region.label)):
        yn, xn = props_original[n - 1].centroid_weighted
        plt.plot([x0, xn], [y0, yn], 'r-')

# Plot the regions and borders of a region
border = labelRAG.nodes[2]["border"]
region = labelRAG.nodes[2]["region"]

for r in region:
    plt.plot(r[1], r[0], 'rx')

for b in border:
    plt.plot(b[1], b[0], 'c.')

plt.tight_layout()

# Perform hierarchical merging
future.graph.merge_hierarchical(labels, labelRAG, 0, False, True, weight_func=merge_weight, merge_func=merge_regions)

# Create a new label graph
labels_merged = np.zeros(labels.shape, dtype=int)
for n in labelRAG:
    for p in labelRAG.nodes[n]["region"]:
        labels_merged[p] = n
    for b in labelRAG.nodes[n]["border"]:
        labels_merged[b] = 0
plt.title('Original')

# Compute region properties for plotting
props_all_merged = measure.regionprops(labels_merged, segment_data)

# Plot post-merged network
plt.subplot(122)
image_label_overlay = color.label2rgb(labels_merged, image=segment_data, bg_label=0)
plt.imshow(image_label_overlay)
for region in props_all_merged:
    y0, x0 = region.centroid_weighted
    plt.plot(x0, y0, 'ok', markersize=10, markerfacecolor="k", markeredgecolor="w")
    plt.text(x0-1, y0-1, region.label, size=20, color="k", bbox=dict(facecolor='white', edgecolor='none', alpha=0.5))
    minr, minc, maxr, maxc = region.bbox
    bx = (minc, maxc, maxc, minc, minc)
    by = (minr, minr, maxr, maxr, minr)
    plt.plot(bx, by, '-b', linewidth=2.5)

    for n in list(labelRAG.neighbors(region.label)):
        for p in props_all_merged:
            if p["label"] == n:
                yn, xn = p.centroid_weighted
                plt.plot([x0, xn], [y0, yn], 'r-')

# Plot the regions and borders of a region
border = labelRAG.nodes[4]["border"]
region = labelRAG.nodes[4]["region"]

for r in region:
    plt.plot(r[1], r[0], 'rx')

for b in border:
    plt.plot(b[1], b[0], 'c.')

# plt.tight_layout()
plt.title('Merged')
plt.suptitle('Merging Process', size=25)


# Display RAG adjacency graph
print('Final Adjacency Graph')
for n in labelRAG:
    print("    Label " + str(n) + " connects to: " + str(list(labelRAG.neighbors(n))))
print(' ')

# Display unique edges
print('Final Edge Weights')
for u, v, weight in labelRAG.edges.data("weight"):
    print(str(tuple([u, v])) + " weight = " + str(weight))

# Show Figures
plt.show()
