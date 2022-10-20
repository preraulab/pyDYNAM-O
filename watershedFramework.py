from skimage.io import imread, imshow
from skimage import measure, segmentation, future, color, data, morphology
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import interpolate as it


def edge_weight(graph_rag, graph_edge, graph_data):
    """
    Computes the edge weight between two regions

    :type graph_data: numpy.ndarray
    :type graph_edge: tuple
    :type graph_rag: skimage.future.graph.rag.RAG
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

    w = np.max([w_ij, w_ji])
    
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
    graph_rag.nodes[dst]["region"] = graph_rag.nodes[dst]["region"].union(graph_rag.nodes[src]["region"])
    graph_rag.nodes[dst]["border"] = graph_rag.nodes[dst]["border"].symmetric_difference(graph_rag.nodes[src]["border"])


def merge_weight(graph_rag, src, dst, n):
    n_weight = edge_weight(graph_rag, list([dst, n]), img)
    return {'weight': n_weight}


# Load in an image
img = imread('https://i.stack.imgur.com/nkQpj.png')

# Run watershed segmentation with empty border regions
labels = segmentation.watershed(-img, connectivity=2, watershed_line=True)

# Expand labels by 1 to join them
join_labels = segmentation.expand_labels(labels, distance=1)

# Create a region adjacency graph based
rag = future.graph.RAG(join_labels, connectivity=2)
for n in rag.nodes():
    rag.nodes[n]['labels'] = [n]

# Get all border and region pixels
for n in rag:
    lregion = (labels == n)
    bx, by = np.where(morphology.dilation(lregion, np.ones([3, 3])) & (labels == 0))
    border = set(zip(bx, by))
    rx, ry = np.where(lregion)
    region = set(zip(rx, ry))
    for b in border:
        region.add(b)
    rag.nodes[n]["border"] = border
    rag.nodes[n]["region"] = region

# Compute Sample Weight
for edge in rag.edges:
    weight = edge_weight(rag, edge, img)
    rag.edges[edge]["weight"] = weight

# Show region boundaries with holes
mbounds = segmentation.mark_boundaries(img, labels, color=(1, 0, 1), outline_color=None, mode='outer',
                                       background_label=0)

# Compute the region properties table
props = measure.regionprops_table(labels, img, properties=('centroid_weighted',
                                                 'bbox',
                                                 'intensity_min',
                                                 'intensity_max'))
# Display region properties
print('Region Props:')
print(pd.DataFrame(props).to_string())
print(' ')

props_all = measure.regionprops(labels, img)


# Display RAG adjacency graph
print('Adjacency Graph')
for n in rag:
    print("    Label " + str(n) + " connects to: " + str(list(rag.neighbors(n))))
print(' ')

# Display unique edges
print('Edge Weights')
for u, v, weight in rag.edges.data("weight"):
    print(str(tuple([u, v])) + " weight = " + str(weight))

# Plot results
plots = {'Original': img, 'Watershed Labels': labels, 'Joined Labels': join_labels, 'Overlay': mbounds}
fig, ax = plt.subplots(1, len(plots))
for n, (title, img_plt) in enumerate(plots.items()):
    cmap = plt.cm.gnuplot if n == len(plots) - 1 else plt.cm.gray
    if n == 1 or n == 2:
        cmap = 'jet'

    ax[n].imshow(img_plt, cmap=cmap)
    ax[n].axis('off')
    ax[n].set_title(title)

plt.tight_layout()

plt.figure(2)
plt.subplot(121)
image_label_overlay = color.label2rgb(labels, image=img, bg_label=0)
plt.imshow(image_label_overlay)
for region in props_all:
    y0, x0 = region.centroid_weighted
    plt.plot(x0, y0, '.m', markersize=15)
    minr, minc, maxr, maxc = region.bbox
    bx = (minc, maxc, maxc, minc, minc)
    by = (minr, minr, maxr, maxr, minr)
    plt.plot(bx, by, '-b', linewidth=2.5)

    for n in list(rag.neighbors(region.label)):
        yn, xn = props_all[n-1].centroid_weighted
        plt.plot([x0, xn], [y0, yn], 'r-')

# Plot the regions and borders of a region
region_num = list(rag.nodes)[0]
border = rag.nodes[region_num]["border"]
region = rag.nodes[region_num]["region"]

for r in region:
    plt.plot(r[1], r[0], 'rx')

for b in border:
    plt.plot(b[1], b[0], 'c.')

plt.tight_layout()

future.graph.merge_hierarchical(labels, rag, -30000, False, True, weight_func=merge_weight, merge_func=merge_regions)

label_merged = np.zeros(labels.shape, dtype=int)
for n in rag:
    for p in rag.nodes[n]["region"]:
        label_merged[p] = n
    for b in rag.nodes[n]["border"]:
        label_merged[b] = 0
plt.title('Original')

props_all_merged = measure.regionprops(label_merged, img)

plt.subplot(122)
image_label_overlay = color.label2rgb(label_merged, image=img, bg_label=0)
plt.imshow(image_label_overlay)
for region in props_all_merged:
    y0, x0 = region.centroid_weighted
    plt.plot(x0, y0, '.m', markersize=15)
    minr, minc, maxr, maxc = region.bbox
    bx = (minc, maxc, maxc, minc, minc)
    by = (minr, minr, maxr, maxr, minr)
    plt.plot(bx, by, '-b', linewidth=2.5)

    for n in list(rag.neighbors(region.label)):
        for p in props_all_merged:
            if p["label"] == n:
                yn, xn = p.centroid_weighted
                plt.plot([x0, xn], [y0, yn], 'r-')

# Plot the regions and borders of a region
region_num = list(rag.nodes)[0]
border = rag.nodes[region_num]["border"]
region = rag.nodes[region_num]["region"]

for r in region:
    plt.plot(r[1], r[0], 'rx')

for b in border:
    plt.plot(b[1], b[0], 'c.')

plt.tight_layout()
plt.title('Merged')
plt.show()
