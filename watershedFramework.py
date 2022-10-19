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



# Load in an image
img = imread('https://i.stack.imgur.com/nkQpj.png')

# Run watershed segmentation with empty border regions
labels = segmentation.watershed(-img, connectivity=2, watershed_line=True)

# Expand labels by 1 to join them
join_labels = segmentation.expand_labels(labels, distance=1)

# Create a region adjacency graph based
rag = future.graph.RAG(join_labels, connectivity=2)

# Get all border and region pixels
for n in rag:
    lregion = labels == n
    bx, by = np.where(morphology.dilation(lregion, np.ones([3, 3])) & (labels == 0))
    border = set(zip(bx, by))
    rx, ry = np.where(lregion)
    region = set(zip(rx, ry))
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
region_num = 3
border = list(rag.nodes[region_num]["border"])
region = list(rag.nodes[region_num]["region"])
plt.plot(border[1], border[0], 'c.')
plt.plot(region[1], region[0], 'rx')
plt.tight_layout()
plt.show()

