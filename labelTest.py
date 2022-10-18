from skimage.io import imread, imshow
from skimage.filters import gaussian, threshold_otsu
from skimage import measure, segmentation, color, data
import matplotlib.pyplot as plt
import pandas as pd
from skimage import future


original = imread('https://i.stack.imgur.com/nkQpj.png')
blurred = gaussian(original, sigma=.8)
binary = blurred > threshold_otsu(original)

labels = segmentation.watershed(-original, connectivity=2, watershed_line=True)
join_labels = segmentation.watershed(-original, connectivity=2, watershed_line=False)

bounds = segmentation.find_boundaries(labels, connectivity=2, mode='subpixel', background=0)
mbounds = segmentation.mark_boundaries(original, labels,  color=(1, 0, 1), outline_color=None, mode='outer',
                                       background_label=0)

plots = {'Original': original, 'Watershed Labels': labels, 'Joined Labels': join_labels, 'Overlay': mbounds}
fig, ax = plt.subplots(1, len(plots))
for n, (title, img) in enumerate(plots.items()):
    cmap = plt.cm.gnuplot if n == len(plots) - 1 else plt.cm.gray
    if n == 1 or n == 2:
        cmap = 'jet'

    ax[n].imshow(img, cmap=cmap)
    ax[n].axis('off')
    ax[n].set_title(title)
fig.tight_layout()
plt.show()

props = measure.regionprops_table(join_labels, original,
                                  properties=['label', 'centroid_weighted', 'bbox', 'intensity_max', 'intensity_min'])

print('Region Props"')
print(pd.DataFrame(props).to_string())
print(' ')
rag = future.graph.RAG(join_labels, connectivity=2)
print(' ')
print('Adjacency Graph')
for i in range(join_labels.min(), join_labels.max()):
    print("    Label " + str(i) + " connects to: " + str(list(rag.neighbors(i))))




# CUSTOM RAG WEIGHTS
# def rag(image, labels):
#    #initialize the RAG
#    graph = RAG(labels, connectivity=2)
#
#    #lets say we want for each node on the graph a label, a pixel count and a total color
#    for n in graph:
#        graph.node[n].update({'labels': [n],'pixel count': 0,
#                              'total color': np.array([0, 0, 0],
#                              dtype=np.double)})
#    #give them values
#    for index in np.ndindex(labels.shape):
#        current = labels[index]
#        graph.node[current]['pixel count'] += 1
#        graph.node[current]['total color'] += image[index]
#
#    #calculate your own weights here
#    for x, y, d in graph.edges(data=True):
#        my_weight = "do whatever"
#        d['weight'] = my_weight
#
#    return graph
