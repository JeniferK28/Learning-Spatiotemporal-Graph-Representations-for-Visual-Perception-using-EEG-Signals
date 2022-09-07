from sklearn.manifold import TSNE
from time import time
import numpy as np
from sklearn.manifold import MDS
from scipy.cluster.hierarchy import dendrogram
from matplotlib import pyplot as plt
from sklearn.manifold import SpectralEmbedding
from mpl_toolkits.mplot3d import Axes3D


def tsne(x,y):
    t0 = time()
    #TSNE
    pt= TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300).fit_transform(x)

    #Spectral Embedding
    #pt = SpectralEmbedding(n_components=3).fit_transform(x)

    #MPI DB
    category_to_color = {0: 'blue', 1: 'red'}
    category_to_label = {0: 'Animals', 1: 'Tools'}

    #SU DB
    #category_to_color = {0: 'blue', 1: 'red', 2: 'green', 3: 'cyan', 4: 'orange', 5: 'magenta'}
    #category_to_label = {0: 'HB', 1: 'HF', 2: 'AB', 3: 'AF', 4: 'FV', 5: 'IO'}

    # plot each category with a distinct label
    fig, ax = plt.subplots(1, 1)

    for category, color in category_to_color.items():
        mask = y == category
        ax.plot(pt[mask, 0], pt[mask, 1], 'o',
                color=color, label=category_to_label[category])

    ax.legend(loc='best')
    plt.show()

    # 3-d plots
    ax = plt.figure(figsize=(16, 10)).gca(projection='3d')
    for category, color in category_to_color.items():
        mask = y == category
        ax.plot(pt[mask, 0], pt[mask, 1], pt[mask, 2], 'o',
                color=color, label=category_to_label[category])

    ax.legend(loc='best')
    plt.show()
    plt.show()

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)



