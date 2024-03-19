import numpy

import hdbscan

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt


class HDBSCAN:
    """ Finds optimal number of clusters using HDBSCAN.
    """
    def __init__(self, **kwargs):
        """ Initializer for optimal clustering.

            :param kwargs (keyword arguments)
                keyword arguments passed to HDBSCAN.
        """
        self.pca_ = PCA(n_components=2)
        self.scores_ = []
        self.clusterer_ = hdbscan.HDBSCAN(**kwargs)

    def run_clustering(self, data: numpy.ndarray) -> numpy.ndarray:
        """ Runs HDSCAN on provided data.

            :param data (numpy.ndarray)
                2D array with data.
        """
        labels = self.clusterer_.fit_predict(data)

        return labels

    def run_subclustering(self,
                          data: numpy.ndarray, labels: numpy.ndarray,
                          k: int) -> numpy.ndarray:
        """ Performs sub-clustering using K-Means.

            :param data (numpy.ndarray)
                2-D array with data.
            :param labels (numpy.ndarray)
                1-D array with data labels (like the output of
                routing `run_clustering()`)
            :param k (int)
                number of sub-clusters to generate.
        """
        kmeans = KMeans(n_clusters=k)
        new_labels = numpy.zeros_like(labels)
        for label in numpy.unique(labels):
            cluster_mask = labels == label
            cluster_data = data[cluster_mask]
            cluster_labels = kmeans.fit_predict(cluster_data)
            new_labels[cluster_mask] = label * k + cluster_labels
            print(numpy.unique(new_labels))

        return new_labels

    def plot_clusters(self,
                      data: numpy.ndarray, labels: numpy.ndarray):
        """ Plot clusters in 2-D space.
        """
        twod = self.pca_.fit_transform(data)
        clustered = labels > -1
        dmin = min(numpy.min(twod[clustered, 0]),
                    numpy.min(twod[clustered, 1]))
        dmax = max(numpy.max(twod[clustered, 0]),
                    numpy.max(twod[clustered, 1]))
        plt.clf()
        plt.scatter(twod[~clustered, 0],
                    twod[~clustered, 1],
                    color=(0.5, 0.5, 0.5),
                    s=0.2,
                    alpha=0.5)
        plt.scatter(twod[clustered, 0],
                    twod[clustered, 1],
                    c=labels[clustered],
                    s=0.5,
                    alpha=0.5,
                    cmap='rainbow')
        plt.xlim(dmin, dmax)
        plt.ylim(dmin, dmax)
        plt.savefig('clusters.png')


if __name__ == '__main__':

    import time
    from sklearn.datasets import make_blobs

    tic = time.time()
    x, y = make_blobs(1000000, n_features=2, centers=4, cluster_std=0.3)
    data_min = min(numpy.min(x[:, 0]), numpy.min(x[:, 1]))
    data_max = max(numpy.max(x[:, 0]), numpy.max(x[:, 1]))
    toc = time.time()
    print(f'elapsed time generating data: {toc - tic:.4f}')

    tic = time.time()
    clusterer = HDBSCAN(min_samples=10)
    labels_global = clusterer.run_clustering(x)
     # labels_sub = clusterer.run_subclustering(x, labels_global, 4)
    toc = time.time()
    print(f'elapsed time clustering using HDBSCAN: {toc - tic:.4f}')
    '''
    fig, axes = plt.subplots(1, 2)
    # plot with global clusters
    axes[0].scatter(x[:, 0], x[:, 1], alpha=0.6, s=0.6, cmap='rainbow', c=labels_global)
    axes[0].set_xlim(data_min, data_max)
    axes[0].set_ylim(data_min, data_max)
    # plot with sub-clusters
    axes[1].scatter(x[:, 0], x[:, 1], alpha=0.6, s=0.6, cmap='rainbow', c=labels_sub)
    axes[1].set_xlim(data_min, data_max)
    axes[1].set_ylim(data_min, data_max)
    plt.show()
    '''
