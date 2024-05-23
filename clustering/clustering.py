from PIL import Image
import tensorflow as tf
import numpy as np
import collections
from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from matplotlib_venn import venn3

def gen_dummy_data(num_samples, num_features):
    """
    Generates a set of dummy data for k-means to run on
    :param num_samples: m, number of samples to generate
    :param num_features: c, number of components
    :return: pca_vector: dummy np array of size (m, c) where m is number of images and c is number of principal components
    :return: id_vector: np array of size (m, ) where the i'th entry of pca_vector corresponds to image id_vector[i]
    """
    pca_vector, true_labels = make_blobs(
   n_samples=num_samples,
   n_features=num_features,
   centers=2,
   cluster_std=.6
    )
    id_vector = np.arange(num_samples)
    return pca_vector, id_vector



def kmeans(pca_vector, id_vector, num_clusters):
    """
        Takes in principal components of image vectors paired with an ID vector and returns a list of the clusters found after k-means

        :param pca_vector: np array of size (m, c) where m is number of images and c is number of principal components
        :param id_vector: np array of size (m, ) where the i'th entry of pca_vector corresponds to image id_vector[i]
        :param num_clusters: number of clusters to split the data into
        
    """
    m = pca_vector.shape[0]
    # c = pca_vector.shape[1]
    assert m == id_vector.shape[0]
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto").fit(pca_vector)
    # labels is an np array of size (m, ) with each value referring to the number of the cluster at each index
    labels = kmeans.labels_
    cluster_list = []
    for i in range(num_clusters):
        indices = np.where(labels == i)[0]
        indices = id_vector[indices]
        cluster_list.append(indices)
    return cluster_list

def plot_venn(low_pca, low_id, med_pca, med_id, high_pca, high_id, num_clusters, lN = None, mN = None, hN = None, incl_PCA = False):
    low_kmeans = KMeansModel(low_pca, low_id, num_clusters)
    med_kmeans = KMeansModel(med_pca, med_id, num_clusters)
    high_kmeans = KMeansModel(high_pca, high_id, num_clusters)
    if incl_PCA:
        low_kmeans = KMeansModel(low_pca, low_id, num_clusters)
        med_kmeans = KMeansModel(med_pca, med_id, num_clusters)
        high_kmeans = KMeansModel(high_pca, high_id, num_clusters)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].set_title('Low Spuriosity PCA')
        axes[0].scatter(low_kmeans.X[:, 0], low_kmeans.X[:, 1], c=low_kmeans.y_kmeans, s=50, cmap='viridis')
        for (k,(i, j)) in enumerate(zip(low_kmeans.X[:, 0], low_kmeans.X[:, 1])):
            name = lN[k]
            axes[0].text(i, j, name)
        axes[1].set_title('Medium Spuriosity PCA')
        axes[1].scatter(med_kmeans.X[:, 0], med_kmeans.X[:, 1], c=med_kmeans.y_kmeans, s=50, cmap='viridis')
        for (k,(i, j)) in enumerate(zip(med_kmeans.X[:, 0], med_kmeans.X[:, 1])):
            name = mN[k]
            axes[1].text(i, j, name)
        axes[2].set_title('High Spuriosity PCA')
        axes[2].scatter(high_kmeans.X[:, 0], high_kmeans.X[:, 1], c=high_kmeans.y_kmeans, s=50, cmap='viridis')
        for (k,(i, j)) in enumerate(zip(high_kmeans.X[:, 0], high_kmeans.X[:, 1])):
            name = hN[k]
            axes[2].text(i, j, name)
        for i in range(3):
            axes[i].set_xlabel('Principal Component 1')
            axes[i].set_ylabel('Principal Component 2')
            axes[i].grid(True)
        plt.tight_layout()
        plt.show()
    
    low_set = set()
    med_set = set()
    high_set = set()

    if incl_PCA:
        low_set = set(low_kmeans.return_cluster_list()[0].flatten())
        print(f"low set 1 is {low_set}")
        med_set = set(med_kmeans.return_cluster_list()[0].flatten())
        print(f"med set 1 is {med_set}")
        high_set = set(high_kmeans.return_cluster_list()[0].flatten())
        print(f"high set 1 is {high_set}")
    else:
        low_set = set(low_kmeans.return_cluster_list()[0].flatten())
        low_names = [lN[x] for x in low_set]
        print(f"low set 1 is {low_names}")
        low_set = set(low_kmeans.return_cluster_list()[1].flatten())
        low_names = [lN[x] for x in low_set]
        print(f"low set 2 is {low_names}")

        med_set = set(med_kmeans.return_cluster_list()[0].flatten())
        med_names = [mN[x] for x in med_set]
        print(f"med set 1 is {med_names}")
        med_set = set(med_kmeans.return_cluster_list()[1].flatten())
        med_names = [mN[x] for x in med_set]
        print(f"med set 2 is {med_names}")

        high_set = set(high_kmeans.return_cluster_list()[0].flatten())
        high_names = [hN[x] for x in high_set]
        print(f"high set 1 is {high_names}")
        high_set = set(high_kmeans.return_cluster_list()[1].flatten())
        high_names = [hN[x] for x in high_set]
        print(f"high set 2 is {high_names}")

    venn3([low_set, med_set, high_set], ('low spuriosity', 'medium spuriosity', 'high spuriosity'))
    plt.show()
        
class KMeansModel:
    def __init__(self, pca_vector, id_vector, num_clusters):
        self.id_vector = id_vector
        self.X = pca_vector
        self.num_clusters = num_clusters
        self.m = pca_vector.shape[0]
        self.c = pca_vector.shape[1]
        assert self.m == id_vector.shape[0]
        kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto")
        kmeans.fit(pca_vector)
        self.y_kmeans = kmeans.predict(self.X)
        self.labels = kmeans.labels_
        self.cluster_centers = kmeans.cluster_centers_

        
    def plot_kmeans_clusters_2d(self):
        '''
        Plots the kmeans clusters, only works on 2D PCA results
        '''
        y_kmeans = self.y_kmeans
        plt.scatter(self.X[:, 0], self.X[:, 1], c=y_kmeans, s=50, cmap='viridis')
        centers = self.cluster_centers
        plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
        plt.show()

    def return_cluster_list(self):
        '''
        Returns a cluster list
        :return: cluster_list: list of clusters resulting from k-means, with each cluster consisting of the ID's of relevant images
        '''
        # labels is an np array of size (m, ) with each value referring to the number of the cluster at each index
        labels = self.labels
        cluster_list = []
        for i in range(self.num_clusters):
            indices = np.where(labels == i)[0]
            indices = self.id_vector[indices]
            cluster_list.append(indices)
        return cluster_list
        

    def train(self, train_captions, train_image_features, padding_index, batch_size=30):
        """
        Runs through one epoch - all training examples.

        :param model: the initialized model to use for forward and backward pass
        :param train_captions: train data captions (all data for training) 
        :param train_images: train image features (all data for training) 
        :param padding_index: the padding index, the id of *PAD* token. This integer is used when masking padding labels.
        :return: None
        """
        pass



if __name__ == '__main__':
    m = 200
    c = 2
    low_pca, low_id = gen_dummy_data(m, c)
    med_pca, med_id = gen_dummy_data(m, c)
    high_pca, high_id = gen_dummy_data(m, c)
    plot_venn(low_pca, low_id, med_pca, med_id, high_pca, high_id, 2)