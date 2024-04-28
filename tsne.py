import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

"""tsne guidelines: https://www.kaggle.com/code/agsam23/pca-vs-t-sne"""

"""
csv_filepath1 -> low spuriosity dataset
csv_filepath2 -> medium spuriosity dataset
csv_filepath3 -> high spuriosity dataset
"""
def pca_by_class(csv_filepath1, csv_filepath2, csv_filepath3, class_index):
    data1 = pd.read_csv(csv_filepath1)
    data2 = pd.read_csv(csv_filepath2)
    data3 = pd.read_csv(csv_filepath3)

    # class_indices1 = data1["Input.class_index"]
    # file_names1 = data1["Image.file_name"]
    class_images1 = data1[data1['Input.class_index'] == class_index]
    class_images2 = data2[data2['Input.class_index'] == class_index]
    class_images3 = data3[data3['Input.class_index'] == class_index]
    
    principal_components1 = pca_helper(class_images1)
    principal_components2 = pca_helper(class_images2)
    principal_components3 = pca_helper(class_images3)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].set_title('Low Spuriosity PCA')
    axes[0].scatter(principal_components1[:,0], principal_components1[:, 1])
    axes[1].set_title('Medium Spuriosity PCA')
    axes[1].scatter(principal_components2[:, 0], principal_components2[:, 1])
    axes[2].set_title('High Spuriosity PCA')
    axes[2].scatter(principal_components3[:, 0], principal_components3[:, 1])
    for i in range(3):
        axes[i].set_xlabel('Principal Component 1')
        axes[i].set_ylabel('Principal Component 2')
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.show()



def pca_helper(images_features):
    #standardize data
    images_features = StandardScaler().fit_transform(images_features)
    #perform pca
    pca = PCA(n_components=2,svd_solver="auto")
    principal_components = pca.fit_transform(images_features)
    return principal_components

