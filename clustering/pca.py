import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
    id1 = data1[data1['Input.class_index'] == class_index].index.to_numpy()
    id2 = data2[data2['Input.class_index'] == class_index].index.to_numpy()
    id3 = data3[data3['Input.class_index'] == class_index].index.to_numpy()

    class_images1 = data1[data1['Input.class_index'] == class_index].iloc[:,2:]
    class_images2 = data2[data2['Input.class_index'] == class_index].iloc[:,2:]
    class_images3 = data3[data3['Input.class_index'] == class_index].iloc[:,2:]
    
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
    return principal_components1, principal_components2, principal_components3, id1, id2, id3

def flt_by_class(csv_filepath1, csv_filepath2, csv_filepath3, class_index):
    data1 = pd.read_csv(csv_filepath1)
    data2 = pd.read_csv(csv_filepath2)
    data3 = pd.read_csv(csv_filepath3)

    # class_indices1 = data1["Input.class_index"]
    # file_names1 = data1["Image.file_name"]
    id1 = data1[data1['Input.class_index'] == class_index].index.to_numpy()
    id2 = data2[data2['Input.class_index'] == class_index].index.to_numpy()
    id3 = data3[data3['Input.class_index'] == class_index].index.to_numpy()

    class_images1 = data1[data1['Input.class_index'] == class_index].iloc[:,2:]
    class_images2 = data2[data2['Input.class_index'] == class_index].iloc[:,2:]
    class_images3 = data3[data3['Input.class_index'] == class_index].iloc[:,2:]

    return class_images1, class_images2, class_images3, id1, id2, id3



def pca_helper(images_features):
    #standardize data
    images_features = StandardScaler().fit_transform(images_features)
    #perform pca
    pca = PCA(n_components=2,svd_solver="auto")
    principal_components = pca.fit_transform(images_features)
    '''
    var = pca.explained_variance_ratio_
    plt.bar(list(range(var.shape[0])),var)
    feature = range(pca.n_components_)
    plt.xlabel('PCA features')
    plt.ylabel('variance %')
    plt.xticks(feature)
    '''
    return principal_components

