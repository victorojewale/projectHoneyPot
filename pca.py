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

    row1 = data1.iloc[class_index][2:].values #assumes class_index and file_name are first 2 columns
    row2 = data2.iloc[class_index][2:].values
    row3 = data3.iloc[class_index][2:].values
    
    principal_components1 = pca_helper(row1)
    principal_components2 = pca_helper(row2)
    principal_components3 = pca_helper(row3)

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



def pca_helper(feature_row):
    #standardize data
    feature_row = np.array(feature_row)
    print(feature_row.shape)
    feature_row = feature_row.reshape(1,-1)
    print(feature_row.shape)
    feature_row = StandardScaler().fit_transform(feature_row)
    #perform pca
    pca = PCA(n_components=2,svd_solver="auto")
    principal_components = pca.fit_transform(feature_row)
    return principal_components

pca_by_class("test1.csv","test2.csv","test3.csv",2)