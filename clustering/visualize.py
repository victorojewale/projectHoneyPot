from PIL import Image
import tensorflow as tf
import numpy as np
import collections
from tqdm import tqdm
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from matplotlib_venn import venn3_circles

from pca import pca_by_class, flt_by_class
from tsne import tsne_by_class
from clustering import KMeansModel, kmeans, plot_venn

# TODO: edit class to be the class you are looking to analyze
class_to_plot = 701
# TODO: choose whether or not to include PCA in the process
INCLUDE_PCA = True

# TODO: Put in links to the set of .csv's to analyze
highC = 'model1_train_701_spurious.csv'
medC = 'model2_train_701_spurious.csv'
lowC = 'model3_train_701_spurious.csv'


high_C = pd.read_csv(highC).iloc[:,1].to_numpy()
med_C = pd.read_csv(medC).iloc[:,1].to_numpy()
low_C = pd.read_csv(lowC).iloc[:,1].to_numpy()

if INCLUDE_PCA:
    high_C = pd.read_csv(highC)
    high_C = high_C[high_C['Input.class_index'] == class_to_plot].iloc[:,1].to_numpy()
    med_C = pd.read_csv(medC)
    med_C = med_C[med_C['Input.class_index'] == class_to_plot].iloc[:,1].to_numpy()
    low_C = pd.read_csv(lowC)
    low_C = low_C[low_C['Input.class_index'] == class_to_plot].iloc[:,1].to_numpy()


def splitter(word):
    dot = word.index('.')
    under = word.index('_')
    return word[under + 5:dot]
high_C = np.array([splitter(xi) for xi in high_C])
med_C = np.array([splitter(xi) for xi in med_C])
low_C = np.array([splitter(xi) for xi in low_C])


if INCLUDE_PCA:
    pc1C, pc2C, pc3C, id1, id2, id3 = pca_by_class(lowC, medC, highC, class_to_plot)
    plot_venn(pc1C, id1, pc2C, id2, pc3C, id3, 2, lN = low_C, mN = med_C, hN = high_C, incl_PCA=True)
else:
    pc1C, pc2C, pc3C, id1, id2, id3 = flt_by_class(lowC, medC, highC, class_to_plot)
    plot_venn(pc1C, id1, pc2C, id2, pc3C, id3, 2, lN = low_C, mN = med_C, hN = high_C, incl_PCA=False)
print(f"low spurious has {id1.shape[0]}, med has {id2.shape[0]}, high has {id3.shape[0]}")

