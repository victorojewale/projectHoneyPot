import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

"""
csv_filepath1 -> low spuriosity dataset
csv_filepath2 -> medium spuriosity dataset
csv_filepath3 -> high spuriosity dataset
"""
def tsne_by_class(csv_filepath1, csv_filepath2, csv_filepath3, class_index):
    data1 = pd.read_csv(csv_filepath1)
    data2 = pd.read_csv(csv_filepath2)
    data3 = pd.read_csv(csv_filepath3)

    df_labels1 = data1.label
    df_labels2 = data2.label
    df_labels3 = data3.label

    # class_indices1 = data1["Input.class_index"]
    # file_names1 = data1["Image.file_name"]
    class_images1 = data1[data1['Input.class_index'] == class_index].iloc[:,2:]
    class_images2 = data2[data2['Input.class_index'] == class_index].iloc[:,2:]
    class_images3 = data3[data3['Input.class_index'] == class_index].iloc[:,2:]
    
    transformed_data1 = tsne_helper(class_images1)
    transformed_data2 = tsne_helper(class_images2)
    transformed_data3 = tsne_helper(class_images3)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    plot_df1 = pd.DataFrame(np.column_stack((transformed_data1, df_labels1)),
                            columns=['x','y','labels'])
    grid1 = sns.scatterplot(data=plot_df1, x='x', y='y', hue='labels', ax=axes[0])
    grid1.set_title('Low Spuriosity t-SNE')

    plot_df2 = pd.DataFrame(np.column_stack((transformed_data2, df_labels2)),
                            columns=['x','y','labels'])
    grid2 = sns.scatterplot(data=plot_df2, x='x', y='y', hue='labels', ax=axes[1])
    grid2.set_title('Medium Spuriosity t-SNE')

    plot_df3 = pd.DataFrame(np.column_stack((transformed_data3, df_labels3)),
                            columns=['x','y','labels'])
    grid3 = sns.scatterplot(data=plot_df3, x='x', y='y', hue='labels', ax=axes[2])
    grid3.set_title('High Spuriosity t-SNE')

    plt.tight_layout()
    plt.show()


def tsne_helper(images_features):
    #standardize data
    images_features = StandardScaler().fit_transform(images_features)
    #perform tsne
    tsne = TSNE(n_components=2)
    transformed_data = tsne.fit_transform(images_features)
    return transformed_data

