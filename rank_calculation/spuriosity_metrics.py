####
# Author - Vipul Sharma
# Function - feature activations' z-score based spuriosity calculation
# input is the feature activations given by inference.py 
# output is spuriosity value for each image within class
'''
For each class c:
    for all images x: 
        find activation r_i(x) of the i^{th} robust feature
        S(c): set of spurious features for class c
        spuriosity of image x for class c: 
        1/len(S(C)) * sum((r_i(x) - mean_ic)/std_ic)
'''
####

import numpy as np
import pandas as pd
import torch
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

def calc_spurious_features_by_class(aggregated_human_labels): 
    '''
    aggregated_human_labels is the csv file spurious_imagenet.csv in data_annotations. 
    This will return a dictionary with key as class index (0-999) and value as the feature index of spurious features in penultimate layer.
    '''
    class_ftr_idx = aggregated_human_labels[aggregated_human_labels['Answer.main']==1].groupby('Input.class_index')['Input.feature_index'].agg(list)
    class_ftr_idx = dict(class_ftr_idx)
    return class_ftr_idx

def count_rows(filename):
    with open(filename, 'r') as file:
        num_rows = sum(1 for line in file)
    return num_rows

def cache_data(cache_path, data_to_cache):
    '''
    schema of output cache: 
    Input.class_index,image_name,spuriosity
    '''
    data_to_cache.to_csv(cache_path, mode='a', header=not os.path.exists(cache_path), index=False)

def calculate_spuriosity_per_class(ftr_activations_path, output_path, spurious_features_by_class):
    '''
    spurious_features_by_class: a dictionary of class label (0-999) and spurious feature index (0-2047)    
    '''
    feature_activations = pd.read_csv(ftr_activations_path)
    
    for class_idx in spurious_features_by_class: 
        spurious_features_idx = np.array(spurious_features_by_class[class_idx])
        class_data = feature_activations[feature_activations['Input.class_index']==class_idx]
        feature_activations_of_class = class_data.iloc[:, 2:].iloc[:,spurious_features_idx]
        zscore_activations = (feature_activations_of_class - feature_activations_of_class.mean())/feature_activations_of_class.std()
        spuriosity = zscore_activations.mean(axis=1)
        spuriosity_df = pd.DataFrame({
            'Input.class_index': class_data['Input.class_index'],
            'image_name': class_data['Image.file_name'],
            'spuriosity': spuriosity
        })
        cache_data(output_path, spuriosity_df)
        del spuriosity_df, spuriosity, zscore_activations
    return count_rows(output_path)

def bin_by_spuriosity(spuriosity_path, output_path, spurious_features_by_class): 
    '''
    bin the data per class by 0-25 percentile as low spurious, 25-75 as medium 
    and 75-100 percentile spuriosity as high spurious
    spurious_features_by_class: dict, key class, value spurious feature index
    '''
    wordnet_data = pd.read_csv('../data_annotations/imagenet_class_metadata.csv')
    sp_vals_data = pd.read_csv(spuriosity_path)
    sp_vals_data.merge(wordnet_data[['Input.class_index', 'Input.wordnet_id']], how='left', on='Input.class_index')
    for class_idx in spurious_features_by_class: 
        spuriosity_val_class = sp_vals_data[sp_vals_data['Input.class_index'] == class_idx]
        percentiles = np.percentile(spuriosity_val_class['spuriosity'], [25, 75]) # percentils[0] 25%, percentiles[1] 75%
        result_df = spuriosity_val_class[['Input.wordnet_id', 'Input.class_index', 'image_name']]
        result_df['bin_type'] = spuriosity_val_class['spuriosity'].apply(lambda x: 0 if x<=percentiles[0] else x)
        result_df['bin_type'] = spuriosity_val_class['spuriosity'].apply(lambda x: 1 if (x>percentiles[0] & x<=percentiles[1]) else x)
        result_df['bin_type'] = spuriosity_val_class['spuriosity'].apply(lambda x: 2 if x>percentiles[1] else x)
        
        cache_data(output_path, result_df)
        del result_df
    return count_rows(output_path)

if __name__ == '__main__': 
    aggregated_data_path = '../data_annotations/spurious_imagenet_classes.csv'
    aggregated_human_labels = pd.read_csv(aggregated_data_path)
    spurious_features_by_class = calc_spurious_features_by_class(aggregated_human_labels)
    #sanity check: for 5 random classes, get the spurious feature indices
    print([spurious_features_by_class[i] for i in list(spurious_features_by_class.keys())[:5]])
    print('output check', len(spurious_features_by_class), ', should be 357 for imagenet.')