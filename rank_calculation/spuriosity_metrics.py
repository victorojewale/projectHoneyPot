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
    aggregated_human_labels is the csv file aggregated_imagenet_mturk.csv in data_annotations. 
    This will return a dictionary with key as class index (0-999) and value as the feature index of spurious features in penultimate layer.
    '''
    class_ftr_idx = aggregated_human_labels[aggregated_human_labels['Answer.main']==1].groupby('Input.class_index')['Input.feature_index'].agg(list)
    class_ftr_idx = dict(class_ftr_idx)
    return class_ftr_idx

def calculate_spuriosity_per_class(feature_activations, class_labels, spurious_features_by_class):
    '''
    inputs: 
    feature_activations: a tensor/array of num_of_images, 2048 feature activations of resnet for imagenet trained resnet
    class_labels: a tensor/array of num_of_images, class label (0-999)
    spurious_features_by_class: a dictionary of class label (0-999) and spurious feature index (0-2047)    
    
    output: 
    spuriosity_values_of_data: a tensor/array of num_of_images, 1 (spuriosity value within class). Same order as class_labels.
    '''
    spuriosity_values_of_data = np.zeros((len(class_labels), 1))
    
    for class_idx in spurious_features_by_class: 
        spurious_features_idx = spurious_features_by_class[class_idx]
        image_idx_of_class = np.where(class_labels == class_idx)[0]
        
        feature_activations_of_class = feature_activations[image_idx_of_class]
        mean_feature_activations_of_class = np.mean(feature_activations_of_class, axis=0)
        std_feature_activations_of_class = np.std(feature_activations_of_class, axis=0)
        z_score_feature_activations_of_class = (feature_activations_of_class - mean_feature_activations_of_class)/std_feature_activations_of_class
        
        spuriosity_values_of_class = np.average(z_score_feature_activations_of_class[:, spurious_features_idx], axis=1)
        spuriosity_values_of_class = np.reshape(spuriosity_values_of_class, (-1, 1))

        spuriosity_values_of_data[image_idx_of_class] = spuriosity_values_of_class
        
    return spuriosity_values_of_data

def bin_by_spuriosity(spuriosity_values_of_data, class_labels, spurious_features_by_class): 
    '''
    bin the data per class by 0-25 percentile as low spurious, 25-75 as medium 
    and 75-100 percentile spuriosity as high spurious
    
    spurious_features_by_class: dict, key class, value spurious feature index
    class_labels: array num_images, (0-999) class idx
    spuriosity_values_of_data: array num_images, spuriosity value as calculated within class
    
    output: dict with spurious feature containing class as key and array value (3, idx of images of a class in original data)
    '''
    result = {}
    for class_idx in spurious_features_by_class: 
        image_idx_of_class = np.where(class_labels == class_idx)[0]
        spurious_vals_class = spuriosity_values_of_data[image_idx_of_class]
        percentiles = np.percentile(spurious_vals_class, [25,75])
        low_bin = image_idx_of_class[np.where(spurious_vals_class<=percentiles[0])]
        mid_bin = image_idx_of_class[np.where((spurious_vals_class>percentiles[0]) & (spurious_vals_class<=percentiles[1]))]
        high_bin = image_idx_of_class[np.where(spurious_vals_class>percentiles[1])]
        result[class_idx] = np.array([low_bin, mid_bin, high_bin])
    return result

if __name__ == '__main__': 
    #the file paths work if you run from root and not within the module folder
    aggregated_data_path = 'data_annotations/aggregated_imagenet_mturk.csv'
    aggregated_human_labels = pd.read_csv(aggregated_data_path)
    spurious_features_by_class = calc_spurious_features_by_class(aggregated_human_labels)
    print([spurious_features_by_class[i] for i in list(spurious_features_by_class.keys())[:5]])
    print('output check', len(spurious_features_by_class), ', should be 357 for imagenet.')