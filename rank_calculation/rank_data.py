###
# Author: Vipul Sharma
# Function: calculate feature activations and then spuriosity of each image based on it and bin the data into low,med and high spuriosity
###

# solve the problem of relative imports in python
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from compute_feature_activations import * 
from spuriosity_metrics import * 
from configs.config import Config

if __name__ == '__main__': 
    config = Config()

    #get the spurious features by class
    aggregated_data_path = '../data_annotations/spurious_imagenet_classes.csv'
    aggregated_human_labels = pd.read_csv(aggregated_data_path)
    spurious_features_by_class = calc_spurious_features_by_class(aggregated_human_labels)
    print('output check', len(spurious_features_by_class), ', should be 357 for imagenet.')

    #validation spuriosity calcuation
    val_raw_feature_act_path = '../feature_activations_data/robust_resnet_50_imagenet_complete/feature_activations_valid_resnet50_.csv'
    val_spuriosity_path = '../data_annotations/validation_imagenet_spuriosity.csv'
    rows_processed = calculate_spuriosity_per_class(val_raw_feature_act_path, val_spuriosity_path, spurious_features_by_class)
    print("Generated spuriosity for validation data, rows produced", rows_processed)
    if rows_processed: 
        binned_img_idx_valid = bin_by_spuriosity(val_spuriosity_path, config.bin_file_path_val, spurious_features_by_class)
        print("Processed binning of", binned_img_idx_valid, "rows of validation spuriosity data.")
    
    #train spuriosity calcuation
    train_raw_feature_act_path = '../feature_activations_data/robust_resnet_50_imagenet_complete/feature_activations_train_resnet50_.csv'
    train_spuriosity_path = '../data_annotations/train_imagenet_spuriosity.csv'
    rows_processed = calculate_spuriosity_per_class(train_raw_feature_act_path, train_spuriosity_path, spurious_features_by_class)
    print("Generated spuriosity for train data, rows produced", rows_processed)
    if rows_processed: 
        binned_img_idx_train = bin_by_spuriosity(train_spuriosity_path, config.bin_file_path_train, spurious_features_by_class)
        print("Processed binning of", binned_img_idx_train, "rows of train spuriosity data.")
    