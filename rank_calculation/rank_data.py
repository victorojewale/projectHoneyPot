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

if __name__ == '__main__': 
    #get the feature activations
    model_path = '../models/robust_resnet50.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    architecture = 'resnet50'
    encoder = get_encoder(model_path, device, architecture)
    #setup_data_loaders(split='val', shuffle=False, bin=None, rank_calculation=False):
    train = setup_data_loaders()
    valid = setup_data_loaders()
    cache_fname_train = '../cached_feature_activations_train'
    cache_fname_valid = '../cached_feature_activations_valid'
    print("Calculating training images feature activations")
    train_ftrs, train_labels = calculate_feature_activations(encoder, train, cache_fname_train, device)
    print("Calculating validation images feature activations")
    valid_ftrs, valid_labels = calculate_feature_activations(encoder, valid, cache_fname_valid, device)
    print('train:',train_ftrs.shape, train_labels.shape)
    print('valid:',valid_ftrs.shape, valid_labels.shape)
    

    #get the spurious features by class
    aggregated_data_path = 'data_annotations/spurious_imagenet_classes.csv'
    aggregated_human_labels = pd.read_csv(aggregated_data_path)
    spurious_features_by_class = calc_spurious_features_by_class(aggregated_human_labels)
    print([spurious_features_by_class[i] for i in list(spurious_features_by_class.keys())[:5]])
    print('output check', len(spurious_features_by_class), ', should be 357 for imagenet.')

    #train spuriosity calculation 
    #spuriosity_data_train = calculate_spuriosity_per_class(train_ftrs, train_labels, spurious_features_by_class)
    #binned_img_idx_train = bin_by_spuriosity(spuriosity_data_train, train_labels, spurious_features_by_class)
    np.save('../data_annotations/train_spuriosity_per_image.npy', spuriosity_data_train)
    np.save('../data_annotations/train_binned_image_indices.npy', binned_img_idx_train)
    
    #validation spuriosity calcuation
    spuriosity_data_valid = calculate_spuriosity_per_class(valid_ftrs, valid_labels, spurious_features_by_class)
    binned_img_idx_valid = bin_by_spuriosity(spuriosity_data_valid, valid_labels, spurious_features_by_class)
    np.save('../data_annotations/valid_spuriosity_per_image.npy', spuriosity_data_valid)
    np.save('../data_annotations/valid_binned_image_indices.npy', binned_img_idx_valid)

