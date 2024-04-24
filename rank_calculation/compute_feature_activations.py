####
# Author - Le Sharma de Vipul
# Function - given a dataset path and a model, compute the pairs (feature activations of penultimate layer, class label)
# input: dataset path, model path
# output: 2 numpy objects as described below,
#         feature_activations ->  (num_of_images, feature_activations of penultimate layer)
#         class_labels -> (num_of_images, 1), values in  0-999 for imagenet
####

import numpy as np
import pandas as pd
import torch
import torchvision
import pickle
from tqdm import tqdm
# solve the problem of relative imports in python
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from data_handler.data_loader import setup_data_loaders
from collections import OrderedDict

'''below 3 functions are not my own, they are copied from https://github.com/mmoayeri/spuriosity-rankings/blob/main/spuriosity_rankings.py'''
def cache_data(cache_path, data_to_cache):
    os.makedirs('/'.join(cache_path.split('/')[:-1]), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(data_to_cache, f)

def load_cached_data(cache_path):
    with open(cache_path, 'rb') as f:
        dat = pickle.load(f)
    return dat

def get_encoder(model_path = '../models/robust_resnet50.pth', device='cuda', architecture='resnet50'): 
        full_model_dict = torch.load(model_path, map_location=torch.device(device))['model']
        model = torchvision.models.get_model(architecture)

        # Reformat model_dict to be compatible with torchvision
        model_keys = [k for k in full_model_dict if 'attacker' not in k and 'normalizer' not in k]
        model_dict = dict({k.split('module.model.')[-1]:full_model_dict[k] for k in model_keys})
        model.load_state_dict(model_dict)
        
        normalizer_keys = [k for k in full_model_dict if 'attacker' not in k and 'normalizer' in k]
        normalizer_dict = dict({k.split('_')[-1]:full_model_dict[k] for k in normalizer_keys})
        normalizer = torchvision.transforms.Normalize(mean=normalizer_dict['mean'], std=normalizer_dict['std'])
        
        encoder = torch.nn.Sequential(
            OrderedDict([
                ('normalizer',normalizer), 
                *list(model.named_children())[:-1], 
                ('flatten', torch.nn.Flatten())
            ])
        )
        return encoder.eval().to(device)


def calculate_feature_activations(encoder, loader, cache_fname, device='cuda'): 
    """
    Expects model to have a fn 'forward_features' that maps inputs to features.
    Models from the timm library already have this built-in.

    Expects loader to return (inputs, labels) from a classification dataset.
    """
    if not os.path.exists(cache_fname):
        all_ftrs, labels = [], []
        encoder = encoder.eval().to(device)
        for dat in loader:
            x, y = dat['image'].to(device), dat['label'].to(device)
            with torch.no_grad():
                ftrs = encoder(x.to(device)).flatten(1)
                all_ftrs.extend(ftrs.detach().cpu().numpy())
                labels.extend(y)
        ftrs, labels = [np.array(x) for x in [all_ftrs, labels]]
        # encoder = encoder.cpu()

        dat = dict({'ftrs': ftrs, 'labels': labels})
        cache_data(cache_fname, dat)
    else:
        dat = load_cached_data(cache_fname)
        ftrs, labels = [dat[x] for x in ['ftrs', 'labels']]
    return ftrs, labels

if __name__ == '__main__':
    model_path = '../models/robust_resnet50.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    architecture = 'resnet50'
    encoder = get_encoder(model_path, device, architecture)
    train, valid = setup_data_loaders()
    cache_fname_train = '../cached_feature_activations_train'
    cache_fname_valid = '../cached_feature_activations_valid'
    #train_ftrs, train_labels = calculate_feature_activations(encoder, train, cache_fname_train, device)
    valid_ftrs, valid_labels = calculate_feature_activations(encoder, valid, cache_fname_valid, device)
    #print('train:',train_ftrs.shape, train_labels.shape)
    print('valid:',valid_ftrs.shape, valid_labels.shape)
    