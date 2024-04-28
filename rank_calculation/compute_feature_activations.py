####
# Author - Le Sharma de Vipul
# Function - given a dataset path and a model, compute the pairs (feature activations of penultimate layer, class label)
# input: dataset path, model path
# output: 2 numpy objects as described below,
#         feature_activations ->  (num_of_images, feature_activations of penultimate layer)
#         class_labels -> (num_of_images, 1), values in  0-999 for imagenet
####
import gc
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
from data_handler.salient_imagenet_data_loader import setup_data_loaders
from collections import OrderedDict

'''below 3 functions are modified from https://github.com/mmoayeri/spuriosity-rankings/blob/main/spuriosity_rankings.py'''
def cache_data(cache_path, data_to_cache):
    '''
    input: dat = dict({'ftrs': ftrs, 'labels': y, 'fnames': fname})
    output: a csv file with below columns
    Input.class_index,Image.file_name,Input.feature0,Input.feature1,Input.feature2,Input.feature3,...,Input.feature2047
    '''
    # Generate column names for the 2048 columns
    column_names = [f'Input.feature{i}' for i in range(data_to_cache['ftrs'].shape[1])]
    df = pd.DataFrame(data_to_cache['ftrs'], columns=column_names)
    df['Input.class_index'] = data_to_cache['labels']
    df['Image.file_name'] =  data_to_cache['fnames']
    
    df = df[['Input.class_index', 'Image.file_name'] + column_names]
    # Append DataFrame to CSV file
    df.to_csv(cache_path, mode='a', header=not os.path.exists(cache_path), index=False)

def count_rows(filename):
    with open(filename, 'r') as file:
        num_rows = sum(1 for line in file)
    return num_rows


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
        encoder = encoder.eval().to(device)
        batch_num = 0
        #for train loader, the shuffle needs to be set to false
        for dat in loader:
            x, y, fname = dat[0].to(device), dat[1], dat[2]
            with torch.no_grad():
                ftrs = encoder(x.to(device)).flatten(1)
            ftrs = np.array(ftrs.detach().cpu().numpy())
            y = np.array(y)
            fname = np.array(fname)
            print(f"Batch {batch_num} processed.")
            batch_num+=1
            #print(ftrs.shape, fname.shape, y.shape) #(batch_sz, 2048) (batch_sz,) (batch_sz,)
            dat = dict({'ftrs': ftrs, 'labels': y, 'fnames': fname})
            cache_data(cache_fname, dat)
            del dat
            gc.collect()
    num_rows = count_rows(cache_fname)
    print("Number of rows in", cache_fname, ":", num_rows)
    gc.collect()
    return num_rows

if __name__ == '__main__':
    model_path = '../models/robust_resnet50.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    architecture = 'resnet50'
    encoder = get_encoder(model_path, device, architecture)
    
    finetune_setting = ''
    train_loader, val_loader = setup_data_loaders(rank_calculation=True)
    cache_fname_train = '../feature_activations_data/samples/feature_activations_train_' + architecture + '_' + finetune_setting + '.csv'
    cache_fname_valid = '../feature_activations_data/samples/feature_activations_valid_' + architecture + '_' + finetune_setting + '.csv'
    #feature_activations_train_resnet50_.csv
    #Input.class_index,Image.file_name,Input.feature0,Input.feature1,Input.feature2,Input.feature3,...,Input.feature2047
    print("Calculating training images feature activations")
    _ = calculate_feature_activations(encoder, train_loader, cache_fname_train, device)
    if _: 
        print("Training set feature activation completed and stored in", cache_fname_train)
    
    print("Calculating validation images feature activations")
    _ = calculate_feature_activations(encoder, val_loader, cache_fname_valid, device)
    if _: 
        print("Validation set feature activation completed and stored in", cache_fname_valid)
    