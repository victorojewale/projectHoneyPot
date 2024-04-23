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


def get_encoder(model_path = 'models/robust_resnet50.pth'): 
        #self.model.eval()
        #with torch.no_grad():
        #    outputs = self.model(inputs)
        #return outputs
        return None


def calculate_feature_activations(dataset_path, model_path): 
    return None

if __name__ == '__main__':
    model_path = 'models/robust_resnet50.pth'
    encoder = get_encoder(model_path)