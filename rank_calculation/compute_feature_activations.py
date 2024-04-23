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
import os
import pickle
from tqdm import tqdm

def load_dataset(dataset_path): 
    


def calculate_feature_activations(dataset_path, model_path): 
    

if __name__ == '__main__':
    dataset_path = 'imagenet path'
    model_path = 'robust resnet for imagenet'
    