import gc
import os
import torch
import torchvision
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict
from data_handler.salient_imagenet_data_loader import setup_data_loaders


def cache_data(cache_path, data_to_cache):
    """Cache feature activations and labels into a CSV file."""
    column_names = [f'Input.feature{i}' for i in range(data_to_cache['ftrs'].shape[1])]
    df = pd.DataFrame(data_to_cache['ftrs'], columns=column_names)
    df['Input.class_index'] = data_to_cache['labels']
    df['Image.file_name'] = data_to_cache['fnames']
    df = df[['Input.class_index', 'Image.file_name'] + column_names]
    df.to_csv(cache_path, mode='a', header=not os.path.exists(cache_path), index=False)


def count_rows(filename):
    """Count the number of rows in a CSV file."""
    with open(filename, 'r') as file:
        num_rows = sum(1 for line in file)
    return num_rows

# Function to initialize an encoder model by excluding the classification layer
def get_encoder(model_path, architecture, device='cuda'):
    """Initialize and return an encoder model by removing the final classification layer."""
    # Load the model architecture without pre-trained weights
    model = getattr(torchvision.models, architecture)(pretrained=False)
    full_model_dict = torch.load(model_path, map_location=torch.device(device))
    
    # Load state dictionary from the trained model file
    model_keys = [k for k in full_model_dict if 'attacker' not in k and 'normalizer' not in k]
    model_dict = dict({k.split('module.')[-1]:full_model_dict[k] for k in model_keys})
    model.load_state_dict(model_dict)

    encoder = torch.nn.Sequential(
        OrderedDict([
            *list(model.named_children())[:-1],  # Exclude the final classification layer
            ('flatten', torch.nn.Flatten())     
        ])
    )


    return encoder.eval().to(device)

# Function to calculate feature activations and cache them
def calculate_feature_activations(encoder, loader, cache_fname, device='cuda'):
    """Calculate feature activations and save them in a CSV."""
    if not os.path.exists(cache_fname):
        encoder = encoder.eval().to(device)
        batch_num = 0
        for batch in loader:
            inputs, labels, fnames = batch['image'].to(device), batch['label'], batch['file_name']
            with torch.no_grad():
                features = encoder(inputs).flatten(1)
            features = np.array(features.cpu().numpy())
            labels = np.array(labels)
            fnames = np.array(fnames)

            print(f"Batch {batch_num} processed.")
            batch_num += 1

            data = {'ftrs': features, 'labels': labels, 'fnames': fnames}
            cache_data(cache_fname, data)

            del data
            gc.collect()

    num_rows = count_rows(cache_fname)
    print(f"Number of rows in {cache_fname}: {num_rows}")
    gc.collect()
    return num_rows

if __name__ == '__main__':
    # Paths to the fine-tuned models
    model_paths = [
        './models/resnet50_high_best.pth',
        './models/resnet50_mid_best.pth',
        './models/resnet50_low_best.pth'
    ]

    # Set device and architecture
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    architecture = 'resnet50'

    # Control: Feature activations for validation bin 1
    val_bin = 1
    val_loader_bin1, _ = setup_data_loaders(bin=val_bin)

    for i, model_path in enumerate(model_paths):
        encoder = get_encoder(model_path, architecture, device)
        cache_fname_bin1 = f'./feature_activations_data/model{i + 1}_bin_{val_bin}.csv'
        print(f"Calculating feature activations for bin {val_bin} using model {i + 1}")
        _ = calculate_feature_activations(encoder, val_loader_bin1, cache_fname_bin1, device)
        if _:
            print(f"Feature activations saved to {cache_fname_bin1}")

    # Test: Feature activations for all spurious validation sets
    val_loader_spurious, _ = setup_data_loaders(rank_calculation=True)

    for i, model_path in enumerate(model_paths):
        encoder = get_encoder(model_path, architecture, device)
        cache_fname_spurious = f'./feature_activations_data/model{i + 1}_spurious.csv'
        print(f"Calculating feature activations for spurious sets using model {i + 1}")
        _ = calculate_feature_activations(encoder, val_loader_spurious, cache_fname_spurious, device)
        if _:
            print(f"Feature activations saved to {cache_fname_spurious}")
