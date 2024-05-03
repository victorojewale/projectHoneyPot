# main.py

from models.model_manager import ModelManager
from visualizations.visualization import plot_accuracies
from configs.config import Config
from data_handler.salient_imagenet_data_loader import setup_data_loaders
import os
import torch.distributed as dist
import torch.multiprocessing as mp 
import torch.nn as nn 
import torch.optim as optim 
from torch.nn.parallel import DistributedDataParallel as DDP 



def main():
    config = Config()

    bin_type = 0  
    print("starting loader...")
    train_loader, val_loader = setup_data_loaders(bin=bin_type)
    print("Loader done loading...")
    manager = ModelManager(config, train_loader, val_loader)

    manager.train_model()  


    plot_save_path = os.path.join(os.getcwd(), f'training_validation_accuracy_bin_{bin_type}.png')
    plot_accuracies(manager.train_accuracies, manager.val_accuracies, save_path=plot_save_path)

    print(f"Plot has been saved to: {plot_save_path}")

if __name__ == "__main__":
    print("starting code...")
    main()


#torch>=1.7.1
#torchvision>=0.8.2
#datasets>=1.2.1
#matplotlib>=3.3.3
#Pillow
