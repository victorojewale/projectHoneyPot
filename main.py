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
import torch


def main(rank, world_size):
    setup(rank, world_size)
    #os.environ['CUDA_VISIBLE_DEVICES'] = str(rank) 
    torch.cuda.set_device(rank)
    config = Config()

    bin_type = 2  
    print("starting loader...")
    train_loaderFull, val_loaderFull = setup_data_loaders()
    train_loader, val_loader = setup_data_loaders(bin=bin_type)
    print(f"Loader done loading..., len of train in process with rank {rank} is ", len(train_loader))
    manager = ModelManager(config, train_loader, val_loaderFull, rank)
    print("This is the ranking", rank)
    print(f"Process {rank}: CUDA device ID:", torch.cuda.current_device())
    manager.train_model(rank, world_size)  


    plot_save_path = os.path.join(os.getcwd(), f'training_validation_accuracy_bin_{bin_type}.png')
    plot_accuracies(manager.train_accuracies, manager.val_accuracies, save_path=plot_save_path)

    print(f"Plot has been saved to: {plot_save_path}")
    cleanup()

def setup(rank, world_size): 
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('gloo', rank=rank, world_size=world_size)

def cleanup(): 
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print("starting code...with gpus:", world_size)
    mp.spawn(main, args=(world_size,), nprocs=world_size)

#torch>=1.7.1
#torchvision>=0.8.2
#datasets>=1.2.1
#matplotlib>=3.3.3
#Pillow
