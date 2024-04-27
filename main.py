# main.py

from models.model_manager import ModelManager
from visualizations.visualization import plot_accuracies
from configs.config import Config
from data_handler.data_loader import setup_data_loaders
import os

def main():
    config = Config()
    bin_type = 0
    train_loader, val_loader = setup_data_loaders(bin=bin_type)
    manager = ModelManager(config)
    manager.set_loaders(train_loader, val_loader)
    manager.train_model()
    #plot_save_path = os.path.join(os.getcwd(), 'training_validation_accuracy.png')
    #plot_accuracies(manager.train_accuracies, manager.val_accuracies, save_path=plot_save_path)

if __name__ == "__main__":
    main()




#torch>=1.7.1
#torchvision>=0.8.2
#datasets>=1.2.1
#matplotlib>=3.3.3
#Pillow
