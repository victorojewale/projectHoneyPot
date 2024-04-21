# main.py

from models.model_manager import ModelManager
from visualizations.visualization import plot_accuracies
from configs.config import Config
import os

def main():
    config = Config()
    manager = ModelManager(config)
    manager.train_model()
    plot_accuracies(manager.train_accuracies, manager.val_accuracies)

if __name__ == "__main__":
    main()




#torch>=1.7.1
#torchvision>=0.8.2
#datasets>=1.2.1
#matplotlib>=3.3.3
#Pillow
