# models/model_manager.py
import torch
import os
from torch import nn, optim
from torchvision import models
import time
from torch.nn.parallel import DistributedDataParallel as DDP 
import torch.multiprocessing as mp 

#from data_handler.salient_imagenet_data_loader import setup_data_loaders

class ModelManager:
    def __init__(self, config, train_loader, val_loader):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optim.Adam(self.model.fc.parameters(), lr=config.learning_rate, momentum = config.weights_decay)
        self.criterion = nn.CrossEntropyLoss()
        self.num_epochs = config.num_epochs
        self.early_stopping_limit = config.early_stopping_limit
        self.train_accuracies = []
        self.val_accuracies = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_model(self, model_path = './models/robust_resnet50.pth', architecture='resnet50'): 
            full_model_dict = torch.load(model_path, map_location=torch.device(self.device))['model']
            model = models.get_model(architecture)

            # Reformat model_dict to be compatible with torchvision
            model_keys = [k for k in full_model_dict if 'attacker' not in k and 'normalizer' not in k]
            model_dict = dict({k.split('module.model.')[-1]:full_model_dict[k] for k in model_keys})
            model.load_state_dict(model_dict)
            for param in model.parameters():
                param.requires_grad = False
            model.fc.requires_grad = True
            model = DDP(model, device_ids = [self.device])
            return model
    
    #def load_model(self, model_name, num_classes):
        #model = getattr(models, model_name)(pretrained=True)
        #for param in model.parameters():
        #    param.requires_grad = False
        #model.fc = nn.Linear(model.fc.in_features, num_classes)
        #model.fc.requires_grad = True
        #return model.to(self.device)



    def train_model(self, rank, world_size):
        init_acc = self.evaluate(self.val_loader)
        #epochs_no_improve = 0

        for epoch in range(self.num_epochs):
            start_time = time.time()
            self.model.train()
            batch_num = 0
            for batch in self.train_loader:
                inputs, labels = batch['image'].to(rank), batch['label'].to(rank)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                print("Batch processed, num", batch_num)
                print("BATCH END | Current Timestamp:", time.time())
                batch_num += 1
        

            train_acc = self.evaluate(self.train_loader)
            val_acc = self.evaluate(self.val_loader)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            end_time = time.time()
            time_taken = end_time - start_time
            print("EPOCH END | Current Timestamp:", time.time())
            print("EPOCH END | Time taken:", time_taken, "seconds")
            if val_acc > init_acc - 5:
                #best_acc = val_acc
                self.save_model(f"models/{self.config.model_name}_best.pth")  # Save the best model
                #epochs_no_improve = 0
            else:
                #epochs_no_improve += 1
                #if epochs_no_improve == self.early_stopping_limit:
                print('Early stopping!')
                break

            print(f'Epoch {epoch+1}: Train Acc = {train_acc}%, Val Acc = {val_acc}%')

    def evaluate(self, loader):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in loader:
                inputs, labels = batch['image'].to(self.device), batch['label'].to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total

    def save_model(self, save_path):
        """ Save the currently loaded model's state_dict to a file. """
        full_path = os.path.join(self.config.model_dir, save_path)
        torch.save(self.model.state_dict(), full_path)
        print(f"Model saved to {full_path}")
