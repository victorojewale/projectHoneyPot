# models/model_manager.py
import torch
import os
from torch import nn, optim
from torchvision import models
import time
from torch.nn.parallel import DistributedDataParallel as DDP 
import torch.multiprocessing as mp 
from sklearn.metrics import accuracy_score


#from data_handler.salient_imagenet_data_loader import setup_data_loaders

class ModelManager:
    def __init__(self, config, train_loader, val_loader, rank):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu", rank)
        self.rank = rank
        self.model = self.load_model()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optim.Adam(self.model.module.fc.parameters(), lr=config.learning_rate, weight_decay = config.weights_decay)
        self.criterion = nn.CrossEntropyLoss()
        self.num_epochs = config.num_epochs
        self.early_stopping_limit = config.early_stopping_limit
        self.train_accuracies = []
        self.val_accuracies = []
        print("Model management with device and rank:", self.device, self.rank)
        

    def load_model(self, model_path = './models/robust_resnet50.pth', architecture='resnet50'): 
            full_model_dict = torch.load(model_path, map_location=torch.device(self.device))['model']
            model = models.get_model(architecture)

            # Reformat model_dict to be compatible with torchvision
            model_keys = [k for k in full_model_dict if 'attacker' not in k and 'normalizer' not in k]
            model_dict = dict({k.split('module.model.')[-1]:full_model_dict[k] for k in model_keys})
            model.load_state_dict(model_dict)


            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
            model.to(self.rank)
            model = DDP(model, device_ids = [self.rank])
            return model
    
    #def load_model(self, model_name, num_classes):
        #model = getattr(models, model_name)(pretrained=True)
        #for param in model.parameters():
        #    param.requires_grad = False
        #model.fc = nn.Linear(model.fc.in_features, num_classes)
        #model.fc.requires_grad = True
        #return model.to(self.device)



    def train_model(self, rank, world_size):
        best_acc = 0.0
        epochs_no_improve = 0
        init_acc = self.evaluate(self.val_loader)
        print("Init Accuracy", init_acc)


        for epoch in range(self.num_epochs):
            start_time = time.time()
            self.model.train()
            batch_num = 0
            train_preds, train_labels = [], []
            for batch in self.train_loader:
                try: 
                    inputs, labels = batch['image'].to(self.device), batch['label'].to(self.device)
                except: 
                    raise Exception("Fails in this process with rank:", rank)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                train_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                train_labels.extend(labels.cpu().numpy())


                print(f"Batch processed, num {batch_num}, rank {rank}")
                print("BATCH END | Total Time Elapsed:", (time.time()-start_time)/60.0)
                batch_num += 1
        

            train_acc = accuracy_score(train_labels, train_preds)

            # Evaluate validation set
            val_acc = self.evaluate(self.val_loader)

            # Append accuracies to lists
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)


            if val_acc > best_acc:
                best_acc = val_acc
                epochs_no_improve = 0  
                model_save_path = f"{self.config.model_name}_best.pth"
                self.save_model(model_save_path)
                print(f"Best model updated: {model_save_path} with accuracy: {best_acc}%")
            else:
                epochs_no_improve += 1

                # Implement early stopping if no improvement is detected
                if epochs_no_improve == self.early_stopping_limit:
                    print("Early stopping!")
                    break

            end_time = time.time()
            time_taken = end_time - start_time
            print("EPOCH END | Current Timestamp:", time.time())
            print("EPOCH END | Time taken:", time_taken / 60.0, "minutes")
            print(f'Epoch {epoch + 1}: Train Acc = {train_acc}%, Val Acc = {val_acc}%')



    def evaluate(self, loader):
        self.model.eval()
        print("Evaluating...")
        correct, total = 0, 0
        batch_num = 0
        with torch.no_grad():
            start_time = time.time()
            for batch in loader:
                inputs, labels = batch['image'].to(self.device), batch['label'].to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                print(f"Batch processed, num {batch_num}")
                print("BATCH END | Total Time Elapsed:", (time.time()-start_time)/60.0)
                batch_num += 1
        accuracy = 100 * correct / total
        print(f'Accuracy: {accuracy:.2f}%')
        return accuracy

    def save_model(self, save_path):
        """ Save the currently loaded model's state_dict to a file. """
        full_path = os.path.join(self.config.model_dir, save_path)
        torch.save(self.model.state_dict(), full_path)
        print(f"Model saved to {full_path}")
