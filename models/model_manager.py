# models/model_manager.py
import torch
import os
from torch import nn, optim
from torchvision import models
import time
from torch.nn.parallel import DistributedDataParallel as DDP 
import torch.multiprocessing as mp 
from sklearn.metrics import accuracy_score
import wandb

#from data_handler.salient_imagenet_data_loader import setup_data_loaders

class ModelManager:
    def __init__(self, config, train_loader, val_loader, top_val_loader, bottom_val_loader, rank):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu", rank)
        self.rank = rank
        self.model = self.load_model()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.top_val_loader = top_val_loader
        self.bottom_val_loader = bottom_val_loader
        self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=config.learning_rate, momentum = config.weights_decay)
        self.criterion = nn.CrossEntropyLoss()
        self.num_epochs = config.num_epochs
        self.early_stopping_limit = config.early_stopping_limit
        self.train_accuracies = []
        self.val_accuracies = []
        print("Model management with device and rank:", self.device, self.rank)
        wandb.init(project="honeyPot", config=config)
        wandb.watch(self.model, log="all")

        

    def load_model(self, model_path = './models/robust_resnet50.pth', architecture='resnet50'): 
            full_model_dict = torch.load(model_path, map_location=torch.device(self.device))['model']
            model = models.get_model(architecture)

            # Reformat model_dict to be compatible with torchvision
            model_keys = [k for k in full_model_dict if 'attacker' not in k and 'normalizer' not in k]
            model_dict = dict({k.split('module.model.')[-1]:full_model_dict[k] for k in model_keys})
            model.load_state_dict(model_dict)


            # layer3 (third-last residual block), layer4 (last residual block), and the fully connected (fc) layer
            for param in model.layer3.parameters():
                param.requires_grad = False
            for param in model.layer4.parameters():
                param.requires_grad = True
            for param in model.fc.parameters():
                param.requires_grad = True

            # Verify that there are trainable parameters before applying DDP
            trainable_params = sum(p.requires_grad for p in model.parameters())
            if trainable_params == 0:
                raise ValueError("No trainable parameters detected for fine-tuning.")

            model.to(self.device)
            model = DDP(model, device_ids=[self.rank])
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
        init_acc = self.evaluate(self.val_loader, rank)
        
        print(f"{rank}: Init Validation Accuracy", init_acc)
        init_train_acc = self.evaluate(self.train_loader, rank, 'train')
        print(f"{rank}: Init Training Accuracy", init_train_acc)
        top_eval = self.evaluate(self.top_val_loader, rank)
        bottom_eval = self.evaluate(self.bottom_val_loader, rank)
        spuriosity_gap = top_eval - bottom_eval
        print(f"{rank}: Initial spuriosity gap: {spuriosity_gap}, top {top_eval}, bottom {bottom_eval}")
        wandb.log({
                "epoch":  0,
                "train_accuracy": init_train_acc,
                "val_accuracy": init_acc,
                "spuriosity gap": spuriosity_gap
            })
        for epoch in range(self.num_epochs):
            start_time = time.time()
            self.model.train()
            batch_num = 0
            total, correct = 0, 0
            total_loss = 0
            for batch in self.train_loader:
                inputs, labels = batch['image'].to(self.device), batch['label'].to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                batch_total = labels.size(0)
                total += batch_total
                batch_correct = (predicted == labels).sum().item()
                correct += batch_correct
                batch_acc = 100*batch_correct/batch_total
                print(f'Rank {rank}, Batch: {batch_num} Loss: {loss.item()}, Accuracy: {batch_acc}, Time elapsed: {(time.time()-start_time)/60.0}')
                self.train_accuracies.append(batch_acc)
                batch_num += 1
            
            train_acc = 100 * correct / total

            # Evaluate validation set
            val_acc = self.evaluate(self.val_loader, rank)
            top_eval = self.evaluate(self.top_val_loader, rank, mode='sg')
            bottom_eval = self.evaluate(self.bottom_val_loader, rank, mode='sg')
            spuriosity_gap = top_eval - bottom_eval

            wandb.log({
                "epoch": epoch + 1,
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
                "loss": total_loss / len(self.train_loader),
                "spuriosity gap": spuriosity_gap
            })


            #if spuriosity_gap <= 5.0:  
            #    model_save_path = f"{self.config.model_name}_best.pth"
            #    self.save_model(model_save_path)
            #    print(f"Best model updated: {model_save_path} with train acc {train_acc}, val_acc {val_acc}, loss {total_loss/len(self.train_loader)}, sg {spuriosity_gap}%")
            #    break
            model_save_path = f"{self.config.model_name}_best.pth"
            self.save_model(model_save_path)
            end_time = time.time()
            time_taken = end_time - start_time
            print(f'{rank}: Epoch {epoch+1}: Train Acc = {train_acc}%, Val Acc = {val_acc}%, Loss = {total_loss/len(self.train_loader)}, Spuriosity Gap: {spuriosity_gap}, Time taken:{time_taken / 60.0}mins')



    def evaluate(self, loader, rank, mode='val'):
        self.model.eval()
        print(f"{rank} Evaluation in frozen mode, data {mode}")
        correct, total = 0, 0
        batch_num = 0
        with torch.no_grad():
            start_time = time.time()
            for batch in loader:
                inputs, labels = batch['image'].to(self.device), batch['label'].to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                batch_total = labels.size(0)
                total += batch_total
                batch_correct = (predicted == labels).sum().item()
                correct += batch_correct
                batch_acc = 100*batch_correct/batch_total
                if mode=='val': 
                    self.val_accuracies.append(batch_acc)
                elif mode=='train': 
                    self.train_accuracies.append(batch_acc)
                print(f'Rank {rank},batch {batch_num}, Accuracy: {batch_acc}, Time elapsed: {(time.time()-start_time)/60.0}')
                batch_num += 1
        accuracy = 100 * correct / total

        print(f'{rank}: {mode} Accuracy: {accuracy:.2f}%')
        return accuracy

    def save_model(self, save_path):
        """ Save the currently loaded model's state_dict to a file. """
        full_path = os.path.join(self.config.model_dir, save_path)
        torch.save(self.model.state_dict(), full_path)
        print(f"Model saved to {full_path}")
