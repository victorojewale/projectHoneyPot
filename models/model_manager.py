# models/model_manager.py
import torch
from torch import nn, optim
from torchvision import models
from data_handler.data_loader import load_data

class ModelManager:
    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(config.model_name, config.weights_path, config.num_classes)
        self.train_loader, self.val_loader = load_data(config.dataset_name, config.batch_size)
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=config.learning_rate
        )
        self.criterion = nn.CrossEntropyLoss()
        self.num_epochs = config.num_epochs
        self.early_stopping_limit = config.early_stopping_limit
        self.train_accuracies = []
        self.val_accuracies = []

    def load_model(self, model_name, weights_path, num_classes):
        model = getattr(models, model_name)(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        for param in model.parameters():
            param.requires_grad = False
        model.fc.requires_grad = True
        return model

    def train_model(self):
        best_acc = 0
        epochs_no_improve = 0

        for epoch in range(self.num_epochs):
            self.model.train()
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            train_acc = self.evaluate(self.train_loader)
            val_acc = self.evaluate(self.val_loader)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

            if val_acc > best_acc:
                best_acc = val_acc
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve == self.early_stopping_limit:
                    print('Early stopping!')
                    break

            print(f'Epoch {epoch+1}: Train Acc = {train_acc}%, Val Acc = {val_acc}%')

    def evaluate(self, loader):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total



#    def load_model(self, model_name, num_classes):
#        model = getattr(models, model_name)(pretrained=True)
#        for param in model.parameters():
#            param.requires_grad = False
#        model.fc = nn.Linear(model.fc.in_features, num_classes)
#        model.fc.requires_grad = True
#        return model.to(self.device)
