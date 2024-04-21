import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage, Resize, CenterCrop, Normalize, Compose, ToTensor, Lambda
from PIL import Image

# from Hugging Face datasets
dataset = load_dataset("zh-plus/tiny-imagenet")
print(dataset.keys())

# transformations and preprocessing
transform = Compose([
    Lambda(lambda x: x.convert("RGB") if x.mode != "RGB" else x),
    Resize(224),
    CenterCrop(224),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def preprocess_data(examples):
    if isinstance(examples['image'][0], Image.Image):
        # If the images are already PIL images, just apply the transform without ToPILImage()
        examples['image'] = [transform(image) for image in examples['image']]
    else:
        # If the images are not PIL images, convert them to PIL first 
        examples['image'] = [transform(ToPILImage()(image)) for image in examples['image']]
    return examples


dataset = dataset.with_transform(preprocess_data)

#dataset.keys()

# Prepare data loaders
train_loader = DataLoader(dataset['train'], batch_size=32, shuffle=True)
val_loader = DataLoader(dataset['valid'], batch_size=32, shuffle=False)


# loading intheb model

model = models.resnet50(pretrained=True)


# Configure the model for fine-tuning
for param in model.parameters():
    param.requires_grad = False

# replacing the last fully connected layer
model.fc = nn.Linear(model.fc.in_features, 200)  # tinyImgnet has 200 classes


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# optimizer and loss function for the last layer
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# with early stopping
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, early_stopping_limit=5):
    best_acc = 0
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            inputs, labels = batch['image'].to(device), batch['label'].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch['image'].to(device), batch['label'].to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total

        # Early stopping
        if accuracy > best_acc:
            best_acc = accuracy
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == early_stopping_limit:
                print('Early stopping!')
                break

        print(f'Epoch {epoch+1}: Accuracy = {accuracy}%, Loss = {loss.item()}')

train_model(model, train_loader, val_loader, criterion, optimizer)

