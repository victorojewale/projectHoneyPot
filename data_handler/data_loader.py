from datasets import load_dataset
from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize, Compose, Lambda
from PIL import Image
from torch.utils.data import DataLoader
import io

def get_transform():
    return Compose([
        Lambda(lambda x: Image.open(io.BytesIO(x)) if isinstance(x, bytes) else x),
        Lambda(lambda x: x.convert("RGB") if x.mode != "RGB" else x),
        Resize(224),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def preprocess_data(examples):
    transform = get_transform()
    examples['image'] = [transform(image) for image in examples['image']]
    return examples

def load_data(dataset_name, batch_size=32):
    dataset = load_dataset(dataset_name)
    dataset = dataset.map(preprocess_data, batched=True)
    dataset.set_format(type='torch', columns=['image', 'label'])
    
    train_loader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset['valid'], batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
