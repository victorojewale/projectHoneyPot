# Import necessary libraries
from datasets import load_dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, Lambda, ToPILImage
from torch.utils.data import DataLoader
from PIL import Image
import io


from configs.config import Config


def setup_data_loaders():
    config = Config()  

    transform = Compose([
        Lambda(lambda x: x.convert("RGB") if x.mode != "RGB" else x),
        Resize(224),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    def preprocess_data(examples):
        # check if images are already PIL Images
        if isinstance(examples['image'][0], Image.Image):
            examples['image'] = [transform(image) for image in examples['image']]
        else:

            examples['image'] = [transform(ToPILImage()(image)) for image in examples['image']]
        return examples


    dataset = load_dataset(config.dataset_name)
    print(dataset.keys())


    dataset = dataset.with_transform(preprocess_data)

    train_loader = DataLoader(dataset['train'], batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(dataset['valid'], batch_size=config.batch_size, shuffle=False)

    return train_loader, val_loader

