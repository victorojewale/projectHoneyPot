
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, Lambda, ToPILImage
from torch.utils.data import DataLoader
from PIL import Image
from configs.config import Config
from salient_imagenet_data_loader import SalientImageNet  

def setup_data_loaders(bin):
    config = Config()  

    # Define image transformations
    transform = Compose([
        Lambda(lambda x: x.convert("RGB") if x.mode != "RGB" else x),
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    if bin not in [0, 1, 2]:
        raise ValueError(f"Invalid bin type '{bin}'. Expected 0 (low), 1 (medium), or 2 (high).")



    train_dataset = SalientImageNet(
        bin=bin,
        bin_file_path=config.bin_file_path_train,
        rank_calculation=False,
        spurious_classes_path=config.spurious_classes_path,
        root=config.local_data_path,
        split='train',
        transform=transform
    )
    val_dataset = SalientImageNet(
        bin=bin,
        bin_file_path=config.bin_file_path_val,
        rank_calculation=False, 
        spurious_classes_path=config.spurious_classes_path,
        root=config.local_data_path,
        split='val',
        transform=transform
    )


    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    return train_loader, val_loader
