# Import necessary libraries
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, Lambda, ToPILImage
from torch.utils.data import DataLoader
from PIL import Image
import io
from torchvision.datasets import ImageNet

class subsetImageNet(ImageNet):
    '''
    Only 357 out of 1k classes are needed for analysis, so don't load rest of the "junk" data.
    '''
    ImageFolder
    is_valid_file -> image_path and return valid or invalid. #willl raise empty class error if done like this. 
    # rewrite my own DatasetFolder and copy and put in current folder. 
    # 
    def __init__(self, spurious_annotations, **kwargs): 
        super().__init__(**kwargs)
    
class SpuriosityCalculationImageNet(subsetImageNet): 
    '''
    This dataloader will be used for calculation of the spuriosity of images class wise. 
    It creates a csv which has class_idx, wordnetID, image name and spuriosity columns. 
    ''' 
    
    __getitem__ use full path and split it to find the last token to get fname (pathlib split)
    def __init__(self, **kwargs): 
        super().__init__(**kwargs)
    
class FineTuningImageNet(subsetImageNet):
    '''
    This dataloader will be used for selecting images as per the spuriosity bins.  
    '''
    def __init__(self, **kwargs): 
        super().__init__(**kwargs)
    is_valid_file -> image_path and return valid or invalid (check on sub-class)

    
def setup_data_loaders(purpose='fine tuning', spilt='val', shuffle=False):
    '''
    Input split (train or val) and shuffle (False for spuriosity calculation, true for training, false for validation).
    Get the corresponding data loader (torch.utils.data.DataLoader) as output.
    '''
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
    
    if purpose=='fine tuning': 
        imagenet_data = FineTuningImageNet(config.local_data_path,
                                                split=spilt,
                                                transform=preprocess_data)
    elif purpose=='spuriosity calculation': 
        imagenet_data = SpuriosityCalculationImageNet(config.local_data_path,
                                                split=spilt,
                                                transform=preprocess_data)
        
    data_loader = torch.utils.data.DataLoader(imagenet_data,
                                          batch_size=config.batch_size,
                                          shuffle=shuffle)
    return data_loader

