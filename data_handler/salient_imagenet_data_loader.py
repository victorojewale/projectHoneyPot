# Import necessary libraries
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, Lambda, ToPILImage
from torch.utils.data import DataLoader
from PIL import Image
from .torchvision_override.imagenet import ImageNet
from configs.config import Config
import pandas as pd
import os
import torch
from torch.utils.data.distributed import DistributedSampler

# is_valid_file(path) takes the whole path, i.e., includes the wordnetID for classes.
#    this function will only use images for the 357 classes which are spurious

# allow_empty(boolean/True) some classes won't have any images, i.e., classes with no spurious features.

# __getitem__ return image file name to keep track of spuriousity rankings for calculating spuriousity, \
#    otherwise just return image, label as normal

class SalientImageNet(ImageNet): 
    def __init__(self, bin=None, bin_file_path=None, rank_calculation=False, spurious_classes_path=None, spuriosity_gap=False, k=10, val_spuriosity_path=None, portion='top', **kwargs): 
        '''
        bin = 'low', 'medium' or 'high'
        bin_file_path = path to csv file containing class, image file_name and the spuriosity rank and the bin to which this class belongs to
        rank_calculation = False or True, if True, get feature activations for 357 spurious classes and assume no binning information available
        
        Either use as a dataloader to get some specific bin of salient imagenet. 
        Or use to to get all spurious classes' data and to calculate binning metadata. 
        Or use it as a vanilla imagenet data loader.
        '''
        #binning functionality
        self.bin = bin
        self.bin_file_path = bin_file_path
        self.image_bin_mapping = None
        #spuriosity gap params
        self.spuriosity_gap = spuriosity_gap
        self.k = k
        self.val_spuriosity_path = val_spuriosity_path
        self.portion = portion
        #spuriousity calculation functionality
        self.rank_calculation = rank_calculation
        self.spurious_classes_path = spurious_classes_path
        self.spurious_classes_features = None
        #spurious classes set, only load images of these classes
        self.spurious_classes_wordnetID = set()
        #print("Init salient imagenet inside of it now!")
        if self.bin is not None: 
            if self.bin_file_path is None: 
                raise Exception("Path to binning information not provided. Use Vanilla imagenet functionality or provide path to csv.")
            self.image_bin_mapping = pd.read_csv(self.bin_file_path)
            #Input.wordnet_id,Input.class_index,image_name,bin_type
            self.spurious_classes_wordnetID = set(self.image_bin_mapping['Input.wordnet_id'])
            if not self.bin in set(self.image_bin_mapping['bin_type']): 
                raise Exception("Invalid bin type passed. Please use 0 for low spurious, 1 for medium and 2 for high spurious.")
            self.image_bin_mapping = self.image_bin_mapping.set_index('image_name')['bin_type'].to_dict()
        #print("Binning if condition ran, now if of rank calculation. Bin file path csv loaded.")
        if self.rank_calculation: 
            if self.spurious_classes_path is None: 
                raise Exception("Path to spurious classes csv not provided. Do you want to use vanilla imagenet instead?")
            self.spurious_classes_features = pd.read_csv(self.spurious_classes_path)
            #Input.wordnet_id,Input.class_index,Input.feature_index,Input.feature_rank,Answer.main
            self.spurious_classes_wordnetID = set(self.spurious_classes_features['Input.wordnet_id'])
        if self.spuriosity_gap: 
            if self.val_spuriosity_path is None: 
                raise Exception("Path to Val spuriousity data csv not provided. Do you want to use vanilla imagenet instead?")
            self.spurious_classes_features = pd.read_csv(self.spurious_classes_path)
            self.spurious_classes_wordnetID = set(self.spurious_classes_features['Input.wordnet_id'])
            class_wordnet_map = self.spurious_classes_features.set_index('Input.class_index')['Input.wordnet_id'].to_dict()
            self.val_spuriosity = pd.read_csv(self.val_spuriosity_path).groupby('Input.class_index')
            self.spuriosity_gap_dict = {} #key is wordnet id and value is a set of valid image names
            for group_name, group_data in self.val_spuriosity: 
                wordnet_key = class_wordnet_map[group_name]
                self.spuriosity_gap_dict[wordnet_key] = self.top_or_bottom(group_data, self.k, self.portion)
            
        if (self.bin is None) and (not self.rank_calculation) and (not self.spuriosity_gap): 
            #neither binning nor rank calculation, vanilla imagenet behavior
            valid_image_check = None
            allow_empty = False
        else: 
            valid_image_check = self.is_valid_file
            allow_empty = True
        
        super().__init__(is_valid_file=valid_image_check, allow_empty=allow_empty, **kwargs)
    
    def top_or_bottom(self, group, k, portion): 
        if portion=='bottom': 
            bottom_k = set(group.nsmallest(k, 'spuriosity')['image_name'])
            return bottom_k
        elif portion=='top': 
            top_k = set(group.nlargest(k, 'spuriosity')['image_name'])
            return top_k
        
    def is_valid_file(self, path): 
        image_class = os.path.basename(os.path.dirname(path))
        fname = os.path.basename(path)
        #class check, only load spurious classes for everything other than vanilla imagenet
        if not image_class in self.spurious_classes_wordnetID: 
            return False
        #only load images corresponding to user passed value of bin type
        if self.bin is not None: 
            #cols: image_name,bin_type
            #print("Valid file check being performed!")
            image_rows = self.image_bin_mapping.get(fname, None)
            if not (image_rows is None) and self.bin==image_rows: 
                return True
            else: 
                return False
        elif self.rank_calculation: 
            return True
        elif self.spuriosity_gap: 
            if fname in self.spuriosity_gap_dict[image_class]: 
                return True
            else: 
                return False
        else: 
            raise Exception("Both bin and rank calculation tried at the same time, this is invalid.")

    def __getitem__(self, index): 
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
            tuple: (sample, target, file_name (str)) in case of spurious ranking calculation
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return {'image': sample, 'label': target, 'file_name': os.path.basename(path)}
        

def convert_to_rgb(image):
    return image.convert("RGB") if image.mode != "RGB" else image
    
def setup_data_loaders(bin=None, rank_calculation=False, spuriosity_gap=False, k=10):
    '''
    3 possible type of data loaders returned: 
    
    1. bin [int] : some bin type as defined in binned_imagenet_train/val.csv, 0, 1, 2 as default. 
    If bin given, then it will load images of that bin's spuriosity only and obviously only for 357 spurious classes.
    2. rank_calculation [Boolean] : prepare data loader for rank calculation, i.e, feature activation calculation. 
    Only returns data for 357 spurious classes.
    3. neither bin nor rank_calculation passed : this will work as a normal imagenet data loader
    
    Input Args: 
    split ('train' or 'val') 
    
    Get the corresponding data loader (torch.utils.data.DataLoader) as output.
    '''
    distributed = torch.distributed.is_initialized()
    config = Config() 
    
    transform = Compose([
        Lambda(convert_to_rgb),
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if spuriosity_gap: 
        top_val_imagenet_data = SalientImageNet(bin=bin, 
                                    bin_file_path=config.bin_file_path_val,
                                    rank_calculation=rank_calculation, 
                                    spurious_classes_path=config.spurious_classes_path,
                                    spuriosity_gap=spuriosity_gap,
                                    k=k,
                                    val_spuriosity_path=config.val_spuriosity_path,
                                    portion='top',
                                    root=config.local_data_path,
                                    split='val',
                                    transform=transform)    
        top_val_sampler = DistributedSampler(top_val_imagenet_data) if distributed else None 
        top_val_loader = DataLoader(top_val_imagenet_data, batch_size=config.batch_size, sampler=top_val_sampler, num_workers =8)
        
        bottom_val_imagenet_data = SalientImageNet(bin=bin, 
                                    bin_file_path=config.bin_file_path_val,
                                    rank_calculation=rank_calculation, 
                                    spurious_classes_path=config.spurious_classes_path,
                                    spuriosity_gap=spuriosity_gap,
                                    k=k,
                                    val_spuriosity_path=config.val_spuriosity_path,
                                    portion='bottom',
                                    root=config.local_data_path,
                                    split='val',
                                    transform=transform)    
        bottom_val_sampler = DistributedSampler(bottom_val_imagenet_data) if distributed else None 
        bottom_val_loader = DataLoader(bottom_val_imagenet_data, batch_size=config.batch_size, sampler=bottom_val_sampler, num_workers =8)
        print("Top and bottom val loaders for spuriosity gap calculation worked, returning and exiting.")
        return top_val_loader, bottom_val_loader
    #print("Breaking inside setup data loader function")
    val_imagenet_data = SalientImageNet(bin=bin, 
                                    bin_file_path=config.bin_file_path_val,
                                    rank_calculation=rank_calculation, 
                                    spurious_classes_path=config.spurious_classes_path,
                                    root=config.local_data_path,
                                    split='val',
                                    transform=transform)    
    val_sampler = DistributedSampler(val_imagenet_data) if distributed else None 
    #print("Validation imagenet dataset worked, creating train dataset")
    train_imagenet_data = SalientImageNet(bin=bin, 
                                    bin_file_path=config.bin_file_path_train,
                                    rank_calculation=rank_calculation, 
                                    spurious_classes_path=config.spurious_classes_path,
                                    root=config.local_data_path,
                                    split='train',
                                    transform=transform)
    train_sampler = DistributedSampler(train_imagenet_data) if distributed else None    
    #print("Train dataset worked, going onto dataloader")
    val_loader = DataLoader(val_imagenet_data, batch_size=config.batch_size, sampler=val_sampler, num_workers = 8)
    train_loader = DataLoader(train_imagenet_data, batch_size=config.batch_size, sampler=train_sampler, num_workers = 8)
    print("Data loader worked, returning it.")
    return train_loader, val_loader

