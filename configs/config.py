# configs/config.py

class Config:
    def __init__(self):
        self.dataset_name = 'zh-plus/tiny-imagenet'
        self.model_dir = './models' 
        self.weights_path = './models/robust_resnet50.pth' 
        self.model_name = 'resnet50_low'
        self.num_classes = 1000
        self.batch_size = 128
        self.learning_rate = 0.001
        self.weights_decay = 0.9
        self.num_epochs = 20
        self.early_stopping_limit = 5
        #self.local_data_path = './sample_imagenet'
        self.local_data_path = './softlinks'
        self.bin_file_path_train='./data_annotations/binned_imagenet_train.csv'
        self.bin_file_path_val='./data_annotations/binned_imagenet_val.csv'
        self.bin_file_path = None
        self.spurious_classes_path='./data_annotations/spurious_imagenet_classes.csv'
        self.val_spuriosity_path = './data_annotations/validation_imagenet_spuriosity.csv'
        
        
