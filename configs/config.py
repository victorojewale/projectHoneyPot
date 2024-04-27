# configs/config.py

class Config:
    def __init__(self):
        self.dataset_name = 'zh-plus/tiny-imagenet'
        self.weights_path = './models/robust_resnet50.pth' 
        self.model_name = 'resnet50'
        self.num_classes = 200
        self.batch_size = 32
        self.learning_rate = 0.001
        self.num_epochs = 10
        self.early_stopping_limit = 5
        self.local_data_path = './'
        self.bin_file_path_train='./data_annotations/binned_imagenet_train.csv'
        self.bin_file_path_val='./data_annotations/binned_imagenet_val.csv'
        self.bin_file_path = None
        self.spurious_classes_path='./data_annotations/spurious_imagenet_classes.csv'
        