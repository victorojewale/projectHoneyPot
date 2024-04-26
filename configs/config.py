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
        self.local_data_path = 'absolute path of the local imagenet dataset'
        
