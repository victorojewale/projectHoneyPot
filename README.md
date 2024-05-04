# Le documentation:    

## rank_calculation:
	- compute_feature_activations.py -> runs 1 iteration on frozen model and gets the last layer activations     
	- spuriosity_metrics.py -> z-score spuriosity calculation         
	- test_process_feature_labels.ipynb -> Unit tests notebook to explore the functionality of this module.      
## process_feature_labels:        
    - annotations_processor.py -> takes as input multi-human labeling results for each (feature,class) pair and aggregates it class-wise.          
	- test_spuriosity_calculation.ipynb -> Unit tests notebook to explore the functionality of this module.      

## Public code implementation for generating dataset: 
https://github.com/mmoayeri/spuriosity-rankings/blob/main/spuriosity_rankings.py

## Environment setup: 
First try:         
conda env create -f full_conda_without_builds_environment.yml      
Second try:         
conda env create -f conda_environment.yml      
Third try:             
Some packages might not be available on conda, use below if that's the case.       
pip install -r pip_requirements.txt       

## Get robust resnet pretrained on imagenet: 
wget -O robust_resnet50.pth  https://www.dropbox.com/s/knf4uimlqsi1yz8/imagenet_l2_3_0.pt?dl=0

## Binning logic:
1. The train is binned as bottom 100, top 100 and medium 100 as sorted by spuriosity. 
2. For val there's only 50 images per class, so it's sorted as bottom 25 percentile, 25-75 percentile and 75 and above percentile.