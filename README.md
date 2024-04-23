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