# Le documentation:    

## rank_calculation:
	- compute_feature_activations.py -> runs 1 iteration on frozen model and gets the last layer activations     
	- spuriosity_metrics.py -> z-score spuriosity calculation         
## process_feature_labels:        
    - annotations_processor.py -> takes as input multi-human labeling results for each (feature,class) pair and aggregates it class-wise.