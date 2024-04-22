###
# Author - Vipul Sharma 
# Function - Takes as input a csv file containing feature and class wise annotations and aggregates the data across multiple human annotations class-wise

# Example csv: 
# WorkerId,Input.wordnet_id,Input.class_index,Input.feature_index,Input.feature_rank,Answer.main,Answer.confidence,Answer.reasons
# 52,n01440764,0,1696,5,main_object,4.0,focus is on body of tench
# 229,n01440764,0,1696,5,main_object,5.0,The focus of the red region in the Highlighted visual attributes is on tench fish.
# 23,n01440764,0,1696,5,separate_object,5.0,focus in on the hands of a man and tench.
# 82,n01440764,0,1696,5,separate_object,4.0,Focus is on above the fins of the Tinca tinca.
# 186,n01440764,0,1696,5,main_object,5.0,all images have focus on main body part of Tinca tinca (object)

# 52,n01484850,2,2034,1,background,4.0,focus is on water in the background of great white shark
# 82,n01484850,2,1697,2,separate_object,4.0,Focus is on the water of the ocean around the great white shark.

# feature classification logic:
# core feature -> main_object, spurious feature -> background, separate_object

# vote aggregation logic: 
# Moayeri et Al. 2023, Appendix I.2.1: "for which majority of workers selected either background or separate object as the answer were deemed to be spurious"
###

import numpy as np
import pandas as pd

def load_human_labels(filename): 
    df = pd.read_csv(filename)
    return df

def aggregate_class_wise(mturk_labels): 
    return None
def store_aggregated_labels(filename, aggregated_labels): 
    return None



if __name__ == '__main__':
    mturk_data_annotations_path = '../data_annotations/mturk_results_imagenet.csv'
    aggregated_data_path = '../data_annotations/aggregated_imagenet_mturk.csv'
    mturk_labels = load_human_labels(mturk_data_annotations_path)
    print(len(mturk_labels))
    aggregated_labels = aggregate_class_wise(mturk_labels)
    store_aggregated_labels(aggregated_data_path, aggregated_labels)    
    