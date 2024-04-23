###
# Author: Vipul Sharma
# Function: calculate feature activations and then spuriosity of each image based on it and bin the data into low,med and high spuriosity
###

# solve the problem of relative imports in python
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

