#!/bin/bash

#SBATCH --mail-user=victor_ojewale@brown.edu
#SBATCH --mail-type=ALL

#SBATCH --output=/users/vojewale/data_suresh/batch_jobs/output-%j.out
#SBATCH --error=/users/vojewale/data_suresh/batch_jobs/output-%j.err
#SBATCH -p gpu-he 
#SBATCH --gres=gpu:4
#SBATCH --gres-flags=enforce-binding
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH --time=15:00:00
#SBATCH --mem=20G




# Load a CUDA module
module load cuda

module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh 
conda activate honeyPot
export PYTHONUNBUFFERED=TRUE
python3 main.py
