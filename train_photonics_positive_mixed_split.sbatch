#!/bin/bash --login
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100:1
#SBATCH --cpus-per-gpu=4 
#SBATCH --mem=64G 
#SBATCH --constraint=intel
#SBATCH --partition=batch
#SBATCH --mail-type=ALL
#SBATCH --output=./results_photonics_positive_mixed_80_split_correct_param_4/%x-%j-slurm.out
#SBATCH --error=./results_photonics_positive_mixed_80_split_correct_param_4/%x-%j-slurm.err

set -e
module purge
module load cuda/11.2.2
module load cudnn/8.1.1-cuda11.2.2 
# Load software stack
conda activate ./env

python prime_photonics_positive_mixed_split.py