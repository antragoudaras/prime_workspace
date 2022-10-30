#!/bin/bash --login
#SBATCH --time=26:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100:1
#SBATCH --cpus-per-gpu=6 
#SBATCH --mem=64G 
#SBATCH --constraint=intel
#SBATCH --partition=batch
#SBATCH --mail-type=ALL
#SBATCH --output=./photonics_negative_mixed_80_split_logs/%x-%j-slurm.out
#SBATCH --error=./photonics_negative_mixed_80_split_logs/%x-%j-slurm.err

set -e
module purge
module load cuda/11.2.2
module load cudnn/8.1.1-cuda11.2.2 
# Load software stack
conda activate ./env

python prime_photonics_negative_mixed_split.py