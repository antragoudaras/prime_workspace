#!/bin/bash --login
#SBATCH --time=80:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100:1
#SBATCH --cpus-per-gpu=8 
#SBATCH --mem=64G 
#SBATCH --constraint=intel
#SBATCH --partition=batch
#SBATCH --mail-type=ALL
#SBATCH --output=./leave_one_out_contextual_EEG_low_freq_logs/%x-%j-slurm.out
#SBATCH --error=./leave_one_out_contextual_EEG_low_freq_logs/%x-%j-slurm.err

set -e
module purge
module load cuda/11.2.2
module load cudnn/8.1.1-cuda11.2.2 

# Load software stack
conda activate ./env

python leave_one_out_contextual_prime_NAS_EEG_positive_COM_mixed_split_low_freq.py "$@"