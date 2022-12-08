#!/bin/bash

# entire script fails if a single command fails
set -e

# script should be run from the project directory
PROJECT_DIR="$PWD"

# project should have a src directory
SRC_DIR="$PROJECT_DIR"
JOB_RESULTS_DIR="$PROJECT_DIR"/contextual_EEG_high_freq_all_subjs_logs
OPTIMIZED_DATASET_DIR="$PROJECT_DIR"/contextual_all_subjs_EEG_high_freq_optimized_params_december

mkdir -p "$JOB_RESULTS_DIR"
mkdir -p "$OPTIMIZED_DATASET_DIR"

CQL_ALPHA=0.1
INFEASIBLE_ALPHA=0.05
NUM_VOTES=1
TRAIN_STEPS=20001
BATCH_SIZE=512
TOTAL_STEPS=120000

JOB_NAME=PRIME_contextual_all_subjs_EEG_high_freq_${TOTAL_STEPS}_grad_steps_${NUM_VOTES}_votes_${CQL_ALPHA}_cql_alpha_${INFEASIBLE_ALPHA}_infeasible_alpha

sbatch --job-name "$JOB_NAME" "$SRC_DIR"/train_contextual_prime_NAS_EEG_positive_COM_mixed_split_high_freq.sbatch --cql_alpha ${CQL_ALPHA} --infeasible_alpha ${INFEASIBLE_ALPHA} --num_votes ${NUM_VOTES} --train_steps ${TRAIN_STEPS} --batch_size ${BATCH_SIZE}

echo "Contextual training PRIME BCI EEG 2a high-freq for all subjects/multi-context optimization -> 100k grad. steps."