#!/bin/bash

# entire script fails if a single command fails
set -e

# script should be run from the project directory
PROJECT_DIR="$PWD"

# project should have a src directory
SRC_DIR="$PROJECT_DIR"
JOB_RESULTS_DIR="$PROJECT_DIR"/leave_one_out_contextual_ECoG_logs
OPTIMIZED_DATASET_DIR="$PROJECT_DIR"/contextual_leave_one_out_ECoG_optimized_params_december

mkdir -p "$JOB_RESULTS_DIR"
mkdir -p "$OPTIMIZED_DATASET_DIR"

CQL_ALPHA=0.1
INFEASIBLE_ALPHA=0.05
NUM_VOTES=1
TRAIN_STEPS=60001
BATCH_SIZE=512
TOTAL_STEPS=100000

JOB_NAME=leave_one_out_PRIME_contextual_ECoG_${TOTAL_STEPS}_grad_steps_mixed_split_${NUM_VOTES}_votes_${CQL_ALPHA}_cql_alpha_${INFEASIBLE_ALPHA}_infeasible_alpha

sbatch --job-name "$JOB_NAME" "$SRC_DIR"/train_leave_one_out_contextual_prime_NAS_ECoG_positive_COM_mixed_split.sbatch --cql_alpha ${CQL_ALPHA} --infeasible_alpha ${INFEASIBLE_ALPHA} --num_votes ${NUM_VOTES} --train_steps ${TRAIN_STEPS} --batch_size ${BATCH_SIZE}

echo "Contextual training PRIME BCI IV ECoG - leave-one-out optimization"